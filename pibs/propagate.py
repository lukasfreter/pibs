#!/usr/bin/env python
import numpy as np
from time import time, sleep
import sys, os
#from util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect
from pibs.util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect

from scipy.integrate import ode
from scipy.interpolate import interp1d

import multiprocessing
from multiprocessing import shared_memory, Pool

import ray # multiprocessing alternative to try out



@ray.remote # ray actor -> store states of blocks in chunks here
class SharedStates:
    def __init__(self, chunksize,ntimes,blocksizes, initial):
        self._shared_rhos = ([np.zeros((blocksizes[nu], ntimes), dtype='complex') for nu in range(len(blocksizes))])
        self._rho0 = initial
        self.chunksize = chunksize
        self.chunk_status = [0]*len(blocksizes) # keep track of how many chunks are done for each block
        
        # store ecact solver timesteps for each block for better interpolation
        self._solver_times = np.zeros((len(blocksizes), ntimes))
        
        
        #print('INIT', (self._shared_rhos)[1].shape, ntimes)
        
    
    def rho0(self,nu):
        return self._rho0[nu]
    
    def get_shared_rhos(self,nu, start=None,end=None):
        if start is None and end is None:
            return self._shared_rhos[nu] # return whole block
        
        return self._shared_rhos[nu][:,start:end]
    
    def get_all(self):
        return self._shared_rhos
    
    def write_shared_rhos(self, nu, start, end, data):
        #print('WRITE', nu, start, end, data.shape, self._shared_rhos[nu][:,start:end].shape, self._shared_rhos[nu].shape)
        self._shared_rhos[nu][:,start:end] = data
        
    def get_chunk_status(self, nu):
        return self.chunk_status[nu]
    
    def increment_chunk_status(self, nu):
        self.chunk_status[nu] += 1
        
    def solver_times(self,nu, start=None, end=None):
        if start is None and end is None:
            return self._solver_times[nu, start:end]
        return self._solver_times[nu,:]
    
    def write_solver_times(self, nu, start, end, data):
        self._solver_times[nu,start:end] = data
        
        
@ray.remote # store global parameters for ray calculations
class GlobalParameters:
    def __init__(self, L,t):
        self._L = L
        self._t = t
        
    def L0(self,nu):
        return self._L.L0[nu]
    def L1(self, nu):
        return self._L.L1[nu]
    def t(self, start=None,end=None):
        if start is None and end is None:
            return self._t
        else:
            return self._t[start:end]

  
class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []
        
class Progress:
    def __init__(self, total, description='', start_step=0):
        self.description = description
        self.step = start_step
        self.end = total-1
        self.percent = self.calc_percent()
        self.started = False

    def calc_percent(self):
        return int(100*self.step/self.end)

    def update(self, step=None):
        # print a description at the start of the calculation
        if not self.started:
            print('{}{:4d}%'.format(self.description, self.percent), end='', flush=True)
            self.started = True
            return
        # progress one step or to the specified step
        if step is None:
            self.step += 1
        else:
            self.step = step
        percent = self.calc_percent()
        # only waste time printing if % has actually increased one integer
        if percent > self.percent:
            print('\b\b\b\b\b{:4d}%'.format(percent), end='', flush=True)
            self.percent = percent
        if self.step == self.end:
            print('', flush=True)


class TimeEvolve():
    """ Class to handle time evolution of the quantum system. There are two
    main methods to solve the time evolution:
        
    time_evolve_block:
        This method solves the block structure block for block sequentially.
        Meaning, starting with the block of highest excitation number nu_max, we solve
        the whole time evolution 
        
                d/dt rho_nu_max = L0_nu_max * rho_nu_max
                
        All states rho_nu_max are stored. Then we can iteratively solve for the lower
        blocks using
        
                d/dt rho_nu = L0_nu * rho_nu + L1_{nu+1} * rho_{nu+1}
        
        If all blocks have been solved, we can compute the desired expectation values.
        
    time_evolve_chunk:
        This method sovles the block structure in a parallel manner. The time
        evolution is segmented into chunks of given number of time steps. In 
        the first time chunk, we solve the system for rho_nu_max. In the second
        time chunk, we solve the system for rho_nu_max, and additionally we can
        solve for rho_nu_max-1 for the first chunk, by using the result of rho_nu_max.
        That way, we maximize the amount of calculations we can do in parallel.
        Further, we can compute the expectation values for every chunk and then 
        discard the states, such that memory usage can be minimized.
        """
        
    def __init__(self, rho, L, indices, tend, dt , atol=1e-5, rtol=1e-5):
        self.tend = tend
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        
        self.result = Results()
        
        # should we store rho and L like this ? Or give as parameters when calling .time_evolve_block?
        self.rho = rho
        self.L = L
        self.indices = indices
        
    
    
    def evolve_numax(self,rho0,t0, chunksize, store_initial):
        """
        Calculate the time evolution of the block nu_max of highest excitation number.
        Elements in this block only couple to other elements in this block.


        Parameters
        ----------
        rho0 : initial state
        t0 : initial time
        chunksize : Number of time steps of the integration
        store_initial : boolean value that determines, if the initial state is 
                    saved as the first state in the time evolution

        Returns
        -------
        rho : time evolved density matrix

        """
        
        nu_max = len(self.indices.mapping_block) - 1
        blocksize= len(self.indices.mapping_block[nu_max])
        
        # initialize rho
        rho = np.zeros((blocksize, chunksize), dtype = complex)
        t = []
        
        # integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0,t0).set_f_params(self.L.L0[nu_max])
        
        # integrate
        if store_initial:
            rho[:,0] = rho0
            t.append(t0)
            n_t = 1
        else:
            n_t=0
        while r.successful() and n_t < chunksize:
            rho_step = r.integrate(r.t+self.dt)
            rho[:,n_t] = rho_step
            t.append(r.t)
            # reset integrator
            #r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu_max])
            
            n_t += 1
            
        return rho,t
    
    
    def evolve_nu(self, rho0_nu, rho_nup1, t0, nu, chunksize, store_initial):
        """
        Time evolution of block nu, which can couple to block nu+1

        Parameters
        ----------
        rho0_nu : initial state of block nu
        rho_nup1 : states of block nu+1 for each time step
        t0 : initial time
        nu : total excitation number
        chunksize : Number of time steps of the integration
        store_initial : TYPE
            DESCRIPTION.

        Returns
        -------
        rho_nu : boolean value that determines, if the initial state is 
                    saved as the first state in the time evolution

        """
        
        blocksize= len(self.indices.mapping_block[nu])
        # initial values
        rho_nu = np.zeros((blocksize, chunksize), dtype = complex)
        
        # integrator 
        r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0_nu, t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rho_nup1[:,0])
        
        if store_initial:
            rho_nu[:,0] = rho0_nu
            n_t = 1
        else:
            n_t=0
        n_nup1 = 1 #
        while r.successful() and n_t<chunksize:
            rho_step = r.integrate(r.t+self.dt)
            rho_nu[:, n_t] = rho_step
            
            # update integrator if n_up1 smaller than chunksize
            if n_nup1 < chunksize:
                r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf',
                        atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu],self.L.L1[nu],rho_nup1[:,n_nup1])
                n_nup1 += 1
            n_t += 1
        return rho_nu
    
    
    def evolve_nu_interp(self, rho0_nu, rho_nup1_func, t0, nu, chunksize, store_initial):
        """
        Time evolution of block nu, which can couple to block nu+1

        Parameters
        ----------
        rho0_nu : initial state of block nu
        rho_nup1_func : interpolation function of calculated rho values one block above
        t0 : initial time
        nu : total excitation number
        chunksize : Number of time steps of the integration
        store_initial : TYPE
            DESCRIPTION.

        Returns
        -------
        rho_nu : boolean value that determines, if the initial state is 
                    saved as the first state in the time evolution

        """
        
        blocksize= len(self.indices.mapping_block[nu])
        # initial values
        rho_nu = np.zeros((blocksize, chunksize), dtype = complex)
        times = np.zeros(chunksize)
        
        # integrator 
        r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0_nu, t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rho_nup1_func)
        
        if store_initial:
            rho_nu[:,0] = rho0_nu
            times[0] = t0
            n_t = 1
        else:
            n_t=0
        while r.successful() and n_t<chunksize:
            rho_step = r.integrate(r.t+self.dt)
            rho_nu[:, n_t] = rho_step         
            times[n_t] = r.t
            n_t += 1
        return rho_nu, times
    
    
    
    @staticmethod 
    def evolve_numax_parallel(nu_max,chunksize,atol, rtol, write_queue, shm_name,shape,dtype,lock):
        """
        Calculate the time evolution of the block nu_max of highest excitation number.
        Elements in this block only couple to other elements in this block.


        Parameters
        ----------
        rho0 : initial state
        t0 : initial time
        chunksize : Number of time steps of the integration
        store_initial : boolean value that determines, if the initial state is 
                    saved as the first state in the time evolution

        Returns
        -------
        rho : time evolved density matrix

        """
        
        with lock:
            # get shared rhos
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            # Create a NumPy array using the shared memory
            shared_rho = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
            
            
            # initial values from global variables
            rho0 = shared_rho[:,0]
            t0 = t[0]
            dt = t[1]-t[0]
            ntimes = len(t)
    
            
            # integrator
            r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
            r.set_initial_value(rho0,t0).set_f_params(L0[nu_max])
            
            # integrate
            n_t = 1
            while r.successful() and n_t < ntimes:
                #print(n_t, rhos[nu_max][:n_t-1])
                rho_step = r.integrate(r.t+dt)
                shared_rho[:,n_t] = rho_step
                #t[n_t] = r.t
                n_t += 1
                
                # put data in queue if chunksize has been reached
                if n_t % chunksize == 0:
                    write_queue.put(n_t-chunksize+1)
                    
                    print(n_t)
            existing_shm.close()

            
    @staticmethod
    def evolve_nu_parallel(nu, chunksize, atol,rtol, read_queue, write_queue, shm_name,shape,shm_name_ff, shape_ff, dtype,lock):
        """
        Time evolution of block nu, which can couple to block nu+1
        
        """
        with lock:
            # get shared rhos
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            # Create a NumPy array using the shared memory
            shared_rho = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
            
            # feedforward
            existing_shm_ff = shared_memory.SharedMemory(name=shm_name_ff)
            # Create a NumPy array using the shared memory
            shared_rho_ff = np.ndarray(shape_ff, dtype=dtype, buffer=existing_shm_ff.buf)
            
    
            ntimes = len(t)
            dt = t[1]-t[0]
            
            FLAG_STOP=False
    
            while FLAG_STOP is False:
                data = read_queue.get()  # get will wait
                if data is None:
                    # Finish analysis
                    FLAG_STOP = True
                else:
                    # chunk nu+1 has been calculated -> calculate this chunk
                    # initial values from global variables. data from read_queue should be the initial index in the rhos array
                    rho0_nu = shared_rho[:,data]
                    t0 = t[data]
                    end_idx = data+chunksize
                    if end_idx > ntimes:
                        end_idx = ntimes
                    rho_feedforward = shared_rho_ff[:,0:end_idx] # since we want to interpolate, it makes sense to always start from time index 0 !
                    t_ff = t[0:end_idx]
                    feedforward_func = interp1d(t_ff, rho_feedforward[:,0:len(t_ff)], bounds_error=False, fill_value='extrapolate') # interpolation function for feedforward
                    
                    # setup integrator
                    r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
                    r.set_initial_value(rho0_nu, t0).set_f_params(L0[nu], L1[nu], feedforward_func)
                    
                    n_t=data
                    if n_t > 0:
                        n_t += 1
                    #print(n_t)
                    #print('First n_t', n_t)

                    chunk_count = 0
                    while r.successful() and (n_t<ntimes and chunk_count < chunksize):
                        rho_step = r.integrate(r.t+dt)
                        shared_rho[:, n_t] = rho_step         
                        n_t += 1
                        chunk_count += 1
                        #print(n_t, chunk_count)
                        if n_t % chunksize == 0: 
                            write_queue.put(n_t - chunksize+1)
                    print('Last n_t', n_t)
                    
                    if n_t >= ntimes:
                        FLAG_STOP=True
            existing_shm.close()
            existing_shm_ff.close()

        
    
    
    def time_evolve_chunk_parallel(self, expect_oper, chunksize = 50, progress=False):
        """ Parallelize chunk propagation. 
        First attempt: Save all the rho states in a global list of arrays, such that
        every process can access that array. The respective initial states can then
        be conveniently read from that array.
        
        """
        if expect_oper == None:
            self.time_evolve_block(save_states=True, progress=progress)
            return
        
        print('Starting time evolution in chunks, parallel...')
        tstart = time()
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        #self.result.rho = [[] for _ in range(num_blocks)]
        self.result.rho = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
        
        # initialize rhos
        for nu in range(num_blocks):
            self.result.rho[nu][:,0] = self.rho.initial[nu]
            
        # setup shared memory of rho for all processes:
        shared_rhos = []
        shms = []
        for nu in range(num_blocks):
            array = self.result.rho[nu]
            shm=(shared_memory.SharedMemory(create=True,size=array.nbytes))
            shms.append(shm)
            shared_array = np.ndarray(array.shape, dtype=array.dtype,buffer=shm.buf)
            np.copyto(shared_array,array)
            shared_rhos.append(shared_array)
            
        # Create a lock for synchronization
        lock = multiprocessing.Lock()    
            
        # initialize expectation values and time
        self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        self.result.t = np.arange(t0,self.tend+self.dt,self.dt)
        self.result.t[0] = t0
        

        
        global t, L0, L1
        L0 = self.L.L0
        L1 = self.L.L1
        t = self.result.t
        
        # list of queues. every process (except numax) needs two queues.
        # numax needs a write queue. numax-1 needs a read queue, which IS the write queue of numax.
        # numax-1 need a write queue, which IS the read queue of numax-2 and so on.
        # technically, numin also only needs one read queue.
        queues = []
        for nu in range(num_blocks):
            queues.append(multiprocessing.Queue())
        
        # initialize processes
        processes = []
        processes.append(multiprocessing.Process(target=self.evolve_numax_parallel, args=(nu_max,chunksize,self.atol, self.rtol, queues[nu_max], shms[nu_max].name,self.result.rho[nu_max].shape,'complex',lock)))
        
        for nu in range(num_blocks-2, -1,-1):
             processes.append(multiprocessing.Process(target=self.evolve_nu_parallel, args=(nu, chunksize, self.atol,self.rtol, queues[nu+1], queues[nu],shms[nu].name,self.result.rho[nu].shape,shms[nu+1].name,self.result.rho[nu+1].shape,'complex',lock)))
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
         
        for q in queues:
            q.close()
        
        self.result.rho = shared_rhos
        
        # clean up shared memory
        # for shm in shms:
        #     shm.close()
        #     shm.unlink()
        
        
        # Now with all rhos, we can calculate the expectation values:         
        for t_idx in range(ntimes):
            for nu in range(num_blocks):
                self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  np.array(self.expect_comp_block([self.result.rho[nu][:,t_idx]],nu, expect_oper)).flatten()

        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        
        
    @staticmethod
    def evolve_nu_parallel2(args_tuple):
        """
        Handles the time evolution of block nu_max and all other blocks nu, too.
        Distinguish between those two cases by checkin,if a feedforward input is given.
        
        This combined function enables the use of the multiprocessing Pool class.

        """
        tstart = time()
        rho0, rhoff_func, t0, nu, chunksize, store_initial = args_tuple
        rtol = 1e-8
        atol = 1e-8
        dt = t[1]-t[0]
        
        if rhoff_func is None: # no feedforward -> block nu_max
            nu_max = nu
            blocksize= len(mapping_block[nu_max])
            
            # initialize rho and times
            rho = np.zeros((blocksize, chunksize), dtype = complex)
            solver_times = np.zeros(chunksize)
        
            # integrator
            r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
            r.set_initial_value(rho0,t0).set_f_params(L.L0[nu_max])
            
            # integrate
            if store_initial:
                rho[:,0] = rho0
                solver_times[0] = t0
                n_t = 1
            else:
                n_t=0
            while r.successful() and n_t < chunksize:
                rho_step = r.integrate(r.t+dt)
                rho[:,n_t] = rho_step
                solver_times[n_t] = r.t
                n_t += 1
            
            # elapsed = time() -tstart
            # print(f'Block numax={nu}: elapsed {elapsed:.2f}s')
            return (rho, solver_times)
        
        else: # not nu_max
            blocksize= len(mapping_block[nu])
            # initial values
            rho_nu = np.zeros((blocksize, chunksize), dtype = complex)
            solver_times = np.zeros(chunksize)
            
            # integrator 
            r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
            r.set_initial_value(rho0, t0).set_f_params(L.L0[nu], L.L1[nu], rhoff_func)
            
            if store_initial:
                rho_nu[:,0] = rho0
                solver_times[0] = t0
                n_t = 1
            else:
                n_t=0
            while r.successful() and n_t<chunksize:
                rho_step = r.integrate(r.t+dt)
                rho_nu[:, n_t] = rho_step         
                solver_times[n_t] = r.t
                n_t += 1
            
            # elapsed = time() - tstart
            # print(f'Block nu={nu}: elapsed {elapsed:.2f}s')


            return (rho_nu, solver_times)
  
        
  
    def time_evolve_chunk_parallel2(self, expect_oper, chunksize = 50, progress=False, save_states=False):
        """ Parallelize and minimize the amount of stored states. In this function, the synchronization
        between processes is done 'by hand'. Meaning, similar to the 'time_evolve_chunk' function,
        we loop through chunks, and for each iteration of the loop, a new parallel pool is being set up
        that makes use of the results from the previous chunk. To minimize memory usage, only one past chunk
        is being saved for each nu.
        """
        
        if expect_oper == None:
            self.time_evolve_block(save_states=True, progress=progress)
            return
       
        print(f'Starting time evolution in chunks (parallel 2), chunk size {chunksize}...')
        tstart = time()
               
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        num_evolutions = int(np.ceil(ntimes/chunksize))+nu_max
        if progress:
            bar = Progress(num_evolutions, description='Time evolution total progress...', start_step=0)
        
        if save_states:
            #self.result.rho = [[] for _ in range(num_blocks)]
            self.result.rho = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
            rhos_lengths = [0] * (num_blocks) # keep track of lengths of rhos, if the states are being saved

        # setup rhos as 2 times the chunksize, such that there is space for one chunk for feedforward,
        # and one chunk for calculating the future time evolution.
        saved_chunks=2 # number of stored chunks. Minimum amound is 2: One for feedforward, and one for the next time evolution
        rhos_chunk = [np.zeros((len(self.indices.mapping_block[i]),saved_chunks*chunksize), dtype=complex) for i in range(num_blocks)] 
        
        # if we want to save the exact solver time steps:
        times_chunk = np.zeros((num_blocks, saved_chunks*chunksize))
        
        # can maybe get rid of this, not sure
        #times =[np.zeros(ntimes) for _ in range(num_blocks)] # keep track of times from all blocks for interpolation (because the solver has variable time steps. Though the result is very very small, it is noticable)
        
        # keep track, where to write and where to read for each block
        write = [0] * len(self.indices.mapping_block) # zero means, write in first chunk. One means, write in second chunk
            
        # initial states for each block. Needs to be updated after each chunk calculation
        initial = [np.copy(self.rho.initial[nu]) for nu in range(num_blocks)]
        
        # if we want to save the exact solver time steps:
        initial_times = [t0] * num_blocks
        
        # initialize expectation values and time
        self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        self.result.t = np.zeros(ntimes)
        self.result.t = np.arange(t0, self.tend+self.dt,self.dt)
        
        
        # make static variables global for parallel processes to use
        global L, mapping_block, t
        L = self.L
        mapping_block = self.indices.mapping_block
        t = self.result.t
        
        
        # keep track of which blocks are finished:
        finished = [0]*num_blocks

        # Loop until a chunk is outside of ntimes
        count = 0
        
        # The calculation for block nu can only be started after (nu_max-nu) chunks.
        # total computation steps: ntimes + (nu_max-1) * chunksize
        while count * chunksize - (nu_max)*chunksize <= ntimes:    # - (nu_max)*chunktimes, because the nu blocks have to be solved with a time delay      
            # setup arglist for multiprocessing pool
            arglist = [] #rho0, rhoff_func, t0,nu, chunksize, store_initial
            
            # first calculate block nu_max
            if count == 0:
                t0_numax = t0+count*chunksize*self.dt
            else: 
                t0_numax = t0+count*chunksize*self.dt - self.dt # because only in the first chunk, the initial condition does not need to be stored again. 
                                                                # The initial condition of second chunk is the last time of the previous chunk
            
            # IF we store exact solver time steps:
            t0_numax = initial_times[nu_max]                                                    
            
            # only calculate time evolution, if tend has not been reached yet
            if t0_numax < self.tend:
               # print(t0_numax)
                # In the initial chunk, save the initial state. Otherwise, don't save initial state,
                # because it has already been calculated as the final state of the previous chunk
                if count == 0:
                    save_initial = True
                else:
                    save_initial = False
                
                arglist.append((initial[nu_max], None, t0_numax, nu_max, chunksize, save_initial))
            else:
                finished[nu_max] = 1
                

            # get the number of blocks, one can simultaneously propagate in current chunk
            # cannot exceet num_blocks, otherwise grows linearly with number of chunks
            parallel_blocks = min(count+1, num_blocks) 
            
            # repeat what has been done already for nu_max
            for nu in range(nu_max -1,nu_max - parallel_blocks, -1):
                if count - (nu_max - nu) == 0:
                    # for the first chunk, the initial value is already given, so we do not need to calculate them again
                    # that is why t0 and feedforward are shifted by one
                    t0_nu = t0 + (count - (nu_max - nu)) * chunksize*self.dt
                    feedforward = rhos_chunk[nu+1][:,write[nu] * chunksize:(write[nu]+1)*chunksize]
                    t_ff = times_chunk[nu+1,write[nu] * chunksize:(write[nu]+1)*chunksize]
                else:
                    # for all but the first chunk, the initial conditions are stored in the previous chunk,
                    # so the initial conditions are shifted by one timestep
                    t0_nu = t0 + (count - (nu_max - nu)) * chunksize*self.dt-self.dt
                    
                    end_idx = (write[nu]+1)*chunksize-1
                    start_idx = write[nu] * chunksize-1
                    if start_idx < 0 : # initial state is always the last state of previous chunk. If that chunk is written in the second half of rhos_chunk, we need to take care that a negative first index means: take the last element of the array
                        feedforward = np.concatenate((rhos_chunk[nu+1][:,-1:], rhos_chunk[nu+1][:,:end_idx]), axis=1)
                        t_ff = np.concatenate((times_chunk[nu+1, -1:], times_chunk[nu+1,:end_idx]))
                    else:
                        feedforward = rhos_chunk[nu+1][:, start_idx:end_idx]
                        t_ff = times_chunk[nu+1, start_idx:end_idx]
                        
                # IF we store exact solver time steps:
                t0_nu = initial_times[nu]

                #print(nu, count-nu_max+nu)
                if t0_nu < self.tend and not np.isclose(t0_nu, self.tend):
                    # only store initial state, if it is the very first state. Otherwise, initial state is already stored as last state in previous chunk
                    if count - (nu_max-nu) == 0:
                        save_initial = True
                    else:
                        save_initial = False
                    
                    #t_ff2 = np.linspace(t0_nu, t0_nu+chunksize*self.dt-self.dt, chunksize)
                    #print('t_ff', t_ff[0], t_ff[-1], 'ff', feedforward[:,0], feedforward[:,-1])

                    
                    feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate')
                    
                    arglist.append((initial[nu], feedforward_func, t0_nu, nu, chunksize, save_initial))
                else:
                    finished[nu] = 1
                  
                    
            # setup parallel pool
            with Pool() as pool:
                results = pool.map(self.evolve_nu_parallel2, arglist)
            
            
            # Think about if it is necessary, to get the exact solver time steps for the interpolation, or is it fine to just use
            # a linearly spaced t-array for all blocks
            rhos=[]
            solver_times = []
            for res in results:
                rhos.append(res[0])
                solver_times.append(res[1])
            
                
            # how many blocks are already finished?
            f = sum(finished)
            
            # loop through results
            k = 0 # index of result
            for nu in range(nu_max-f, nu_max-len(rhos)-f, -1): # nu_max is first element in rhos, therefore need to loop through in reverse order
                rhos_chunk[nu][:, write[nu] * chunksize:(write[nu]+1)*chunksize] = rhos[k]
                
                # IF we store exact solver time steps:
                times_chunk[nu, write[nu]* chunksize:(write[nu]+1)*chunksize] = solver_times[k]
                    
                k += 1
                
                initial[nu] = rhos_chunk[nu][:,(write[nu]+1) * chunksize-1]
                
                # IF we store exact solver time steps:
                initial_times[nu] = times_chunk[nu,(write[nu]+1) * chunksize-1]
                
                # get part of expectation value corresponding to nu of that chunk
                i=0
                for t_idx in range((count-(nu_max-nu))*chunksize, (count-(nu_max-nu)+1)*chunksize):
                    if t_idx >= ntimes:
                        continue
                    read_index = write[nu] * chunksize
                    self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                        np.array(self.expect_comp_block([rhos_chunk[nu][:,read_index+ i]],nu, expect_oper)).flatten())
                    i = i+1
                # update write variable
                write[nu] = (write[nu] + 1) % saved_chunks # if 1 make it 0, if 0 make it 1
            
            count += 1
            if progress:
                bar.update()

        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        
    


    def time_evolve_chunk_ray(self, expect_oper, chunksize,num_cpus=None, save_states=False):
        """ Parallelize chunk time evolution using module 'ray'
        
        
        Preliminary observations:
            - If only looking at the time evolution of nu_max (i.e. if initial condition is superfluorescent and no dissipation)
            then this ray method is EXACTLY the same as the serial block solver.
            - Do not use global variables in remote function! Somehow, ray exoirts the function to its storage. Better: pack global
            variables in actor classes https://discuss.ray.io/t/an-error-with-function-size-threshold/7361
            
            
        
        """
        
        print(f'Starting time evolution in chunks (RAY), chunk size {chunksize}...')
        tstart = time()
               
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        ray.init(num_cpus=num_cpus) # initialize ray, takes about 3s
    
        # initialize expectation values and time
        self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        self.result.t = np.arange(t0, self.tend+self.dt,self.dt)
        
        # store number of elements in each block
        blocksizes = [len(self.indices.mapping_block[nu]) for nu in range(num_blocks)]
        
        # shared parameters include Liouvillian L and time vector. 
        # It is necessary in ray to store global variables, that are to be shared
        # between processes, in such 'Actor' classes
        shared_params = GlobalParameters.remote(self.L, self.result.t)
        
        # shared memory for the states
        shared_rhos = SharedStates.remote(chunksize, ntimes,blocksizes, self.rho.initial)
        
        arglist = []
        # (is_block_numax, nu,blocksize, chunksize, shared_rho, shared_params)
        arglist.append((True, nu_max, blocksizes[nu_max], chunksize, shared_rhos,shared_params))
        
        for nu in range(nu_max - 1, -1 , -1):
            arglist.append((False, nu, blocksizes[nu], chunksize, shared_rhos, shared_params))
        
        #object_references = [evolve_nu_ray.remote(arglist[nu]) for nu in range(num_blocks)]
        # use exact solver times:
        object_references = [evolve_nu_ray_solver_times.remote(arglist[nu]) for nu in range(num_blocks)]

        
        #object_references = evolve_nu_ray.remote(arglist[0])
        
        ray.get(object_references) # wait until all tasks are done
        
        # get states from shared memory
        self.result.rho = [np.copy(ray.get(shared_rhos.get_shared_rhos.remote(nu))) for nu in range(num_blocks)]
        
        # get part of expectation value corresponding to nu of that chunk
        for t_idx in range(ntimes):
            for nu in range(num_blocks):
                self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([self.result.rho[nu][:,t_idx]],nu, expect_oper)).flatten()
           
        
        ray.shutdown()
        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        
  
    
    
    def time_evolve_chunk(self, expect_oper, chunksize = 50, progress=False, save_states=False):
        """ Parallelize and minimize the amount of stored states """
        if expect_oper == None:
            self.time_evolve_block(save_states=True, progress=progress)
            return
        
        print('Starting time evolution in chunks...')
        tstart = time()
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        if save_states:
            #self.result.rho = [[] for _ in range(num_blocks)]
            self.result.rho = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
            rhos_lengths = [0] * (num_blocks) # keep track of lengths of rhos, if the states are being saved

        # setup rhos as 3 times the chunksize, such that there is space for one chunk for feedforward,
        # and one chunk for calculating the future time evolution.
        saved_chunks=3 # number of stored chunks. Must be no less than 3
        rhos_chunk = [np.zeros((len(self.indices.mapping_block[i]),saved_chunks*chunksize), dtype=complex) for i in range(num_blocks)] 
        
        # can maybe get rid of this, not sure
        #times =[np.zeros(ntimes) for _ in range(num_blocks)] # keep track of times from all blocks for interpolation (because the solver has variable time steps. Though the result is very very small, it is noticable)
        
        # keep track, where to write and where to read for each block
        write = [0] * len(self.indices.mapping_block) # zero means, write in first chunk. One means, write in second chunk
            
        # initial states for each block. Needs to be updated after each chunk calculation
        initial = [np.copy(self.rho.initial[nu]) for nu in range(num_blocks)]

        # initialize expectation values and time
        self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        self.result.t = np.zeros(ntimes)

        # Loop until a chunk is outside of ntimes
        count = 0
        
        # The calculation for block nu can only be started after (nu_max-nu) chunks.
        # total computation steps: ntimes + (nu_max-1) * chunksize
        while count * chunksize - (nu_max)*chunksize <= ntimes:    # - (nu_max)*chunktimes, because the nu blocks have to be solved with a time delay      
            # first calculate block nu_max
            if count == 0:
                t0_numax = t0+count*chunksize*self.dt
            else: 
                t0_numax = t0+count*chunksize*self.dt - self.dt # because only in the first chunk, the initial condition does not need to be stored again. 
                                                                # The initial condition of second chunk is the last time of the previous chunk
            # only calculate time evolution, if tend has not been reached yet
            if t0_numax <= self.tend:
                # In the initial chunk, save the initial state. Otherwise, don't save initial state,
                # because it has already been calculated as the final state of the previous chunk
                if count == 0:
                    save_initial = True
                else:
                    save_initial = False
                    
                # do the time evolution and store it in the appropriate indices of rho_chunk
                newchunk, tchunk = self.evolve_numax(initial[nu_max], t0_numax, chunksize, save_initial)
                rhos_chunk[nu_max][:,write[nu_max]*chunksize:(write[nu_max]+1)*chunksize] = newchunk

                
                # write data from the current chunk into t and rho arrays. Need to take care of boundary
                if (count+1)*chunksize < ntimes:
                    if save_states:
                        self.result.rho[nu_max][:,count*chunksize:(count+1)*chunksize] = newchunk
                    self.result.t[count*chunksize:(count+1)*chunksize] = tchunk
                    #times[nu_max][count*chunksize:(count+1)*chunksize] = tchunk
                    rhos_lengths[nu_max]+=chunksize
                else:
                    idx = 0
                    while rhos_lengths[nu_max] < ntimes:
                        if save_states:
                            self.result.rho[nu_max][:,rhos_lengths[nu_max]] = newchunk[:,idx]
                        self.result.t[rhos_lengths[nu_max]] = tchunk[idx]
                       # times[nu_max][rhos_lengths[nu_max]] = tchunk[idx]
                        rhos_lengths[nu_max]+=1
                        idx+=1

                # update initial value for next chunk
                initial[nu_max] = rhos_chunk[nu_max][:,(write[nu_max]+1) * chunksize-1]

                # get part of expectation value corresponding to nu_max of that chunk
                i=0
                for t_idx in range(count*chunksize, (count+1)*chunksize):
                    if t_idx >= ntimes:
                        continue
                    read_index = write[nu_max] * chunksize

                    self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                        np.array(self.expect_comp_block([rhos_chunk[nu_max][:,read_index+ i]],nu_max, expect_oper)).flatten())
                    i = i+1
                
                # update write variable
                write[nu_max] = (write[nu_max] + 1) % saved_chunks # if 1 make it 0, if 0 make it 1
                               

            # get the number of blocks, one can simultaneously propagate in current chunk
            # cannot exceet num_blocks, otherwise grows linearly with number of chunks
            parallel_blocks = min(count+1, num_blocks) 
            
            # repeat what has been done already for nu_max
            for nu in range(nu_max -1,nu_max - parallel_blocks, -1):
                if count - (nu_max - nu) == 0:
                    # for the first chunk, the initial value is already given, so we do not need to calculate them again
                    # that is why t0 and feedforward are shifted by one
                    t0_nu = t0 + (count - (nu_max - nu)) * chunksize*self.dt
                    feedforward = rhos_chunk[nu+1][:,write[nu] * chunksize:(write[nu]+1)*chunksize]
                else:
                    # for all but the first chunk, the initial conditions are stored in the previous chunk,
                    # so the initial conditions are shifted by one timestep
                    t0_nu = t0 + (count - (nu_max - nu)) * chunksize*self.dt-self.dt
                    
                    # since we store the rho results in an array of length chunksize*saved_chunks, the following can happen:
                    # We want to feedforward the last saved chunk from nu=1, to calculate nu=0. The last saved chunk in nu=1 is in the
                    # first slot of rhos_chunk, meaning the initial rho_nu=1 for nu=0 is actually the last element of the rhos_chunk array.
                    start_idx = write[nu] * chunksize-1
                    end_idx = (write[nu]+1)*chunksize-1
                    
                    if start_idx < 0: # when it is negative 1
                        feedforward = np.array(rhos_chunk[nu+1][:, -1:])
                        feedforward = np.concatenate((feedforward, rhos_chunk[nu+1][:,start_idx+1:end_idx]),axis=1)
                    else:
                        feedforward = rhos_chunk[nu+1][:,write[nu] * chunksize-1:(write[nu]+1)*chunksize-1]
                
                #print(nu, count-nu_max+nu)
                if t0_nu <= self.tend:
                    # only store initial state, if it is the very first state. Otherwise, initial state is already stored as last state in previous chunk
                    if count - (nu_max-nu) == 0:
                        save_initial = True
                    else:
                        save_initial = False
                    
                    # interpolate feed forward from block nu above
                    idx_t0_nu = (np.abs(self.result.t - t0_nu)).argmin() # find index in time array closest to t0_nu
                    #idx_t0_nu = (np.abs(times[nu+1] - t0_nu)).argmin() # find index in time array closest to t0_nu
                    end_idx = idx_t0_nu + chunksize
                    if end_idx > ntimes: 
                        end_idx = ntimes
                    t_ff = self.result.t[idx_t0_nu:end_idx] 
                    #t_ff = times[nu+1][idx_t0_nu:end_idx]
                    feedforward_func = interp1d(t_ff, feedforward[:,0:len(t_ff)], bounds_error=False, fill_value='extrapolate')
                    
                    global interp_chunk
                    interp_chunk = {'interp': feedforward_func,
                                    't': t_ff,
                                    'y': feedforward}
                    
                    newchunk, tchunk = self.evolve_nu_interp(initial[nu],feedforward_func, t0_nu, nu, chunksize, save_initial)
                    rhos_chunk[nu][:, write[nu] * chunksize:(write[nu]+1)*chunksize] = newchunk
                    
                    # save state in result
                    if save_states:
                        if (count - (nu_max - nu)+1)*chunksize < ntimes:
                            self.result.rho[nu][:,(count - (nu_max - nu))*chunksize:(count - (nu_max - nu)+1)*chunksize] = newchunk
                            #times[nu][(count - (nu_max - nu))*chunksize:(count - (nu_max - nu)+1)*chunksize] = tchunk
                            rhos_lengths[nu]+= chunksize
                        else:
                            idx = 0
                            while rhos_lengths[nu] < ntimes:
                                self.result.rho[nu][:,rhos_lengths[nu]] = newchunk[:,idx]
                               # times[nu][rhos_lengths[nu]] = tchunk[idx]
                                rhos_lengths[nu]+=1
                                idx+=1
                            
                            
                    # update initial value for next chunk
                    initial[nu] = rhos_chunk[nu][:,(write[nu]+1) * chunksize-1]
                    
                    # get part of expectation value corresponding to nu of that chunk
                    i=0
                    for t_idx in range((count-(nu_max-nu))*chunksize, (count-(nu_max-nu)+1)*chunksize):
                        if t_idx >= ntimes:
                            continue
                        read_index = write[nu] * chunksize
                        self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                            np.array(self.expect_comp_block([rhos_chunk[nu][:,read_index+ i]],nu, expect_oper)).flatten())
                        i = i+1
                    # update write variable
                    write[nu] = (write[nu] + 1) % saved_chunks # if 1 make it 0, if 0 make it 1

            count = count+1

        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        

    def time_evolve_block(self,expect_oper=None, save_states=None, progress=False):
        """Time evolve initial state using L0, L1 block structure Liouvillian matrices.
        This only works for weak U(1) symmetry.
    
        expect_oper should be a list of operators that each either act on the photon
        (dim_lp Z dim_lp), the photon and one spin (dim_lp*dim_ls X dim_lp*dim_ls), the
        photon and two spins... etc. setup_convert_rho_nrs(X) must have been run with
        X = 0, 1, 2,... prior to the calculation.
    
        progress==True writes progress in % for the time evolution
        
        DO NOT USE THIS FUNCTION. ONLY LEFT HERE FOR COMPARISON. USE time_evolve_block_interp!
        """
        
        print('Starting time evolution serial block ...')
        tstart = time()
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
            
        rhos= [[] for _ in range(num_blocks)] # store all rho for feed forward
        
        if progress:
            bar = Progress((ntimes-1)*num_blocks, description='Time evolution under L...', start_step=1)
        if save_states is None:
            save_states = True if expect_oper is None else False
        if not save_states and expect_oper is None:
            print('Warning: Not recording states or any observables. Only initial and final'\
                    ' compressed state will be returned.')
        
        # first calculate block nu_max
        nu = num_blocks - 1
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu])
        
        #Record initial values
        self.result.t.append(r.t)
        rhos[nu].append(self.rho.initial[nu])

        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+self.dt)
            self.result.t.append(r.t)
            rhos[nu].append(rho)
            
            # reset integrator
           # r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu])
            
            n_t += 1
            
            if progress:
                bar.update()
   
        # Maybe worth including a check for gamma=kappa=0, then we only need to time evolve one block
        # (depending on the initial condition of course)
        
        # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block
        for nu in range(num_blocks-2, -1,-1):
            r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
            r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rhos[nu+1][0])
            
            #Record initial values
            rhos[nu].append(self.rho.initial[nu])
            
            n_t=1
            while r.successful() and n_t<ntimes:
                rho = r.integrate(r.t+self.dt)
                rhos[nu].append(rho)
                
                # update integrator:
                r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf',
                        atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu],self.L.L1[nu],rhos[nu+1][n_t])
                n_t += 1
    
                if progress:
                    bar.update()
                    

        # Now with all rhos, we can calculate the expectation values:
        if expect_oper is not None:
            self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
            if progress:
                bar = Progress(ntimes, description='Calculating expectation values...', start_step=1)
                
            for t_idx in range(ntimes):
                for nu in range(num_blocks):
                    self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  np.array(self.expect_comp_block([rhos[nu][t_idx]],nu, expect_oper)).flatten()
                if progress:
                    bar.update()
        if save_states:
            self.result.rho = rhos
        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        
        
    def time_evolve_block_interp(self,expect_oper=None, save_states=None, progress=False):
        """ Time evolution of the block structure without resetting the solver at each step.
        Do so by interpolating feedforward."""
        
        print('Starting time evolution serial block (interpolation)...')
        tstart = time()
               
        # store number of elements in each block
        num_blocks = len(self.indices.mapping_block)
        blocksizes = [len(self.indices.mapping_block[nu]) for nu in range(num_blocks)]
        nu_max = num_blocks -1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        if progress:
            bar = Progress(2*(ntimes-1)*num_blocks, description='Time evolution under L...', start_step=1)
        if save_states is None:
            save_states = True if expect_oper is None else False
        if not save_states and expect_oper is None:
            print('Warning: Not recording states or any observables. Only initial and final'\
                    ' compressed state will be returned.')
                

        # set up results:
        self.result.t = np.zeros(ntimes)
        if save_states:
            self.result.rho = [np.zeros((blocksizes[nu], ntimes), dtype=complex) for nu in range(num_blocks)]
        else: # only record initial and final states
            self.result.rho = [np.zeros((blocksizes[nu], 2), dtype=complex) for nu in range(num_blocks)]
            
        if expect_oper is not None:
            self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        
        #Record initial values
        self.result.t[0] = t0            
        
        # first calculate block nu_max. Setup integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(self.rho.initial[nu_max],t0).set_f_params(self.L.L0[nu_max])
        
        # temporary variable to store states
        #rhos = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
        #rhos[nu][:,0] = self.rho.initial[nu]

        rho_nu = np.zeros((blocksizes[nu_max], ntimes), dtype = complex)
        rho_nu[:,0] = self.rho.initial[nu_max] 
        
        # using the exact solver times for the feedforward interpolation instead of using linearly spaced time array makes a (small) difference
        solver_times = np.zeros(ntimes)
        solver_times[0] = t0
    
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+self.dt)
            self.result.t[n_t] = r.t
            solver_times[n_t] = r.t
            #rhos[nu][:,n_t] = rho
            rho_nu[:,n_t] = rho
            n_t += 1
            
            if progress:
                bar.update()
                
        # calculate nu_max part of expectation values
        if expect_oper is not None:
            # if progress:
            #     bar = Progress(ntimes, description='Calculating expectation values...', start_step=1)
                
            for t_idx in range(ntimes):
                #self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
                self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rho_nu[:,t_idx]],nu_max, expect_oper)).flatten()
                if progress:
                    bar.update()
        if save_states:
            self.result.rho[nu_max] = rho_nu
        else: # store initial and final states
            self.result.rho[nu_max][:,0] = rho_nu[:,0]
            self.result.rho[nu_max][:,1] = rho_nu[:,-1] 
            
        #self.result.t = np.arange(t0, self.tend+self.dt,self.dt)
   
        
        # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block_interp
        for nu in range(num_blocks-2, -1,-1):           
            #rho_interp = interp1d(self.result.t, rhos[nu+1], bounds_error=False, fill_value="extrapolate") # extrapolate results from previous block
            rho_interp = interp1d(solver_times, rho_nu, bounds_error=False, fill_value="extrapolate") # interpolate results from previous block, rho_nu                  
                       
            r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
            r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rho_interp)
            
            #Record initial value
            #rhos[nu][:,0] = (self.rho.initial[nu])
            # Update rho_nu variable for current block
            rho_nu = np.zeros((blocksizes[nu], ntimes), dtype=complex)
            rho_nu[:,0] = self.rho.initial[nu]
            solver_times[0] = t0
            
            # integrate
            n_t=1
            while r.successful() and n_t<ntimes:
                rho = r.integrate(r.t+self.dt)
                #rhos[nu][:,n_t] = rho
                solver_times[n_t] = r.t
                rho_nu[:,n_t] = rho
                n_t += 1
    
                if progress:
                    bar.update()
                    
            # calculate contribution of block nu to expectation values
            if expect_oper is not None:
                for t_idx in range(ntimes):
                    #self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
                    self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rho_nu[:,t_idx]],nu, expect_oper)).flatten()
                    if progress:
                        bar.update()
            if save_states:
                self.result.rho[nu] = rho_nu
            else:
                self.result.rho[nu][:,0] = rho_nu[:,0]
                self.result.rho[nu][:,1] = rho_nu[:,-1] 

        # # Now with all rhos, we can calculate the expectation values:
        # if expect_oper is not None:
        #     self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        #     if progress:
        #         bar = Progress(ntimes, description='Calculating expectation values...', start_step=1)
                
        #     for t_idx in range(ntimes):
        #         for nu in range(num_blocks):
        #             self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
        #         if progress:
        #             bar.update()
        # if save_states:
        #     self.result.rho = rhos
        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        
    
    
    
    def expect_comp_block(self, rho_list, nu, ops):
        """ Calculate expectation value of ops in block nu"""
        
        num_blocks = len(self.rho.convert_rho_block_dic)

        # converted densities matrices (different operators may have different number of
        # spins; we need a list of reduced density matrices for each number of spins)
        rhos_converted_dic = {}
        
        output = []
        for op in ops:
            # number of spins in the target rdm
            nrs = int(np.log(op.shape[0]//self.indices.ldim_p)/np.log(self.indices.ldim_s))

            if nrs not in self.rho.convert_rho_block_dic:
                raise TypeError('need to run setup_convert_rho_nrs({})'.format(nrs))

            # Only convert compressed matrices in rho
            if nrs not in rhos_converted_dic:
                rhos_converted_dic[nrs] = []
                for count in range(len(rho_list)):
                    rho_nrs = self.rho.convert_rho_block_dic[nrs][nu].dot(rho_list[count])
                    rho_nrs = vector_to_operator(rho_nrs)
                    rhos_converted_dic[nrs].append(rho_nrs)
            
            one_output = [] 
            rhos_converted = rhos_converted_dic[nrs]
            for count in range(len(rho_list)):
                one_output.append(expect(rhos_converted[count], op))
            output.append(one_output)
        return output
    
    
def _intfunc(t, y, L):
    return (L.dot(y))

def _intfunc_block(t,y, L0, L1, y1):
    """ For blocks in block structure that couple do different excitation, described
    by L1 and y1"""    
    return(L0.dot(y) + L1.dot(y1))

def _intfunc_block_interp(t,y,L0,L1,y1_func):
    """ Same as _intfunc_block, but where y1 is given as a function of time"""
    return (L0.dot(y) + L1.dot(y1_func(t)))




@ray.remote
def evolve_nu_ray(arglist):
    """
    Handles the time evolution of block nu_max and all other blocks nu, too.
    Distinguish between those two cases by checkin,if a feedforward input is given.
    
    Used for RAY parallelization

    """
    is_nu_max, nu,blocksize, chunksize, shared_rho, shared_params = arglist
    rtol = 1e-8
    atol = 1e-8
    
    rho0 = ray.get(shared_rho.rho0.remote(nu)) # initial state from shared_rho
    t = ray.get(shared_params.t.remote()) # get time array
    dt = t[1]-t[0]
    t0 = t[0]
    ntimes = len(t)    
        
    if is_nu_max: # then nu=nu_max, so no need for feedforward
        nu_max = nu     
        
        L0 = ray.get(shared_params.L0.remote(nu_max)) # L0 from shared_params
        # initialize rho
        rho = np.zeros((blocksize, ntimes), dtype = complex)
        rho[:,0] = rho0
    
        # integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(rho0,t0).set_f_params(L0)
        
        written = 0 # keep track of last written index
        
        # integrate
        n_t = 1
        while r.successful() and n_t < ntimes:
            rho_step = r.integrate(r.t+dt)
            rho[:,n_t] = rho_step
            
            if n_t % chunksize == 0:
                # write to shared memory and give signal that next block
                # can use data
                shared_rho.write_shared_rhos.remote(nu,n_t-chunksize,n_t, rho[:,n_t-chunksize:n_t])
                shared_rho.increment_chunk_status.remote(nu)
                written += chunksize
                
            n_t += 1
        if written < ntimes: # if last chunk is not exactly full, write the partially filled chunk into the shared memory
            #print('Fill up', written, ntimes, rho.shape)
            shared_rho.write_shared_rhos.remote(nu, written, ntimes, rho[:,written:ntimes])
            shared_rho.increment_chunk_status.remote(nu)
        
    
    else: # not nu_max -> need feedforward from block nu+1
        
        L0 = ray.get(shared_params.L0.remote(nu)) # L0 from shared_params
        L1 = ray.get(shared_params.L1.remote(nu)) # L1 from shared_params

        # initial values
        rho_nu = np.zeros((blocksize, ntimes), dtype = complex)
        rho_nu[:,0] = rho0[nu]
        
        written = 0 # keep track on where in shared array we wrote data to last
        
        # include check: is block above ready? Yes -> calculate rhoff_func. No -> wait
        while ray.get(shared_rho.get_chunk_status.remote(nu)) >= ray.get(shared_rho.get_chunk_status.remote(nu+1)):
            #print(f'Waiting initial {nu}...')
            continue#sleep(0.01)
        
        # after while loop: calculate next block
        start =0 # first time index for interpolation of nu+1 -> 
        end = ray.get(shared_rho.get_chunk_status.remote(nu))*chunksize + chunksize # last time index for interpolation
        if end > ntimes:    # check if time is larger than max. time
            end = ntimes
        feedforward = ray.get(shared_rho.get_shared_rhos.remote(nu+1,start,end)) # feedforward data from nu+1
        t_ff = t[start:end]
        
        # calculate interpolation function of feedforward data to feed into differential equation for block nu
        feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate')
        
        # integrator 
        r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(rho0, t0).set_f_params(L0, L1, feedforward_func)
        
        n_t = 1
        while r.successful() and n_t<ntimes:
            # integrate
            rho_step = r.integrate(r.t+dt)
            rho_nu[:, n_t] = rho_step  
            
            if n_t % chunksize == 0: # update shared data in chunks
                shared_rho.write_shared_rhos.remote(nu,n_t-chunksize,n_t, rho_nu[:,n_t-chunksize:n_t]) # write chunk into shared data
                shared_rho.increment_chunk_status.remote(nu)    # increment the number of chunks that have been calculated
                written += chunksize  # update write index variable
                
                # while loop: wait until block nu+1 has a larger number of solved chunks thatn block nu
                while ray.get(shared_rho.get_chunk_status.remote(nu)) >= ray.get(shared_rho.get_chunk_status.remote(nu+1)):
                    #print(f'Waiting {nu}...')
                    continue#sleep(0.1)
                
                # after while loop: calculate next block by updating integrator
                start =0
                end = ray.get(shared_rho.get_chunk_status.remote(nu))*chunksize + chunksize
                if end > ntimes:
                    end = ntimes
                feedforward = ray.get(shared_rho.get_shared_rhos.remote(nu+1,start,end))
                t_ff = t[start:end]
                
                feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate')
                r = ode(_intfunc_block_interp).set_integrator('zvode', method='bdf', atol=atol, rtol=rtol).set_initial_value(r.y,r.t).set_f_params(L0,L1, feedforward_func)
                     
            n_t += 1    
        if written < ntimes: # if last chunk is not exactly full, write the partially filled chunk into the shared memory
            shared_rho.write_shared_rhos.remote(nu, written, ntimes, rho_nu[:,written:ntimes])
            shared_rho.increment_chunk_status.remote(nu)
            
            
            
@ray.remote
def evolve_nu_ray_solver_times(arglist):
    """
    Handles the time evolution of block nu_max and all other blocks nu, too.
    Distinguish between those two cases by checkin,if a feedforward input is given.
    
    Used for RAY parallelization

    """
    is_nu_max, nu,blocksize, chunksize, shared_rho, shared_params = arglist
    rtol = 1e-8
    atol = 1e-8
    
    rho0 = ray.get(shared_rho.rho0.remote(nu)) # initial state from shared_rho
    t = ray.get(shared_params.t.remote()) # get time array
    dt = t[1]-t[0]
    t0 = t[0]
    ntimes = len(t)    
        
    if is_nu_max: # then nu=nu_max, so no need for feedforward
        nu_max = nu     
        
        L0 = ray.get(shared_params.L0.remote(nu_max)) # L0 from shared_params
        # initialize rho
        rho = np.zeros((blocksize, ntimes), dtype = complex)
        rho[:,0] = rho0
        t_nu = np.zeros(ntimes)
        t_nu[0] = t0
    
        # integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(rho0,t0).set_f_params(L0)
        
        written = 0 # keep track of last written index
        
        # integrate
        n_t = 1
        while r.successful() and n_t < ntimes:
            rho_step = r.integrate(r.t+dt)
            rho[:,n_t] = rho_step
            t_nu[n_t] = r.t
            
            if n_t % chunksize == 0:
                # write to shared memory and give signal that next block
                # can use data
                shared_rho.write_shared_rhos.remote(nu,n_t-chunksize,n_t, rho[:,n_t-chunksize:n_t])
                shared_rho.write_solver_times.remote(nu, n_t-chunksize,n_t, t_nu[n_t-chunksize:n_t])
                shared_rho.increment_chunk_status.remote(nu)
                written += chunksize
                
            n_t += 1
        if written < ntimes: # if last chunk is not exactly full, write the partially filled chunk into the shared memory
            #print('Fill up', written, ntimes, rho.shape)
            shared_rho.write_shared_rhos.remote(nu, written, ntimes, rho[:,written:ntimes])
            shared_rho.write_solver_times.remote(nu, written, ntimes, t_nu[written:ntimes])
            shared_rho.increment_chunk_status.remote(nu)
        
    
    else: # not nu_max -> need feedforward from block nu+1
        
        L0 = ray.get(shared_params.L0.remote(nu)) # L0 from shared_params
        L1 = ray.get(shared_params.L1.remote(nu)) # L1 from shared_params

        # initial values
        rho_nu = np.zeros((blocksize, ntimes), dtype = complex)
        rho_nu[:,0] = rho0[nu]
        t_nu = np.zeros(ntimes)
        t_nu[0] = t0
        
        written = 0 # keep track on where in shared array we wrote data to last
        
        # include check: is block above ready? Yes -> calculate rhoff_func. No -> wait
        while ray.get(shared_rho.get_chunk_status.remote(nu)) >= ray.get(shared_rho.get_chunk_status.remote(nu+1)):
            #print(f'Waiting initial {nu}...')
            continue#sleep(0.01)
        
        # after while loop: calculate next block
        start =0 # first time index for interpolation of nu+1 -> 
        end = ray.get(shared_rho.get_chunk_status.remote(nu))*chunksize + chunksize # last time index for interpolation
        if end > ntimes:    # check if time is larger than max. time
            end = ntimes
        feedforward = ray.get(shared_rho.get_shared_rhos.remote(nu+1,start,end)) # feedforward data from nu+1
        t_nu_plus1 = ray.get(shared_rho.solver_times.remote(nu+1))
        t_ff = t_nu_plus1[start:end]
        #print('t_ff', t_ff[0], t_ff[-1], 'ff', feedforward[:,0], feedforward[:,-1])
        
        
        # calculate interpolation function of feedforward data to feed into differential equation for block nu
        feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate')
        
        # integrator 
        r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(rho0, t0).set_f_params(L0, L1, feedforward_func)
        
        n_t = 1
        while r.successful() and n_t<ntimes:
            # integrate
            rho_step = r.integrate(r.t+dt)
            rho_nu[:, n_t] = rho_step 
            t_nu[n_t] = r.t
            
            if n_t % chunksize == 0: # update shared data in chunks
                shared_rho.write_shared_rhos.remote(nu,n_t-chunksize,n_t, rho_nu[:,n_t-chunksize:n_t]) # write chunk into shared data
                shared_rho.increment_chunk_status.remote(nu)    # increment the number of chunks that have been calculated
                shared_rho.write_solver_times.remote(nu, n_t-chunksize, n_t, t_nu[n_t-chunksize:n_t])
                written += chunksize  # update write index variable
                
                # while loop: wait until block nu+1 has a larger number of solved chunks thatn block nu
                while ray.get(shared_rho.get_chunk_status.remote(nu)) >= ray.get(shared_rho.get_chunk_status.remote(nu+1)):
                    #print(f'Waiting {nu}...')
                    continue#sleep(0.1)
                
                # after while loop: calculate next block by updating integrator
                
                # INTERPOLATION OPTION. from start or from prev block
                
                start =0#ray.get(shared_rho.get_chunk_status.remote(nu))*chunksize-1
                end = ray.get(shared_rho.get_chunk_status.remote(nu))*chunksize + chunksize-1
                if start ==-1:
                    start+=1
                    end+=1
                if end > ntimes:
                    end = ntimes
                feedforward = ray.get(shared_rho.get_shared_rhos.remote(nu+1,start,end))
                t_nu_plus1 = ray.get(shared_rho.solver_times.remote(nu+1))
                t_ff = t_nu_plus1[start:end]
                
               # print('t_ff', t_ff[0], t_ff[-1], 'ff', feedforward[:,0], feedforward[:,-1])
                
                feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate')
                r = ode(_intfunc_block_interp).set_integrator('zvode', method='bdf', atol=atol, rtol=rtol).set_initial_value(r.y,r.t).set_f_params(L0,L1, feedforward_func)
                     
            n_t += 1    
        if written < ntimes: # if last chunk is not exactly full, write the partially filled chunk into the shared memory
            shared_rho.write_shared_rhos.remote(nu, written, ntimes, rho_nu[:,written:ntimes])
            shared_rho.write_solver_times.remote(nu, written,ntimes, t_nu[written:ntimes])

            shared_rho.increment_chunk_status.remote(nu)


if __name__ =='__main__':
    # testing
    from setup import Indices, Rho
    ntls = 7
    nphot = ntls+1
    indi = Indices(ntls)
    rho = Rho(basis(nphot,0), basis(2,0), indi) # initial condition with zero photons and all spins up.
    t = TimeEvolve(rho, 1, indi,1,1)
