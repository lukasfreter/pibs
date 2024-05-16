#!/usr/bin/env python
import numpy as np
from time import time
import sys, os
from util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect
from scipy.integrate import ode
from scipy.interpolate import interp1d

import multiprocessing
interp1 = []
interp_chunk=[]

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
    
    
    
    
    def evolve_numax_parallel(self,rho0,t0, chunksize, store_initial, result_queue):
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
        
        # integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0,t0).set_f_params(self.L.L0[nu_max])
        
        # integrate
        if store_initial:
            rho[:,0] = rho0
            n_t = 1
        else:
            n_t=0
        while r.successful() and n_t < chunksize:
            rho_step = r.integrate(r.t+self.dt)
            rho[:,n_t] = rho_step
            n_t += 1
            
        return rho
    
    
    def time_evolve_chunk_parallel(self, expect_oper, chunksize = 50, progress=False):
        """ Parallelize and minimize the amount of stored states """
        if expect_oper == None:
            self.time_evolve_block(save_states=True, progress=progress)
            return
        
        print('Starting time evolution in chunks, parallel...')
        tstart = time()
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = int(self.tend/self.dt)+1
        
        # setup result for each block
        result_nu = []
        for nu in num_blocks:
            result_nu.append(Results())
            result_nu[nu].expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        
        # setup rhos as double the chunksize, such that there is space for one chunk for feedforward,
        # and one chunk for calculating the future time evolution.
        rhos_chunk = [np.zeros((len(self.indices.mapping_block[i]),2*chunksize), dtype=complex) for i in range(num_blocks)] 
        
        # keep track, where to write and where to read for each block
        write = [0] * len(self.indices.mapping_block) # zero means, write in first chunk. One means, write in second chunk
            
        # initial states for each block. Needs to be updated after each chunk calculation
        initial = [np.copy(self.rho.initial[nu]) for nu in range(num_blocks)]
        
        count = 0       
        # The calculation for block nu can only be started after (nu_max-nu) chunks.
        # total computation steps: ntimes + (nu_max-1) * chunksize
        while count * chunksize - (nu_max-1)*chunksize < ntimes:    # - (nu_max-1)*chunktimes, because the nu blocks have to be solved with a time delay      
           
            # get the number of blocks, one can simultaneously propagate in current chunk
            # cannot exceet num_blocks, otherwise grows linearly with number of chunks
            parallel_blocks = min(count+1, num_blocks) 
            
            processes = []
           # result_queues = [multiprocessing.Queue()

            
            # first calculate block nu_max
            t0_numax = t0+count*chunksize*self.dt
            
            # only calculate time evolution, if tend has not been reached yet
            if t0_numax <= self.tend:
                # In the initial chunk, save the initial state. Otherwise, don't save initial state,
                # because it has already been calculated as the final state of the previous chunk
                if count == 0:
                    save_initial = True
                else:
                    save_initial = False
                    
                # do the time evolution and store it in the appropriate indices of rho_chunk 
                args = (initial[nu_max], t0_numax, chunksize, save_initial, result_queue)
                p = multiprocessing.Process(target = self.evolve_numax_parallel, args=args)
                processes.append(p)

                # get part of expectation value corresponding to nu_max of that chunk
                i=0
                for t_idx in range(count*chunksize, (count+1)*chunksize):
                    if t_idx >= ntimes:
                        continue
                    read_index = write[nu_max] * chunksize

                    self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                        np.array(self.expect_comp_block([rhos_chunk[nu_max][:,read_index+ i]],nu_max, expect_oper)).flatten())
                    self.result.t.append(t_idx*self.dt)
                    i = i+1
                
                # update write variable
                write[nu_max] = (write[nu_max] + 1) % 2 # if 1 make it 0, if 0 make it 1
                
                # x = rhos_chunk[nu_max][:,:]     
                # y = self.result.expect                                  


            
            # repeat what has been done already for nu_max
            for nu in range(nu_max -1,nu_max - parallel_blocks, -1):              
                t0_nu = t0 + (count - (nu_max - nu)) * chunksize*self.dt
                
                if t0_nu <= self.tend:
                    if count - (nu_max-nu) == 0:
                        save_initial = True
                    else:
                        save_initial = False
                    rhos_chunk[nu][:, write[nu] * chunksize:(write[nu]+1)*chunksize] = (
                        self.evolve_nu(initial[nu],rhos_chunk[nu+1][:,write[nu] * chunksize:(write[nu]+1)*chunksize], t0_nu, nu, chunksize, save_initial))
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
                    write[nu] = (write[nu] + 1) % 2 # if 1 make it 0, if 0 make it 1
            
            
            
            # update initial value for next chunk
            initial[nu_max] = rhos_chunk[nu_max][:,(write[nu_max]+1) * chunksize-1]
            
            
            count = count+1

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
        ntimes = int(self.tend/self.dt)+1
        
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
        """
        
        print('Starting time evolution serial block ...')
        tstart = time()
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        t0 = 0
        ntimes = int(self.tend/self.dt)+1
            
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
               
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        t0 = 0
        ntimes = int(self.tend/self.dt)+1
            
        #rhos= [[] for _ in range(num_blocks)] # store all rho for feed forward
        rhos = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
        
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
        self.result.t = np.zeros(ntimes)
        self.result.t[0] = t0
        rhos[nu][:,0] = self.rho.initial[nu]

        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+self.dt)
            self.result.t[n_t] = r.t
            rhos[nu][:,n_t] = rho
            n_t += 1
            
            if progress:
                bar.update()
   
        # Maybe worth including a check for gamma=kappa=0, then we only need to time evolve one block
        # (depending on the initial condition of course)
        
        # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block
        for nu in range(num_blocks-2, -1,-1):           
            rho_interp = interp1d(self.result.t, rhos[nu+1], bounds_error=False, fill_value="extrapolate") # extrapolate results from previous block
            global interp1
            interp1 = {'interp':rho_interp,
                       't': self.result.t,
                       'y':rhos[nu+1]}
                       
                       
            r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
            r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rho_interp)
            
            #Record initial values
            rhos[nu][:,0] = (self.rho.initial[nu])
            
            n_t=1
            while r.successful() and n_t<ntimes:
                rho = r.integrate(r.t+self.dt)
                rhos[nu][:,n_t] = rho
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
                    self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
                if progress:
                    bar.update()
        if save_states:
            self.result.rho = rhos
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




if __name__ =='__main__':
    # testing
    from setup import Indices, Rho
    ntls = 7
    nphot = ntls+1
    indi = Indices(ntls)
    rho = Rho(basis(nphot,0), basis(2,0), indi) # initial condition with zero photons and all spins up.
    t = TimeEvolve(rho, 1, indi,1,1)