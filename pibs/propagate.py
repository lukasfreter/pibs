#!/usr/bin/env python
import numpy as np
from time import time
from .util import  basis, vector_to_operator, expect, Progress
#from pibs.util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect

from scipy.integrate import ode
from scipy.interpolate import interp1d

from multiprocessing import Pool

 
class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []
        self.expect_per_nu = []
        
class TimeEvolve():
    """ Class to handle time evolution of the quantum system. There are two
    main methods to solve the time evolution:
        
    time_evolve_block_interp:
        This method solves the block structure block for block sequentially.
        Meaning, starting with the block of highest excitation number nu_max, we solve
        the whole time evolution 
        
                d/dt rho_nu_max = L0_nu_max * rho_nu_max
                
        All states rho_nu_max are stored. Then we can iteratively solve for the lower
        blocks using
        
                d/dt rho_nu = L0_nu * rho_nu + L1_{nu+1} * rho_{nu+1}
        
        
    time_evolve_chunk_parallel2:
        This method sovles the block structure in a parallel manner. The time
        evolution is segmented into chunks of given number of time steps. In 
        the first time chunk, we solve the system for rho_nu_max. In the second
        time chunk, we solve the system for rho_nu_max, and additionally we can
        solve for rho_nu_max-1 for the first chunk, by using the result of rho_nu_max.
        That way, we maximize the amount of calculations we can do in parallel.
        Further, we can compute the expectation values for every chunk and then 
        discard the states, such that memory usage can be minimized.
        """
        
    def __init__(self, rho, L, tend, dt, atol=1e-5, rtol=1e-5, nsteps=500):
        """
        Initialize

        Parameters
        ----------
        rho : setup.Rho
            Density matrix object
        L : setup.Models
            Liouvillian object
        tend : float
            Final time for time evolution
        dt : float
            Time step
        atol : float, optional
            Absolute tolerance for integrator. The default is 1e-5.
        rtol : float, optional
            Relative tolerance for integrator. The default is 1e-5.
        nsteps : int, optional
            Max iterations for integrator. The default is 500.

        Returns
        -------
        None.

        """
        self.tend = tend
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        
        self.result = Results()
        
        # should we store rho and L like this ? Or give as parameters when calling .time_evolve_block?
        self.rho = rho # this makes just a reference to the rho object.
        self.L = L
        
        # fix this later
        self.indices = rho.indices
        
        self.nsteps=nsteps # number of steps per solver iteration. Standard value is 500 see https://stackoverflow.com/questions/40788747/internal-working-of-scipy-integrate-ode
        
        
        
    @staticmethod
    def evolve_nu_parallel2(args_tuple):
        """
        Handles the time evolution of block nu_max and all other blocks nu, too.
        Distinguish between those two cases by checking,if a feedforward input is given.
        
        This combined function enables the use of the multiprocessing Pool class.

        """
        tstart = time()
        rho0, rhoff_func, t0, nu, chunksize, store_initial, rtol, atol, method = args_tuple
        dt = t[1]-t[0]
        
        if rhoff_func is None: # no feedforward -> block nu_max
            nu_max = nu
            blocksize= len(mapping_block[nu_max])
            
            # initialize rho and times
            rho = np.zeros((blocksize, chunksize), dtype = complex)
            solver_times = np.zeros(chunksize)
        
            # integrator
            r = ode(_intfunc).set_integrator('zvode', method = method, atol=atol, rtol=rtol)
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
  
        
  
    def time_evolve_chunk_parallel(self, expect_oper, chunksize = 50, progress=False, \
                        save_states=False, num_cpus=None, method='bdf', expect_per_nu=False, start_block=None\
                            ,verbose = True):
        """ 
            Parallel time evolution function
        
            Parameters
            ----------
            expect_oper : list
                List of operators for which expectation values are calculated at each time step
            chunksize : int
                Number of time steps to calculate serially in one chunk
            save_states : bool
                True: save states at all times, False: only save initial and final state
            progress : bool
                True: Enable progress bar
            num_cpus : int
                Number of CPUs for parallel computation
            method : string
                Method for integrator
                'bdf' : stiff
                'adams' : non-stiff
            expect_per_nu : bool
                True: Calculate the expectation values for each block separately
            start_block : int
                specify excitation number corresponding to block at which the time evolution should start.
                If None, start_block is set to nu_max, which gives exact result.
            verbose : bool
                True: print status messages
            
            Returns
            -------
            None
        
        Parallelize and minimize the amount of stored states. In this function, the synchronization
        between processes is done 'by hand'. Meaning we loop through chunks, and for each iteration 
        of the loop, a new parallel pool is being set up that makes use of the results from the previous chunk. 
        To minimize memory usage, only one past chunk is being saved for each nu.
        
        """
        
        if self.indices.only_numax:
            print('Parallel time evolution not implemented for only_numax. Starting serial time evolution instead...')
            self.time_evolve_block_interp(expect_oper, progress = progress, save_states=save_states, method=method)
        
        # if expect_oper == None:
        #     self.time_evolve_block(save_states=True, progress=progress)
        #     return
       

               
        num_blocks = len(self.indices.mapping_block)
        nu_max = num_blocks-1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        # determine start block
        if start_block == None or start_block == nu_max:
            start_block = nu_max
        if start_block > nu_max or start_block < 0:
            raise ValueError(f"Start block exciation must be within 0 and {nu_max}.")
        if not isinstance(start_block, (int, np.integer)):
            raise ValueError('Start block excitation must be an integer.')
        if verbose:
            print(f'Start block set to {start_block}/{nu_max}.')
            print(f'Starting time evolution in chunks (parallel 2), chunk size {chunksize}...')
        tstart = time()
        
        num_evolutions = int(np.ceil(ntimes/chunksize))+nu_max # number of loops for progress bar
        # some weird edge case, if chunksize divides ntimes
        if ntimes % chunksize == 0:  
            num_evolutions += 1
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
        if expect_per_nu:
            self.result.expect_per_nu = np.zeros((num_blocks, len(expect_oper),ntimes), dtype=complex)  # idx: (nu, operator, time)
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
                
                arglist.append((initial[nu_max], None, t0_numax, nu_max, chunksize, save_initial, self.rtol, self.atol, method))
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

                    
                    feedforward_func = interp1d(t_ff, feedforward, bounds_error=False, fill_value='extrapolate', kind='cubic')
                    
                    arglist.append((initial[nu], feedforward_func, t0_nu, nu, chunksize, save_initial, self.rtol,self.atol,method))
                else:
                    finished[nu] = 1
                  
                    
            # setup parallel pool
            with Pool(processes = num_cpus) as pool:
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
                    
                    expect_nu = np.array(self.expect_comp_block([rhos_chunk[nu][:,read_index+ i]],nu, expect_oper)).flatten()
                    self.result.expect[:,t_idx] +=  expect_nu 
                    
                    if expect_per_nu:
                        self.result.expect_per_nu[nu, :, t_idx] = expect_nu
                    
                    # self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                    #     np.array(self.expect_comp_block([rhos_chunk[nu][:,read_index+ i]],nu, expect_oper)).flatten())
                    i = i+1
                # update write variable
                write[nu] = (write[nu] + 1) % saved_chunks # if 1 make it 0, if 0 make it 1
            
            count += 1
            if progress:
                bar.update()

        elapsed = time()-tstart
        print(f'Complete {elapsed:.0f}s', flush=True)
        

        
    def time_evolve_block_interp(self,expect_oper=None, save_states=None, progress=False, \
                        method='bdf', expect_per_nu=False, start_block=None, verbose=True):
        """ Serial time evolution function
        
            Parameters
            ----------
            expect_oper : list
                List of operators for which expectation values are calculated at each time step
            save_states : bool
                True: save states at all times, False: only save initial and final state
            progress : bool
                True: Enable progress bar
            method : string
                Method for integrator
                'bdf' : stiff
                'adams' : non-stiff
            expect_per_nu : bool
                True: Calculate the expectation values for each block separately
            start_block : int
                specify excitation number corresponding to block at which the time evolution should start.
                If None, start_block is set to nu_max, which gives exact result.
            verbose : bool
                True: print status messages
            
            Returns
            -------
            None
            
        
        """      
        

               
        # store number of elements in each block
        num_blocks = len(self.indices.mapping_block)
        blocksizes = [len(self.indices.mapping_block[nu]) for nu in range(num_blocks)]
        nu_max = num_blocks -1
        t0 = 0
        ntimes = round(self.tend/self.dt)+1
        
        # determine start block
        if start_block == None or start_block == nu_max:
            start_block = nu_max
        else:
            if self.indices.only_numax:
                print('Warning: Cannot simultaneously set "only_numax" evolution and starting time evolution at block not equal to nu_max.')
                print('Setting "start_block" to nu_max = {nu_max}')
                start_block = nu_max    
                
            
            if start_block > nu_max or start_block < 0:
                raise ValueError(f"Start block exciation must be within 0 and {nu_max}.")
            if not isinstance(start_block, (int, np.integer)):
                raise ValueError('Start block excitation must be an integer.')
            if verbose:
                print(f'Start block set to {start_block}/{nu_max}.')
            
        if verbose:
            print('Starting time evolution serial block (interpolation)...')
        tstart = time()
        
        if progress:
            if self.indices.only_numax:
                bar = Progress(2*ntimes, description='Time evolution under L (only numax)...', start_step=1)
            elif start_block < nu_max:
                bar = Progress(2*(ntimes-1)*(start_block+1), description=f'Time evolution under L (start block: {start_block})...', start_step=1)
            else:
                bar = Progress(2*(ntimes-1)*num_blocks, description='Time evolution under L...', start_step=1)
        if save_states is None:
            save_states = True if expect_oper is None else False
        if not save_states and expect_oper is None:
            print('Warning: Not recording states or any observables. Only initial and final'\
                    ' states will be returned.')
                

        # set up results:
        self.result.t = np.zeros(ntimes)
        if save_states:
            self.result.rho = [np.zeros((blocksizes[nu], ntimes), dtype=complex) for nu in range(num_blocks)]
        else: # only record initial and final states
            self.result.rho = [np.zeros((blocksizes[nu], 2), dtype=complex) for nu in range(num_blocks)]
            
        if expect_oper is not None:
            self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
            
            if expect_per_nu:
                self.result.expect_per_nu = np.zeros((num_blocks, len(expect_oper),ntimes), dtype=complex)  # idx: (nu, operator, time)
        
        #Record initial values
        self.result.t[0] = t0            
        
        # first calculate block nu_max. Setup integrator
        # If start_block is set by user, start with start_block
        r = ode(_intfunc).set_integrator('zvode', method = method, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps)
        
        if self.indices.only_numax:
            r.set_initial_value(self.rho.initial[nu_max],t0).set_f_params(self.L.L0[0])
        else:
            # r.set_initial_value(self.rho.initial[nu_max],t0).set_f_params(self.L.L0[nu_max])
            r.set_initial_value(self.rho.initial[start_block],t0).set_f_params(self.L.L0[start_block])

        
        # temporary variable to store states
        #rhos = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
        #rhos[nu][:,0] = self.rho.initial[nu]

        rho_nu = np.zeros((blocksizes[start_block], ntimes), dtype = complex)
        rho_nu[:,0] = self.rho.initial[start_block] 
        
        # using the exact solver times for the feedforward interpolation instead of using linearly spaced time array makes a (small) difference
        solver_times = np.zeros(ntimes)
        solver_times[0] = t0
    
        n_t=1
        while r.successful() and n_t < ntimes:
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
                expect_nu = np.array(self.expect_comp_block([rho_nu[:,t_idx]],start_block, expect_oper)).flatten()
                self.result.expect[:,t_idx] +=  expect_nu #np.array(self.expect_comp_block([rho_nu[:,t_idx]],nu_max, expect_oper)).flatten()
                
                if expect_per_nu:
                    self.result.expect_per_nu[start_block, :, t_idx] = expect_nu

                
                if progress:
                    bar.update()
        if save_states:
            self.result.rho[start_block] = rho_nu
        else: # store initial and final states
            self.result.rho[start_block][:,0] = rho_nu[:,0]
            self.result.rho[start_block][:,1] = rho_nu[:,-1] 
            
        #self.result.t = np.arange(t0, self.tend+self.dt,self.dt)
        if self.indices.only_numax:
            elapsed = time()-tstart
            
            if verbose:
                print(f'Complete {elapsed:.0f}s', flush=True)
            return
        
        # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block_interp
        for nu in range(start_block - 1, -1,-1):           
            #rho_interp = interp1d(self.result.t, rhos[nu+1], bounds_error=False, fill_value="extrapolate") # extrapolate results from previous block
            rho_interp = interp1d(solver_times, rho_nu, bounds_error=False, fill_value="extrapolate", kind='cubic') # interpolate results from previous block, rho_nu                  
                       
            r = ode(_intfunc_block_interp).set_integrator('zvode', method = method, atol=self.atol, rtol=self.rtol, nsteps=self.nsteps)
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
                    expect_nu = np.array(self.expect_comp_block([rho_nu[:,t_idx]],nu, expect_oper)).flatten()
                    self.result.expect[:,t_idx] +=  expect_nu
                    
                    if expect_per_nu:
                        self.result.expect_per_nu[nu, :, t_idx] = expect_nu
                    
                    if progress:
                        bar.update()
            if save_states:
                self.result.rho[nu] = rho_nu
            else:
                self.result.rho[nu][:,0] = rho_nu[:,0]
                self.result.rho[nu][:,1] = rho_nu[:,-1] 

        elapsed = time()-tstart
        if verbose:
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
                    # rho_nrs = self.rho.convert_rho_block_dic[nrs][nu].dot(rho_list[count]) # .dot
                    rho_nrs = self.rho.convert_rho_block_dic[nrs][nu]@rho_list[count]   # @
                    rho_nrs = vector_to_operator(rho_nrs)
                    rhos_converted_dic[nrs].append(rho_nrs)
            
            one_output = [] 
            rhos_converted = rhos_converted_dic[nrs]
            for count in range(len(rho_list)):
                one_output.append(expect(rhos_converted[count], op) / self.rho.scale_rho)    # added scale functionality
            output.append(one_output)
        return output
    
    
def _intfunc(t, y, L):
    """ Integration funciton for block nu_max"""
    return (L@y)
   

def _intfunc_block_interp(t,y,L0,L1,y1_func):
    """ Integration function for blocks nu < nu_max with coupling L1 to higher blocks"""
    # return (L0.dot(y) + L1.dot(y1_func(t)))
    return (L0 @ y + L1 @ y1_func(t))



