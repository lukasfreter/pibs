#!/usr/bin/env python
import numpy as np
from time import time
import sys, os
from util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect
from scipy.integrate import ode
import multiprocessing


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
    """ Class to handle time evolution of the quantum system """
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
        
    
    
    def evolve_numax(self,rho0,t0, chunksize):
        
        nu_max = len(self.indices.mapping_block) - 1
        blocksize= len(self.indices.mapping_block[nu_max])
        
        #initial values
        rho = np.zeros((blocksize, chunksize), dtype = complex)
        
        # integrator
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0,t0).set_f_params(self.L.L0[nu_max])
        
        # integrate
        n_t=0
        while r.successful() and n_t<chunksize:
            rho_step = r.integrate(r.t+self.dt)
            rho[:,n_t] = rho_step
            n_t += 1
            
        return rho
    
    
    def evolve_nu(self, rho0_nu, rho0_nup1, t0, nu, chunksize):
        """ Time evolution of block nu with initial value rho0. rho0_nup1 is an array
        of length chunksize, which is the coupling to the block nu+1."""
        
        # initial values
        rho = [rho0_nu]
        t = [t0]
        
        # integrator 
        r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(rho0_nu, t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rho0_nup1[0])
        
        n_t=1
        while r.successful() and n_t<chunksize:
            rho.append(r.integrate(r.t+self.dt))
            
            # update integrator:
            r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf',
                    atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu],self.L.L1[nu],rho0_nup1[n_t])
            n_t += 1
        return rho
        
    
    def time_evolve_chunk(self, expect_oper, progress=False):
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
        
        # number of timesteps to be solved in one go
        chunksize = 50
        
        # setup rhos as double the chunksize, such that there is space for one chunk for feedforward,
        # and one chunk for calculating the future time evolution.
        rhos_chunk= [np.zeros((len(self.indices.mapping_block[i]),2*chunksize), dtype=complex) for i in range(num_blocks)] 
        
        # keep track, where to write and where to read for each block
        write = [1] * len(self.indices.mapping_block) # zero means, write in first chunk. One means, write in second chunk
        
        # initialize rhos_chunk. Put initial state at the end of the first chunksize block,
        # such that it can act as initial state of the second chunksize block. 
        # After that, the last element of the second chunksize blick acts as initial state,
        # and the future states will be written in the first chunksize block
        for nu in range(num_blocks):
            rhos_chunk[nu][:,chunksize-1] = self.rho.initial[nu]
        
        # initialize expectation values
        self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)
        self.result.expect[:,0] =  np.array(self.expect_comp_block([self.rho.initial[nu_max]],nu_max, expect_oper)).flatten()
        self.result.t= [t0]
        
        # Loop until a chunk is outside of ntimes
        count = 0
        while count * chunksize < ntimes:          
            # first calculate block nu_max
            initial = rhos_chunk[nu_max][:,((write[nu_max]+1)%2)*chunksize+chunksize-1]
            
            rhos_chunk[nu_max][:,write[nu_max]*chunksize:(write[nu_max]+1)*chunksize] = (
                self.evolve_numax(initial, t0+count*chunksize*self.dt, chunksize))
            
            
            parallel_blocks = min(count, num_blocks) # number of blocks we can simultaneously calculate in current chunk
                                                    # cannot exceet num_blocks, otherwise grows linearly with number of chunks
            # Now the other blocks: take care of when it is allowed to calculate a block.
            # e.g. to calculate block nu, block nu+1 must have been calculated in previous block.
            x = rhos_chunk[nu_max][:,:]                                       
            # for nu in range(parallel_blocks):
            #     initial_nu = 
            
            
            # Calculate expectation values of all blocks that are available
            i=0
            for t_idx in range(count*chunksize+1, (count+1)*chunksize+1):
                if t_idx >= ntimes:
                    continue
                read_index = write[nu_max] *chunksize
                self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  (
                    np.array(self.expect_comp_block([rhos_chunk[nu_max][:,read_index+ i]],nu, expect_oper)).flatten())
                self.result.t.append(t_idx*self.dt)
                i = i+1
            write[nu_max] = (write[nu_max] + 1) % 2 # if 1 make it 0, if 0 make it 1

            count = count+1

            # else:
                
            #     # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block
            #     for nu in range(num_blocks-2, -1,-1):
            #         r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
            #         r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rhos[nu+1][0])
                    
            #         #Record initial values
            #         rhos[nu].append(self.rho.initial[nu])
                    
            #         n_t=1
            #         while r.successful() and n_t<ntimes:
            #             rho = r.integrate(r.t+self.dt)
            #             rhos[nu].append(rho)
                        
            #             # update integrator:
            #             r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf',
            #                     atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu],self.L.L1[nu],rhos[nu+1][n_t])
            #             n_t += 1
               

        # Now with all rhos, we can calculate the expectation values:
        # if expect_oper is not None:
        #     self.result.expect = np.zeros((len(expect_oper), ntimes), dtype=complex)

        #     for t_idx in range(ntimes):
        #         for nu in range(num_blocks):
        #             self.result.expect[:,t_idx] = self.result.expect[:,t_idx] +  np.array(self.expect_comp_block([rhos[nu][t_idx]],nu, expect_oper)).flatten()

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
        
        print('Starting time evolution...')
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


if __name__ =='__main__':
    # testing
    from setup import Indices, Rho
    ntls = 7
    nphot = ntls+1
    indi = Indices(ntls)
    rho = Rho(basis(nphot,0), basis(2,0), indi) # initial condition with zero photons and all spins up.
    t = TimeEvolve(rho, 1, indi,1,1)