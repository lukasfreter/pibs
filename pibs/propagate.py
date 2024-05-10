#!/usr/bin/env python
import numpy as np
from time import time
import sys, os
from util import tensor, qeye, destroy, create, sigmap, sigmam, basis, sigmaz, vector_to_operator, expect
from scipy.integrate import ode



class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []


class TimeEvolve():
    def __init__(self, rho, L, indices, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5, save_states=None):
        self.tend = tend
        self.dt = dt
        self.expect_oper = expect_oper
        self.atol = atol
        self.rtol = rtol
        self.save_states = save_states
        
        # should I store rho and L like this ?
        self.rho = rho
        self.L = L
        self.indices = indices
        
        # time evolve
        print('Starting time evolution...')
        t0 = time()
        self.result = self.time_evolve_block()
        elapsed = time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
        
        


    def time_evolve_block(self):
        """Time evolve initial state using L0, L1 block structure Liouvillian matrices.
        This only works for weak U(1) symmetry.
    
        expect_oper should be a list of operators that each either act on the photon
        (dim_lp Z dim_lp), the photon and one spin (dim_lp*dim_ls X dim_lp*dim_ls), the
        photon and two spins... etc. setup_convert_rho_nrs(X) must have been run with
        X = 0, 1, 2,... prior to the calculation.
    
        progress==True writes progress in % for the time evolution
        """
       
        dim_rho_compressed = self.indices.ldim_p**2 * len(self.indices.indices_elements)
        num_blocks = len(self.indices.mapping_block)
        t0 = 0
        ntimes = int(self.tend/self.dt)+1
            
        output = Results()
        rhos= [[] for _ in range(num_blocks)] # store all rho for feed forward
        
        # first calculate block nu_max
        nu = num_blocks - 1
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
        r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu])
        #Record initial values
        output.t.append(r.t)
        rhos[nu].append(self.rho.initial[nu])
        
        # if progress:
        #     bar = Progress(ntimes, description='Time evolution under L...', start_step=1)
        if self.save_states is None:
            self.save_states = True if self.expect_oper is None else False
        if not self.save_states and self.expect_oper is None:
            print('Warning: Not recording states or any observables. Only initial and final'\
                    ' compressed state will be returned.')
                
        
        # FOR LATER: what if expect_oper = []
        
        # if expect_oper == None:
        #     while r.successful() and r.t < tend:
        #         rho = r.integrate(r.t+dt)
        #         if save_states:
        #             output_nu[nu].rho.append(rho)
        #         output_nu[nu].t.append(r.t)
        #         if progress:
        #             bar.update()
        
        # else:
        #output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
        #output.expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[nu], nu)], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+self.dt)
          #  output.expect[:,n_t] = array(expect_comp([rho_block_to_compressed(rho,nu)], expect_oper)).flatten()
            output.t.append(r.t)
            rhos[nu].append(rho)
            n_t += 1
            # if progress:
            #     bar.update()
            #if not save_states:
            #    output_nu[nu].rho.append(rho) # record final state in this case (otherwise already recorded)
        
        
        # # INCLUDE CHECK IF COUPLING TO DIFFERENT NU IS ZERO
        
        # Now, do the feed forward for all other blocks. Need different integration function,
        # that for this -> _intfunc_block
        
        for nu in range(num_blocks-2, -1,-1):
            r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=self.atol, rtol=self.rtol)
            r.set_initial_value(self.rho.initial[nu],t0).set_f_params(self.L.L0[nu], self.L.L1[nu], rhos[nu+1][0])
            #Record initial values
            rhos[nu].append(self.rho.initial[nu])
            
            
            # FOR LATER
            # if expect_oper == None:
            #     while r.successful() and r.t < tend:
            #         rho = r.integrate(r.t+dt)
            #         if save_states:
            #             output_nu[nu].rho.append(rho)
            #         output_nu[nu].t.append(r.t)
            #         if progress:
            #             bar.update()
            
            # else:
                #output_nu[nu].expect = zeros((len(expect_oper), ntimes), dtype=complex)
            #output_nu[nu].expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[nu],nu)], expect_oper)).flatten()
            n_t=1
            while r.successful() and n_t<ntimes:
                rho = r.integrate(r.t+self.dt)
                #output_nu[nu].expect[:,n_t] = array(expect_comp([rho_block_to_compressed(rho,nu)], expect_oper)).flatten()
                #output_nu[nu].t.append(r.t)
                #if save_states:
                rhos[nu].append(rho)
                # update integrator:
                r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf',
                        atol=self.atol, rtol=self.rtol).set_initial_value(r.y,r.t).set_f_params(self.L.L0[nu],self.L.L1[nu],rhos[nu+1][n_t])
                n_t += 1
    
                # if progress:
                #     bar.update()
                
            # if not save_states:
                # output_nu[nu].rho.append(rho) # record final state in this case (otherwise already recorded)
        
        # Now with all rhos, we can calculate the expectation values:
        output.expect = np.zeros((len(self.expect_oper), ntimes), dtype=complex)
        for t_idx in range(ntimes):
            for nu in range(num_blocks):
                output.expect[:,t_idx] = output.expect[:,t_idx] +  np.array(self.expect_comp_block([rhos[nu][t_idx]],nu, self.expect_oper)).flatten()

        return output
    
    
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