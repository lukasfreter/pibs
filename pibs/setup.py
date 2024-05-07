#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool
from util import export, timeit
import os, sys, logging
import pickle
from time import time

class Indices:
    """Indices for and mappings between compressed and supercompressed 
    density matrices containing unique representative elements for each
    permutation invariant subspace of nspins identical spins. The 
    representative elements are chosen such that, if i=0,1,...,nspins-1
    enumerates the spins one has zeta_0 <= zeta_1 <=... is monotonically
    increasing where zeta is the simple finite pairing
    function
        zeta_i = (dim_s) * s^L_i + s^R_i
    for site (spin) i having element |s^L_i><s^R_i| with left and right 
    spin values s^L_i and s^R_i. Nominally, dim_s = 2 i.e. s^L_i, s^R_i
    = 0 or 1 (spin-1/2).

    In the compressed form, the spin indices are multiplied by the left/right
    photon part of the density matrices.
    In the supercompressed form, the (smaller) set of elements are grouped
    according to the total excitation number nu
    [can we get rid of compressed form entirely?]
    """
    def __init__(self, nspins, nphot=None,spin_dim=None, verbose=True):
        if nphot is None:
            nphot = nspins + 1
        if spin_dim is None:
            spin_dim = 2 # spin 1/2 system
        self.nspins, self.ldim_p, self.ldim_s = nspins, nphot, spin_dim
        self.indices_elements = []
        self.indices_elements_inv = {}
        self.mapping_block = []
        self.elements_block = []
        
        # check if an object with the same arguments already exists in data/indices/ folder
        index_path = 'data/indices/'
        index_files = os.listdir(index_path)
        filename = f'indices_Ntls{self.nspins}_Nphot{self.ldim_p}_spindim{self.ldim_s}.pkl'
        if (any([f == filename for f in index_files])):
            self.load(index_path+filename)
        else:
            
            # setup indices
            print(f'Running setup indices with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
            t0 = time()
            self.list_equivalent_elements()
            elapsed = time()-t0
            print(f'Complete {elapsed:.0f}s', flush=True)
    
            # setup mapping block
            print(f'Running setup indices block with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
            t0 = time()
            self.setup_mapping_block(parallel=True)
            elapsed = time()-t0
            print(f'Complete {elapsed:.0f}s', flush=True)
            
            # export for future use
            self.export(index_path + filename)
                
    
    def list_equivalent_elements(self):
        """Generate basis list, needs to be run at the beginning of
        each calculation"""
        from numpy import concatenate, copy, array

        count = 0
        
        #get minimal list of left and right spin indices (in combined form)
        spins = self.setup_spin_indices()
        
        left =[]
        right = []
        
        #split combined indices into left/right form
        for count in range(len(spins)):
            leftadd, rightadd = self._to_hilbert(spins[count])
            left.append(leftadd)
            right.append(rightadd)

        
        left = array(left)
        right = array(right)

        #loop over each photon state and each spin configuration
        for count in range(len(spins)):
            #calculate element and index 
            element = concatenate((left[count], right[count]))
            
            #add appropriate entries to dictionaries
            self.indices_elements.append(copy(element))
            self.indices_elements_inv[self._comp_tuple(element)] = count
            
                    
    def setup_spin_indices(self):
        """get minimal list of left and right spin indices"""
        from numpy import concatenate, array, copy

        spin_indices = []
        spin_indices_temp = []
        
        #construct all combinations for one spin
        for count in range(self.ldim_s**2):
            spin_indices_temp.append([count])
        spin_indices_temp = array(spin_indices_temp)
        spin_indices = [array(x) for x in spin_indices_temp] # Used if ns == 1
        
        #loop over all other spins
        for count in range(self.nspins-1):
            #make sure spin indices is empty 
            spin_indices = []   
            #loop over all states with count-1 spins
            for index_count in range(len(spin_indices_temp)):
             
                #add all numbers equal to or less than the last value in the current list
                for to_add in range(spin_indices_temp[index_count, -1]+1):
                    spin_indices.append(concatenate((spin_indices_temp[index_count, :], [to_add])))
            spin_indices_temp = copy(spin_indices)
        
        return spin_indices
    
    
    
    def mapping_task(self, args_tuple):
        """ Function to parallelize setup_mapping_block"""
        nu_max = self.nspins
        count_p1, count_p2, count = args_tuple
        num_elements = len(self.indices_elements)
        element = self.indices_elements[count]
        element_index = self.ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
        
        # get left and right states (photon numbers and spin states)
        left = element[0:self.nspins]
        right = element[self.nspins:2*self.nspins]
        
        # calculate total excitation number. Note: spin up is 0, spin down is 1!
        m_left = self.nspins-sum(left)
        m_right = self.nspins-sum(right)
        nu_left = m_left + count_p1
        nu_right = m_right + count_p2
        
        # if left and right excitation numbers are equal and below maximum, add them to list
        if nu_left == nu_right and nu_left <= nu_max:                   
            el = np.concatenate(([count_p1], left, [count_p2],right))
            return (nu_left, element_index, el)
    
    
    def setup_mapping_block(self, parallel=False):
        """
        Generate mapping between reduced representation of density matrix and
        the block structure, which is grouped in different numbers of total excitations
        of photons + spins. Note: 0 in spin array means spin up!
        For now, nu_max (maximum excitation) is set to nspins, because the initial 
        condition is always all spins up and zero photons in the cavity.
        
        Structure of mapping_block = [ [indices of nu=0] , [indices of nu=1], ... [indices of nu_max] ]

        """     
        num_elements = len(self.indices_elements)
        nu_max = self.nspins # maximum excitation number IF initial state is all spins up and zero photons
        
        self.mapping_block = [ [] for _ in range(nu_max+1)] # list of nu_max+1 empty lists
        self.elements_block = [ [] for _ in range(nu_max+1)]
        
        if parallel:  # parallel version  
            arglist = []
            t0 = time()
            for count_p1, count_p2, count in product(range(self.ldim_p), range(self.ldim_p), range(num_elements)):
                arglist.append((count_p1, count_p2, count))
        
            with Pool() as p:
                results = p.map(self.mapping_task, arglist)
        
            #for nu, element_index in results: # do we know how long block will be at each nu? 
            for result in results:
                if result is None:
                    continue
                # try to avoid this?
                self.mapping_block[result[0]].append(result[1])
                self.elements_block[result[0]].append(result[2])
            

        else: # serial version
            for count_p1 in range(self.ldim_p):
                for count_p2 in range(self.ldim_p):
                    for count in range(num_elements):
                        element = self.indices_elements[count]
                        element_index = self.ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                        left = element[0:self.nspins]
                        right = element[self.nspins:2*self.nspins]
                        
                        # calculate excitations. Important: ZEOR MEANS SPIN UP, ONE MEANS SPIN DOWN.
                        m_left = self.nspins-sum(left)
                        m_right = self.nspins-sum(right)
                        # calculate nu
                        nu_left = m_left + count_p1
                        nu_right = m_right + count_p2
                        if nu_left == nu_right and nu_left <= nu_max:                   
                            el = np.concatenate(([count_p1], left, [count_p2],right))
                            self.mapping_block[nu_left].append(element_index)
                            self.elements_block[nu_left].append(el)

    
    
    def _to_hilbert(self,combined):
        """convert to Hilbert space index"""
        left = []
        right = []
        for count in range(len(combined)):
            leftadd, rightadd = self._combined_to_full(combined[count])
            left.append(leftadd)
            right.append(rightadd)
        return left, right
    
    def _combined_to_full(self, combined):
        """create left and right Hilbert space indices from combined index"""
        right = combined%self.ldim_s
        left = (combined - right)//self.ldim_s
        
        return left, right
    
    def _comp_tuple(self, element):
        """compress the tuple used in the dictionary"""        
        element_comp = []
        
        for count in range(self.nspins):
            element_comp.append(element[count]*self.ldim_s + element[count+self.nspins])
        return tuple(element_comp)



        


    def export(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)


    def load(self, filepath):
        with open(filepath, 'rb') as handle:
            indices_load = pickle.load(handle)
        self.indices_elements = indices_load.indices_elements
        self. indices_elements_inv = indices_load.indices_elements_inv
        self.mapping_block = indices_load.mapping_block
        self.elements_block = indices_load.elements_block
        # do some checks
        # at least tell user what they loaded
        print(f'Loaded index file with ntls={self.nspins}, nphot={self.ldim_p}, spin_dim={self.ldim_s}')



# class BlockL:
#     """Description"""
#     def __init__(self):
#         # initialisation

if __name__ == '__main__':
    # Testing purposes
    indi = Indices(5)
