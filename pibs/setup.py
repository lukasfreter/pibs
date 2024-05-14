#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool
from util import export, timeit, tensor, qeye, destroy, create, sigmap, sigmam, basis
from util import sigmaz, degeneracy_spin_gamma, degeneracy_gamma_changing_block_efficient
from util import states_compatible, permute_compatible, degeneracy_outer_invariant_optimized
from util import _multinominal
import os, sys, logging
import pickle
from time import time
import scipy.sparse as sp
from itertools import permutations


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
        # make some checks for validity of nspins, nphot, spin_dim
        if (not isinstance(nspins, (int, np.integer))) or nspins <= 0:
            raise ValueError("Number of spins must be integer N > 0")
        if nphot is not None and ((not isinstance(nphot, (int, np.integer))) or nphot <= 0):
            raise ValueError("Number of photon states must be integer > 0")
        if spin_dim is not None and ((not isinstance(spin_dim, (int, np.integer))) or spin_dim <= 1):
             raise ValueError("Spin dimension must be integer > 1")
            
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
        print(f'Storing Indices for later use in {filepath}')


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



class BlockL:
    """Calculate Liouvillian basis block form. As a requirement, the Master
    Equation for the system must have weak U(1) symmetry, such that density-matrix
    elements couple to density matrix elements that differ in the total number of
    excitations (photons + spin) by at most one. We label the total number of excitation
    by nu.
    
    The Liouvillian block structure consists of 2 sets of matrices: L0 and L1.
    L0 contains square matrices, which couple density matrix elements from one
    block to the same block (preserve total excitation number).
    L1 contains matrices, which couple elements from block nu to elements in block
    nu+1. There are nu_max matrices in L0, and nu_max-1 matrices in L1, because
    the block of nu_max cannot couple to any higher nu.
    
    In this first version, we calculate the Liouvillian for the Dicke model with
    photon loss L[a], individual dephasing L[sigma_z] and individual exciton loss
    L[sigma_m], as well as a Hamiltonian of the form 
    H = wc*adag*a + sum_k { w0*sigmaz_k  + g*(a*sigmap_k + adag*sigmam_k) }
    This means we have 6 parameters, i.e. we need a basis of 6 elements. These
    we can then later scale by the parameters and add up, to get the desired system
    Liouvillian
    
    This is convenient, because we can calculate the basis for a given N, Nphot
    once, and reuse them for all different values of the dissipation rates or energies
    in the hamiltonian.
    """
    def __init__(self, indices, parallel= False):
        # initialisation
        self.L0_basis = {'sigmaz': [],
                         'sigmam': [],
                         'a': [],
                         'H_n': [],
                         'H_sigmaz': [],
                         'H_g': []}
        self.L1_basis = {'sigmam': [],
                         'a': []}
        

        # check if an object with the same arguments already exists in data/liouvillian/ folder
        liouv_path = 'data/liouvillians/'
        liouv_files = os.listdir(liouv_path)
        filename = f'liouvillian_dicke_Ntls{indices.nspins}_Nphot{indices.ldim_p}_spindim{indices.ldim_s}.pkl'
        if (any([f == filename for f in liouv_files])):
            self.load(liouv_path+filename, indices)
        else:
            # if not, calculate them

            t0 = time()
            if parallel:
                print('Calculating normalized L parallel ...')
                self.setup_L_block_basis_parallel(indices)
            else:
                print('Calculating normalized L serial ...')
                self.setup_L_block_basis(indices)
            elapsed = time()-t0
            print(f'Complete {elapsed:.0f}s', flush=True)
            # export normalized Liouvillians for later use
            #self.export(liouv_path+filename)
        
        
    
    def export(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
        print(f'Storing Liouvillian for later use in {filepath}')
            
    def load(self, filepath,ind):
        with open(filepath, 'rb') as handle:
            L_load = pickle.load(handle)
            
        self.L0_basis = L_load.L0_basis
        self.L1_basis = L_load.L1_basis
        
        # at least tell user what they loaded
        print(f'Loaded Liouvillian file with ntls={ind.nspins}, nphot={ind.ldim_p}, spin_dim={ind.ldim_s}')
    
        
    
    def setup_L_block_basis(self, indices):
       """ Calculate Liouvillian basis in block form"""
       num_blocks = len(indices.mapping_block)
       
       #------------------------------------------------------
       # First, get L0 part -> coupling to same block, 
       #------------------------------------------------------
       
       # loop through all elements in block structure
       for nu_element in range(num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           
           # rewrite in dictionary
           L0_new ={'sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'sigmam': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'a': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_n': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_g': np.zeros((current_blocksize, current_blocksize), dtype=complex)}
           
           if nu_element < num_blocks-1:
               next_blocksize = len(indices.mapping_block[nu_element+1])
               L1_new = {'sigmam': np.zeros((current_blocksize, next_blocksize), dtype=complex),
                         'a': np.zeros((current_blocksize, next_blocksize), dtype=complex)}

           
           # Loop through all elements in the same block
           for count_in in range(current_blocksize):
               # get element, of which we want the time derivative
               element = indices.elements_block[nu_element][count_in]
               left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
               right = element[indices.nspins+1:2*indices.nspins+2] # right state
               
               # now loop through all matrix elements in the same block, to get L0 couplings
               for count_out in range(current_blocksize):
                   # get "to couple" element
                   element_to_couple = indices.elements_block[nu_element][count_out]
                   left_to_couple = element_to_couple[0:indices.nspins+1]
                   right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                   
                   # elements which differ in photon number by 2 will never couple:
                   if abs(left_to_couple[0] - left[0]) > 1 or abs(right_to_couple[0] - right[0]) > 1:
                       continue
                   
                   #-----------------------------
                   # get Liouvillian elements
                   #-----------------------------
                  
                   # L0 part from Hamiltonian
                   # Diagonal part
                   if (right_to_couple == right).all() and (left_to_couple == left).all():
                       s_down_right = sum(right[1:])
                       s_down_left = sum(left[1:])
                       L0_new['H_n'][count_in, count_out] = -1j * (left[0]-right[0])
                       L0_new['H_sigmaz'][count_in, count_out] = 1j*(s_down_left-s_down_right)
                   
                   # offdiagonal parts
                   elif(states_compatible(right, right_to_couple)):
                        # if they are compatible, permute left_to_couple appropriately for proper H element
                        left_to_couple_permute = np.copy(left_to_couple)
                        if not (right_to_couple == right).all():
                            # if they are compatible but not equal, we need to permute left_to_couple appropriately, to get correct matrix element of H
                            left_to_couple_permute[1:] = permute_compatible(right[1:],right_to_couple[1:],left_to_couple[1:])
                            
                        # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                        if (left[1:]==left_to_couple_permute[1:]).sum() != indices.nspins-1:
                            continue
                        
                        deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], left_to_couple_permute[1:]) # degeneracy from simulatneous spin permutations, which leave outer spins invariant
                        # check if photon number in left state increases or decreases and
                        # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                        if (left[0] - left_to_couple[0]) == 1 and sum(left[1:])-sum(left_to_couple[1:]) == 1: # need matrix element of adag*sigmam
                            L0_new['H_g'][count_in, count_out] = L0_new['H_g'][count_in, count_out]  - 1j*deg * np.sqrt(left[0])
                        elif left[0] - left_to_couple[0] == -1 and sum(left[1:])-sum(left_to_couple[1:]) == -1 : # need matrix element of a*sigmap
                            L0_new['H_g'][count_in, count_out] = L0_new['H_g'][count_in, count_out] - 1j*deg * np.sqrt(left[0]+1)   
                               
                   elif(states_compatible(left, left_to_couple)):            
                        # if they are compatible, permute right_to_couple appropriately for proper H element
                        right_to_couple_permute = np.copy(right_to_couple)
                        if not (left_to_couple == left).all():
                            right_to_couple_permute[1:] = permute_compatible(left[1:],left_to_couple[1:],right_to_couple[1:])
                            
                        # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                        if (right[1:]==right_to_couple_permute[1:]).sum() != indices.nspins-1:
                            continue
                        deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], right_to_couple_permute[1:])
                        # check if photon number in right state increases or decreases and
                        # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                        if (right[0] - right_to_couple[0]) == 1 and sum(right[1:])-sum(right_to_couple[1:]) == 1: # need matrix element of a*sigmap
                            L0_new['H_g'][count_in, count_out] = L0_new['H_g'][count_in, count_out] + 1j*deg * np.sqrt(right[0])
                        elif right[0] - right_to_couple[0] == -1 and sum(right[1:])-sum(right_to_couple[1:]) == -1: # need matrix element of adag*sigmam
                            L0_new['H_g'][count_in, count_out] = L0_new['H_g'][count_in, count_out] + 1j*deg * np.sqrt(right[0]+1)

                   
                   
                   # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
                   # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                   # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                   if (right_to_couple == right).all() and (left_to_couple == left).all():
                       deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                       deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                       L0_new['sigmam'][count_in,count_out] = - 1/2 * (deg_left+deg_right)
                   
                   # L0 part from L[sigmaz] -> whole dissipator
                   # Left and right states must be equal, because sigmaz is diagonal in the spins.
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                       L0_new['sigmaz'][count_in][count_out] = 2*(equal - indices.nspins)
                       
                   # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       L0_new['a'][count_in][count_out] = -1/2*(left[0] + right[0]) 


                   
               if nu_element == num_blocks -1:
                   continue
                
               # Now get L1 part -> coupling from nu_element to nu_element+1
               # loop through all matrix elements in the next block we want to couple to
               for count_out in range(next_blocksize):
                   # get "to couple" element
                   element_to_couple = indices.elements_block[nu_element+1][count_out]
                   left_to_couple = element_to_couple[0:indices.nspins+1]
                   right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                   
                   # elements which differ in photon number by 2 will never couple:
                   if abs(left_to_couple[0] - left[0]) > 1 or abs(right_to_couple[0] - right[0]) > 1:
                       continue
                   
                   #---------------------------------
                   # get Liouvillian elements
                   #--------------------------------
                   
                   # L1 part from L[sigmam] -> sigmam * rho * sigmap
                   # Photons must remain the same
                   if (left[0] == left_to_couple[0] and right[0] == right_to_couple[0]):
                       # we have to compute matrix elements of sigma^- and sigma^+. Therefore, check first if 
                       # number of spin up in "right" and "right_to_couple" as well as "left" and "left_to_coupole" vary by one
                       if (sum(left[1:]) - sum(left_to_couple[1:]) == 1) and (sum(right[1:]) - sum(right_to_couple[1:]) == 1):       
                           # Get the number of permutations, that contribute.                             
                           deg = degeneracy_gamma_changing_block_efficient(left[1:], right[1:], left_to_couple[1:], right_to_couple[1:])                
                           L1_new['sigmam'][count_in,count_out] = deg
                   
                   # L1 part from L[a] -> a * rho* adag
                   # since spins remain the same, first check if spin states match
                   # if spins match, then the element can couple, because we are looping through the block nu+1. Therefore
                   # the coupled-to-elements necessarily have one more excitation, which for this case is in the photon state.
                   if (left[1:] == left_to_couple[1:]).all() and (right[1:]==right_to_couple[1:]).all():
                       L1_new['a'][count_in][count_out] = np.sqrt((left[0]+1)*(right[0] + 1))
           
            
           # append new blocks to the basis
           for name in self.L0_basis:
                self.L0_basis[name].append(sp.csr_matrix(L0_new[name]))
           
           if nu_element < num_blocks-1: 
               for name in self.L1_basis:
                   self.L1_basis[name].append(sp.csr_matrix(L1_new[name]))
                   
                   
    
                
    # functions for parallelization
    def calculate_L0_line(self,args_tuple):
        """ Calculate L0 part of element count_in in block nu_element """
        indices, count_in, nu_element = args_tuple
        current_blocksize = len(indices.mapping_block[nu_element])
        
        # get element, of which we want the time derivative
        element = indices.elements_block[nu_element][count_in]
        left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
        right = element[indices.nspins+1:2*indices.nspins+2] # right state

        
        # initialize Liouvillian rows for element count_in
        L0_line = {
            'sigmaz': np.zeros((1, current_blocksize), dtype=complex),
            'sigmam': np.zeros((1, current_blocksize), dtype=complex),
            'a': np.zeros((1, current_blocksize), dtype=complex),
            'H_n': np.zeros((1, current_blocksize), dtype=complex),
            'H_sigmaz': np.zeros((1, current_blocksize), dtype=complex),
            'H_g': np.zeros((1, current_blocksize), dtype=complex)
            }
        
        # now loop through all matrix elements in the same block, to get L0 couplings
        for count_out in range(current_blocksize):
            # get "to couple" element
            element_to_couple = indices.elements_block[nu_element][count_out]
            left_to_couple = element_to_couple[0:indices.nspins+1]
            right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
            
            # elements which differ in photon number by 2 will never couple:
            if abs(left_to_couple[0] - left[0]) > 1 or abs(right_to_couple[0] - right[0]) > 1:
                continue
            
            #-----------------------------
            # get Liouvillian elements
            #-----------------------------
           
            # L0 part from Hamiltonian
            # Diagonal part
            if (right_to_couple == right).all() and (left_to_couple == left).all():
                s_down_right = sum(right[1:])
                s_down_left = sum(left[1:])
                L0_line['H_n'][0, count_out] = -1j * (left[0]-right[0])
                L0_line['H_sigmaz'][0, count_out] = 1j*(s_down_left-s_down_right)
            
            # offdiagonal parts
            elif(states_compatible(right, right_to_couple)):
                 # if they are compatible, permute left_to_couple appropriately for proper H element
                 left_to_couple_permute = np.copy(left_to_couple)
                 if not (right_to_couple == right).all():
                     # if they are compatible but not equal, we need to permute left_to_couple appropriately, to get correct matrix element of H
                     left_to_couple_permute[1:] = permute_compatible(right[1:],right_to_couple[1:],left_to_couple[1:])
                     
                 # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                 if (left[1:]==left_to_couple_permute[1:]).sum() != indices.nspins-1:
                     continue
                 
                 deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], left_to_couple_permute[1:]) # degeneracy from simulatneous spin permutations, which leave outer spins invariant
                 # check if photon number in left state increases or decreases and
                 # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                 if (left[0] - left_to_couple[0]) == 1 and sum(left[1:])-sum(left_to_couple[1:]) == 1: # need matrix element of adag*sigmam
                     L0_line['H_g'][0, count_out] = L0_line['H_g'][0, count_out]  - 1j*deg * np.sqrt(left[0])
                 elif left[0] - left_to_couple[0] == -1 and sum(left[1:])-sum(left_to_couple[1:]) == -1 : # need matrix element of a*sigmap
                     L0_line['H_g'][0, count_out] = L0_line['H_g'][0, count_out] - 1j*deg * np.sqrt(left[0]+1)   
                        
            elif(states_compatible(left, left_to_couple)):            
                 # if they are compatible, permute right_to_couple appropriately for proper H element
                 right_to_couple_permute = np.copy(right_to_couple)
                 if not (left_to_couple == left).all():
                     right_to_couple_permute[1:] = permute_compatible(left[1:],left_to_couple[1:],right_to_couple[1:])
                     
                 # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                 if (right[1:]==right_to_couple_permute[1:]).sum() != indices.nspins-1:
                     continue
                 deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], right_to_couple_permute[1:])
                 # check if photon number in right state increases or decreases and
                 # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                 if (right[0] - right_to_couple[0]) == 1 and sum(right[1:])-sum(right_to_couple[1:]) == 1: # need matrix element of a*sigmap
                     L0_line['H_g'][0, count_out] = L0_line['H_g'][0, count_out] + 1j*deg * np.sqrt(right[0])
                 elif right[0] - right_to_couple[0] == -1 and sum(right[1:])-sum(right_to_couple[1:]) == -1: # need matrix element of adag*sigmam
                     L0_line['H_g'][0, count_out] = L0_line['H_g'][0, count_out] + 1j*deg * np.sqrt(right[0]+1)

            
            
            # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
            # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
            # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
            if (right_to_couple == right).all() and (left_to_couple == left).all():
                deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                L0_line['sigmam'][0,count_out] = - 1/2 * (deg_left+deg_right)
            
            # L0 part from L[sigmaz] -> whole dissipator
            # Left and right states must be equal, because sigmaz is diagonal in the spins.
            if (left_to_couple == left).all() and (right_to_couple == right).all():
                equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                L0_line['sigmaz'][0,count_out] = 2*(equal - indices.nspins)
                
            # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
            if (left_to_couple == left).all() and (right_to_couple == right).all():
                L0_line['a'][0,count_out] = -1/2*(left[0] + right[0]) 
        return L0_line

        
        
    
    def calculate_L1_line(self,args_tuple):
        """ Calculate L1 part of element count_in in block nu_element """
        
        indices,count_in, nu_element = args_tuple
        
        current_blocksize = len(indices.mapping_block[nu_element])
        next_blocksize = len(indices.mapping_block[nu_element+1])

        # get element, of which we want the time derivative
        element = indices.elements_block[nu_element][count_in]
        left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
        right = element[indices.nspins+1:2*indices.nspins+2] # right state
            
        # initialize L1 rows
        L1_line = {'sigmam': np.zeros((1, next_blocksize), dtype=complex),
                  'a': np.zeros((1, next_blocksize), dtype=complex)}
        
        # Now get L1 part -> coupling from nu_element to nu_element+1
        # loop through all matrix elements in the next block we want to couple to
        for count_out in range(next_blocksize):
            # get "to couple" element
            element_to_couple = indices.elements_block[nu_element+1][count_out]
            left_to_couple = element_to_couple[0:indices.nspins+1]
            right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
            
            # elements which differ in photon number by 2 will never couple:
            if abs(left_to_couple[0] - left[0]) > 1 or abs(right_to_couple[0] - right[0]) > 1:
                continue
            
            #---------------------------------
            # get Liouvillian elements
            #--------------------------------
            
            # L1 part from L[sigmam] -> sigmam * rho * sigmap
            # Photons must remain the same
            if (left[0] == left_to_couple[0] and right[0] == right_to_couple[0]):
                # we have to compute matrix elements of sigma^- and sigma^+. Therefore, check first if 
                # number of spin up in "right" and "right_to_couple" as well as "left" and "left_to_coupole" vary by one
                if (sum(left[1:]) - sum(left_to_couple[1:]) == 1) and (sum(right[1:]) - sum(right_to_couple[1:]) == 1):       
                    # Get the number of permutations, that contribute.                             
                    deg = degeneracy_gamma_changing_block_efficient(left[1:], right[1:], left_to_couple[1:], right_to_couple[1:])                
                    L1_line['sigmam'][0,count_out] = deg
            
            # L1 part from L[a] -> a * rho* adag
            # since spins remain the same, first check if spin states match
            # if spins match, then the element can couple, because we are looping through the block nu+1. Therefore
            # the coupled-to-elements necessarily have one more excitation, which for this case is in the photon state.
            if (left[1:] == left_to_couple[1:]).all() and (right[1:]==right_to_couple[1:]).all():
                L1_line['a'][0,count_out] = np.sqrt((left[0]+1)*(right[0] + 1))

        return L1_line
    
    
    def setup_L_block_basis_parallel(self, indices):
       """ Calculate Liouvillian basis in block form. Parallelize the calculation
       of rows of the Liouvillian"""
       num_blocks = len(indices.mapping_block)
       
       # loop through all elements in block structure
       for nu_element in range(num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           
           # rewrite in dictionary
           L0_new ={'sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'sigmam': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'a': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_n': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                    'H_g': np.zeros((current_blocksize, current_blocksize), dtype=complex)}
           
           if nu_element < num_blocks-1:
               next_blocksize = len(indices.mapping_block[nu_element+1])
               L1_new = {'sigmam': np.zeros((current_blocksize, next_blocksize), dtype=complex),
                         'a': np.zeros((current_blocksize, next_blocksize), dtype=complex)}

           
           arglist = []
           for count_in in range(current_blocksize):
               arglist.append((indices, count_in, nu_element))
           with Pool() as pool:
               list_of_lines = pool.map(self.calculate_L0_line, arglist)
           

           for name in L0_new:
               for count_in in range(current_blocksize):
                    L0_new[name][count_in,:] = list_of_lines[count_in][name]
               self.L0_basis[name].append(sp.csr_matrix(L0_new[name]))
           
           if nu_element < num_blocks -1:
               with Pool() as pool:
                   list_of_lines = pool.map(self.calculate_L1_line, arglist)

               for name in L1_new:
                   for count_in in range(current_blocksize):
                       L1_new[name][count_in,:] = list_of_lines[count_in][name]
                   self.L1_basis[name].append(sp.csr_matrix(L1_new[name]))
           # Loop through all elements in the same block
           # for count_in in range(current_blocksize):
           #     L0_line = self.calculate_L0_line(indices, count_in, nu_element)
               
           #     for name in L0_new:
           #         L0_new[name][count_in,:] = L0_line[name]
               
           #     if nu_element < num_blocks -1:
           #         L1_line = self.calculate_L1_line(indices, count_in, nu_element)
           #         for name in L1_new:
           #             L1_new[name][count_in,:] = L1_line[name]
            
           # append new blocks to the basis
           # for name in self.L0_basis:
           #      self.L0_basis[name].append(sp.csr_matrix(L0_new[name]))
           
           # if nu_element < num_blocks-1: 
           #     for name in self.L1_basis:
           #         self.L1_basis[name].append(sp.csr_matrix(L1_new[name]))
            
    
    
    def L0_nu_task(self, args_tuple):
        indices, nu_element = args_tuple
        num_blocks = len(indices.mapping_block)
        current_blocksize = len(indices.mapping_block[nu_element])
        # setup the Liouvillians for the current block

        L0_new ={'sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'sigmam': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'a': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_n': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_g': np.zeros((current_blocksize, current_blocksize), dtype=complex)}
        
        
        for count_in in range(current_blocksize):
            L0_line = self.calculate_L0_line((indices, count_in, nu_element))
            
            for name in L0_new:
                L0_new[name][count_in,:] = L0_line[name]
    
        return L0_new

                
                
    def L1_nu_task(self, args_tuple):
        indices, nu_element = args_tuple
        num_blocks = len(indices.mapping_block)
        current_blocksize = len(indices.mapping_block[nu_element])
        next_blocksize = len(indices.mapping_block[nu_element+1])
        L1_new = {'sigmam': np.zeros((current_blocksize, next_blocksize), dtype=complex),
                  'a': np.zeros((current_blocksize, next_blocksize), dtype=complex)}

        
        for count_in in range(current_blocksize):
            L1_line = self.calculate_L1_line((indices, count_in, nu_element))
            for name in L1_new:
                L1_new[name][count_in,:] = L1_line[name]
        
        return L1_new

        
        
        
    
    def setup_L_block_basis_parallel2(self, indices):
       """ Calculate Liouvillian basis in block form. Parallelized the calculation
       of each block"""
       num_blocks = len(indices.mapping_block)
       
       # loop through all elements in block structure
       arglist = []
       for nu in range(num_blocks):
           arglist.append((indices, nu))
       
       with Pool(14) as pool:
           L0s = pool.map(self.L0_nu_task, arglist)
       
       for name in self.L0_basis:
           for nu in range(num_blocks):
               self.L0_basis[name].append(L0s[nu][name])
       
       
       arglist = []
       for nu in range(num_blocks-1):
           arglist.append((indices, nu))
       with Pool(14) as pool:
           L1s = pool.map(self.L1_nu_task,arglist)
           
       for name in self.L1_basis:
           for nu in range(num_blocks-1):
               self.L1_basis[name].append(L1s[nu][name])
           

           
       print('done')
       
     
       
       
       
       
                        
class BlockDicke(BlockL):
    """ Calculates the specific Liouvillian of the Tavis Cummings model. This class
    inherits the Liouvillian basis from BlockL. With the specified parameters of the 
    Hamiltonian and the collapse operators, one can calculate the Liouvillian.
    
    Model:
        H = wc*adag*a + sum_k { w0*sigmaz_k  + g*(a*sigmap_k + adag*sigmam_k) }
        d/dt rho = -i[H,rho] + kappa*L[a] + gamma*L[sigmam] + gamma_phi*L[sigmaz]"""
    def __init__(self,wc,w0,g, kappa, gamma_phi, gamma, indices, parallel=False):
        # specify rates according to what part of Hamiltonian or collapse operators
        # they scale
        self.rates = {'H_n': wc,
                      'H_sigmaz': w0,
                      'H_g': g,
                      'a': kappa,
                      'sigmaz': gamma_phi,
                      'sigmam': gamma}
        self.w0 = w0
        self.wc = wc
        self.g = g
        self.kappa = kappa
        self.gamma = gamma
        self.gamma_phi = gamma_phi
        self.L0 = []
        self.L1 = []
        super().__init__(indices, parallel)
        
        self.setup_L(indices)

    def setup_L(self, indices):
        """ From the basic parts of the Liouvillian, get the whole Liouvillian
        by proper scaling."""
        
        # use dictionary notation
        num_blocks = len(indices.mapping_block)
        for nu in range(num_blocks):
            current_blocksize = len(indices.mapping_block[nu])
            L0_scale = np.zeros((current_blocksize, current_blocksize), dtype=complex)
            for name in self.L0_basis:
                L0_scale = L0_scale + self.rates[name] * self.L0_basis[name][nu]
            self.L0.append( sp.csr_matrix(L0_scale ))
            
            if nu < num_blocks -1:
                next_blocksize = len(indices.mapping_block[nu+1])
                L1_scale = np.zeros((current_blocksize, next_blocksize), dtype=complex)
                for name in self.L1_basis:
                    L1_scale = L1_scale + self.rates[name] * self.L1_basis[name][nu]
                self.L1.append( sp.csr_matrix(L1_scale))                     




class Rho:
    """ Functionality related to density matrix:
        Initial state
        Reduced density matrix
        Calculation of expectation values
    """
        
    def __init__(self, rho_p, rho_s, indices, nrs=1):
        assert type(nrs) == int, "Argument 'nrs' must be int"
        assert nrs >= 0, "Argument 'nrs' must be non-negative"
        assert indices.nspins >= nrs, "Number of spins in reduced density matrix ({}) cannot "\
                "exceed total number of spins ({})".format(nrs, indices.nspins)
        
        self.nrs = nrs # number of spins in reduced density matrix
        self.indices = indices
        self.convert_rho_block_dic = {}
        self.initial= []
        
        
        # setup initial state
        t0 = time()
        print('Set up initial and reduced density matrix...')
        self.initial = self.setup_initial(rho_p, rho_s)
        
        # setup reduced density matrix
        self.setup_convert_rho_block_nrs()
        elapsed= time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
        
        
    
    def setup_initial(self,rho_p, rho_s):
        """Calculate the block representation of the initial state 
        with photon in state rho_p and all spins in state rho_s"""
        indices = self.indices
        num_elements = len(indices.indices_elements)
        blocks = len(indices.mapping_block)
        
        # Check for superfluoresence initial condition, i.e. zero photons and all spins up. 
        # This is very easily initialized by all blocks zero, instead of the first entry of the last block
        if np.isclose(rho_p[0,0],1) and np.isclose(rho_s[0,0],1):
            rho_vec = [np.zeros(len(i)) for i in indices.mapping_block]
            rho_vec[blocks-1][0] = 1
            return rho_vec
                
        
        rho_vec = np.zeros(indices.ldim_p*indices.ldim_p*num_elements, dtype = complex)    
        for count_p1 in range(indices.ldim_p):
            for count_p2 in range(indices.ldim_p):
                for count in range(num_elements):
                    element = indices.indices_elements[count]
                    element_index = indices.ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                    left = element[0:indices.nspins]
                    right = element[indices.nspins:2*indices.nspins]
                    rho_vec[element_index] = rho_p[count_p1, count_p2]
                    for count_ns in range(indices.nspins):
                        rho_vec[element_index] *= rho_s[left[count_ns], right[count_ns]]
                        
        # Now use the mapping list to get the desired block structure from the whole rho_vec:
        rho_vec_block = []
        for count in range(blocks):
            rho_vec_block.append(rho_vec[indices.mapping_block[count]])
        
        return rho_vec_block   
    
    def setup_convert_rho_block_nrs(self):
        indices = self.indices
        # Setup reduced density matrix
        nrs, ldim_s, ldim_p = self.nrs, indices.ldim_s, indices.ldim_p
        indices_elements, nspins = indices.indices_elements, indices.nspins
        num_blocks = len(indices.mapping_block)
        num_elements = [len(block) for block in indices.mapping_block]
        nu_max = num_blocks - 1

        convert_rho_block = [
                sp.lil_matrix(((ldim_p*ldim_s**nrs)**2, num), dtype=float)
                for num in num_elements
                ]

        self.convert_rho_block_dic[nrs] = convert_rho_block

        for count_p1 in range(ldim_p):
            for count_p2 in range(ldim_p):
                for count in range(len(indices_elements)):
                    left = indices_elements[count][0:nspins]
                    right = indices_elements[count][nspins:2*nspins]
                    m_left = nspins-sum(left)
                    m_right = nspins-sum(right)
                    nu_left = m_left + count_p1
                    nu_right = m_right + count_p2
                    if nu_left != nu_right or nu_left > nu_max:                   
                        continue
                    nu = nu_left
                    diff_arg = np.asarray(left != right).nonzero()[0] # indices where bra and ket differ (axis=0)
                    diff_num = len(diff_arg) # number of different spin elements
                    if diff_num > nrs:
                        continue
                    diff_left = left[diff_arg]
                    diff_right = right[diff_arg]
                    same = np.delete(left, diff_arg) # common elements
                    element_index = ldim_p*len(indices_elements)*count_p1 + len(indices_elements)*count_p2 + count
                    block_element_index = next((i for i, index in enumerate(indices.mapping_block[nu]) if index == element_index), None)
                    if block_element_index is None:
                        print('CRITICAL: mapping_block at nu={nu} is no index {element_index}!')
                        sys.exit(1)
                    # fill all matrix elements in column element_index according to different and same spins
                    self.add_all_block(nrs, nu, count_p1, count_p2, diff_left, diff_right, same, block_element_index)
        #for nu in range(num_blocks):
        #    #for element_index in mapping_block[nu]:
        #        #if element_index
        #    for count in range(len(indices_elements)):
        #        left = indices_elements[count][0:nspins]
        #        right = indices_elements[count][nspins:2*nspins]
        #        count_p1 = nu - (nspins - sum(left)) # number of photons in left
        #        count_p2 = nu - (nspins - sum(right)) # number of photons in right
        #        if count_p1 < 0 or count_p2 < 0 or count_p1 > nu_max or count_p2 > nu_max:
        #            print(f'photon count incompatible with nu={nu}!')
        #            continue
        #        element_index = ldim_p*len(indices_elements)*count_p1 + len(indices_elements)*count_p2 + count
        #        if element_index not in mapping_block[nu]:
        #            print(f'{element_index} not in mapping block at nu={nu}!')
        #            continue
        #        diff_arg = np.asarray(left != right).nonzero()[0] # indices where bra and ket differ (axis=0)
        #        diff_num = len(diff_arg) # number of different spin elements
        #        if diff_num > nrs:
        #            continue
        #        # get elements that differ
        #        diff_left = left[diff_arg]
        #        diff_right = right[diff_arg]
        #        same = np.delete(left, diff_arg) # common elements
        #        element_index = count
        #        # fill all matrix elements in column element_index according to different and same spins
        #        add_all_block(nrs, nu, count_p1, count_p2, diff_left, diff_right, same, element_index)
        
        convert_rho_block = [block.tocsr() for block in convert_rho_block]

        return convert_rho_block
    
    
    def add_all_block(self, nrs, nu, count_p1, count_p2, left, right, same, block_element_index, s_start=0):
        """Populate all entries in conversion_matrix with row indices associated with permutations of spin values
        |left> and <right| and column index element_index according to the number of permutations of spin values in 
        'same'.

        nrs is the number of spins in the target reduced density matrix ('number reduced spins').
        """
        if len(left) == nrs:
            # add contributions from same to rdm at |bra><ket|
            self.add_to_convert_rho_block_dic(nrs, nu, count_p1, count_p2,
                                         left, right, same, block_element_index)
            return
        # current |left> too short for rdm, so move element from same to |left> (and <right|)
        # iterate through all possible values of spin...
        for s in range(s_start, self.indices.ldim_s):
            s_index = next((i for i,sa in enumerate(same) if sa==s), None)
            # ...but only act on the spins that are actually in same
            if s_index is None:
                continue
            # extract spin value from same, append to bra and ket
            tmp_same = np.delete(same, s_index)
            tmp_left = np.append(left, s)
            tmp_right = np.append(right, s)
            # repeat until |left> and <right| are correct length for rdm
            self.add_all_block(nrs, nu, count_p1, count_p2, tmp_left, tmp_right, tmp_same, block_element_index, s_start=s)
    
    def add_to_convert_rho_block_dic(self,nrs, nu, count_p1, count_p2, diff_left, diff_right, same, block_element_index):
        convert_rho_block = self.convert_rho_block_dic[nrs]
        # number of permutations of spins in same, each of which contributes one unit 
        combinations = _multinominal(np.bincount(same))
        row_indices = self.get_all_row_indices(count_p1, count_p2, diff_left, diff_right)
        for row_index in row_indices:
            convert_rho_block[nu][row_index, block_element_index] = combinations
            
    def get_all_row_indices(self,count_p1, count_p2, spin_bra, spin_ket):
        """Get all row indices of the conversion matrix corresponding to |count_p1><count_p2|
        for the photon state and |diff_bra><diff_ket| for the spin states."""
        assert len(spin_bra)==len(spin_ket)
        nrs = len(spin_bra)
        s_indices = np.arange(nrs)
        row_indices = []
        for perm_indices in permutations(s_indices):
            index_list = list(perm_indices)
            row_indices.append(self.get_rdm_index(count_p1, count_p2,
                                             spin_bra[index_list],
                                             spin_ket[index_list]))
        return row_indices
    
    def get_rdm_index(self,count_p1, count_p2, spin_bra, spin_ket):
        """Calculate row index in conversion matrix for element |count_p1><count_p2| for the 
        photon part and |spin_bra><spin_ket| for the spin part.

        This index is according to column-stacking convention used by qutip - see for example

        A=qutip.Qobj(numpy.arange(4).reshape((2, 2))
        print(qutip.operator_to_vector(A))
        """
        bra = np.concatenate(([count_p1],spin_bra))
        ket = np.concatenate(([count_p2],spin_ket))
        row = 0
        column = 0
        nrs = len(bra)-1
        for i in range(nrs+1):
            j = nrs-i
            row += bra[j] * self.indices.ldim_s**i
            column += ket[j] * self.indices.ldim_s**i
        return row + column * self.indices.ldim_p * self.indices.ldim_s**nrs
    
    
    

if __name__ == '__main__':
    # Testing purposes
    
    # same parameters as in Peter Kirton's code.
    ntls =2#number 2LS
    nphot = ntls+1
    w0 = 1.0
    wc = 0.65
    Omega = 0.4
    g = Omega / np.sqrt(ntls)
    kappa = 0.011
    gamma = 0.02
    gamma_phi = 0.03
    indi = Indices(ntls)
    L = BlockDicke(wc, w0,g, kappa, gamma_phi/4,gamma, indi)
    rho = Rho(basis(nphot,0), basis(2,0), indi) # initial condition with zero photons and all spins up.
    













