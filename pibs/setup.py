#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool
# from pibs.util import export, timeit, tensor, qeye, destroy, create, sigmap, sigmam, basis
# from pibs.util import sigmaz, degeneracy_spin_gamma, degeneracy_gamma_changing_block_efficient
# from pibs.util import states_compatible, permute_compatible, degeneracy_outer_invariant_optimized
# from pibs.util import _multinominal
# from pibs.propagate import Progress

from util import export, timeit, tensor, qeye, destroy, create, sigmap, sigmam, basis
from util import sigmaz, degeneracy_spin_gamma, degeneracy_gamma_changing_block_efficient
from util import states_compatible, permute_compatible, degeneracy_outer_invariant_optimized
from util import _multinominal
from util import Progress

import os, sys, logging
import pickle
from time import time
import scipy.sparse as sp
from itertools import permutations


import multiprocessing

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
    def __init__(self, nspins, nphot=None,spin_dim=None, verbose=True, debug=False, save=True, index_path=None, suppress_output=False):
        """ debug: Do not load existing file, always calculate new set of indices"""
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
        self.difference_block = []
        self.difference_block_inv = []
        
        # loading/saving paths
        if index_path is None:
            index_path = 'data/indices/'
        filename = f'indices_Ntls{self.nspins}_Nphot{self.ldim_p}_spindim{self.ldim_s}.pkl'
        
        if debug is False:
            # check if an object with the same arguments already exists in data/indices/ folder
            index_files = os.listdir(index_path)
            if (any([f == filename for f in index_files])):
                self.load(index_path+filename)
                return
            
        # debug true -> always calculate spin indices anew, or if not save file is found
        # setup indices
        if not suppress_output:
            print(f'Running setup indices with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
        t0 = time()
        self.list_equivalent_elements()
        elapsed = time()-t0
        if not suppress_output:
            print(f'Complete {elapsed:.0f}s', flush=True)

        # setup mapping block
        if not suppress_output:
            print(f'Running setup indices block with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
        t0 = time()
        self.setup_mapping_block()
        elapsed = time()-t0
        if not suppress_output:
            print(f'Complete {elapsed:.0f}s', flush=True)
        
        if save:
            # export for future use, if save is true
            self.export(index_path + filename)
        for nu in range(len(self.mapping_block)):
            assert len(self.mapping_block[nu]) == len(self.elements_block[nu])
                
    
    def list_equivalent_elements(self):
        """Generate a list of elements [left, right] representing all permutation
        distinct spin elements |left><right|"""
        #get minimal list of left and right spin indices - in combined form (i.e. as a list of zetas)
        #Generate list of all unique zeta strings in reverse lexicographic order
        all_zetas = [np.zeros(self.nspins, dtype=int)]
        max_zeta = 3 * (self.ldim_s-1) # e.g. spin-1/2 -> s = 0,1 and zeta = 0, 1, 2, 3 = 2s_L + s_R
        self.recurse_lexi(all_zetas, 0, max_zeta)
       
        for count, zetas in enumerate(all_zetas):
            left, right = self._to_hilbert(zetas)
            spin_element = np.concatenate((left,right))
            self.indices_elements.append(spin_element) # elements are stored as np.array of spin values for left (bra) then for right (ket)
            self.indices_elements_inv[tuple(zetas)] = count # mapping from combined form to index of indices_elements

    def recurse_lexi(self, all_zetas, current_index, max_zeta):
        """Generate successive strings of zetas, appending to all_zetas list"""
        previous_element = all_zetas[-1]
        for zeta in range(1, max_zeta+1):
            next_element = np.copy(previous_element)
            next_element[current_index] = zeta
            all_zetas.append(next_element)
            if current_index < self.nspins-1:
                self.recurse_lexi(all_zetas, current_index+1, zeta) 

    
    def setup_mapping_block(self):
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
        self.difference_block = [ [] for _ in range(nu_max+1)] # used to create difference_block_inv
        self.difference_block_inv = {nu:[] for nu in range(nu_max+1)} # useful for rdm conversion matrix calculations

        for count in range(num_elements):
            element = self.indices_elements[count]
            left = element[0:self.nspins]
            right = element[self.nspins:2*self.nspins]
            m_left = self.nspins-sum(left)
            m_right = self.nspins-sum(right)
            num_diff = sum(left != right)
            nu_min = max(m_left, m_right) # can't have fewer than m_left+0 photons (or m_right+0photons) excitations
            for nu in range(nu_min, nu_max+1):
                count_p1 = nu - m_left
                count_p2 = nu - m_right
                element_index = self.ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                el = np.concatenate(([count_p1], left, [count_p2], right))
                # can't use inverse here we sort the mapping block below :(
                #self.difference_block_inv[num_diff].append((nu, len(self.mapping_block[nu])))
                self.mapping_block[nu].append(element_index)
                self.elements_block[nu].append(el)
                self.difference_block[nu].append(num_diff)
        # Re-order to match that of earlier implementations
        for nu in range(nu_max+1):
            # zip-sort-zip - a personal favourite Python One-Liner
            self.mapping_block[nu], self.elements_block[nu], \
            self.difference_block[nu] =\
                    zip(*sorted(zip(self.mapping_block[nu],
                                    self.elements_block[nu],
                                    self.difference_block[nu])))
            # have to populate inv AFTER sort
            for i, num_diff in enumerate(self.difference_block[nu]):
                self.difference_block_inv[num_diff].append((nu, i))
    
    def _to_hilbert(self, combined):
        """Convert zeta-string to |left> and <right| spin values"""
        right = combined % self.ldim_s
        left = (combined - right)//self.ldim_s
        return left, right


    def export(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
            
        print(f'Storing Indices for later use in {filepath}')


    def load(self, filepath):
        with open(filepath, 'rb') as handle:
            indices_load = pickle.load(handle)
        self.__dict__ = indices_load.__dict__
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
    def __init__(self, indices, parallel=0,num_cpus=None, debug=False, save=True, progress=False, liouv_path=None):
        # initialisation
        self.L0_basis = {'sigmaz': [],
                         'sigmam': [],
                         'a': [],
                         'H_n': [],
                         'H_sigmaz': [],
                         'H_g': []}
        self.L1_basis = {'sigmam': [],
                         'a': []}
        self.num_cpus = num_cpus
        
        if liouv_path is None:
            liouv_path = 'data/liouvillians/'
        filename = f'liouvillian_dicke_Ntls{indices.nspins}_Nphot{indices.ldim_p}_spindim{indices.ldim_s}.pkl'
        
        if debug is False:
            # check if an object with the same arguments already exists in data/liouvillian/ folder
            liouv_files = os.listdir(liouv_path)
            if (any([f == filename for f in liouv_files])):
                self._load(liouv_path+filename, indices)
                return
            
        # if not, calculate them
        t0 = time()
        pname = {0:'serial', 1:'parallel', 2:'parallel2 (WARNING: memory inefficient, testing only)'}
        pfunc = {0: self.setup_L_block_basis, 1: self.setup_L_block_basis_parallel,
                 2: self.setup_L_block_basis_parallel2}
        try:
            print(f'Calculating normalised Liouvillian {pname[parallel]}...')
            pfunc[parallel](indices, progress)
        except KeyError as e:
            print('Argument parallel={parallel} not recognised')
            raise e
        elapsed = time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
        
        if save:
            # export normalized Liouvillians for later use, if save is true
            self.export(liouv_path+filename)
        
        
    
    def export(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
        print(f'Storing Liouvillian basis for later use in {filepath}')
            
    def _load(self, filepath,ind):
        with open(filepath, 'rb') as handle:
            L_load = pickle.load(handle)
            
        self.L0_basis = L_load.L0_basis
        self.L1_basis = L_load.L1_basis
        
        # at least tell user what they loaded
        print(f'Loaded Liouvillian basis file with ntls={ind.nspins}, nphot={ind.ldim_p}, spin_dim={ind.ldim_s}')
    
    @staticmethod    
    def sparse_constructor_dic(shape):
        # (data, (coords_x, coords_y)
        return {'data':[], 'coords':[[],[]], 'shape':shape}
    @staticmethod
    def new_entry(L_dic, name, count_in, count_out, data):
        # function to add data and coords to target L dictionary and name
        L_dic[name]['data'].append(data)
        L_dic[name]['coords'][0].append(count_in)
        L_dic[name]['coords'][1].append(count_out)
    
    def setup_L_block_basis(self, indices, progress):
       """ Calculate Liouvillian basis in block form"""
       num_blocks = len(indices.mapping_block)
       
       if progress:
           num_elements = sum([len(indices.mapping_block[nu]) for nu in range(num_blocks)])
           bar = Progress(num_elements, 'Calculate L basis...')
       
       #------------------------------------------------------
       # First, get L0 part -> coupling to same block, 
       #------------------------------------------------------
       
       # loop through all elements in block structure
       for nu_element in range(num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']
           L0_new ={name:self.sparse_constructor_dic((current_blocksize, current_blocksize)) for name in names}
           if nu_element < num_blocks-1:
               next_blocksize = len(indices.mapping_block[nu_element+1])
               # Liouvillian terms coupling to next block
               names = ['sigmam', 'a']
               L1_new ={name:self.sparse_constructor_dic((current_blocksize, next_blocksize)) for name in names}
           
           # Loop through all elements in the same block
           for count_in in range(current_blocksize):
               if progress:
                   bar.update()
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
                       self.new_entry(L0_new, 'H_n', count_in, count_out, -1j * (left[0]-right[0]))
                       self.new_entry(L0_new, 'H_sigmaz', count_in, count_out, 1j*(s_down_left-s_down_right))
                   
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
                            self.new_entry(L0_new, 'H_g', count_in, count_out, - 1j*deg * np.sqrt(left[0]))
                        elif left[0] - left_to_couple[0] == -1 and sum(left[1:])-sum(left_to_couple[1:]) == -1 : # need matrix element of a*sigmap
                            self.new_entry(L0_new, 'H_g', count_in, count_out, - 1j*deg * np.sqrt(left[0]+1))
                               
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
                            self.new_entry(L0_new, 'H_g', count_in, count_out,  1j*deg * np.sqrt(right[0]))
                        elif right[0] - right_to_couple[0] == -1 and sum(right[1:])-sum(right_to_couple[1:]) == -1: # need matrix element of adag*sigmam
                            self.new_entry(L0_new, 'H_g', count_in, count_out,  1j*deg * np.sqrt(right[0]+1))

                   
                   
                   # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
                   # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                   # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                   if (right_to_couple == right).all() and (left_to_couple == left).all():
                       deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                       deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                       self.new_entry(L0_new, 'sigmam', count_in, count_out,  - 1/2 * (deg_left+deg_right))
                   
                   # L0 part from L[sigmaz] -> whole dissipator
                   # Left and right states must be equal, because sigmaz is diagonal in the spins.
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                       self.new_entry(L0_new, 'sigmaz', count_in, count_out, 2*(equal - indices.nspins))
                       
                   # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       self.new_entry(L0_new, 'a', count_in, count_out, -1/2*(left[0] + right[0]))


                   
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
                           self.new_entry(L1_new, 'sigmam', count_in, count_out, deg)
                   
                   # L1 part from L[a] -> a * rho* adag
                   # since spins remain the same, first check if spin states match
                   # if spins match, then the element can couple, because we are looping through the block nu+1. Therefore
                   # the coupled-to-elements necessarily have one more excitation, which for this case is in the photon state.
                   if (left[1:] == left_to_couple[1:]).all() and (right[1:]==right_to_couple[1:]).all():
                       self.new_entry(L1_new, 'a', count_in, count_out,  np.sqrt((left[0]+1)*(right[0] + 1)))
           
            
           # append new blocks to the basis as sparse matrices (CSR format)
           for name in self.L0_basis:
               Lnew = L0_new[name]
               data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
               self.L0_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())
           
           if nu_element < num_blocks-1: 
               for name in self.L1_basis:
                   Lnew = L1_new[name]
                   data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
                   self.L1_basis[name].append(sp.coo_matrix((data,coords), shape=shape).tocsr())
    
                
    # functions for parallelization
    @staticmethod
    def calculate_L0_line(args_tuple):
        """ Calculate L0 part of element count_in in block nu_element """
        global elements_block, new_entry, sparse_constructor_dic
        
        nu_element, count_in = args_tuple

        current_element_block = elements_block[nu_element]
        current_blocksize = len(current_element_block)
        # get element, of which we want the time derivative
        element = current_element_block[count_in]
        left = element[0:nspins+1] # left state, first index is photon number, rest is spin states
        right = element[nspins+1:2*nspins+2] # right state
    
        
        # initialize Liouvillian rows for element count_in
        names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']
        L0_line = {name:sparse_constructor_dic((current_blocksize, current_blocksize)) for name in names}
        new_entry_func = lambda name, count_out, val: new_entry(L0_line, name, count_in, count_out, val)
        
        # now loop through all matrix elements in the same block, to get L0 couplings
        for count_out in range(current_blocksize):
            # get "to couple" element
            element_to_couple = current_element_block[count_out]
            left_to_couple = element_to_couple[0:nspins+1]
            right_to_couple = element_to_couple[nspins+1:2*nspins+2]
            
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
                new_entry_func('H_n', count_out, -1j * (left[0]-right[0]))
                new_entry_func('H_sigmaz', count_out, 1j*(s_down_left-s_down_right))
            
            # offdiagonal parts
            elif(states_compatible(right, right_to_couple)):
                 # if they are compatible, permute left_to_couple appropriately for proper H element
                 left_to_couple_permute = np.copy(left_to_couple)
                 if not (right_to_couple == right).all():
                     # if they are compatible but not equal, we need to permute left_to_couple appropriately, to get correct matrix element of H
                     left_to_couple_permute[1:] = permute_compatible(right[1:],right_to_couple[1:],left_to_couple[1:])
                     
                 # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                 if (left[1:]==left_to_couple_permute[1:]).sum() != nspins-1:
                     continue
                 
                 deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], left_to_couple_permute[1:]) # degeneracy from simulatneous spin permutations, which leave outer spins invariant
                 # check if photon number in left state increases or decreases and
                 # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                 if (left[0] - left_to_couple[0]) == 1 and sum(left[1:])-sum(left_to_couple[1:]) == 1: # need matrix element of adag*sigmam
                     new_entry_func('H_g', count_out, - 1j*deg * np.sqrt(left[0]))

                 elif (left[0] - left_to_couple[0] == -1) and sum(left[1:])-sum(left_to_couple[1:]) == -1 : # need matrix element of a*sigmap
                     new_entry_func('H_g', count_out,- 1j*deg * np.sqrt(left[0]+1))
                        
            elif(states_compatible(left, left_to_couple)):            
                 # if they are compatible, permute right_to_couple appropriately for proper H element
                 right_to_couple_permute = np.copy(right_to_couple)
                 if not (left_to_couple == left).all():
                     right_to_couple_permute[1:] = permute_compatible(left[1:],left_to_couple[1:],right_to_couple[1:])
                     
                 # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                 if (right[1:]==right_to_couple_permute[1:]).sum() != nspins-1:
                     continue
                 deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], right_to_couple_permute[1:])
                 # check if photon number in right state increases or decreases and
                 # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                 if (right[0] - right_to_couple[0]) == 1 and sum(right[1:])-sum(right_to_couple[1:]) == 1: # need matrix element of a*sigmap
                     new_entry_func('H_g', count_out, 1j*deg * np.sqrt(right[0]))
                 elif right[0] - right_to_couple[0] == -1 and sum(right[1:])-sum(right_to_couple[1:]) == -1: # need matrix element of adag*sigmam
                     new_entry_func('H_g', count_out, 1j*deg * np.sqrt(right[0]+1))
    
            
            
            # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
            # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
            # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
            if (right_to_couple == right).all() and (left_to_couple == left).all():
                deg_right = degeneracy_spin_gamma(right_to_couple[1:nspins+1], right[1:nspins+1]) # degeneracy: because all spin up elements contribute equally
                deg_left = degeneracy_spin_gamma(left_to_couple[1:nspins+1], left[1:nspins+1])
                new_entry_func('sigmam', count_out, - 1/2 * (deg_left+deg_right))
            
            # L0 part from L[sigmaz] -> whole dissipator
            # Left and right states must be equal, because sigmaz is diagonal in the spins.
            if (left_to_couple == left).all() and (right_to_couple == right).all():
                equal = (left[1:nspins+1] == right[1:nspins+1]).sum()
                new_entry_func('sigmaz', count_out,  2*(equal - nspins))
                
            # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
            if (left_to_couple == left).all() and (right_to_couple == right).all():
                new_entry_func('a', count_out, -1/2*(left[0] + right[0]))
        return L0_line

    @staticmethod
    def calculate_L1_line(args_tuple):
        """ Calculate L1 part of element count_in in block nu_element """
        
        #indices,count_in, nu_element = args_tuple
        nu_element, count_in = args_tuple
        global elements_block, new_entry, sparse_constructor_dic
        
        # get element, of which we want the time derivative
        current_element = elements_block[nu_element][count_in]
        current_blocksize = len(elements_block[nu_element])
    
        left = current_element[0:nspins+1] # left state, first index is photon number, rest is spin states
        right = current_element[nspins+1:2*nspins+2] # right state
            
        
        # Now get L1 part -> coupling from nu_element to nu_element+1
        # loop through all matrix elements in the next block we want to couple to
        next_element_block = elements_block[nu_element+1]
        next_blocksize = len(elements_block[nu_element+1])

        names = ['sigmam', 'a']
        L1_line = {name:sparse_constructor_dic((current_blocksize, next_blocksize)) for name in names}
        new_entry_func = lambda name, count_out, val: new_entry(L1_line, name, count_in, count_out, val)
        for count_out in range(next_blocksize):
            # get "to couple" element
            element_to_couple = next_element_block[count_out]
            left_to_couple = element_to_couple[0:nspins+1]
            right_to_couple = element_to_couple[nspins+1:2*nspins+2]
            
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
                    new_entry_func('sigmam', count_out, deg)
            
            # L1 part from L[a] -> a * rho* adag
            # since spins remain the same, first check if spin states match
            # if spins match, then the element can couple, because we are looping through the block nu+1. Therefore
            # the coupled-to-elements necessarily have one more excitation, which for this case is in the photon state.
            if (left[1:] == left_to_couple[1:]).all() and (right[1:]==right_to_couple[1:]).all():
                new_entry_func('a', count_out, np.sqrt((left[0]+1)*(right[0] + 1)))
    
        return L1_line
    
    def setup_L_block_basis_parallel(self, indices, progress):
       """ Calculate Liouvillian basis in block form. Parallelize the calculation
       of rows of the Liouvillian"""
       num_blocks = len(indices.mapping_block)
       #multiprocessing.set_start_method('fork')
       if progress:
           num_elements = sum([len(indices.mapping_block[nu]) for nu in range(num_blocks)])
           bar = Progress(2*num_elements, 'Calculate L basis...') # 2 updates per block
       # loop through all elements in block structure
       for nu_element in range(num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           

           L0_names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']
           L0_new = {name:self.sparse_constructor_dic((current_blocksize, current_blocksize))
                     for name in L0_names}
           
           arglist = []
           global nspins, elements_block, sparse_constructor_dic, new_entry
           nspins  = indices.nspins
           elements_block = indices.elements_block
           sparse_constructor_dic = self.sparse_constructor_dic
           new_entry = self.new_entry
        #nu_element, count_in = args_tuple
           for count_in in range(current_blocksize):
               arglist.append((nu_element, count_in))
           #print(f'Block {nu_element}/{num_blocks}: {len(arglist)} args')
           with Pool(processes=self.num_cpus) as pool:
               #print('Number of processes:', pool._processes)
               for L0_data in pool.imap(self.calculate_L0_line, arglist):
                   for name in L0_names:
                       L0_new[name]['data'].extend(L0_data[name]['data'])
                       L0_new[name]['coords'][0].extend(L0_data[name]['coords'][0])
                       L0_new[name]['coords'][1].extend(L0_data[name]['coords'][1])
                   if progress:
                       bar.update()
           for name in L0_names:
               Lnew = L0_new[name]
               data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
               self.L0_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())
           
           if nu_element < num_blocks -1:
               next_blocksize = len(indices.mapping_block[nu_element+1])
               L1_names = ['sigmam', 'a']
               L1_new = {name:self.sparse_constructor_dic((current_blocksize, next_blocksize))
                         for name in L1_names}
               with Pool(processes=self.num_cpus) as pool:
                   #print('Number of processes:', pool._processes)
                   for L1_data in pool.imap(self.calculate_L1_line, arglist):
                       for name in L1_names:
                           L1_new[name]['data'].extend(L1_data[name]['data'])
                           L1_new[name]['coords'][0].extend(L1_data[name]['coords'][0])
                           L1_new[name]['coords'][1].extend(L1_data[name]['coords'][1])
                   if progress:
                       bar.update()
               for name in L1_names:
                   Lnew = L1_new[name]
                   data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
                   self.L1_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())
           else:
               if progress:
                   bar.update(2*num_elements-1)
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
            
    
    
    @staticmethod
    def L0_nu_task(nu_element):
        current_blocksize = len(elements_block[nu_element])
        # setup the Liouvillians for the current block

        L0_new ={'sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'sigmam': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'a': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_n': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_sigmaz': np.zeros((current_blocksize, current_blocksize), dtype=complex),
                 'H_g': np.zeros((current_blocksize, current_blocksize), dtype=complex)}
        
        
        for count_in in range(current_blocksize):
            L0_line = BlockL.calculate_L0_line((nu_element, count_in))
            
            for name in L0_new:
                L0_new[name][count_in,:] = L0_line[name]
    
        return L0_new

                
    @staticmethod     
    def L1_nu_task(nu_element):
        current_blocksize = len(elements_block[nu_element])
        next_blocksize = len(elements_block[nu_element+1])
        L1_new = {'sigmam': np.zeros((current_blocksize, next_blocksize), dtype=complex),
                  'a': np.zeros((current_blocksize, next_blocksize), dtype=complex)}

        
        for count_in in range(current_blocksize):
            L1_line = BlockL.calculate_L1_line((nu_element, count_in))
            for name in L1_new:
                L1_new[name][count_in,:] = L1_line[name]
        
        return L1_new

    def setup_L_block_basis_parallel2(self, indices, progress):
       """ Calculate Liouvillian basis in block form. Parallelized the calculation
       of each block"""
       num_blocks = len(indices.mapping_block)
       global nspins, elements_block
       nspins  = indices.nspins
       elements_block = indices.elements_block
       
       # loop through all elements in block structure
       arglist = [nu for nu in range(num_blocks)]
       
       with Pool(processes=self.num_cpus) as pool:
           L0s = pool.map(self.L0_nu_task, arglist)
       
       for name in self.L0_basis:
           for nu in range(num_blocks):
               self.L0_basis[name].append(L0s[nu][name])
       
       arglist = arglist[:-1] # not nu_max 
       with Pool(processes=self.num_cpus) as pool:
           L1s = pool.map(self.L1_nu_task, arglist)
           
       for name in self.L1_basis:
           for nu in range(num_blocks-1):
               self.L1_basis[name].append(L1s[nu][name])

       print('done')
       
       
   
                        
# class BlockDicke(BlockL):
#     """ Calculates the specific Liouvillian of the Tavis Cummings model. This class
#     inherits the Liouvillian basis from BlockL. With the specified parameters of the 
#     Hamiltonian and the collapse operators, one can calculate the Liouvillian.
    
#     Model:
#         H = wc*adag*a + sum_k { w0*sigmaz_k  + g*(a*sigmap_k + adag*sigmam_k) }
#         d/dt rho = -i[H,rho] + kappa*L[a] + gamma*L[sigmam] + gamma_phi*L[sigmaz]"""
#     def __init__(self,wc,w0,g, kappa, gamma_phi, gamma, indices, parallel=0,progress=False, debug=False):
#         # specify rates according to what part of Hamiltonian or collapse operators
#         # they scale
#         self.rates = {'H_n': wc,
#                       'H_sigmaz': w0,
#                       'H_g': g,
#                       'a': kappa,
#                       'sigmaz': gamma_phi,
#                       'sigmam': gamma}
#         self.w0 = w0
#         self.wc = wc
#         self.g = g
#         self.kappa = kappa
#         self.gamma = gamma
#         self.gamma_phi = gamma_phi
#         self.L0 = []
#         self.L1 = []
#         super().__init__(indices, parallel,debug)
        
#         t0 = time()
#         print('Calculating Liouvillian from basis...', flush =True)
#         self.setup_L(indices, progress)
#         elapsed = time()-t0
#         print(f'Complete {elapsed:.0f}s', flush=True)

#     def setup_L(self, indices, progress):
#         """ From the basic parts of the Liouvillian, get the whole Liouvillian
#         by proper scaling."""
        
#         num_blocks = len(indices.mapping_block)
        
#         if progress: # progress bar
#             bar = Progress(2*num_blocks-1,'Louvillian: ')
        
#         for nu in range(num_blocks):
#             current_blocksize = len(indices.mapping_block[nu])
#             L0_scale = np.zeros((current_blocksize, current_blocksize), dtype=complex)
#             for name in self.L0_basis:
#                 L0_scale = L0_scale + self.rates[name] * self.L0_basis[name][nu]
#             self.L0.append( sp.csr_matrix(L0_scale ))
            
#             if progress:
#                 bar.update()
            
#             if nu < num_blocks -1:
#                 next_blocksize = len(indices.mapping_block[nu+1])
#                 L1_scale = np.zeros((current_blocksize, next_blocksize), dtype=complex)
#                 for name in self.L1_basis:
#                     L1_scale = L1_scale + self.rates[name] * self.L1_basis[name][nu]
#                 self.L1.append( sp.csr_matrix(L1_scale))   
                
#                 if progress:
#                     bar.update()                  




class Models(BlockL):
    """ This class contains information about the exact model at hand and
    calculates the Liouvillian from the basis elements from BlockL.
    
    Demanding weak U(1) symmetry and no gain, the most general model of N spins
    interacting with a common photon mode is described by the Master equation
        
        d/dt rho = -i[H,rho] + kappa*L[a] + sum_k{ gamma*L[sigmam_k] + gamma_phi * L[sigmaz_k] }
        
    with Hamiltonian H = wc*adag*a + w0/2*sum_k{ sigmaz_k } + g*sum_k{adag*sigmam_k + a*sigmap_k }
    
    where the light-matter coupling g is assumed real.
    
    """
    def __init__(self,wc,w0,g, kappa, gamma_phi, gamma, indices, parallel=0,progress=False, debug=False, save=True, num_cpus=None, liouv_path=None):
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
        self.indices = indices
        self.L0 = []
        self.L1 = []
        super().__init__(indices=indices, parallel=parallel,num_cpus=num_cpus, debug=debug, save=save, progress=progress,liouv_path=liouv_path)
    
    def setup_L_Tavis_Cummings(self, progress=False, save_path=None):
        t0 = time()
        print('Calculating Liouvillian for TC model from basis ...', flush =True)
        
        self.L0 = []
        self.L1 = []
        
        num_blocks = len(self.indices.mapping_block)
        
        if progress: # progress bar
            bar = Progress(2*num_blocks-1,'Liouvillian: ')
        
        for nu in range(num_blocks):
            current_blocksize = len(self.indices.mapping_block[nu])
            L0_scale = sp.csr_matrix(np.zeros((current_blocksize, current_blocksize), dtype=complex))
            for name in self.L0_basis:
                L0_scale = L0_scale + self.rates[name] * self.L0_basis[name][nu]
            self.L0.append( L0_scale)
            
            if progress:
                bar.update()
            
            if nu < num_blocks -1:
                next_blocksize = len(self.indices.mapping_block[nu+1])
                L1_scale = sp.csr_matrix(np.zeros((current_blocksize, next_blocksize), dtype=complex))
                for name in self.L1_basis:
                    L1_scale = L1_scale + self.rates[name] * self.L1_basis[name][nu]
                self.L1.append(L1_scale)   
                
                if progress:
                    bar.update()
 
        elapsed = time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
        if save_path is not None:
            with open(save_path, 'wb') as handle:
                pickle.dump(self, handle)
            print(f'Wrote full model to {save_path}.')

    @classmethod
    def load(cls, filepath):
        """Load a previously saved model from .pkl file"""
        with open(filepath, 'rb') as handle:
            obj = pickle.load(handle)
        # check it is actually a valid Model object...
        print(f'Loaded {type(cls)} object from {filepath}')
        return obj

class Rho:
    """ Functionality related to density matrix:
        Initial state
        Reduced density matrix
        Calculation of expectation values
    """
        
    def __init__(self, rho_p, rho_s, indices, max_nrs=1):
        assert type(max_nrs) == int, "Argument 'max_nrs' must be int"
        assert max_nrs >= 0, "Argument 'max_nrs' must be non-negative"
        assert indices.nspins >= max_nrs, "Number of spins in reduced density matrix "\
                "(max_nrs) cannot exceed total number of spins ({indices.nspins})"
        
        self.max_nrs = max_nrs # maximum number of spins in reduced density matrix
        self.indices = indices
        self.convert_rho_block_dic = {}
        self.initial= []
        
        
        # setup initial state
        t0 = time()
        print('Set up initial density matrix...')
        self.initial = self.setup_initial(rho_p, rho_s)
        
        # setup reduced density matrix
        print('Set up mappings to reduced density matrices at...')
        for nrs in range(max_nrs+1):
            print(f'nrs = {nrs}...')
            self.setup_convert_rho_block_nrs(nrs)
        elapsed= time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
    
    def setup_initial(self, rho_p, rho_s):
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
    
    def setup_convert_rho_block_nrs(self, nrs):
        """Setup conversion matrix from supercompressed vector to vector form of
        reduced density matrix of the photon plus nrs (possibly 0) spins

        N.B. Takes advantage of counts of how many spins are different between
        ket (left) and bra (right) for a state with a given excitation nu and
        block_index, as stored in indices.different_block_inv

        Fills in self.convert_rho_block_dic with an entry with key nrs
        """
        indices = self.indices
        nspins = indices.nspins
        num_elements = [len(block) for block in indices.mapping_block]

        # initialise empty matrices for each block of the supercompressed form
        convert_rho_block = [
                sp.lil_matrix(((indices.ldim_p*indices.ldim_s**nrs)**2, num), dtype=float)
                for num in num_elements
                ]

        self.convert_rho_block_dic[nrs] = convert_rho_block # modified in place 

        # for reduced density matrix involving nrs spins, must consider elements
        # with a most nrs differences between ket (left) and bra (right) spin
        # states
        for num_diff in range(nrs+1):
            # difference_block_inv has a tuple (nu, block_index) describing all the
            # block elements and indices with a given num_diff differences
            block_element_tuples = indices.difference_block_inv[num_diff]
            for nu, block_index in block_element_tuples:
                # to calculation contribution to rdm, we need element
                element = indices.elements_block[nu][block_index]
                count_p1, left, count_p2, right = \
                        element[0], element[1:nspins+1], element[nspins+1], element[nspins+2:] 
                diff_arg = (left != right).nonzero()[0] # location of spins with different ket, bra states 
                same = np.delete(left, diff_arg) # spins with states same in ket and bra
                self.add_all_for_block_element(nrs, nu, block_index,
                                               count_p1, left[diff_arg],
                                               count_p2, right[diff_arg],
                                               same)
        convert_rho_block = [block.tocsr() for block in convert_rho_block]
    
    def add_all_for_block_element(self, nrs, nu, block_index,
                                  count_p1, left, count_p2, right, same, 
                                  s_start=0):
        """Populate all entries in conversion matrix at nu with reduced density matrix indices
        associated with permutations of spin values |left> and <right| and column index
        block_index, according to the number of permutations of spin values in 'same'.

        nrs is the number of spins in the target reduced density matrix ('number reduced spins').
        """
        if len(left) == nrs:
            # add contributions from 'same' to rdm at |left><right|
            self.add_to_convert_rho_block_dic(nrs, nu, block_index,
                                              count_p1, left,
                                              count_p2, right,
                                              same)
            return # end of recursion
        # current |left> too short for rdm, so move element from 'same' to |left> (and <right|)
        # iterate through all possible values of spin...
        for s in range(s_start, self.indices.ldim_s):
            s_index = next((i for i,sa in enumerate(same) if sa==s), None)
            # ...but only act on the spins that are actually in 'same'
            if s_index is None:
                continue
            # extract spin value from same, append to left and right
            tmp_same = np.delete(same, s_index)
            tmp_left = np.append(left, s)
            tmp_right = np.append(right, s)
            # repeat until |left> and <right| are correct length for rdm
            self.add_all_for_block_element(nrs, nu, block_index,
                                           count_p1, tmp_left,
                                           count_p2, tmp_right,
                                           tmp_same, s_start=s) # can skip up to s in next function call
    
    def add_to_convert_rho_block_dic(self, nrs, nu, block_index,
                                     count_p1, diff_left,
                                     count_p2, diff_right,
                                     same):
        """Calculate contribution to reduced density matrix element with spin state
        |diff_left><diff_right| (photon |count_p1><count_p1|) according to free
        permutations of 'same' """
        convert_rho_block = self.convert_rho_block_dic[nrs]
        # number of permutations of spins in same, each of which contributes one unit under trace 
        combinations = _multinominal(np.bincount(same))
        # get all vectorised reduced density matrix indices for element
        s_indices = np.arange(nrs)
        rdm_indices = []
        for perm_indices in permutations(s_indices):
            # spins in rdm are still identical, so need to populate elements are
            # all rdm indices associated with (distinct) permutations of the nrs spin
            # N.B. duplication occurs here, use more_itertools.distinct_permutations() to
            # avoid; only costly for large nrs (?)
            index_list = list(perm_indices)
            rdm_indices.append(self.get_rdm_index(count_p1, diff_left[index_list],
                                                  count_p2, diff_right[index_list]))
        for rdm_index in rdm_indices:
            convert_rho_block[nu][rdm_index, block_index] = combinations
            
    def get_rdm_index(self, count_p1, left, count_p2, right):
        """Calculate index in vectorised reduced density matrix 
        for element |count_p1><count_p2|(X)|left><right|

        This index is according to column-stacking convention used by qutip; see e.g.

        A=qutip.Qobj(numpy.arange(4).reshape((2, 2))
        print(qutip.operator_to_vector(A))

        I can't remember writing this magic - pfw
        """
        ket = np.concatenate(([count_p1], left))
        bra = np.concatenate(([count_p2], right))
        row = 0
        column = 0
        nrs = len(ket)-1
        for i in range(nrs+1):
            j = nrs-i
            row += ket[j] * self.indices.ldim_s**i
            column += bra[j] * self.indices.ldim_s**i
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













