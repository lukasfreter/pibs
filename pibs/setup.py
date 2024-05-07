#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool
from util import export, timeit, tensor, qeye, destroy, create, sigmap, sigmam, sigmaz, degeneracy_spin_gamma, degeneracy_gamma_changing_block_efficient
import os, sys, logging
import pickle
from time import time
import scipy.sparse as sp

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
    """Liouvillian for a given system in Block form. As a requirement, the Master
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
    L[sigma_m].
    
    Strategy: Calculate the Liouvillian for the collapse operators L[a],L[sigma_m],
    L[sigma_z] once for unit dissipation rates, and store them for later use.
    """
    def __init__(self, kappa, gamma, gamma_phi, indices, H):
        """
        kappa: photon loss rate
        gamma: exciton loss rate
        gamma_phi: dephasing rate
        indices: Object of Indices class
        H: Object of a Hamiltonian class, for example DickeH"""
        
        # initialisation
        self.kappa = kappa
        self.gamma = gamma
        self.gamma_phi = gamma_phi
        self.L0_sigmaz = []
        self.L0_sigmam = []
        self.L1_sigmam = []
        self.L0_a = []
        self.L1_a = []
        self.L0_H = []
        
        # check if Liouvillians already exist in file
        
        
        # if not, calculate them
        print('Calculating L...')
        t0 = time()
        self.setup_L_block(indices)
        elapsed = time()-t0
        print(f'Complete {elapsed:.0f}s', flush=True)
   
    
   
    
    def setup_L_block(self, indices):
       """ Calculate Liouvillian in block form"""
       num_blocks = len(indices.mapping_block)
       
       # First, get L0 part -> coupling to same block, 
       # loop through all elements in block structure
       for nu_element in range(num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           L0_sigmam_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)
           L0_sigmaz_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)
           L0_a_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)
           L0_H_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)

           
           # Loop through all elements in the block
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
                   
                   #-----------------------------
                   # get Liouvillian elements
                   #-----------------------------
                  
                   # L0 part from Hamiltonian
                                       
                   
                   # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
                   # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                   # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                   if (right_to_couple == right).all() and (left_to_couple == left).all():
                       deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                       deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                       L0_sigmam_nu[count_in,count_out] = - 1/2 * (deg_left+deg_right)
                   
                   # L0 part from L[sigmaz] -> whole dissipator
                   # Left and right states must be equal, because sigmaz is diagonal in the spins.
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                       L0_sigmaz_nu[count_in][count_out] = 2*(equal - indices.nspins)
                       
                   # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
                   if (left_to_couple == left).all() and (right_to_couple == right).all():
                       L0_a_nu[count_in][count_out] = -1/2*(left[0] + right[0]) 
           self.L0_sigmam.append(sp.csr_matrix(L0_sigmam_nu))
           self.L0_sigmaz.append(sp.csr_matrix(L0_sigmaz_nu))
           self.L0_a.append(sp.csr_matrix(L0_a_nu))
                   
       
       # Now get L1 part -> coupling from nu_element to nu_element+1
       for nu_element in range(num_blocks):
           if nu_element == num_blocks-1:
               continue
           current_blocksize = len(indices.mapping_block[nu_element])
           next_blocksize = len(indices.mapping_block[nu_element+1])
           L1_sigmam_nu = np.zeros((current_blocksize, next_blocksize), dtype=complex)
           L1_a_nu = np.zeros((current_blocksize, next_blocksize), dtype=complex)

           
           for count_in in range(current_blocksize):
               # get element, of which we want the time derivative
               element = indices.elements_block[nu_element][count_in]
               left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
               right = element[indices.nspins+1:2*indices.nspins+2] # right state
               
               # now loop through all matrix elements in the next block we want to couple to
               for count_out in range(next_blocksize):
                   # get "to couple" element
                   element_to_couple = indices.elements_block[nu_element+1][count_out]
                   left_to_couple = element_to_couple[0:indices.nspins+1]
                   right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                   
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
                           L1_sigmam_nu[count_in,count_out] = deg
                   
                   # L1 part from L[a] -> a * rho* adag
                   # since spins remain the same, first check if spin states match
                   # if spins match, then the element can couple, because we are looping through the block nu+1. Therefore
                   # the coupled-to-elements necessarily have one more excitation, which for this case is in the photon state.
                   if (left[1:] == left_to_couple[1:]).all() and (right[1:]==right_to_couple[1:]).all():
                       L1_a_nu[count_in][count_out] = np.sqrt((left[0]+1)*(right[0] + 1))
           self.L1_sigmam.append(sp.csr_matrix(L1_sigmam_nu))
           self.L1_a.append(sp.csr_matrix(L1_a_nu))

        
    
    def setup_L_block_H(self):
        """ Get Liouvillian part corresponding to -i[H,rho] """
        
    def setup_L_block_sigmaz(self, indices):
        """ Get Liouvillian part corresponding to L[sigmaz] for gamma_phi=1.
        This liouvillian can then be scaled by any gamma_phi."""
        num_blocks = len(indices.mapping_block)

        # loop through all elements in block structure
        for nu_element in range(num_blocks):
            current_blocksize = len(indices.mapping_block[nu_element])
            L0_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)
            
            # For each nu, calculate part of the liouvillian that couples
            # to same nu, stored in L0
            for count_in in range(current_blocksize):
                # get element, of which we want the time derivative
                element = indices.elements_block[nu_element][count_in]
                left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
                right = element[indices.nspins+1:2*indices.nspins+2] # right state
                
                # now loop through all matrix elements, that could couple to "element"
                # for L[sigmaz], they are all in the same block.
                for count_out in range(current_blocksize):
                    # get "to couple" element
                    element_to_couple = indices.elements_block[nu_element][count_out]
                    left_to_couple = element_to_couple[0:indices.nspins+1]
                    right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                    
                    # get Liouvillian element. Left and right states must be equal,
                    # because sigmaz is diagonal in the spins.
                    if (left_to_couple == left).all() and (right_to_couple == right).all():
                        equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                        L0_nu[count_in][count_out] = 2*(equal - indices.nspins)
            self.L0_sigmaz.append(sp.csr_matrix(L0_nu))
        
    def setup_L_block_sigmam(self, indices):
        """ Get Liouvillian part corresponding to L[sigmam] """
        num_blocks = len(indices.mapping_block)
        
        # First, get L0_sigmam -> coupling to same block, from terms -sigmap*sigmam*rho - rho*sigmap*sigmam
        # loop through all elements in block structure
        for nu_element in range(num_blocks):
            current_blocksize = len(indices.mapping_block[nu_element])
            L0_nu = np.zeros((current_blocksize, current_blocksize), dtype=complex)
            
            # For each nu, calculate part of the liouvillian that couples
            # to same nu, stored in L0
            for count_in in range(current_blocksize):
                # get element, of which we want the time derivative
                element = indices.elements_block[nu_element][count_in]
                left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
                right = element[indices.nspins+1:2*indices.nspins+2] # right state
                
                # now loop through all matrix elements in that block
                for count_out in range(current_blocksize):
                    # get "to couple" element
                    element_to_couple = indices.elements_block[nu_element][count_out]
                    left_to_couple = element_to_couple[0:indices.nspins+1]
                    right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                    
                    # get Liouvillian element
                    # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                    # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                    if (right_to_couple == right).all() and (left_to_couple == left).all():
                        deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                        deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                        L0_nu[count_in,count_out] = - 1/2 * (deg_left+deg_right)
            self.L0_sigmam.append(sp.csr_matrix(L0_nu))
        
        # Now get L1_sigmam, which couples elements from nu to nu+1
        for nu_element in range(num_blocks):
            if nu_element == num_blocks-1:
                continue
            current_blocksize = len(indices.mapping_block[nu_element])
            next_blocksize = len(indices.mapping_block[nu_element+1])
            L1_nu = np.zeros((current_blocksize, next_blocksize), dtype=complex)
            
            for count_in in range(current_blocksize):
                # get element, of which we want the time derivative
                element = indices.elements_block[nu_element][count_in]
                left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
                right = element[indices.nspins+1:2*indices.nspins+2] # right state
                
                # now loop through all matrix elements in the next block we want to couple to
                for count_out in range(next_blocksize):
                    # get "to couple" element
                    element_to_couple = indices.elements_block[nu_element+1][count_out]
                    left_to_couple = element_to_couple[0:indices.nspins+1]
                    right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                    
                    # get Liouvillian element
                    # L[sigma^-] contribution sigma^- * rho * sigma^+. changes spin excitation number.
                    # Photons must remain the same
                    if (left[0] == left_to_couple[0] and right[0] == right_to_couple[0]):
                        # we have to compute matrix elements of sigma^- and sigma^+. Therefore, check first if 
                        # number of spin up in "right" and "right_to_couple" as well as "left" and "left_to_coupole" vary by one
                        if (sum(left[1:]) - sum(left_to_couple[1:]) == 1) and (sum(right[1:]) - sum(right_to_couple[1:]) == 1):       
                            # Get the number of permutations, that contribute.                             
                            deg = degeneracy_gamma_changing_block_efficient(left[1:], right[1:], left_to_couple[1:], right_to_couple[1:])                
                            L1_nu[count_in,count_out] = deg
            self.L1_sigmam.append(sp.csr_matrix(L1_nu))
            
            
                       
                        
                        

        
class DickeH:
    """ Dicke Hamiltonian.
    H = wc*adag*a + w0*sz  + g*(a*sp + adag*sm) 
    c_ops = kappa L[a] + gamma_phi L[sigmaz] + gamma L[sigmam]"""
    
    def __init__(self, w0, wc, Omega, indices):
        self.w0 = w0
        self.wc = wc
        self.g = Omega / indices.nspins
        self.H = []
        
        # setup Hamiltonian
        self.setup_Dicke(indices)
         
        
    def setup_Dicke(self,indices):
        """ Setup Hamiltonian"""
        num = create(indices.ldim_p)*destroy(indices.ldim_p)
        #note terms with just photon operators need to be divided by nspins
        self.H = self.wc*tensor(num, qeye(indices.ldim_s))/indices.nspins + self.w0*tensor(qeye(indices.ldim_p), sigmaz()) 
        self.H = self.H + self.g*(tensor(create(indices.ldim_p), sigmam()) +  tensor(destroy(indices.ldim_p), sigmap()))
       

if __name__ == '__main__':
    # Testing purposes
    ntls =5#number 2LS
    w0 = 1.0
    wc = 0.65
    Omega = 0.4
    indi = Indices(ntls)
    dicke = DickeH(w0, wc, Omega, indi)
    L = BlockL(0, 0, 0, indi, dicke)













