#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool

from .util import degeneracy_spin_gamma, degeneracy_gamma_changing_block_efficient
from .util import states_compatible, permute_compatible, degeneracy_outer_invariant_optimized
from .util import _multinominal
from .util import Progress

import os
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
    def __init__(self, nspins, nphot=None,spin_dim=2, verbose=True, debug=False,\
                 save=False, index_path=None, only_numax = False):
        """

        Parameters
        ----------
        nspins : int
            Number of two-level systems
        nphot : int, optional
            Photon space dimension. If None, nphot=nspins+1
        spin_dim : int, optional
            Dimension of spin space. The default is 2 (only implemented value)
        verbose : bool, optional
            Print status messages. The default is True.
        debug : bool, optional
            If True, do not load existing file, always calculate new set of indices. The default is False.
        save : bool, optional
            If True, save spin indices for later use. The default is False.
        index_path : string, optional
            Save path for index file. The default is None.
        only_numax : bool, optional
            If true, calculate indices only for block with maximum excitation numbers. 
            This is interesting when solving a system with no losses, then we stay in the block 
            determined by initial conditions (which for sf initial conditions is nu_max). The default is False.


        Returns
        -------
        None.

        """
        
        # make some checks for validity of nspins, nphot, spin_dim
        if (not isinstance(nspins, (int, np.integer))) or nspins <= 0:
            raise ValueError("Number of spins must be integer N > 0")
        if nphot is not None and ((not isinstance(nphot, (int, np.integer))) or nphot <= 0):
            raise ValueError("Number of photon states must be integer > 0")
        if spin_dim is not None and ((not isinstance(spin_dim, (int, np.integer))) or spin_dim <= 1):
             raise ValueError("Spin dimension must be integer > 1")

            
        if nphot is None:
            nphot = nspins + 1
            if verbose:
                print('Photon dimension set to ', nphot)
        if spin_dim is None:
            spin_dim = 2 # spin 1/2 system
            if verbose:
                print('Spin dimension set to ', spin_dim)
            
        self.nspins, self.ldim_p, self.ldim_s = nspins, nphot, spin_dim
        self.indices_elements = []
        self.indices_elements_inv = {}
        self.mapping_block = []
        self.elements_block = []
        self.difference_block_inv = []
        self.coupled_photon_block = []
        
        self.only_numax = only_numax
        
        self.verbose = verbose
        
        # loading/saving paths
        if index_path is None:
            index_path = 'data/indices/'
        if not os.path.isdir(index_path):
            os.makedirs(index_path)
            print('Created directory ', index_path)
        filename = f'indices_Ntls{self.nspins}_Nphot{self.ldim_p}_spindim{self.ldim_s}.pkl'
        fname_numax = f'indices_numax_Ntls{self.nspins}_Nphot{self.ldim_p}_spindim{self.ldim_s}.pkl' # for the case only_numax =True
        
        
        if debug is False:
            if only_numax: # check data/indices/numax
                if os.path.isdir(index_path+'numax/'):
                    index_files = os.listdir(index_path + 'numax/')
                    if any([f == fname_numax for f in index_files]):
                        self.load(index_path + 'numax/' + fname_numax)
                        return
                else:
                    os.makedirs(index_path + 'numax/')
                
            # check if an object with the same arguments already exists in data/indices/ folder
            index_files = os.listdir(index_path)
            if (any([f == filename for f in index_files])):
                self.load(index_path+filename)
                return
            
        # debug true -> always calculate spin indices anew, or if not save file is found
        # setup indices
        if verbose:
            print(f'Setting up spin indices with nspins={self.nspins},spin_dim={self.ldim_s}...', flush=True)
        t0 = time()
        self.list_equivalent_elements()
        elapsed = time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)

        # setup mapping block
        if verbose:
            if only_numax:
                print(f'Running setup indices block (only nu_max) with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
            else:
                print(f'Running setup indices block with nspins={self.nspins},nphot={self.ldim_p}...', flush=True)
        t0 = time()
        self.setup_mapping_block()
        elapsed = time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
        
        if save:
            # export for future use, if save is true
            if only_numax:
                self.export(index_path + 'numax/' + fname_numax)
            else:
                self.export(index_path + filename)
                
        for nu in range(len(self.mapping_block)):
            assert len(self.mapping_block[nu]) == len(self.elements_block[nu])
                
    
    def list_equivalent_elements(self):
        """Generate a list of elements [left, right] representing all permutation
        distinct spin elements |left><right|"""
        #get minimal list of left and right spin indices - in combined form (i.e. as a list of zetas)
        #Generate list of all unique zeta strings in reverse lexicographic order
        all_zetas = [np.zeros(self.nspins, dtype=int)]
        # e.g. spin-1/2 -> s = 0,1 and zeta = 0, 1, 2, 3 = 2s_L + s_R
        max_zeta = self.ldim_s**2 - 1 # (d_s-1)(d_s+1)
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
        
        IF self.only_numax == True -> only get the block with nu_max. This block has each
        element of self.indices_elements exactly once, which is because any spin configuration is allowed (in max. excitation block)
        and the photon numbers then get just matched such that left and right excitations equal nu_max.

        """     
        num_elements = len(self.indices_elements) # number of distinct spin states
        nu_max = self.nspins # maximum excitation number IF initial state is all spins up and zero photons
                             # TODO: depending on the initial state of spins and photons, adapt nu_max properly 
        
        self.mapping_block = [ [] for _ in range(nu_max+1)] # list of nu_max+1 empty lists
        self.elements_block = [ [] for _ in range(nu_max+1)]
        self.elements_block_inv = [ {} for _ in range(nu_max+1)]
        difference_block = [ [] for _ in range(nu_max+1)] # used to create difference_block_inv
        self.difference_block_inv = {nu:[] for nu in range(nu_max+1)} # used for rdm conversion calculations,
        # contains index and difference between left and right spin states for each spin element
        self.coupled_photon_block = [{} for _ in range(nu_max+1)] # at each nu, dictionary of key-values 
        # where key is a photon state as a tuple i.e. |p1><p2| -> (p1,p2) and
        # values is a pair of lists [same_block, below_block], where same_block
        # is to contain all the photon states (tuples) that may couple to |p1><p2| at nu, 
        # and below_block those states (tuples) that couple in the block below (via decay)
        # [Construction lends to needing the state in excitation below rather than above]
        coupled_photon_counts = {} # dictionary, elements under key (p1,p2) describe all photon 
        # states (tuples) which |p1><p2| couples to
        
        # for initial state calculation:
        # self.elements_photon_diag = {nu: [] for nu in range(nu_max+1)}

        def get_coupled_counts(p1, p2):
            """List all states coupled to |p1><p2| by physical processes
            Returns [ same_block_counts, below_block_counts ], where same and below counts
            indicate states in same and one lower excitation block, respectively
            """
            eye = (p1,p2) # act identically
            dleft = (max(p1-1,0),p2) # destroy left (e.g. a sigma^+ in LM coupling)
            cleft = (min(p1+1,nu_max),p2) # create left
            dright = (p1,min(p2+1,nu_max)) # destroy right
            cright = (p1,max(p2-1,0)) # create right
            same_block_counts = list(set((eye, dleft, cleft, dright, cright))) # set removes any duplicates
            below_block_counts = [eye] # photon count stays the same, spin decays
            if p1 > 0 and p2 > 0: # e.g. |p1><p2|=|0><1| does NOT decay to |-1><0| !
                below_block_counts.append((p1-1, p2-1)) # photon decay
            return [ same_block_counts, below_block_counts ]
        
        # populate coupled_photon_counts dictionary and initialise coupled_photon_block 
        for count_tuple in product(range(nu_max+1), range(nu_max+1)):
            coupled_photon_counts[count_tuple] = get_coupled_counts(*count_tuple)
            for nu in range(max(count_tuple), nu_max+1):
                self.coupled_photon_block[nu][count_tuple] = [[],[]]

        for count in range(num_elements): # loop through spin states
            element = self.indices_elements[count]  # spin-element in compressed form
            left = element[0:self.nspins]   # left spin indices
            right = element[self.nspins:2*self.nspins] # right spin indices
            zetas = 2 * left + right 
            m_left = self.nspins-sum(left) # excitations in left spins
            m_right = self.nspins-sum(right) # excitations in right spins
            num_diff = sum(left != right)
            nu_min = max(m_left, m_right) # can't have fewer than m_left+0 photons (or m_right+0photons) excitations
            
            # nu_max_loop can at most be nu_max (because no element has more excitations than nu_max)
            # but in the case where nphot != ntls+1, it could be that nu_max_loop < nu_max, because
            # the photon space is restricted. Then, nu_max_loop = maximum excitation possible with left or right spin index plus photon excitations.
            nu_max_loop = min(max(m_left, m_right) + self.ldim_p-1, nu_max)
            
            if self.only_numax == True:
                nu_min = nu_max
            
            for nu in range(nu_min, nu_max_loop+1):
                count_p1 = nu - m_left
                count_p2 = nu - m_right
                if count_p1 >= self.ldim_p or count_p2 >= self.ldim_p: # disregard states, where photon number in left or right index exceeds photon space dimension
                    continue
                
                element_index = self.ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                el = np.concatenate(([count_p1], left, [count_p2], right))
                self.mapping_block[nu].append(element_index)
                self.elements_block[nu].append(el)
                # can't use inverse here we sort the mapping block below - instead
                # note difference and fill difference_block_inv AFTER sorting
                # below. 
                #self.difference_block_inv[num_diff].append((nu, len(self.mapping_block[nu])))
                difference_block[nu].append(num_diff)
                
                # for general initial state: store indices of elements, which have the same 
                # number of photons in left and right state
                # if count_p1 == count_p2:
                #     self.elements_photon_diag[count_p1].append((nu, len(self.mapping_block[nu])-1))

        # Re-order to match that of earlier implementations
        if self.only_numax == True:
            nu_min = nu_max
        else:
            nu_min = 0
        for nu in range(nu_min,nu_max+1): # need to go through all nu here
            # zip-sort-zip - a personal favourite Python One-Liner
            self.mapping_block[nu], self.elements_block[nu], \
            difference_block[nu] =\
                    zip(*sorted(zip(self.mapping_block[nu],
                                    self.elements_block[nu],
                                    difference_block[nu])))
            # have to populate difference and coupled photon indices AFTER sort
            for i, num_diff in enumerate(difference_block[nu]):
                self.difference_block_inv[num_diff].append((nu, i))
            for i, element in enumerate(self.elements_block[nu]):
                count_tuple = (element[0], element[self.nspins+1])
                same_block_counts, below_block_counts = coupled_photon_counts[count_tuple]
                for target_tuple in same_block_counts:
                    try:
                        self.coupled_photon_block[nu][target_tuple][0].append(i)
                    except KeyError:
                        pass # target_tuple does not exist in this excitation block!
                        # e.g. |2><1| couples to |3><1| (create photon
                        # in left state) at nu=3 but NOT nu=2
                if nu == 0:
                    # at lowest block already
                    continue
                for target_tuple in below_block_counts:
                    try:
                        # state at target_tuple in block below couples to count_tuple, so add index
                        self.coupled_photon_block[nu-1][target_tuple][1].append(i)
                    except KeyError:
                        pass # e.g. |2><2| at nu=3 spin decay to nu=2 is valid
                        # but |2><2| at nu=2 spin decay to nu=1 is NOT (since
                        # |2><2| is not in the nu=1 block)
    
    def _to_hilbert(self, combined):
        """Convert zeta-string to |left> and <right| spin values"""
        right = combined % self.ldim_s
        left = (combined - right)//self.ldim_s
        return left, right


    def export(self, filepath):
        if self.verbose:
            print(f'Storing Indices for later use in {filepath} ...')
        t0 = time()
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
        elapsed = time() - t0
        if self.verbose:
            print(f'Storing complete {elapsed:.1f}s')
            


    def load(self, filepath):
        if self.verbose:
            print(f'Loading indices file {filepath} ...')
        t0 = time()
        with open(filepath, 'rb') as handle:
            indices_load = pickle.load(handle)
        self.__dict__ = indices_load.__dict__
        elapsed = time() - t0
        # do some checks
        # at least tell user what they loaded
        if not hasattr(self, 'only_numax'):
            self.only_numax = False # if the indices file loaded is an old one that does not have the attribute 'only_numax' yet, then set it to false.

        if self.verbose:            
            if self.only_numax:
                print(f'Loaded index file with ntls={self.nspins}, nphot={self.ldim_p}, spin_dim={self.ldim_s} (only nu_max) {elapsed:.1f}s')
            else:
                print(f'Loaded index file with ntls={self.nspins}, nphot={self.ldim_p}, spin_dim={self.ldim_s} {elapsed:.1f}s')
        
        
        
      
    # DEBUGGING FUNCTIONs   
        
    def print_elements(self, numax=None):
        """ Print elements in each block """
        if numax is None:
            numax = len(self.mapping_block)
        for nu in range(numax):
            print(30* '-', f'nu={nu}',30* '-')
            for i in range(len(self.mapping_block[nu])):
                left = self.elements_block[nu][i][1:self.nspins+1]
                right = self.elements_block[nu][i][self.nspins+2:]
                xis = 2*left + right
                print(f'{i}:',self.elements_block[nu][i], xis)
                
    def print_spins(self):
        """ Print all spin elements"""
        print('Printing spins: (0=up, 1=down)')
        count = 0
        for element in self.indices_elements:
            left = element[0:self.nspins]
            right = element[self.nspins+1:]
            xi = 2*left + right
            print(f'{count}: {element}, xi: {xi}')
            count +=1
                
    def element_count(self):
        """ Print the number of elements in each block """
        sizeL0 = np.array([len(block)**2 for block in self.mapping_block])
        sizeL1 = [len(self.mapping_block[nu]) * len(self.mapping_block[nu+1]) for nu in range(len(self.mapping_block)-1)]
        sizeL1.append(0)
        sizeL1 = np.array(sizeL1)
        
        loops_photon_trick = np.zeros(len(self.mapping_block))
        for nu in range(len(self.mapping_block)):
            count_nu = self.coupled_photon_block[nu]
            for key in count_nu:
                loops_photon_trick[nu] += len(count_nu[key][0]) + len(count_nu[key][1])
    
        
        print('Number of elements:', [len(block) for block in self.mapping_block])
        print('Size L0:', sizeL0)
        print('Size L1:', sizeL1)
    
                        


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
    
    
    IF indices.only_numax is true, L0_basis and L0 have only one entry, and L1_basis and L1 have no entries
    (Not like in the spin indices case, where mapping_block e.g. has num_blocks - 1 empty arrays before nu_max block)
    
    L calculation in the case indices.only_numax = True only for
        parallel=0
        parallel=1
    """
    def __init__(self, indices, parallel=0,num_cpus=None, debug=False, save=True,\
                 progress=False, liouv_path=None,verbose=True):
        """
        

        Parameters
        ----------
        indices : setup.Indices
            Indices object
        parallel : int, optional
            Liouvillian construction:
                0: serial, 1: parallel. The default is 0.
        num_cpus : int, optional
            Number of CPUs. The default is None.
        debug : bool, optional
            If True, do not load existing file, always calculate Liouvillian from scratch. The default is False.
        save : bool, optional
            If True, save Liouvillian for later use. The default is True.
        progress : bool, optional
            Progress bar. The default is False.
        liouv_path : string, optional
            Save path for Liouvillian files. The default is None.
        verbose : bool, optional
            Print status messages. The default is True.

        Returns
        -------
        None.

        """
        
        if indices.only_numax:
            if parallel > 1:
                raise ValueError(f'For calculation with only nu_max, use parallel=0 or parallel=1. Parallel = {parallel} is not supported.')
                
        
        # initialisation
        # L0 : does not change excitation number
        # L1 : changes excitation number 
        self.L0_basis = {'sigmaz': [],
                         'sigmam': [],
                         # 'sigmam_collective': [],
                         'a': [],
                         'ad_sigmam' : [],
                         'a_sigmap'  : [],
                         'H_n': [],
                         'H_sigmaz': [],
                         'H_g': []}
        self.L1_basis = {'sigmam': [],
                         # 'sigmam_collective':[],
                         'a': []}
        self.num_cpus = num_cpus
        self.verbose = verbose
        
        
        if liouv_path is None:
            liouv_path = 'data/liouvillians/'
        if not os.path.exists(liouv_path):
            os.makedirs(liouv_path)
            print('Created directory ', liouv_path)
            
        filename = f'liouvillian_TC_Ntls{indices.nspins}_Nphot{indices.ldim_p}_spindim{indices.ldim_s}.pkl'
        fname_numax = f'liouvillian_TC_numax_Ntls{indices.nspins}_Nphot{indices.ldim_p}_spindim{indices.ldim_s}.pkl'
        
        if debug is False:
            if indices.only_numax: # check, if we only need to load nu_max
                if os.path.isdir(liouv_path + 'numax/'):
                    liouv_files= os.listdir(liouv_path + 'numax/')
                    if (any([f == fname_numax for f in liouv_files])):
                        self._load(liouv_path + 'numax/'+fname_numax, indices)
                        return
                else:
                    os.makedirs(liouv_path + 'numax/')
             
            # check if an object with the same arguments already exists in data/liouvillian/ folder
            liouv_files = os.listdir(liouv_path)
            if (any([f == filename for f in liouv_files])):
                self._load(liouv_path+filename, indices)
                return
            
        # if not, calculate them
        t0 = time()
        pname = {0:'serial', 1:'parallel (lines)', 2:'parallel (blocks) (WARNING: memory inefficient, testing only)',3:'ray module (lines)', 4: 'ray module (blocks)'}
        pfunc = {0: self.setup_L_block_basis, 1: self.setup_L_block_basis_parallel,
                 2: self.setup_L_block_basis_parallel2
                 }
        try:
            if verbose:
                if indices.only_numax:
                    print(f'Calculating normalised Liouvillian (only nu_max) {pname[parallel]}...')
                else:
                    print(f'Calculating normalised Liouvillian {pname[parallel]}...')
            pfunc[parallel](indices, progress)
        except KeyError as e:
            print('Argument parallel={parallel} not recognised')
            raise e
        elapsed = time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
        
        if save:
            # export normalized Liouvillians for later use, if save is true
            if indices.only_numax:
                self.export(liouv_path+'numax/'+fname_numax)
            else:
                self.export(liouv_path+filename)
        
        
    
    def export(self, filepath):
        if self.verbose:
            print(f'Storing Liouvillian basis for later use in {filepath} ...', flush=True)
        t0 = time()
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
        elapsed = time() - t0
        if self.verbose:
            print(f'Storing complete {elapsed:.1f}', flush=True)
            
    def _load(self, filepath,ind):
        t0 = time()
        if self.verbose:
            if ind.only_numax:
                print(f'Loading Liouvillian basis (only nu_max) file with ntls={ind.nspins}, nphot={ind.ldim_p}, spin_dim={ind.ldim_s} from file {filepath} ...', flush=True)
            else:
                print(f'Loading Liouvillian basis file with ntls={ind.nspins}, nphot={ind.ldim_p}, spin_dim={ind.ldim_s} from file {filepath} ...', flush = True)

        
        with open(filepath, 'rb') as handle:
            L_load = pickle.load(handle)
        if (not hasattr(L_load.indices, 'only_numax')):
            loaded_only_numax = False # If loaded L does not have attribute only_numax, set to False.
        else:
            loaded_only_numax = L_load.indices.only_numax
        if loaded_only_numax == False and ind.only_numax == True:
            # In this case, we loaded a full L basis, but only need the largest block!
            for names in L_load.L0_basis:
                # print(L_load.L0_basis['sigmam_collective'])
                self.L0_basis[names] = [L_load.L0_basis[names][-1]]
                self.L1_basis = []
        else:
        
            self.L0_basis = L_load.L0_basis
            self.L1_basis = L_load.L1_basis
        
        elapsed = time() - t0
        # at least tell user what they loaded
        if self.verbose:
            print(f'Loading complete {elapsed:.1f}', flush=True)

    
    @staticmethod    
    def sparse_constructor_dic(shape):
        # (data, (coords_x, coords_y)
        return {'data':[], 'coords':[[],[]], 'shape':shape}
    @staticmethod
    def new_entry(L_dic, name, count_in, count_out, data):
        # function to add data and coords to target L dictionary and name
        
        # check first, if an entry at those coordinates already exists
        coord_tuples = [(L_dic[name]['coords'][0][i],L_dic[name]['coords'][1][i]) for i in range(len(L_dic[name]['coords'][0]))]
        if (count_in, count_out) in coord_tuples:
            idx = coord_tuples.index((count_in, count_out))
            L_dic[name]['data'][idx] += data
        
        else: # coords (count_in, count_out) do not yet exist, then just append them
            L_dic[name]['data'].append(data)
            L_dic[name]['coords'][0].append(count_in)
            L_dic[name]['coords'][1].append(count_out)
        
    
    def setup_L_block_basis(self, indices, progress):
       """ Calculate Liouvillian basis in block form, serial version"""
       num_blocks = len(indices.mapping_block)
       
       if progress:
           num_elements = sum([len(indices.mapping_block[nu]) for nu in range(num_blocks)])
           bar = Progress(num_elements, 'Calculate L basis...')
       
       #------------------------------------------------------
       # First, get L0 part -> coupling to same block, 
       #------------------------------------------------------
       
       # loop through all elements in block structure

       if indices.only_numax:
           nu_min = num_blocks -1  # num_blocks = nu_max + 1
       else:
           nu_min = 0
       
       # Loop through all blocks
       for nu_element in range(nu_min, num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz',\
                    'H_g','ad_sigmam','a_sigmap'] # keys for the individual terms in Liouvillian
           names = list(self.L0_basis.keys())
           L0_new ={name:self.sparse_constructor_dic((current_blocksize, current_blocksize)) for name in names}
           if nu_element < num_blocks-1:
               next_blocksize = len(indices.mapping_block[nu_element+1])
               # Liouvillian terms coupling to next block
               names = ['sigmam', 'a']
               names = list(self.L1_basis.keys())
               
               L1_new ={name:self.sparse_constructor_dic((current_blocksize, next_blocksize)) for name in names}
           
           # Loop through all elements in one block

           for count_in in range(current_blocksize):
               if progress:
                   bar.update()
               # get element, of which we want the time derivative
               element = indices.elements_block[nu_element][count_in]
               left = element[0:indices.nspins+1] # left state, first index is photon number, rest is spin states
               right = element[indices.nspins+1:2*indices.nspins+2] # right state
               
               photon_tuple = (left[0], right[0])
               # Loop through all elements in the same block WITH compatible photon counts. Note: For L[ad_sigm], we need to adapt this, because new couplings are possible
               coupled_counts_nu = indices.coupled_photon_block[nu_element][photon_tuple][0]
               # print('WARNING: Serial L construction does not use photon count trick to calculate L[ad*sigmam]')
               for count_out in coupled_counts_nu:
                   
                   # get "to couple" element, that contributes to time derivative of "element"
                   element_to_couple = indices.elements_block[nu_element][count_out]
                   left_to_couple = element_to_couple[0:indices.nspins+1]
                   right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                   
                   #-----------------------------
                   # get Liouvillian elements
                   #-----------------------------
                   
                   right_equal = (right_to_couple == right).all()
                   left_equal = (left_to_couple == left).all()      

                   # Diagonal parts

                   if left_equal and right_equal: 
                       # L0 part from Hamiltonian
                       s_down_right = sum(right[1:])
                       s_down_left = sum(left[1:])
                       self.new_entry(L0_new, 'H_n', count_in, count_out, -1j * (left[0]-right[0]))
                       self.new_entry(L0_new, 'H_sigmaz', count_in, count_out, 1j*(s_down_left-s_down_right))
                       
                       # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
                       # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                       # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                       deg_right = degeneracy_spin_gamma(right_to_couple[1:indices.nspins+1], right[1:indices.nspins+1]) # degeneracy: because all spin up elements contribute equally
                       deg_left = degeneracy_spin_gamma(left_to_couple[1:indices.nspins+1], left[1:indices.nspins+1])
                       self.new_entry(L0_new, 'sigmam', count_in, count_out,  - 1/2 * (deg_left+deg_right))
                       
                       
                       # L0 part from L[sigmaz] -> whole dissipator
                       # Left and right states must be equal, because sigmaz is diagonal in the spins.
                       equal = (left[1:indices.nspins+1] == right[1:indices.nspins+1]).sum()
                       self.new_entry(L0_new, 'sigmaz', count_in, count_out, 2*(equal - indices.nspins))
                       
                       # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
                       self.new_entry(L0_new, 'a', count_in, count_out, -1/2*(left[0] + right[0]))
                   

                   # offdiagonal parts; from commutator part of H_g and from Lindbladian terms X*rho or rho*X
                   elif(states_compatible(right, right_to_couple)): 
                        # if they are compatible, permute left_to_couple appropriately for proper H element
                        left_to_couple_permute = np.copy(left_to_couple)
                        if not right_equal:
                            # if they are compatible but not equal, we need to permute left_to_couple appropriately, to get correct matrix element of H
                            left_to_couple_permute[1:] = permute_compatible(right[1:],right_to_couple[1:],left_to_couple[1:])
                            
                        # FIRST: H_g part of commutator -i * H_g * rho
                        # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                        if (left[1:]==left_to_couple_permute[1:]).sum() == indices.nspins-1:
                        
                            deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], left_to_couple_permute[1:]) # degeneracy from simulatneous spin permutations, which leave outer spins invariant
                            # check if photon number in left state increases or decreases and
                            # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                            left_photon_diff = left[0] - left_to_couple[0]
                            left_spin_sum_diff = sum(left[1:])-sum(left_to_couple[1:])
                            if left_photon_diff == 1 and left_spin_sum_diff == 1: # need matrix element of adag*sigmam
                                self.new_entry(L0_new, 'H_g', count_in, count_out, - 1j*deg * np.sqrt(left[0]))
                            elif left_photon_diff == -1 and left_spin_sum_diff == -1 : # need matrix element of a*sigmap
                                self.new_entry(L0_new, 'H_g', count_in, count_out, - 1j*deg * np.sqrt(left[0]+1))
                        
                               
                   elif(states_compatible(left, left_to_couple)):            
                        # if they are compatible, permute right_to_couple appropriately for proper H element
                        right_to_couple_permute = np.copy(right_to_couple)
                        if not left_equal:
                            right_to_couple_permute[1:] = permute_compatible(left[1:],left_to_couple[1:],right_to_couple[1:])
                            
                        # FIRST: H_g part of commutator i * rho * H_g
                        # Now first check, if the matrix element is nonzero. This is the case, if all the spins but one match up.
                        if (right[1:]==right_to_couple_permute[1:]).sum() == indices.nspins-1:
                            deg = degeneracy_outer_invariant_optimized(left[1:], right[1:], right_to_couple_permute[1:])
                            # check if photon number in right state increases or decreases and
                            # if all but one spin agree, and that the spin that does not agree is down in right and up in right_to_couple
                            right_photon_diff = right[0] - right_to_couple[0]
                            right_spin_sum_diff = sum(right[1:])-sum(right_to_couple[1:])
                            if right_photon_diff == 1 and right_spin_sum_diff == 1: # need matrix element of a*sigmap
                                self.new_entry(L0_new, 'H_g', count_in, count_out,  1j*deg * np.sqrt(right[0]))
                            elif right_photon_diff == -1 and right_spin_sum_diff == -1: # need matrix element of adag*sigmam
                                self.new_entry(L0_new, 'H_g', count_in, count_out,  1j*deg * np.sqrt(right[0]+1))
                            

                   
               if nu_element == num_blocks -1: # no L1 part for highest block
                   continue
                
               # Now get L1 part -> coupling from nu_element to nu_element+1 loop through matrix
               # elements in the next block WITH compatible photon counts (only photon number 
               # unchanged -> spin decay or photon number decreased by one -> photon decay)
               coupled_counts_nu_plus = indices.coupled_photon_block[nu_element][photon_tuple][1]
               for count_out in coupled_counts_nu_plus:    
                   
                   # get "to couple" element
                   element_to_couple = indices.elements_block[nu_element+1][count_out]
                   left_to_couple = element_to_couple[0:indices.nspins+1]
                   right_to_couple = element_to_couple[indices.nspins+1:2*indices.nspins+2]
                   left_to_couple_spins = left_to_couple[1:]
                   right_to_couple_spins = right_to_couple[1:]
                   
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
        global elements_block, new_entry, sparse_constructor_dic, coupled_photon_block_nu
        
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
        
        photon_tuple = (left[0], right[0])
        # Loop through all elements in the same block WITH compatible photon counts
        coupled_counts_nu = coupled_photon_block_nu[photon_tuple][0]
        for count_out in coupled_counts_nu:
            # get "to couple" element
            element_to_couple = current_element_block[count_out]
            left_to_couple = element_to_couple[0:nspins+1]
            right_to_couple = element_to_couple[nspins+1:2*nspins+2]
            
            #-----------------------------
            # get Liouvillian elements
            #-----------------------------
           
            # Diagonal part
            if (right_to_couple == right).all() and (left_to_couple == left).all():
                # L0 part from Hamiltonian
                s_down_right = sum(right[1:])
                s_down_left = sum(left[1:])
                new_entry_func('H_n', count_out, -1j * (left[0]-right[0]))
                new_entry_func('H_sigmaz', count_out, 1j*(s_down_left-s_down_right))
                
                # L0 part from L[sigmam] -> -sigmap*sigmam*rho - rho*sigmap*sigmam
                # make use of the fact that all spin indices contribute only, if left and right spin states in sigma^+sigma^- are both up
                # also make use of the fact that sigma^+sigma^- is diagonal, so the two terms rho*sigma^+sigma^- and sigma^+sigma^-*rho are equal
                deg_right = degeneracy_spin_gamma(right_to_couple[1:nspins+1], right[1:nspins+1]) # degeneracy: because all spin up elements contribute equally
                deg_left = degeneracy_spin_gamma(left_to_couple[1:nspins+1], left[1:nspins+1])
                new_entry_func('sigmam', count_out, - 1/2 * (deg_left+deg_right))
                
                # L0 part from L[sigmaz] -> whole dissipator
                # Left and right states must be equal, because sigmaz is diagonal in the spins.
                equal = (left[1:nspins+1] == right[1:nspins+1]).sum()
                new_entry_func('sigmaz', count_out,  2*(equal - nspins))
                    
                # L0 part from L[a]     -> -adag*a*rho - rho*adag*a
                new_entry_func('a', count_out, -1/2*(left[0] + right[0]))
            
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
    
            
            

        return L0_line

    @staticmethod
    def calculate_L1_line(args_tuple):
        """ Calculate L1 part of element count_in in block nu_element """
        
        #indices,count_in, nu_element = args_tuple
        nu_element, count_in = args_tuple
        global elements_block, new_entry, sparse_constructor_dic, coupled_photon_block_nu
        
        # get element, of which we want the time derivative
        current_element = elements_block[nu_element][count_in]
        current_blocksize = len(elements_block[nu_element])
    
        left = current_element[0:nspins+1] # left state, first index is photon number, rest is spin states
        right = current_element[nspins+1:2*nspins+2] # right state
        photon_tuple = (left[0], right[0])
            
        
        # Now get L1 part -> coupling from nu_element to nu_element+1
        # loop through all matrix elements in the next block we want to couple to
        next_element_block = elements_block[nu_element+1]
        next_blocksize = len(elements_block[nu_element+1])

        names = ['sigmam', 'a']
        L1_line = {name:sparse_constructor_dic((current_blocksize, next_blocksize)) for name in names}
        new_entry_func = lambda name, count_out, val: new_entry(L1_line, name, count_in, count_out, val)
        # Now get L1 part -> coupling from nu_element to nu_element+1 loop through matrix
        # elements in the next block WITH compatible photon counts (only photon number 
        # unchanged -> spin decay or photon number decreased by one -> photon decay)
        coupled_counts_nu_plus = coupled_photon_block_nu[photon_tuple][1]
        for count_out in coupled_counts_nu_plus:                   
            # get "to couple" element
            element_to_couple = next_element_block[count_out]
            left_to_couple = element_to_couple[0:nspins+1]
            right_to_couple = element_to_couple[nspins+1:2*nspins+2]
            
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
       of rows of the Liouvillian. Called when parallel=1.
       """
       num_blocks = len(indices.mapping_block)
       #multiprocessing.set_start_method('fork')
    
       updates = 0
       if progress:
           num_elements = sum([len(indices.mapping_block[nu]) for nu in range(num_blocks)])
           bar = Progress(2*num_elements - len(indices.mapping_block[num_blocks-1]), 'Calculate L basis...') # 2 updates per block, except last block (there is no L1 for last block)
       # loop through all elements in block structure
       
       
       if indices.only_numax:
           nu_min = num_blocks - 1 # only loop through last nu if only_numax=True
       else:
           nu_min = 0
       
       for nu_element in range(nu_min, num_blocks):
           current_blocksize = len(indices.mapping_block[nu_element])
           # setup the Liouvillians for the current block
           

           L0_names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']
           L0_new = {name:self.sparse_constructor_dic((current_blocksize, current_blocksize))
                     for name in L0_names}
           
           arglist = []
           global nspins, elements_block, sparse_constructor_dic, new_entry, coupled_photon_block_nu
           nspins  = indices.nspins
           elements_block = indices.elements_block
           coupled_photon_block_nu = indices.coupled_photon_block[nu_element]
           sparse_constructor_dic = self.sparse_constructor_dic
           new_entry = self.new_entry
        #nu_element, count_in = args_tuple
           for count_in in range(current_blocksize):
               arglist.append((nu_element, count_in))
           # print(f'Block {nu_element}/{num_blocks-1}: {len(arglist)} args')
           with Pool(processes=self.num_cpus) as pool:
               #print('Number of processes:', pool._processes)
               for L0_data in pool.imap(self.calculate_L0_line, arglist):
                   for name in L0_names:
                       L0_new[name]['data'].extend(L0_data[name]['data'])
                       L0_new[name]['coords'][0].extend(L0_data[name]['coords'][0])
                       L0_new[name]['coords'][1].extend(L0_data[name]['coords'][1])
                   if progress:
                       bar.update()
                       updates +=1
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
                           updates+=1
               for name in L1_names:
                   Lnew = L1_new[name]
                   data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
                   self.L1_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())
           # else:
           #     if progress:
           #         bar.update(2*num_elements-1)
       # print(updates, num_elements*2 - len(indices.mapping_block[-1]))

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
        
        L0_names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']
        L0_new ={name:sparse_constructor_dic((current_blocksize, current_blocksize)) for name in L0_names}

        global coupled_photon_block_nu
        coupled_photon_block_nu = coupled_photon_block[nu_element]

        for count_in in range(current_blocksize):
            L0_line = BlockL.calculate_L0_line((nu_element, count_in))
            
            for name in L0_names:
                L0_new[name]['data'].extend(L0_line[name]['data'])
                L0_new[name]['coords'][0].extend(L0_line[name]['coords'][0])
                L0_new[name]['coords'][1].extend(L0_line[name]['coords'][1])
            
        return L0_new

                
    @staticmethod     
    def L1_nu_task(nu_element):
        current_blocksize = len(elements_block[nu_element])
        next_blocksize = len(elements_block[nu_element+1])

        global coupled_photon_block_nu
        coupled_photon_block_nu = coupled_photon_block[nu_element]

        L1_names = ['sigmam', 'a']
        L1_new ={name:sparse_constructor_dic((current_blocksize, next_blocksize)) for name in L1_names}
        
        for count_in in range(current_blocksize):
            L1_line = BlockL.calculate_L1_line((nu_element, count_in))
            for name in L1_names:
                L1_new[name]['data'].extend(L1_line[name]['data'])
                L1_new[name]['coords'][0].extend(L1_line[name]['coords'][0])
                L1_new[name]['coords'][1].extend(L1_line[name]['coords'][1])
        
        return L1_new

    def setup_L_block_basis_parallel2(self, indices, progress):
       """ Calculate Liouvillian basis in block form. Parallelized the calculation
       of each block"""
       num_blocks = len(indices.mapping_block)
       global nspins, elements_block, coupled_photon_block
       nspins  = indices.nspins
       elements_block = indices.elements_block
       coupled_photon_block = indices.coupled_photon_block
       
       # loop through all elements in block structure
       arglist = [nu for nu in range(num_blocks)]
       
       with Pool(processes=self.num_cpus) as pool:
           L0s = pool.map(self.L0_nu_task, arglist)
       
       arglist = arglist[:-1] # not nu_max 
       with Pool(processes=self.num_cpus) as pool:
           L1s = pool.map(self.L1_nu_task, arglist)
           
           
       L0_names = ['sigmaz', 'sigmam', 'a', 'H_n', 'H_sigmaz', 'H_g']   
       L1_names = ['a', 'sigmam']        
       
       for L0_new in L0s:
           for name in L0_names:
               Lnew = L0_new[name]
               data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
               self.L0_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())
               
       for L1_new in L1s:
           for name in L1_names:
               Lnew = L1_new[name]
               data, coords, shape = Lnew['data'], Lnew['coords'], Lnew['shape']
               self.L1_basis[name].append(sp.coo_matrix((data, coords), shape=shape).tocsr())

       print('done')
       
       

class Models(BlockL):
    """ This class contains information about the exact model at hand and
    calculates the Liouvillian from the basis elements from BlockL.
    
    Demanding weak U(1) symmetry and no gain, the most general model of N spins
    interacting with a common photon mode is described by the Master equation
        
        d/dt rho = -i[H,rho] + kappa*L[a] + sum_k{ gamma*L[sigmam_k] + gamma_phi * L[sigmaz_k] }
        
    with Hamiltonian H = wc*adag*a + w0/2*sum_k{ sigmaz_k } + g*sum_k{adag*sigmam_k + a*sigmap_k }
    
    where the light-matter coupling g is assumed real.
    
    """
    def __init__(self,wc,w0,g, kappa, gamma_phi, gamma, indices, parallel=0,progress=False, debug=False, save=True, num_cpus=None, liouv_path=None, verbose=True):
        """
        

        Parameters
        ----------
        wc : float
            Cavity frequency
        w0 : float
            Two-level system level splitting
        g : float
            Light-matter coupling
        kappa : float
            Cavity loss rate
        gamma_phi : float
            Molecular dephasing rate
        gamma : float
            Molecular loss rate
        indices : setup.Indices
            Indices object
        parallel : int, optional
            Liouvillian construction:
                0: serial, 1: parallel. The default is 0.
        num_cpus : int, optional
            Number of CPUs. The default is None.
        debug : bool, optional
            If True, do not load existing file, always calculate Liouvillian from scratch. The default is False.
        save : bool, optional
            If True, save Liouvillian for later use. The default is True.
        progress : bool, optional
            Progress bar. The default is False.
        liouv_path : string, optional
            Save path for Liouvillian files. The default is None.
        verbose : bool, optional
            Print status messages. The default is True.

        Returns
        -------
        None.

        """
        
        # specify rates according to what part of Hamiltonian or collapse operators
        # they scale
        
        if indices.only_numax == True:
            if kappa != 0 or gamma != 0 :
                print(r'WARNING: only_numax is set true, but kappa = {kappa}, gamma = {gamma}. Set kappa = 0, gamma = 0 for physical results.'.format(kappa=kappa,gamma=gamma))
        
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
        self.verbose = verbose
        super().__init__(indices=indices, parallel=parallel,num_cpus=num_cpus, debug=debug, save=save, progress=progress,liouv_path=liouv_path, verbose=verbose)
    
    def setup_L_Tavis_Cummings(self, progress=False, save_path=None):
        t0 = time()
        if self.indices.only_numax:
            if self.verbose:
                print('Calculating Liouvillian for TC model from basis (only nu_max) ...', flush =True)
            if progress:
                progress = False
                if self.verbose:
                    print('Disabled progress bar (only one step)')
        else:
            if self.verbose:
                print('Calculating Liouvillian for TC model from basis ...', flush =True)
        
        names0 = ['H_sigmaz', 'H_n', 'H_g','a', 'sigmam', 'sigmaz']
        names1 = ['sigmam' , 'a']
        
        
        self.L0 = []
        self.L1 = []
        
        num_blocks = len(self.indices.mapping_block)
        
        if progress: # progress bar
            loops = 2*num_blocks-1
            bar = Progress(loops,'Liouvillian: ')
            
        # Adapt loop if only_numax is true
        if self.indices.only_numax:
            nu_min = num_blocks - 1
        else:
            nu_min = 0
        
        for nu in range(nu_min, num_blocks):
            current_blocksize = len(self.indices.mapping_block[nu])
            #L0_scale = sp.csr_matrix(np.zeros((current_blocksize, current_blocksize), dtype=complex))
            L0_scale = sp.csr_matrix((current_blocksize, current_blocksize), dtype=complex)
            for name in names0:
                if self.indices.only_numax:
                    L0_scale = L0_scale + self.rates[name] * self.L0_basis[name][0] # in Liouvillian basis, if only_numax is true, only one matrix is contained (i.e. different than mapping_block, which has nu_max-1 unfilled matrices, such that the old indices still work)
                else:
                    L0_scale = L0_scale + self.rates[name] * self.L0_basis[name][nu]

            self.L0.append( L0_scale)
            
            if progress:
                bar.update()
            
            if nu < num_blocks -1:
                next_blocksize = len(self.indices.mapping_block[nu+1])
                #L1_scale = sp.csr_matrix(np.zeros((current_blocksize, next_blocksize), dtype=complex))
                L1_scale = sp.csr_matrix((current_blocksize, next_blocksize), dtype=complex)
                
                for name in names1:
                    L1_scale = L1_scale + self.rates[name] * self.L1_basis[name][nu]
                self.L1.append(L1_scale)   
                
                if progress:
                    bar.update()
 
        elapsed = time()-t0
        if self.verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
        if save_path is not None:
            with open(save_path, 'wb') as handle:
                pickle.dump(self, handle)
            if self.verbose:
                print(f'Wrote full model to {save_path}.')
            
            
            
    # def setup_L_generic(self,rates,  progress=False, save_path=None):
    #     """ Calculate generic Liouvillian. IGNORES RATES GIVEN IN CONSTRUCTOR OF Models CLASS
    #     Parameters:
    #         rates (dic) : dictionary that contains the dissipation rates.
    #                      Form:
    #                           rates = {'H_n': wc,
    #                                    'H_sigmaz': w0,
    #                                    'H_g': g,
    #                                    'a': kappa,
    #                                    'sigmaz': gamma_phi,
    #                                    'sigmam': gamma,
    #                                    'sigmam_collective' : sigmam_collective,
    #                                    'ad_sigmam': ,
    #                                    'a_sigmap': }
    # """
    #     if self.verbose:
    #         print('Given rates:', rates)
    
    #     t0 = time()
    #     if self.indices.only_numax:
    #         if self.verbose:
    #             print('Calculating Liouvillian from basis (only nu_max) ...', flush =True)
    #         if progress:
    #             progress = False
    #             if self.verbose:
    #                 print('Disabled progress bar (only one step)')
    #     else:
    #         if self.verbose:
    #             print('Calculating Liouvillian from basis ...', flush =True)
        
    #     names0 = ['H_sigmaz', 'H_n', 'H_g','a', 'sigmam', 'sigmaz','ad_sigmam','a_sigmap',\
    #               'sigmam_collective']
    #     names1 = ['sigmam' , 'a', 'sigmam_collective']
        
        
    #     self.L0 = []
    #     self.L1 = []
        
    #     num_blocks = len(self.indices.mapping_block)
        
    #     if progress: # progress bar
    #         loops = 2*num_blocks-1
    #         bar = Progress(loops,'Liouvillian: ')
            
    #     # Adapt loop if only_numax is true
    #     if self.indices.only_numax:
    #         nu_min = num_blocks - 1
    #     else:
    #         nu_min = 0
        
    #     for nu in range(nu_min, num_blocks):
    #         current_blocksize = len(self.indices.mapping_block[nu])
    #         #L0_scale = sp.csr_matrix(np.zeros((current_blocksize, current_blocksize), dtype=complex))
    #         L0_scale = sp.csr_matrix((current_blocksize, current_blocksize), dtype=complex)
    #         for name in names0:
    #             if name in rates: # the key must be contained in rates dic. otherwise assume 0 value
    #                 if self.indices.only_numax:
    #                     L0_scale = L0_scale + rates[name] * self.L0_basis[name][0] # in Liouvillian basis, if only_numax is true, only one matrix is contained (i.e. different than mapping_block, which has nu_max-1 unfilled matrices, such that the old indices still work)
    #                 else:
    #                     L0_scale = L0_scale + rates[name] * self.L0_basis[name][nu]

    #         self.L0.append( L0_scale)
            
    #         if progress:
    #             bar.update()
            
    #         if nu < num_blocks -1:
    #             next_blocksize = len(self.indices.mapping_block[nu+1])
    #             #L1_scale = sp.csr_matrix(np.zeros((current_blocksize, next_blocksize), dtype=complex))
    #             L1_scale = sp.csr_matrix((current_blocksize, next_blocksize), dtype=complex)
                
    #             for name in names1:
    #                 if name in rates:
    #                     L1_scale = L1_scale + rates[name] * self.L1_basis[name][nu]
    #             self.L1.append(L1_scale)   
                
    #             if progress:
    #                 bar.update()
 
    #     elapsed = time()-t0
    #     if self.verbose:
    #         print(f'Complete {elapsed:.0f}s', flush=True)
    #     if save_path is not None:
    #         with open(save_path, 'wb') as handle:
    #             pickle.dump(self, handle)
    #         if self.verbose:
    #             print(f'Wrote full model to {save_path}.')
            

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
        
    def __init__(self, rho_p, rho_s, indices, max_nrs=1, scale_rho=1.0, verbose=True):
        """
        

        Parameters
        ----------
        rho_p : scipy.sparse.csr_matrix
            Density matrix of photons
        rho_s : scipy.sparse.csr_matrix
            2x2 density matrix of a single spin
        indices : setup.Indices
            Indices object
        max_nrs : int, optional
            Maximum number of spins contained in the reduced density matrix. The default is 1.
        scale_rho : float, optional
            Scaling factor of the density matrix. Only for numerical stability. The default is 1.0.
        verbose : bool, optional
            Print status messages. The default is True.

        Returns
        -------
        None.

        """
    
        assert type(max_nrs) == int, "Argument 'max_nrs' must be int"
        assert max_nrs >= 0, "Argument 'max_nrs' must be non-negative"
        assert indices.nspins >= max_nrs, "Number of spins in reduced density matrix "\
                "(max_nrs) cannot exceed total number of spins ({indices.nspins})"
        
        self.max_nrs = max_nrs # maximum number of spins in reduced density matrix
        self.indices = indices
        self.convert_rho_block_dic = {}
        self.initial= []
        self.scale_rho = scale_rho
        
        # debugging variables
        self.initial_full = []
        self.initial_reference = []
        
        
        # setup initial state
        t0 = time()
        if verbose:
            print('Set up initial density matrix...')
        self.initial=self.setup_initial_efficient(rho_p, rho_s)
        elapsed= time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
        
        if self.scale_rho != 1:
            t0 = time()
            if verbose:
                print(f'Scaling initial density matrix by {self.scale_rho} ...')
            for nu in range(len(self.initial)):
                self.initial[nu] = self.initial[nu] * self.scale_rho
        elapsed= time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
        
        
        # setup reduced density matrix
        t0 = time()
        if verbose:
            print('Set up mappings to reduced density matrices at...')
        for nrs in range(max_nrs+1):
            if verbose:
                print(f'nrs = {nrs}...')
            self.setup_convert_rho_block_nrs(nrs)
        elapsed= time()-t0
        if verbose:
            print(f'Complete {elapsed:.0f}s', flush=True)
         
    
    def setup_initial_efficient(self, rho_p, rho_s):
        """
        Setup initial state in block representation, without first calculating
        the initial state in the compressed form
        
        Comparison: N=20 takes 185s in compressed form. Efficient version basically instant.

         Parameters
         ----------
         rho_p : sparse array
             Density matrix of initial photons state
         rho_s : sparse array
             density matrix of initial spin state

         Returns
         -------
         None

        """

        indices = self.indices
        num_blocks = len(indices.mapping_block)
        rho_vec = [np.zeros(len(i), dtype=complex) for i in indices.mapping_block] # initializing initial state

        
        # Check for superfluoresence initial condition, i.e. zero photons and all spins up. 
        # This is very easily initialized by all blocks zero, instead of the first entry of the last block
        if np.isclose(rho_p[0,0],1) and np.isclose(rho_s[0,0],1):
            rho_vec[num_blocks-1][0] = 1
            return rho_vec
        # Check for complementary condition of nu_max photons and all spins down 
        if np.isclose(rho_p[-1,-1],1) and np.isclose(rho_s[-1,-1],1):
            rho_vec[num_blocks-1][-1] = 1
            return rho_vec
            
        
        # Next, check if photon density matrix is trivial, i.e. has a one somewhere on the diagonal.
        # This is the most common use case for us, where there are exactly zero photons in the cavity mode.
        # rho_s can be general
        phot_diag = rho_p.diagonal()
        p = np.where(phot_diag == 1)[0]
        if len(p) == 1: # if there is a diagonal element of value 1 
            num_phot = p[0] # number of photons in the cavity mode -> only states with that number of photons contribute to intial state

            # Now need to loop through all elements.
            for nu in range(num_blocks):
                for element_count in range(len(indices.mapping_block[nu])):
                    element = indices.elements_block[nu][element_count]
                    p_left = element[0] # photons in left state
                    p_right = element[indices.nspins+1] # photons in right state
                    
                    # Where left and right photons match num_phot, the 
                    # entry can be nonzero, depending on the spin state
                    if p_left == p_right and p_left == num_phot:
                        rho_vec[nu][element_count] = 1                        
                        s_left = element[1:indices.nspins+1] # left spin state
                        s_right = element[indices.nspins+2:] # right spin state
                        
                        # loop through spin states and multiply all contributions
                        for count_s in range(len(s_left)):
                            rho_vec[nu][element_count] *= rho_s[s_left[count_s], s_right[count_s]]
            return rho_vec
            
        
        
        # now completely general initial state; probably never needed, because we almost always
        # set the photon number to zero in the beginning.
        # Loop through all elements
        for nu in range(num_blocks):
            for element_count in range(len(indices.mapping_block[nu])):
                element = indices.elements_block[nu][element_count]
                p_left = element[0] # photons in left state
                p_right = element[indices.nspins+1] # photons in right state
                
                rho_vec[nu][element_count] = rho_p[p_left, p_right]

                if rho_vec[nu][element_count] != 0:
                    s_left = element[1:indices.nspins+1] # left spin state
                    s_right = element[indices.nspins+2:] # right spin state
                    
                    # loop through spin states and multiply all contributions
                    for count_s in range(len(s_left)):
                        rho_vec[nu][element_count] *= rho_s[s_left[count_s], s_right[count_s]]
        return rho_vec
                        
                        

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













