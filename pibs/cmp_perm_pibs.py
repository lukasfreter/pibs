#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:31:48 2024

@author: freterl1
"""

# this file is designed to compare results from permutation_fork and pibs

import numpy as np
import sys
from setup import Indices, BlockDicke, Rho
from propagate import TimeEvolve
from util import qeye, create, destroy, sigmam, sigmap, tensor, basis
import matplotlib.pyplot as plt
from time import time
import pickle

sys.path.append('/u/10/freterl1/unix/StAndrews/permutations_fork/')
from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from models import setup_Dicke_block, setup_Dicke, setup_Dicke_block1
from expect import setup_convert_rho_nrs, setup_convert_rho_block_nrs


t0 = time()
# same parameters as in Peter Kirton's code.
ntls = 10 #number 2LS
nphot = ntls+1
w0 = 1.0
wc = 0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 0.011
gamma = 0.02
gamma_phi = 0.03

dt = 0.2 # timestep
tmax = 200.0 - dt # that way it is exactly 1000 points

print(f'Number of spins {ntls}')
print('Pibs:')
indi = Indices(ntls)
L = BlockDicke(wc, w0,g, kappa, gamma_phi/4,gamma, indi)


# kirton
t0_tot = time()

print('Block form optimized')
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block(parallel=True)       # setup mapping between compressed density matrix and block form
setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM
setup_convert_rho_block_nrs(1)
#sys.exit()

# Initial state
t0 = time()
initial_block = setup_rho_block(basis(nphot,0),basis(2,0))
print('setup initial state block in {:.1f}s'.format(time()-t0), flush=True)
#initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
#sys.exit()
t0=time()
L0,L1 = setup_Dicke_block1(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma)
print('setup L block in {:.1f}s'.format(time()-t0), flush=True)


for i in range(len(L0)):
    assert np.allclose(L0[i].todense(), L.L0[i].todense())
for i in range(len(L1)):
    assert(np.allclose(L1[i].todense(), L.L1[i].todense()))
    
print('L0 and L1 from both codes agree')





