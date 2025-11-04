#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard example for calculating time evolution with pibs code.

"""

import numpy as np
import os
from pibs import Indices, Rho, Models
from pibs import TimeEvolve
from pibs import qeye, create, destroy, sigmam, sigmap, tensor, basis
import sys
import matplotlib.pyplot as plt
import scipy.sparse as sp

if __name__ == '__main__':
    # Set parameters
    ntls = 5                    # Number of two-level systems (TLS)
    nphot =ntls+1               # Photon space truncation
    w0 = 0.35                   # Level splitting of TLS
    wc = 0.0                    # Cavity frequency
    Omega =0.4                  # Vacuum Rabi splitting
    g = Omega / np.sqrt(ntls)   # Light-matter coupling 
    kappa = 0.01                # Cavity loss
    gamma = 0.001               # Molecular loss
    gamma_phi=0.0075            # Dephasing
    
    dt = 0.2        # timestep
    tmax = 200
    chunksize=200   # time chunks for parallel evolution
    
    # Solver parameters
    atol=1e-12
    rtol=1e-12
    nsteps=1000
    
    # Setup indices mapping
    indi = Indices(ntls,nphot, debug=False, save =True)
    
    # rotation matrix around x-axis of spin 1/2 : exp(-i*theta*Sx)=exp(-i*theta/2*sigmax) = cos(theta/2)-i*sin(theta/2)*sigmax
    theta = 0.25*np.pi
    rot_x = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
    rot_x_dag = np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],[1j*np.sin(theta/2), np.cos(theta/2)]])
    
    rho_phot = basis(nphot,0) # Initial photon state with zero photons
    rho_spin = sp.csr_matrix(rot_x @ basis(2,0) @ rot_x_dag) # Initial spin state. 0 = up, 1 = down
    
    rho = Rho(rho_phot, rho_spin, indi, max_nrs=1) # Setup density matrix
    
    # Setup Liouvillian
    L = Models(wc, w0,g, kappa, gamma_phi,gamma,indi, parallel=0,progress=False, debug=False,save=True, num_cpus=None)
    L.setup_L_Tavis_Cummings(progress=False)
    
    # Operators for time evolution
    adag = create(nphot)
    a = destroy(nphot)
    n = adag*a
    p = tensor(qeye(nphot), sigmap()*sigmam())
    ops = [n,p] # operators to calculate expectations for
    
    # Time evolution
    evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol, nsteps=nsteps)
    #evolve.time_evolve_block_interp(ops, progress = True, expect_per_nu=False, start_block=None, save_states=False)
    evolve.time_evolve_chunk_parallel(ops, chunksize=chunksize, progress=True, num_cpus=None)
    
    # Get expectation values
    e_phot_tot = evolve.result.expect[0].real
    e_excit_site = evolve.result.expect[1].real
    t = evolve.result.t
    
    
    # Plot
    fig, ax = plt.subplots(1,2, figsize=(8,3))
    ax[0].plot(t, e_phot_tot/ntls)
    ax[1].plot(t, e_excit_site)
    ax[0].set_xlabel('t')
    ax[1].set_xlabel('t')
    ax[0].set_ylabel(r'$\langle n\rangle /N$')
    ax[1].set_ylabel(r'$\langle\sigma^+\sigma^-\rangle$')
    plt.tight_layout()
    plt.show()

    




