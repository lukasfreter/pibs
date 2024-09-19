#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:06:18 2024

@author: freterl1


Standard example for calculating time evolution with pibs code.

"""

import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
sys.path.insert(0, '..')
import numpy as np
from setup import Indices, Rho, Models
from propagate import TimeEvolve
from util import qeye, create, destroy, sigmam, sigmap, tensor, basis
import matplotlib.pyplot as plt
from time import time
import pickle
import multiprocessing
import scipy.sparse as sp


# plt.rcParams.update({'font.size': 18,
#                      'xtick.labelsize' : 18,
#                      'ytick.labelsize' : 18,
#                      'lines.linewidth' : 2,
#                      'lines.markersize': 10,
#                      'figure.figsize': (10,6),
#                      'figure.dpi': 150})

t0 = time()
# same parameters as in Peter Kirton's code.
ntls =20#int(sys.argv[1])#number 2LS
nphot = ntls+1
w0 = 1.0
wc = 0.65
Omega = 0.1N#0.4
g = Omega / np.sqrt(ntls)
kappa = 1e-02
gamma = 1e-03
gamma_phi = 0.0075
gamma_phi_qutip = 4*gamma_phi

dt = 0.2 # timestep
tmax = 200-2*dt # for optimum usage of chunks in parallel evolution
chunksize=200  # time chunks for parallel evolution

atol=1e-10
rtol=1e-10
nsteps=1000


indi = Indices(ntls, debug=True, save = False)

# rotation matrix around x-axis of spin 1/2 : exp(-i*theta*Sx)=exp(-i*theta/2*sigmax) = cos(theta/2)-i*sin(theta/2)*sigmax
theta = 0#np.pi/2
rot_x = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
rot_x_dag = np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],[1j*np.sin(theta/2), np.cos(theta/2)]])

rho_phot = basis(nphot,0) # second argument: number of photons in initial state
rho_spin = sp.csr_matrix(rot_x @ basis(2,0) @ rot_x_dag) # First argument: spin dimension 2. Second argument: 0=up, 1=down

scale = 1e3
rho = Rho(rho_phot, rho_spin, indi) # initial condition with zero photons and all spins up.# sys.exit()


L = Models(wc, w0,g, kappa, gamma_phi,gamma,indi, parallel=1,progress=True, debug=False,save=True, num_cpus=None)
L.setup_L_Tavis_Cummings(progress=True)


# Operators for time evolution
n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol, nsteps=nsteps)

# evolve.time_evolve_block_interp(ops, progress = True)
evolve.time_evolve_chunk_parallel2(ops, chunksize=chunksize, progress=True, num_cpus=None)

e_phot_tot = evolve.result.expect[0].real
e_excit_site = evolve.result.expect[1].real
t = evolve.result.t

runtime = time() - t0

fig, ax = plt.subplots(2)
ax[0].plot(t, e_phot_tot)
# ax[1].plot(t, e_excit_site)
ax[1].plot(t[:-1], np.diff(e_excit_site))

fig.suptitle(r'$N={N}$, chunksize={chunksize}'.format(N=ntls, chunksize=chunksize))
ax[0].set_title(r'$\Delta={delta},\ g\sqrt{{N}}={Omega},\ \kappa={kappa},\ \gamma={gamma},\ \gamma_\phi={gamma_phi},\ \gamma_\phi^{{qutip}}={gamma_phi_qutip}$'.format(delta=wc-w0, Omega=Omega,kappa=kappa,gamma=gamma,gamma_phi=gamma_phi,gamma_phi_qutip=gamma_phi_qutip))


# ax.legend()
plt.show()
# sys.exit()
    



# store results
params = {
    'method': 'pibs',
    'N': ntls,
    'nphot': nphot,
    'w0': w0,
    'wc': wc,
    'Delta': wc- w0,
    'gamma': gamma,
    'gamma_phi': gamma_phi, # value that we actually feed into code
    'gamma_phi_qutip': gamma_phi*4, # gammaphi that is consistent with qutip 
    'kappa': kappa,
    'Omega': Omega,
    'tmax': tmax,
    'dt': dt,
    'theta': theta,
    'chunksize':chunksize,
    'nsteps': nsteps,
    'atol':atol,
    'rtol':rtol,
    
    }
res = {
    't':t,
    'e_phot_tot': e_phot_tot,
    'e_excit_site': e_excit_site,    
        }
data = {
        'params': params,
        'results': res,
        'runtime': runtime}

fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}_tmax{tmax}_theta{theta}_atol{atol}_rtol{rtol}.pkl'
#fname = f'results/{params["method"]}.pkl'
#save results in pickle file
with open(fname, 'wb') as handle:
    pickle.dump(data,handle)
