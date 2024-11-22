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
from util import qeye, create, destroy, sigmam, sigmap, tensor, basis, thermal_dm
import matplotlib.pyplot as plt
from time import time
import pickle
import multiprocessing
import scipy.sparse as sp
from util import wigner_d


plt.rcParams.update({'font.size': 12,
                      'xtick.labelsize' : 12,
                      'ytick.labelsize' : 12,
                      'lines.linewidth' : 1.5,
                      'lines.markersize': 5,
                      'figure.figsize': (10,6),
                      'figure.dpi': 150})

t0 = time()
# same parameters as in Peter Kirton's code.
ntls = 5#int(sys.argv[1])#number 2LS
nphot = ntls+1
w0 = 1.0
wc = 1.0
Omega = 0.0
g = Omega / np.sqrt(ntls)
kappa = 0.05#1e-02
gamma = 0#1e-03
gamma_phi = 0#0.075
gamma_phi_qutip = 4*gamma_phi

dt = 0.2 # timestep
tmax = 200-2*dt # for optimum usage of chunks in parallel evolution
chunksize=200  # time chunks for parallel evolution

atol=1e-12
rtol=1e-12
nsteps=1000


indi = Indices(ntls,nphot, debug=True, save = False)
# indi.print_elements()

# sys.exit()


# rotation matrix around x-axis of spin 1/2 : exp(-i*theta*Sx)=exp(-i*theta/2*sigmax) = cos(theta/2)-i*sin(theta/2)*sigmax
theta = np.pi
rot_x = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
rot_x_dag = np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],[1j*np.sin(theta/2), np.cos(theta/2)]])


# wigner d test
# from math import factorial
# fig, ax = plt.subplots()
# M = np.arange(-ntls/2, ntls/2, 1)
# for m in M:
#     # print(m)
#     if m >= 21:
#         color = 'r'
#     else:
#         color = 'b'
#     ax.scatter(m, wigner_d(theta, ntls/2, m , ntls/2), color=color)
#     # d = np.sqrt(float(factorial(ntls))) / (np.sqrt(float(factorial(int(ntls/2+m)))) * np.sqrt(float(factorial(int(ntls/2-m))))) * np.cos(theta/2)**(ntls/2+m)*np.sin(theta/2)**(ntls/2-m)
#     # N = ntls/2
#     # stirling = np.sqrt( 1/np.sqrt(np.pi) * N**(2*N+1/2) * (N**2 - m**2)**(-N-1/2) * ((N-m) / (N+m))**m)
#     # ax.scatter(m, d, color='r')
#     # ax.scatter(m, stirling, color='b')
    
# ax.set_xlabel('m')
# ax.set_ylabel(r'$d_{m, N/2}^{N/2}(\pi/4)$')
# plt.show()
# sys.exit()

rho_phot = basis(nphot,2) # second argument: number of photons in initial state
# rho_phot = thermal_dm(nphot, nth=1/2)
rho_spin = sp.csr_matrix(rot_x @ basis(2,0) @ rot_x_dag) # First argument: spin dimension 2. Second argument: 0=up, 1=down

# print(rho_phot.todense())
# sys.exit()
scale = 1e3
rho = Rho(rho_phot, rho_spin, indi) # initial condition with zero photons and all spins up.# sys.exit()


L = Models(wc, w0,g, kappa, gamma_phi,gamma,indi, parallel=1,progress=True, debug=False,save=True, num_cpus=None)
L.setup_L_Tavis_Cummings(progress=True)


# Operators for time evolution
adag = tensor(create(nphot), qeye(2))
a = tensor(destroy(nphot), qeye(2))
n = adag*a
n2 = adag*a*adag*a
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p, n2] # operators to calculate expectations for

evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol, nsteps=nsteps)
evolve.time_evolve_block_interp(ops, progress = True, expect_per_nu=False, start_block=None)
# evolve.time_evolve_chunk_parallel2(ops, chunksize=chunksize, progress=True, num_cpus=None)

e_phot_tot = evolve.result.expect[0].real
e_excit_site = evolve.result.expect[1].real
e_phot_n2 = evolve.result.expect[2].real
#expect_per_nu_phot = np.squeeze(evolve.result.expect_per_nu[:,0,:])
t = evolve.result.t

# g2 function: g2(t, 0)
G2 = e_phot_n2 - e_phot_tot
g2 = G2 / e_phot_tot**2


# two time correlations: g1
rho_phot_g1 = rho_phot @ create(nphot)
rho = Rho(rho_phot_g1, rho_spin, indi)
ops = [a]
evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol, nsteps=nsteps)
evolve.time_evolve_block_interp(ops, progress = True, expect_per_nu=False, start_block=None)
G1 = evolve.result.expect[0]
g1 = G1 / np.sqrt(e_phot_tot[0] * e_phot_tot)


# two time correlations: g2
rho_phot_g2 = destroy(nphot) @ rho_phot @ create(nphot)
rho = Rho(rho_phot_g2, rho_spin, indi)
ops = [n]
evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol, nsteps=nsteps)
evolve.time_evolve_block_interp(ops, progress = True, expect_per_nu=False, start_block=None)
G2 = evolve.result.expect[0]
g2 = G2 / (e_phot_tot[0] * e_phot_tot)


runtime = time() - t0

fig, ax = plt.subplots(2,1)
# ax[0].plot(t, e_phot_tot, label='n')
# ax[0].plot(t, e_phot_n2/ntls**2, label='n^2')
ax[0].plot(t, g1,label='g1')

ax[0].plot(t, g2,label='g2')
# ax[0].plot(t, g2_c2, label='g2 c2')
# ax[0].plot(t, G2/e_phot_tot**2)
# ax[0].plot(t, e_phot_tot, ls='--')
# ax[0].plot(t, g2_tau)
# ax[0].plot(t, e_phot_tot/ntls)

ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$\langle n\rangle$')
ax[1].plot(t, e_excit_site)
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$\langle \sigma_i^+\sigma_i^-\rangle$')
fig.suptitle(r'$N={N}$'.format(N=ntls))
ax[0].set_title(r'$\Delta={delta},\ g\sqrt{{N}}={Omega},\ \kappa={kappa},\ \gamma={gamma},\ \gamma_\phi={gamma_phi},\ \theta={theta}$'.format(delta=wc-w0, Omega=Omega,kappa=kappa,gamma=gamma,gamma_phi=gamma_phi,theta=theta))
ax[0].legend()
plt.show()
# sys.exit()










# common_params = {
#     'method': 'pibs',
#     'N': ntls,
#     'nphot': nphot,
#     'w0': w0,
#     'wc': wc,
#     'Delta': wc- w0,
#     'gamma': gamma,
#     'gamma_phi': gamma_phi, # value that we actually feed into code
#     'gamma_phi_qutip': gamma_phi*4, # gammaphi that is consistent with qutip 
#     'kappa': kappa,
#     'Omega': Omega,
#     'tmax': tmax,
#     'dt': dt,
#     'theta': theta,
#     'chunksize':chunksize,
#     'nsteps': nsteps,
#     'atol':atol,
#     'rtol':rtol,
    
#     }

# import itertools
# color = itertools.cycle(('-', '--',':'))
# fname = 'results/pibs_parallel_mp_N100_Delta-0.35_Omega0.4_kappa0.01_gamma0.001_gammaphi0.0075_tmax100.0_atol1e-18_rtol1e-15_solverbdf_nsteps10000_scale1000000_perNuTrue.pkl'
# with open(fname, 'rb') as handle:
#     data= pickle.load(handle)
# t = data['results']['t']
# expect_per_nu_phot = data['results']['e_phot_tot_nu']
# common_params = data['params']


# fig, ax = plt.subplots(1,2)
# count = 0
# for nu in range(common_params['N'],-1,-1):
#     if max(expect_per_nu_phot[nu,:]) < 1e-2 and False:
#         continue
#     count+=1
#     ls = next(color)
#     ax[0].plot(t, expect_per_nu_phot[nu, :].real,ls=ls, label=r'$\nu={nu}$'.format(nu=nu))
#     ax[1].plot(t, expect_per_nu_phot[nu, :].real,ls=ls, label=r'$\nu={nu}$'.format(nu=nu))
# ax[0].set_xlabel(r'$t$')
# ax[0].set_ylabel(r'$\langle n\rangle$')
# ax[1].set_xlabel(r'$t$')
# ax[1].set_ylabel(r'$\langle n\rangle$')
# ax[1].set_yscale('log')
# ax[0].legend(ncol=2)
# fig.suptitle(r'$N={N},\ \Delta={Delta},\ g\sqrt{{N}}={Omega},\ \kappa={kappa},\ \gamma={gamma},\ \gamma_\phi={gamma_phi},\ \theta={theta}$'.format(**common_params))
# plt.tight_layout()
# plt.show()
# print(count)

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
# e_phot_tot = np.sin(1*2*np.pi*t) + np.exp(-1j*2*np.pi*2*t)
res = {
    't':t,
    'e_phot_tot': e_phot_tot,
    'e_excit_site': e_excit_site, 
    # 'e_phot_a' : e_phot_a,
    'e_phot_n2' : e_phot_n2,
    'G2_tau0' : G2,
    'g2_tau0': g2
        }
data = {
        'params': params,
        'results': res,
        'runtime': runtime}

# fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}_tmax{tmax}_theta{theta}_atol{atol}_rtol{rtol}.pkl'
fname = f'results/example.pkl'
# fname = 'results/test_sin.pkl'
#save results in pickle file
with open(fname, 'wb') as handle:
    pickle.dump(data,handle)
