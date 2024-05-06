#!/usr/bin/env python
import numpy as np
from itertools import product
from multiprocessing import Pool
from util import export
import os, sys, logging

class Indices:
    """Indices for and mappings between compressed and supercompressed 
    density matrices containing unique representative elements for each
    permutation invariant subspace of nspins identical spins. The 
    representative elements are chosen such that, if i=0,1,...,nspins-1
    enumerates the spins one has zeta_0 >= zeta_1 >=... is monotonically
    decreasing (increasing?) where zeta is the simple finite pairing
    function
        zeta_i = s^L_i + (dim_s) * s^R_i
    for site (spin) i having element |s^L_i><s^R_i| with left and right 
    spin values s^L_i and s^R_i. Nominally, dim_s = 2 i.e. s^L_i, s^R_i
    = 0 or 1 (spin-1/2).

    In the compressed form, the spin indices are multiplied by the left/right
    photon part of the density matrices.
    In the supercompressed form, the (smaller) set of elements are grouped
    according to the total excitation number nu
    [can we get rid of compressed form entirely?]
    """
    def __init__(self, nspins, nphot=None, verbose=True):
        if nphot is None:
            nphot = nspins + 1
        self.nspins, self.nphot = nspins, nphot
        self.indices_elements = []
        self.indices_elements_inv = {}
        self.mapping_block = []
        # do all setup calculations
        util.timeit(self.list_equivalent_elements, msg='setting up indices_elements')
        self.export()

    def list_equivalent_elements(self):
        # write to
        pool = Pool()
        pool.map()


    def export(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)#??
        from util import export
        export(self, filepath)

    def load(self):
        data = pickle.load
        # do some checks
        # at least tell user what they loaded
        print('ntls={}...')


ind = Indices(20, 21)
Indices.export(filepath)
class BlockL:
    """Description"""
    def __init__(self):
        # initialisation

if __name__ == '__main__':
    # Testing purposes
    indi = Indices()
