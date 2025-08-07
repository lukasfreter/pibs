#!/usr/bin/env python
from time import time
import pickle
import numpy as np
import scipy.sparse as sp
from math import factorial

def get_size(obj):
    """Estimate size of object in MB"""
    return round(len(pickle.dumps(obj))/1024**2, 2)
def export(obj, fp):
    pass

def timeit(func, args=None, msg=''):
    print(f'Running {msg} with arguments {args}...', flush=True)
    t0 = time()
    func(*args)
    elapsed = time()-t0
    print(f'Complete {elapsed:.0f}s', flush=True)

# other helper functions
def degeneracy_outer_invariant_optimized(outer1, outer2, inner):
    """ calculate how many distinct permutations there are of the spins (outer1, inner)
    and (inner, outer2), which leave outer1 and outer2 invariant. Necessary
    for commutator term in Liouvillian -i[H,rho]"""
    xi = outer1+ 2*outer2
    deg = 1
    for i in range(4):
        l = np.where(xi==i)[0]
        # print('loop',i)
        # print(l)
        if len(l) == 0:
            continue
        
        sub_inner = inner[l]
        s = sum(sub_inner)
        factor =  factorial(len(l)) / (factorial(s)*factorial(len(l)-s))
        deg = deg * factor
        
    return int(deg)


def states_compatible(state1, state2):
    """ checks, if state1 and state2 are equivalent up to permutation of spins"""
    if state1[0] != state2[0]:
        return False
    
    from numpy import where, setdiff1d, intersect1d
    
    spins1 = state1[1:]
    spins2 = state2[1:]
    if(sum(state1) != sum(state2)):
        return False
    
    return True

def permute_compatible(comp1, comp2, permute):
    """comp1 and comp2 contain compatible (=equal up to permutation) spin states.
    Find the permutation, that transforms comp2 in comp1 and perform the same
    transformation to permute. This is important to calculate the proper H-element
    in calc_L_line_block1."""
    
    # check indices, where the arrays have ones
    idx1 = np.where(comp1 == 1)[0]
    idx2 = np.where(comp2 == 1)[0]
    
    # find common indices and remove them, because they are already in order
    common_elements = np.intersect1d(idx1,idx2)
    idx_ones1 = np.setdiff1d(idx1, common_elements)
    idx_ones2 = np.setdiff1d(idx2, common_elements)
    
    cp_permute = np.copy(permute)
    for i in range(len(idx_ones1)):
        cp_permute[idx_ones2[i]] = permute[idx_ones1[i]]
        cp_permute[idx_ones1[i]] = permute[idx_ones2[i]]
        
    return cp_permute

def degeneracy_spin_gamma(spin1, spin2):
    """Return number of indices where spin1 and spin2 are up. For L0 part of
    individual loss L[sigmam]"""
    return np.count_nonzero(spin1+spin2==0)

def degeneracy_gamma_changing_block_efficient(outer1, outer2, inner1, inner2):
    """Find simultaneous permutation of inner1 and inner2, such that all but one
    spin index align, and in exactly the same positions. This is necessary
    for calculating the Lindblad operator of sigma minus (individual decay). efficient way """
    from itertools import permutations
    from numpy import where
    Oc = outer1 + 2*outer2
    Ic = inner1 + 2*inner2
    
    outer_num3 = len(where(Oc==3)[0])
    inner_num3 = len(where(Ic==3)[0])
    
    if outer_num3 - inner_num3 == 1:
        return outer_num3
    else:
        return 0


# operators here?

def qeye(N):

    return sp.eye(N, N, dtype=complex, format='csr')

def destroy(N):

    return sp.spdiags(np.sqrt(range(0, N)),
                           1, N, N, format='csr')

def create(N):

    qo = destroy(N)  # create operator using destroy function
    qo = qo.T.tocsr()  # transpose data in Qobj and convert to csr
    return qo

def sigmap():
    return sp.spdiags(np.array([0.0,1.0]),
                      1, 2, 2, format='csr')
                      
def sigmam():
    return sigmap().T.tocsr()

def sigmaz():
    return sp.spdiags(np.array([1.0,-1.0]),
                      0, 2, 2, format='csr')
def sigmax():
    return sigmap() + sigmam()

def sigmay():
    return -1j*sigmap() + 1j*sigmam()
    
def tensor(*args):

    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    # elif len(args) == 1 and isinstance(args[0], Qobj):
    #     # tensor is called with a single input, do nothing
    #     return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args

    out = []

    for n, q in enumerate(qlist):
        if n == 0:
            out = q
        else:
            out = sp.kron(out, q, format='csr')

    return out
    
def basis(N, n=0):
    """Create Fock density matrix for N-level Hilbert space
    with excitation in level n""" 
    
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.integer))):
        raise ValueError("n must be integer n >= 0")

    if n > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    bas = sp.lil_matrix((N, 1))  # column vector of zeros
    bas[n, 0] = 1  # 1 located at position n
    bas = bas.tocsr()

    return tensor(bas, bas.T)

def thermal_dm(N, nth):
    """ Create thermal density matreix for N-level Hilbert space with average
    photon number nth"""
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    dm = np.zeros((N,N))
    x= nth /(1+nth)
    Z = (1 - x**N) / (1 - x)  # partition function
    for i in range(N):
       dm[i,i] = 1.0 / Z * x**i
        
    return sp.csr_matrix(dm)
        
        
    
def expect(oper, state):

    # calculates expectation value via TR(op*rho)
    
    return ((oper@state).toarray()).trace()
    
    # return (oper.dot(state).toarray()).trace()

def vector_to_operator(op):

    n = int(np.sqrt(op.shape[0]))
    q = sp_reshape(op.T, (n, n)).T  # This seems to be quite expensive operation 
    return q


def sp_reshape(A, shape, format='csr'):

    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('Shape must be a list of two integers')

    C = sp.coo_matrix(A)
    nrows, ncols = C.shape
    size = nrows * ncols
    new_size = shape[0] * shape[1]

    if new_size != size:
        raise ValueError('Total size of new array must be unchanged.')

    flat_indices = ncols * C.row + C.col
    new_row, new_col = divmod(flat_indices, shape[1])
    B = sp.coo_matrix((C.data, (new_row, new_col)), shape=shape)

    if format == 'csr':
        return B.tocsr()
    elif format == 'coo':
        return B
    elif format == 'csc':
        return B.tocsc()
    elif format == 'lil':
        return B.tolil()
    else:
        raise ValueError('Return format not valid.')

def _multinominal(bins):
    """calculate multinominal coeffcient"""
    
    from math import factorial
    
    n = sum(bins)
    combinations  = factorial(n)
    for count_bin in range(len(bins)):
        combinations = combinations//factorial(bins[count_bin])
    return combinations


def wigner_d(theta, j, n, m):
    """ Calculate small Wigner d matrix d_{n,m}^j(theta) """
    from math import factorial
    smin = int(max(0, m-n))
    smax = int(min(j+m, j-n))
    print(smin,smax)

    
    #prefactor
    p = np.sqrt( float(factorial(int(j+n)))) * np.sqrt(float(factorial(int(j-n)))) * np.sqrt(float(factorial(int(j+m)))) * np.sqrt(float(factorial(int(j-m)) ))
    
    # sum
    d = 0
    for s in range(smin, smax+1):
        pre = factorial(int(j+m-s))*factorial(int(s))*factorial(int(n-m+s))*factorial(int(j-n-s))
        d += (-1)**(n-m+s) * np.cos(theta/2)**(2*j+m-n-2*s)*np.sin(theta/2)**(n-m+2*s) / pre
    
    d = d * p
    return d
    


class Progress:
    def __init__(self, total, description='', start_step=0):
        self.description = description
        self.step = start_step
        self.end = total-1
        self.percent = self.calc_percent()
        self.started = False

    def calc_percent(self):
        return int(100*self.step/self.end)

    def update(self, step=None):
        # print a description at the start of the calculation
        if not self.started:
            print('{}{:4d}%'.format(self.description, self.percent), end='', flush=True)
            self.started = True
            return
        # progress one step or to the specified step
        if step is None:
            self.step += 1
        else:
            self.step = step
        percent = self.calc_percent()
        # only waste time printing if % has actually increased one integer
        if percent > self.percent:
            print('\b\b\b\b\b{:4d}%'.format(percent), end='', flush=True)
            self.percent = percent
        if self.step == self.end:
            print('', flush=True)
