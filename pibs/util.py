#!/usr/bin/env python
from time import time
import pickle

def export(obj, fp):
    pass

def timeit(func, args=None, msg=''):
    print(f'Running {msg} with arguments {args}...', flush=True)
    t0 = time()
    func(*args)
    elapsed = time()-t0
    print(f'Complete {elapsed:.0f}s', flush=True)

# other helper functions

# operators here?

