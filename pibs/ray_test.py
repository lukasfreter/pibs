#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:59:30 2024

@author: freterl1
"""

# https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

import time
from datetime import datetime
import numpy as np
import sys
import multiprocessing
import ray

NUM_WORKERS = multiprocessing.cpu_count()

# ray init
ray.init(num_cpus=NUM_WORKERS)
np.random.seed(42)
ARRAY_SIZE = int(2e8)
ARRAY_SHAPE = (ARRAY_SIZE,)
data = np.random.random(ARRAY_SIZE)



@ray.remote
def np_sum_ray2(obj_ref, start,stop):
    return np.sum(obj_ref[start:stop])




def benchmark():
    chunk_size = int(ARRAY_SIZE / NUM_WORKERS)
    futures = []
    obj_ref = ray.put(data)
    ts = time.time_ns()
    for i in range(0, NUM_WORKERS):
        start = i + chunk_size if i == 0 else 0
        futures.append(np_sum_ray2.remote(obj_ref, start, i + chunk_size))
    results = ray.get(futures)
    return (time.time_ns() - start_time) / 1_000_000