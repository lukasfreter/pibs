#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:59:30 2024

@author: freterl1
"""
from collections import defaultdict
import numpy as np
import psutil
import ray
import sys

import time

#https://maxpumperla.com/learning_ray/ch_02_ray_core/

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)


@ray.remote
class SharedMemory:
    def __init__(self):
        self._array = np.random.rand(10)
    def array(self):
        return self._array
    def setarray(self, value, index):
        self._array[index] = value
        

    

@ray.remote
def retrieve_task(memory, index):
    value = ray.get(memory.array.remote())[index]
    memory.setarray.remote(0.0, index)
    return value

def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")

memory = SharedMemory.remote()
print(ray.get(memory.array.remote()))

start = time.time()
object_references = [retrieve_task.remote(memory, i) for i in range(10)]
data = ray.get(object_references)
print_runtime(data, start)

print(ray.get(memory.array.remote()))

ray.shutdown()