
import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy

import unittest

## Basic quick test
mat = np.array([range(10)]*6,dtype=np.float32)
cats = np.array([0]*5+[1]*5,dtype=np.float32)

nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)

nv_stats = np.matrix(nv_stats).transpose()
assert sum(abs(nv_stats-np_stats) > 0.1) == 0
assert sum(abs(nv_stats-cl_stats) > 0.1) == 0




## Profile code
def nv_prof():
    N = 20
    M = 20
    counts = 10
    mat = np.array([range(M)]*N,dtype=np.float32)
    cats = np.array([0]*(M/2)+[1]*(M/2),dtype=np.float32)
    for _ in range(counts):
        _naive_mean_permutation_test(mat,cats,1000)

import cProfile
cProfile.run("nv_prof()")

def np_prof():
    N = 20
    M = 20
    counts = 10
    mat = np.array([range(M)]*N,dtype=np.float32)
    cats = np.array([0]*(M/2)+[1]*(M/2),dtype=np.float32)
    for _ in range(counts):
        _np_mean_permutation_test(mat,cats,1000)

import cProfile
cProfile.run("np_prof()")

def cl_prof():
    N = 1000
    M = 1000
    counts = 10
    mat = np.array([range(M)]*N,dtype=np.float32)
    cats = np.array([0]*(M/2)+[1]*(M/2),dtype=np.float32)
    for _ in range(counts):
        _cl_mean_permutation_test(mat,cats,1000)

import cProfile
cProfile.run("cl_prof()")

## Large test
N = 100
M = 100
mat = np.array([range(M)]*N,dtype=np.float32)
cats = np.array([0]*(M/2)+[1]*(M/2),dtype=np.float32)

t1 = time()
counts = 10
for _ in range(counts):
    nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
nv_time = (time()-t1)/counts
t1 = time()
for _ in range(counts):
    np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
np_time = (time()-t1)/counts
t1 = time()
for _ in range(counts):
    cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)
cl_time = (time()-t1)/counts

print "Naive time [s]:", nv_time
print "Numpy time [s]:", np_time
print "GPU compute [s]:", cl_time
