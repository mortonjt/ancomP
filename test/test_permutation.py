
import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy

import unittest

from stats.permutation import (_naive_mean_permutation_test,
                               _np_mean_permutation_test,
                               _cl_mean_permutation_test,
                               _init_device,
                               _two_sample_mean_statistic)

class TestPermutation(unittest.TestCase):

    
    def test_basic1(self):
        ## Basic quick test
        D = 5
        M = 6
        mat = np.array([range(10)]*M,dtype=np.float32)
        cats = np.array([0]*D+[1]*D,dtype=np.float32)

        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
        cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)

        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        self.assertEquals(sum(abs(nv_stats-np_stats) > 0.1), 0)
        self.assertEquals(sum(abs(nv_stats-cl_stats) > 0.1), 0)
        #Check test statistics
        self.assertAlmostEquals(sum(nv_stats-5),0,0.01)
        self.assertAlmostEquals(sum(np_stats-5),0,0.01)
        self.assertAlmostEquals(sum(cl_stats-5),0,0.01)
        #Check for small pvalues
        self.assertEquals(sum(nv_p>0.05),0)
        self.assertEquals(sum(np_p>0.05),0)
        self.assertEquals(sum(cl_p>0.05),0)


    def test_basic2(self):
        ## Basic quick test
        D = 5
        M = 6
        mat = np.array([[0]*D+[10]*D]*M,dtype=np.float32)
        cats = np.array([0]*D+[1]*D,dtype=np.float32)

        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
        cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)

        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        self.assertEquals(sum(abs(nv_stats-np_stats) > 0.1), 0)
        self.assertEquals(sum(abs(nv_stats-cl_stats) > 0.1), 0)
        #Check test statistics
        self.assertAlmostEquals(sum(nv_stats-10),0,0.01)
        self.assertAlmostEquals(sum(np_stats-10),0,0.01)
        self.assertAlmostEquals(sum(cl_stats-10),0,0.01)
        #Check for small pvalues
        self.assertEquals(sum(nv_p>0.05),0)
        self.assertEquals(sum(np_p>0.05),0)
        self.assertEquals(sum(cl_p>0.05),0)

    def test_large(self):
        ## Large test
        N = 10
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N/2)+[1]*(N/2)),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N/2)+[1]*(N/2),dtype=np.float32)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
        cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)
        self.assertEquals(sum(abs(np_stats-cl_stats) > 0.1), 0)
        
    def test_random(self):
        ## Randomized test
        N = 50
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N/2)+[100]*(N/2)),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N/2)+[1]*(N/2),dtype=np.float32)
        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
        cl_stats, cl_p = _cl_mean_permutation_test(mat,cats,1000)
        nv_stats = np.matrix(nv_stats).transpose()
        
        self.assertAlmostEquals(nv_stats[0],100.,4)
        self.assertAlmostEquals(np_stats[0],100.,4)
        self.assertAlmostEquals(cl_stats[0],100.,4)
        self.assertLess(nv_p[0],0.05)
        self.assertLess(np_p[0],0.05)
        self.assertLess(cl_p[0],0.05)

        #Check test statistics
        self.assertEquals(sum(nv_stats[1:]>nv_stats[0]), 0)
        self.assertEquals(sum(np_stats[1:]>np_stats[0]), 0)
        self.assertEquals(sum(cl_stats[1:]>cl_stats[0]), 0)
        
        self.assertEquals(sum(abs(np_stats-cl_stats) > 0.1), 0)
        self.assertEquals(sum(abs(nv_stats-cl_stats) > 0.1), 0)
        
    def test_mean_stat(self):
        N = 20
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N/4)+[1]*(3*N/4)),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N/4)+[1]*(3*N/4), dtype=np.float32)
        d_mat, d_perms = _init_device(mat,cats)
        mean_stats, pvalues = _two_sample_mean_statistic(d_mat, d_perms)
        self.assertEquals(mean_stats.argmax(), 0)
        self.assertEquals(mean_stats.max(), 1)
        self.assertLess(pvalues.min(), 0.05)
        
    def test_init_device(self):
        N = 10
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N/2)+[1]*(N/2)),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N/2)+[1]*(N/2), dtype=np.float32)
        d_mat, d_perms = _init_device(mat,cats)
        self.assertEquals(type(d_mat), pv.pycore.Matrix)
        self.assertEquals(type(d_perms), pv.pycore.Matrix)

        self.assertEquals(d_mat.shape, (7, 10) )
        self.assertEquals(d_perms.shape, (10, 2002) )
        
        
    def test_times(self):
        ## Compare timings between numpy and pyviennacl
        N = 60
        M = 60
        mat = np.array([range(M)]*N,dtype=np.float32)
        cats = np.array([0]*(M/2)+[1]*(M/2),dtype=np.float32)

        t1 = time()
        counts = 3
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
        self.assertGreater(nv_time,cl_time)
        self.assertGreater(np_time,cl_time)
        
unittest.main()
