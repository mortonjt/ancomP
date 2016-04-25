import numpy as np
import scipy.sparse as sp
import pandas as pd
from time import time
import copy

import unittest
import numpy.testing as np_test

from ancomP.stats.permutation import (_init_reciprocal_perms,
                                      _init_categorical_perms,
                                      _np_two_sample_mean_statistic,
                                      _naive_mean_permutation_test,
                                      _np_mean_permutation_test,
                                      _naive_t_permutation_test,
                                      _np_t_permutation_test,
                                      _np_two_sample_t_statistic,
                                      _np_k_sample_f_statistic,
                                      _naive_f_permutation_test
                                      )

class TestPermutation(unittest.TestCase):

    def test_init_perms(self):
        cats = np.array([0, 1, 2, 0, 0, 2, 1])
        perms = _init_categorical_perms(cats, permutations=0)
        np_test.assert_array_equal(perms,
                          np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0]]))

    def test_basic_mean1(self):
        ## Basic quick test
        D = 5
        M = 6
        mat = np.array([range(10)]*M,dtype=np.float32)
        cats = np.array([0]*D+[1]*D,dtype=np.float32)

        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)

        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        nv_stats = np.array(nv_stats)
        np_stats = np.array(np_stats)
        self.assertEqual(sum(abs(nv_stats-np_stats) > 0.1)[0], [0])

        #Check test statistics

        self.assertAlmostEquals(sum(nv_stats-5)[0], 0, 4)
        self.assertAlmostEquals(sum(np_stats-5)[0], 0, 4)

        #Check for small pvalues
        self.assertEquals(sum(nv_p>0.05)[0],0)
        self.assertEquals(sum(np_p>0.05)[0],0)

        np_test.assert_array_almost_equal(nv_stats, np_stats)


    def test_basic_mean2(self):
        ## Basic quick test
        D = 5
        M = 6
        mat = np.array([[0]*D+[10]*D]*M,dtype=np.float32)
        cats = np.array([0]*D+[1]*D,dtype=np.float32)

        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)

        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        self.assertEquals(sum(abs(nv_stats-np_stats) > 0.1), 0)
        #Check test statistics
        self.assertAlmostEquals(sum(nv_stats-10),0,0.01)
        self.assertAlmostEquals(sum(np_stats-10),0,0.01)

        #Check for small pvalues
        self.assertEquals(sum(nv_p>0.05),0)
        self.assertEquals(sum(np_p>0.05),0)

        np_test.assert_array_almost_equal(nv_stats, np_stats)


    def test_large(self):
        ## Large test
        N = 10
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N//2)+[1]*(N//2)),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N//2)+[1]*(N//2),dtype=np.float32)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)


    def test_random_mean_test(self):
        ## Randomized test
        N = 50
        mat = np.array(
            np.matrix(np.vstack((
                np.array([0]*(N//2)+[100]*(N//2)),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*(N//2)+[1]*(N//2),dtype=np.float32)
        nv_stats, nv_p = _naive_mean_permutation_test(mat,cats,1000)
        np_stats, np_p = _np_mean_permutation_test(mat,cats,1000)
        nv_stats = np.matrix(nv_stats).transpose()

        self.assertAlmostEquals(nv_stats[0],100.,4)
        self.assertAlmostEquals(np_stats[0],100.,4)

        self.assertLess(nv_p[0],0.05)
        self.assertLess(np_p[0],0.05)

        #Check test statistics
        self.assertEquals(sum(nv_stats[1:]>nv_stats[0]), 0)
        self.assertEquals(sum(np_stats[1:]>np_stats[0]), 0)

        np_test.assert_array_almost_equal(np_stats, nv_stats)


    def test_t_test_basic1(self):
        np.set_printoptions(precision=3)
        N = 20
        mat = np.array(
            np.matrix(np.vstack((
                np.hstack((np.arange((3*N)//4), np.arange(N//4)+100)),
                np.random.random(N))),dtype=np.float32))
        cats = np.array([0]*((3*N)//4)+[1]*(N//4), dtype=np.float32)
        nv_t_stats, pvalues = _naive_t_permutation_test(mat, cats)
        perms = _init_categorical_perms(cats)
        mat, perms = np.matrix(mat), np.matrix(perms)
        np_t_stats, pvalues = _np_two_sample_t_statistic(mat, perms)
        np_test.assert_array_almost_equal(nv_t_stats, np_t_stats, 5)

    def test_f_test_basic1(self):
        np.set_printoptions(precision=3)
        N = 9
        mat = np.vstack((
                np.hstack((np.arange(N//3),
                           np.arange(N//3)+100,
                           np.arange(N//3)+200)),
                np.hstack((np.arange(N//3)+100,
                           np.arange(N//3)+300,
                           np.arange(N//3)+400))))
        cats = np.array([0]*(N//3)+
                        [1]*(N//3)+
                        [2]*(N//3),
                        dtype=np.float32)
        nv_f_stats, pvalues = _naive_f_permutation_test(mat, cats)
        perms = _init_categorical_perms(cats)
        np_f_stats, pvalues = _np_k_sample_f_statistic(mat, cats, perms)
        np_test.assert_array_almost_equal(nv_f_stats, np_f_stats, 5)

if __name__=='__main__':
    unittest.main()

