
import numpy as np
from ancom.linalg.composition import (CompositionMatrix,
                                      zero_replacement,
                                      closure)

import unittest
import numpy.testing as np_test

class TestComposition(unittest.TestCase):
    
    def test_zero_replacement(self):
        D = 5
        mat = np.matrix(np.vstack((
            np.array(range(1,4) + [0]*1 + [5]),
            np.array(range(1,2) + [0]*2 + range(4,6)),
            np.array(range(1,D+1)))),dtype=np.float32)

        amat = zero_replacement(mat)
        np_test.assert_array_almost_equal(amat,
                          np.matrix(np.vstack((
                              np.array([1, 2, 3, .2, 5]),
                              np.array([1, .4, .4, 4, 5]),
                              np.array([1, 2, 3, 4, 5]))),
                              dtype=np.float32))
        
    def test_closure(self):
        mat = np.matrix(np.vstack((
            np.array([2, 2, 6]),
            np.array([4, 4, 2]))),dtype=np.float32)
        amat = CompositionMatrix(np.matrix([0]))
        amat = closure(mat)

        np_test.assert_array_almost_equal(amat,
                          np.matrix(np.vstack((
                              np.array([.2, .2, .6]),
                              np.array([.4, .4, .2]))),
                              dtype=np.float32))

        D = 5
        mat = np.matrix(np.vstack((
            np.array(range(1,4) + [0]*1 + [5]),
            np.array(range(1,2) + [0]*2 + range(4,6)),
            np.array(range(1,D+1)))),dtype=np.float32)

        amat = closure(zero_replacement(mat))
        
        np_test.assert_array_almost_equal(amat,
                          np.matrix(np.vstack((
                              np.array([1, 2, 3, .2, 5]) / 11.2,
                              np.array([1, .4, .4, 4, 5]) / 10.8,
                              np.array([1, 2, 3, 4, 5]) / 15.)),
                              dtype=np.float32))

        
    def test_perturb(self):
        mat = np.matrix(np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2]))),dtype=np.float32)
        amat = CompositionMatrix(mat)
        pmat = amat + np.array([.5, .5, .5])
        np_test.assert_array_almost_equal(pmat.mat,
                          np.matrix(np.vstack((
                              np.array([.2, .2, .6]),
                              np.array([.4, .4, .2]))),
                              dtype=np.float32))
        
        pmat = amat + np.array([10, 10, 20])
        np_test.assert_array_almost_equal(pmat.mat,
                          np.matrix(np.vstack((
                              np.array([.125, .125, .75]),
                              np.array([1./3, 1./3, 1./3]))),
                              dtype=np.float32))
        
        
    def test_power(self):
        mat = np.matrix(np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2]))),dtype=np.float32)
        amat = CompositionMatrix(mat)
        pmat = amat * 2
        np_test.assert_array_almost_equal(pmat.mat,
                          np.matrix(np.vstack((
                              np.array([.04, .04, .36])/.44,
                              np.array([.16, .16, .04])/.36)),
                              dtype=np.float32))
        
    def test_clr(self):
        pass
    def test_svd(self):
        pass


unittest.main()
