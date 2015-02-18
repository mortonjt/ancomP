
import numpy as np
import scipy.stats as ss
from ancom.linalg.composition import (power, perturb, clr, ilr,
                                      centre, variation_matrix, total_variation,
                                      zero_replacement, closure)

import unittest
import numpy.testing as np_test

class TestComposition(unittest.TestCase):
    
    def test_zero_replacement(self):
        D = 5
        mat = np.vstack((
            np.array(range(1,4) + [0]*1 + [5]),
            np.array(range(1,2) + [0]*2 + range(4,6)),
            np.array(range(1,D+1))))

        amat = zero_replacement(mat)
        np_test.assert_array_almost_equal(amat,
                          np.vstack((
                              np.array([1, 2, 3, .2, 5]),
                              np.array([1, .4, .4, 4, 5]),
                              np.array([1, 2, 3, 4, 5]))))
        
    def test_closure(self):
        mat = np.vstack((
            np.array([2, 2, 6]),
            np.array([4, 4, 2])))
        amat = closure(mat)

        np_test.assert_array_almost_equal(amat,
                          np.vstack((
                              np.array([.2, .2, .6]),
                              np.array([.4, .4, .2]))))

        D = 5
        mat = np.vstack((
            np.array(range(1,4) + [0]*1 + [5]),
            np.array(range(1,2) + [0]*2 + range(4,6)),
            np.array(range(1,D+1))))

        amat = closure(zero_replacement(mat))
        
        np_test.assert_array_almost_equal(amat,
                          np.vstack((
                              np.array([1, 2, 3, .2, 5]) / 11.2,
                              np.array([1, .4, .4, 4, 5]) / 10.8,
                              np.array([1, 2, 3, 4, 5]) / 15.)))

        
    def test_perturb(self):
        mat = np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2])))
        amat = mat
        pmat = perturb(amat, np.array([.5, .5, .5]))
        np_test.assert_array_almost_equal(pmat,
                          np.vstack((
                              np.array([.2, .2, .6]),
                              np.array([.4, .4, .2]))))
        
        pmat = perturb(amat, np.array([10, 10, 20]))
        np_test.assert_array_almost_equal(pmat,
                          np.vstack((
                              np.array([.125, .125, .75]),
                              np.array([1./3, 1./3, 1./3]))))
        
        
    def test_power(self):
        mat = np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2])))
        amat = mat
        pmat = power(amat, 2)
        np_test.assert_array_almost_equal(pmat,
                          np.vstack((
                              np.array([.04, .04, .36])/.44,
                              np.array([.16, .16, .04])/.36)))
        
    def test_clr(self):
        mat = np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2])))
        amat = mat
        cmat = clr(amat)
        A = np.array([.2, .2, .6])
        B = np.array([.4, .4, .2])

        np_test.assert_array_almost_equal(cmat,
                          np.vstack((
                              np.log(A / np.exp(np.log(A).mean())) ,
                              np.log(B / np.exp(np.log(B).mean())) )))
                
unittest.main()
