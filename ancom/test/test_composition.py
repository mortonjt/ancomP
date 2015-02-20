
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
                np.array([[ 0.09056604,  0.18113208,  0.27169811,  0.00377358,  0.45283019],
                          [ 0.09913793,  0.00431034,  0.00431034,  0.39655172,  0.49568966],
                          [ 0.06666667,  0.13333333,  0.2       ,  0.26666667,  0.33333333]]))
    
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

        amat = mat
        pmat = power(amat, np.array([2, 2, 2]))
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
    def test_ilr(self):
        # mat =np.array([[np.exp(1), 1, 1]])
        # np_test.assert_array_almost_equal(ilr(mat),
        #                                   np.array([np.exp(1), 1]))
        
        # mat =np.array([[1, np.exp(1), 1]])
        # np_test.assert_array_almost_equal(ilr(mat),
        #                                   np.array([1, np.exp(1)]))
        pass
unittest.main()
