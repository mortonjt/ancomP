
import numpy as np
from ancom.linalg.composition import CompositionMatrix

import unittest

class TestComposition(unittest.TestCase):
    
    def test_zero_replacement(self):
        D = 5
        mat = np.matrix(np.vstack((
            np.array(range(1,4) + [0]*1 + [5]),
            np.array(range(1,2) + [0]*2 + range(4,6)),
            np.array(range(1,D+1)))),dtype=np.float32)
        amat = CompositionMatrix(np.matrix([0]))
        amat.mat = mat

        amat.zero_replacement()
        self.assertEquals(amat.mat,
                          np.matrix(np.vstack((
                              np.array([1, 2, 3, .2, 5]),
                              np.array([1, .4, .4, 4, 5]),
                              np.array([1, 2, 3, 4, 5]),
                              dtype=np.float32))))
        amat.closure()
        self.assertEquals(amat.mat,
                          np.matrix(np.vstack((
                              np.array([1, 2, 3, .2, 5])/11.2,
                              np.array([1, .4, .4, 4, 5])/10.8,
                              np.array([1, 2, 3, 4, 5])/15.,
                              dtype=np.float32))))
        
    def test_closure(self):
        mat = np.matrix(np.vstack((
            np.array([2, 2, 6]),
            np.array([4, 4, 2]))),dtype=np.float32)
        amat = CompositionMatrix(np.matrix([0]))
        amat.mat = mat
        amat.closure()
        self.assertEquals(amat.mat,
                          np.matrix(np.vstack((
                              np.array([.2, .2, .6]),
                              np.array([.4, .4, .2]),
                              dtype=np.float32))))     
        
    def test_add(self):
        mat = np.matrix(np.vstack((
            np.array([.2, .2, .6]),
            np.array([.4, .4, .2]))),dtype=np.float32)
        amat = CompositionMatrix(mat)
        
        
    def test_mult(self):
        pass
    def test_clr(self):
        pass
    def test_svd(self):
        pass


unittest.main()
