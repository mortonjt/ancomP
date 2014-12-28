import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from stats.pearson import (_cl_pearson_test,
                           _np_pearson_test,
                           _naive_pearson_test)

import unittest

class TestPearson(unittest.TestCase):

    def test_basic(self):
        print "Basic Pearson test"
        mat = np.matrix(np.array(range(10),dtype=np.float32))
        mat = np.matrix(np.random.random(10),dtype=np.float32)
        d = 40
        mat = np.matrix(np.vstack((
            np.array(range(1,d+1)),
            np.array(range(2,d+2)),
            np.array(range(3,d+3)),
            np.array(range(4,d+4)),
            np.array(range(5,d+5)),
            np.random.random(d))),dtype=np.float32)

        x = np.array(range(d),dtype=np.float32)
        np_r, np_p = _np_pearson_test(mat,x)

        mat = np.array(mat)

        nv_r, nv_p = _naive_pearson_test(mat,x)
        nv_r = np.matrix(nv_r).transpose()
        nv_p = np.matrix(nv_p).transpose()
        self.assertEquals(sum(abs(np_r-nv_r) > 0.1), 0)
        self.assertEquals(sum(abs(np_p-nv_p) > 0.1), 0)
unittest.main()
