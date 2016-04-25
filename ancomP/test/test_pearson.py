import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from ancomP.stats.pearson import (_cl_pearson_test,
                                  _np_pearson_test,
                                  _naive_pearson_test)

import unittest

# class TestPearson(unittest.TestCase):

#     def test_basic(self):
#         mat = np.matrix(np.array(range(10),dtype=np.float32))
#         mat = np.matrix(np.random.random(10),dtype=np.float32)
#         d = 40
#         mat = np.matrix(np.vstack((
#             np.array(range(1,d+1)),
#             np.array(range(2,d+2)),
#             np.array(range(3,d+3)),
#             np.array(range(4,d+4)),
#             np.array(range(5,d+5)),
#             np.random.random(d))),dtype=np.float32)

#         x = np.array(range(d),dtype=np.float32)
#         np_r, np_p = _np_pearson_test(mat,x)

#         mat = np.array(mat)

#         nv_r, nv_p = _naive_pearson_test(mat,x)
#         nv_r = np.matrix(nv_r).transpose()
#         nv_p = np.matrix(nv_p).transpose()
#         self.assertEquals(sum(abs(np_r-nv_r) > 0.1), 0)
#         self.assertEquals(sum(abs(np_p-nv_p) > 0.1), 0)

#     def test_zeros(self):
#         mat = np.matrix(np.array(range(10),dtype=np.float32))
#         mat = np.matrix(np.random.random(10),dtype=np.float32)
#         d = 40
#         l = 20

#         np.set_printoptions(precision=3)
#         mat = np.matrix(np.vstack((
#             np.array(range(1,11) + [0]*20 + range(11,21)),
#             np.array(range(1,21) + [0]*10 + range(21,31)),
#             np.array(range(1,11) + [0]*10 + range(11,21) + [0]*10),
#             np.array(range(d)),
#             np.random.random(d))),dtype=np.float32)
#         mat = mat.transpose()
#         mat = np.matrix(np.array(range(1,11) + [0]*20 + range(11,21)),dtype=np.float32)

#         y = np.array(range(d),dtype=np.float32)
#         x = np.array(range(1,21) + [0]*10 + range(21,31), dtype=np.float32)

#         nv_r, nv_p = _naive_pearson_test(mat,x)
#         zv_r, zv_p = _naive_nonzero_pearson_test(mat,x)

if __name__=='__main__':
    #unittest.main()
    pass
