import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from stats.ancom import (ancom_cl,
                         _log_compare,
                         _stationary_log_compare,
                         Holm)

from stats.permutation import (_cl_mean_permutation_test,
                               _np_mean_permutation_test,
                               _naive_mean_permutation_test)

import unittest
class TestANCOM(unittest.TestCase):
    def setUp(self):
        self.samples = 80
        self.half_samples = self.samples/2
        D = self.half_samples
        L = self.samples
        self.data = {'OTU1': map( abs, map(int,
                                  np.concatenate((random.normal(10,1,D),
                                                  random.normal(20,1,D)))
                                                  )),
             'OTU2': map( abs, map(int,
                                   np.concatenate((random.normal(20,1,D),
                                                   random.normal(100000,1,D)))
                                   )),
             'OTU3': map( abs, map( int,random.normal(10,1,L))),
             'OTU4': map( abs, map(int,
                                   np.concatenate((random.normal(10,1,D),
                                                   random.normal(20,1,D)))
                                                   )),
             'OTU5': map( abs, map(int,
                                   np.concatenate((random.normal(20,1,D),
                                                   random.normal(100000,1,D)))
                                   )),
             'OTU6': map( abs, map( int,random.normal(10,1,L))),
             'OTU7': map( abs, map(int,
                                   np.concatenate((random.normal(10,1,D),
                                                   random.normal(20,1,D)))
                                                   )),
             'OTU8': map( abs, map( int,random.normal(10,1,L))),
             'OTU9': map( abs, map( int,random.normal(10,1,L))),
             'GRP': array([0]*D + [1]*D)
             }
        self.bigdata = dict({'OTU1': map( abs, map(int,
                                   np.concatenate((random.normal(10,1,D),
                                                   random.normal(20,1,D)))
                                                   )),
             'OTU2': map( abs, map(int,
                                   np.concatenate((random.normal(20,1,D),
                                                   random.normal(100000,1,D)))
                                   )),
             'OTU3': map( abs, map(int,random.normal(10,1,L))),
             'OTU4': map( abs, map(int,
                                   np.concatenate((random.normal(10,1,D),
                                                   random.normal(20,1,D)))
                                                   )),
             'OTU5': map( abs, map(int,
                                   np.concatenate((random.normal(20,1,D),
                                                   random.normal(100000,1,D)))
                                   )),
             'OTU6': map( abs, map( int,random.normal(10,1,L))),
             'OTU7': map( abs, map(int,
                                   np.concatenate((random.normal(10,1,D),
                                                   random.normal(20,1,D)))
                                                   )),
             'OTU8': map( abs, map( int,random.normal(10,1,L))),
             'OTU9': map( abs, map( int,random.normal(10,1,L)))}.items()+
             {'OTU%d'%i: map( abs, map( int,random.normal(10,1,L))) for i in range(0,40)}.items()+
             {'GRP': array([0]*D + [1]*D)}.items()
             )
        
    def test_holm(self):
        p = [0.005, 0.011, 0.02, 0.04, 0.13]
        corrected_p = p * np.arange(1,6)[::-1]
        guessed_p = Holm(p)
        for a,b in zip(corrected_p,guessed_p):
            self.assertAlmostEqual(a,b)
        
    def test_ancomCL(self):
        print("Test ancom cl")
        D = self.half_samples
        otu_table = DataFrame(self.data)
        
        otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
        cats = array([0]*D + [1]*D)
        sig_otus = ancom_cl(otu_table,cats,0.05,1000)
        self.assertItemsEqual(sig_otus,
                              ['OTU7', 'OTU5', 'OTU4', 'OTU2', 'OTU1', 'GRP'])

    
    def test_speed(self):
        print("Test speed")
        import time
        D = self.half_samples
        L = self.samples

        otu_table = DataFrame(self.bigdata)

        mat = otu_table.as_matrix().transpose()
        mat = mat.astype(np.float32)
        cats = array([0]*D + [1]*D)
        counts = 3
        t1 = time.time()
        for _ in range(counts):
            sig_otus = _log_compare(mat, cats,
                                    stat_test=_np_mean_permutation_test,
                                    permutations=1000)
        np_time = (time.time()-t1)/counts
        t1 = time.time()
        for _ in range(counts):
            sig_otus = _log_compare(mat, cats,
                                    stat_test=_cl_mean_permutation_test,
                                    permutations=1000)
        cl_time = (time.time()-t1)/counts
        t1 = time.time()
        for _ in range(counts):
            sig_otus = _stationary_log_compare(mat,cats,permutations=1000,gpu=False)
        np_s_time = (time.time()-t1)/counts

        t1 = time.time()
        for _ in range(counts):
            sig_otus = _stationary_log_compare(mat,cats,permutations=1000,gpu=True)
        cl_s_time = (time.time()-t1)/counts

        print("Repeated numpy permutation time [s]",np_time)
        print("Repeated opencl permutation time [s]",cl_time)
        print("Stationary numpy permutation time [s]",np_s_time)
        print("Stationary opencl permutation time [s]",cl_s_time)

unittest.main()
