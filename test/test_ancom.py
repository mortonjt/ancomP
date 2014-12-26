import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from stats.ancom import (_log_compare,
                         ancom_cl,
                         _init_device,
                         _mean_stat,
                         Holm)



import unittest
class TestANCOM(unittest.TestCase):
    def setUp(self):
        self.samples = 20
        self.half_samples = 10
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
             'OTU9': map( abs, map( int,random.normal(10,1,L)))}.items()+
             {'OTU%d'%i: map( abs, map( int,random.normal(10,1,L))) for i in range(0,80)}.items()+
             {'GRP': array([0]*D + [1]*D)}.items()
             )
    def test_holm(self):
        p = [0.005, 0.011, 0.02, 0.04, 0.13]
        corrected_p = p * np.arange(1,6)[::-1]
        guessed_p = Holm(p)
        for a,b in zip(corrected_p,guessed_p):
            self.assertAlmostEqual(a,b)
        
    def test_ancomCL(self):
        D = self.half_samples
        otu_table = DataFrame(self.data)
        otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
        cats = array([0]*D + [1]*D)
        sig_otus = ancom_cl(otu_table,cats,0.05,10000)
        self.assertItemsEqual(sig_otus,
                              ['OTU7', 'OTU5', 'OTU4', 'OTU2', 'OTU1', 'GRP'])

    def test_init_device(self):
        
    # def test_speed(self):
    #     import time
    #     otu_table = DataFrame(self.bigdata)
    #     otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
    #     counts = 3
    #     t1 = time.time()
    #     for _ in range(counts):
    #         sig_otus = ancom_R(otu_table,0.05,1,True)
    #     approx_time = (time.time()-t1)/counts
    #     t1 = time.time()
    #     for _ in range(counts):
    #         sig_otus = ancom_R(otu_table,0.05,1,False)
    #     for _ in range(counts):
    #         exact_time = (time.time()-t1)/counts
    #     t1 = time.time()
    #     for _ in range(counts):
    #         sig_otus = ancom_cl(otu_table,0.05,1,False)
    #     gpu_time = (time.time()-t1)/counts

    #     print("Approx U-test [s]",approx_time)
    #     print("Exact U-test [s]",exact_time)
    #     print("Exact GPU [s]",gpu_time)
    #     self.assertGreater(approx_time,gpu_time)
    #     self.assertGreater(exact_time,gpu_time)

unittest.main()
