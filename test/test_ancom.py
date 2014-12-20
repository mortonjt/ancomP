import numpy as np
from numpy import random, array
from pandas import DataFrame, Series



import unittest
class TestANCOM(unittest.TestCase):
    def setUp(self):
        L = 1000
        D = 500
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

    def test_ancomR(self):
        otu_table = DataFrame(self.data)
        otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
        sig_otus = ancom_R(otu_table,0.05,2,False)
        self.assertItemsEqual(sig_otus['OTU Significant at FDR = 0.05'],
                              ['OTU7', 'OTU5', 'OTU4', 'OTU2', 'OTU1'])
    def test_ancomCL(self):
        otu_table = DataFrame(self.data)
        otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
        cats = array([0]*D + [1]*D)
        sig_otus = ancom_cl(otu_table,cats,0.05,10000)
        self.assertItemsEqual(sig_otus,
                              ['OTU7', 'OTU5', 'OTU4', 'OTU2', 'OTU1', 'GRP'])
    
    def test_speed(self):
        import time
        otu_table = DataFrame(self.bigdata)
        otu_table = otu_table.reindex_axis(sorted(otu_table.columns,reverse=True), axis=1)
        counts = 10
        t1 = time.time()
        for _ in range(counts):
            sig_otus = ancom_R(otu_table,0.05,1,True)
        approx_time = (time.time()-t1)/counts
        t1 = time.time()
        for _ in range(counts):
            sig_otus = ancom_R(otu_table,0.05,1,False)
        for _ in range(counts):
            exact_time = (time.time()-t1)/counts
        t1 = time.time()
        for _ in range(counts):
            sig_otus = ancom_cl(otu_table,0.05,1,False)
        gpu_time = (time.time()-t1)/counts

        print("Approx U-test [s]",approx_time)
        print("Exact U-test [s]",exact_time)
        print("Exact GPU [s]",gpu_time)
        self.assertGreaterThan(approx_time,gpu_time)
        self.assertGreaterThan(exact_time,gpu_time)

unittest.main()
