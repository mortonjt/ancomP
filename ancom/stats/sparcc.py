import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy
import math


def sparcc_statistic(mat x, iters = 100):
    """
    mat: numpy.ndarray
    x: numpy.ndarray
    """
    _M = copy.deepcopy(mat)
    _x = copy.deepcopy(x)
    vM = mat.var(axis=0)
    vx = np.var(x)
    _,D = vM.shape
    d = D - 1
    t_ix = d*vx + vM.sum() - vx
    _p = (vx + vM - t_ix) / (2*math.sqrt(vx*vM))
    for _ in iters:
        i = _p.argmax()
        _x[i] = _x[ [:i]+[i+1:] ]
        _M[i,:] = _M[ [:i]+[i+1:] , :]
        
        vM = _M.var(axis=0)
        vx = np.var(_x)
        _,D = vM.shape
        d = D - 1
        t_ix = d*vx + vM.sum() - vx
        _p = (vx + vM - t_ix) / (2*math.sqrt(vx*vM))
        
    return _p

def _naive_sparcc_test(mat, x, permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    x:   numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of sparcc coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    the naive approach

    Reference
    [1] http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002687
    """
    n_feats, cols = mat.shape
    pvalues = np.zeros(n_feats)
    test_stats = np.zeros(n_feats)
    test_stat,_ = sparcc(mat,x) 
    perm_stats = np.empty(permutations, dtype=np.float64)
    for i in range(permutations):
        perm_cats = np.random.permutation(x)
        test_stat,_ = pearsonr(values,x) 
    p_value = ((perm_stats > test_stat).sum() + 1) / (permutations + 1)
    pvalues[r] = p_value
    test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues
