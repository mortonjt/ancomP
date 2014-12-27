import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy
#from statsmodels.sandbox.stats.multicomp import multipletests 

def _init_perms(cats, permutations=1000):
    """
    Creates a permutation matrix
    
    cats: numpy.array
       List of binary class assignments
    """
    num_cats = 2 #number of distinct categories
    c = len(cats)
    copy_cats = copy.deepcopy(cats)
    copy_cats = np.matrix(copy_cats).transpose() #convert to matrix
    perms = np.matrix(np.zeros((c,num_cats*(permutations+1))))
    _ones = np.matrix(np.ones(c)).transpose()
    for m in range(permutations+1):
        perms[:,2*m] = copy_cats 
        perms[:,2*m+1] = _ones - copy_cats 
        np.random.shuffle(copy_cats)
    return perms

def _init_device(mat, cats, permutations=1000):
    """
    Creates a permutation matrix and loads it on device
    
    mat: numpy.ndarray
       Feature matrix
    cats: numpy.array
       List of binary class assignments
    """
    num_cats = 2 #number of distinct categories
    r,c = mat.shape
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c,num_cats*(permutations+1)),dtype=mat.dtype))
    _samp_ones = np.array(np.ones(c),dtype=mat.dtype).transpose()
    _feat_ones = np.array(np.ones(r),dtype=mat.dtype).transpose()
    for m in range(permutations+1):
        perms[:,2*m] = copy_cats / float(copy_cats.sum())
        perms[:,2*m+1] = (_samp_ones - copy_cats) / float((_samp_ones - copy_cats).sum())
        np.random.shuffle(copy_cats)
    
    d_perms = pv.Matrix(perms)
    d_mat = pv.Matrix(mat)
    return d_mat, d_perms


def _naive_mean_permutation_test(mat,cats,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now
    
    Return
    ------
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values
    
    This module will conduct a mean permutation test using
    the naive approach
    """
    def _mean_test(values,cats):
        #calculates mean for binary categories
        return abs(values[cats==0].mean()-values[cats==1].mean())
    
    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _mean_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _mean_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues


def _np_mean_permutation_test(mat, cats, permutations=1000, perms = None):
    """
    mat: numpy.ndarray or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now
    
    Return
    ------
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    
    ## Create a permutation matrix
    num_cats = 2 #number of distinct categories
    r,c = mat.shape
    if perms == None:
        perms = _init_perms(cats, permutations)
        
    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    sums = mat * perms
    avgs = sums / perms.sum(axis=0)
    ## Calculate the mean statistic
    idx = np.array([i for i in range(0, (permutations+1)*num_cats,2)])
    mean_stat = abs(avgs[:, idx+1] - avgs[:, idx])

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
        
    #_,pvalues,_,_ = multipletests(pvalues)
    return map(np.array,[mean_stat[:,0],pvalues])

#@profile
def _two_sample_mean_statistic(d_mat, d_perms):
    """
    Calculates a permutative mean statistic just looking at binary classes
    
    d_mat: pv.pycore.Matrix
        Device based feature matrix
    d_perm: pv.pycore.Matrix
        Device based permutation matrix

    Returns
    =======
    test_mean_stats: numpy.array
        Mean statistics for each feature
    pvalues: numpy.array
        Type I error calculations for each mean statistics
    """
    
    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    num_cats = 2
    n_otus, c = d_perms.shape
    permutations = (c-num_cats) / num_cats
    d_avgs = d_mat * d_perms

    ## Transfer results back to host
    avgs = np.matrix(d_avgs.value)

    ## Calculate the mean statistic
    idx = np.array( [i for i in xrange(0, (permutations+1) * num_cats, num_cats)] )
    mean_stat = abs(avgs[:,idx+1] - avgs[:,idx])    

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = ( cmps.sum(axis=1) + 1.) / (permutations+1. )
    test_mean_stats = mean_stat[:,0]
    return map(np.array, [test_mean_stats,pvalues] )


def _cl_mean_permutation_test(mat, cats, permutations=1000, num_cats=2):
    """
    mat: numpy 2-d array,  numpy.float32
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now
    
    Return
    ------
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    opencl matrix multiplication
    """
    d_mat, d_perms = _init_device(mat, cats, permutations)
    return _two_sample_mean_statistic(d_mat, d_perms)


    
