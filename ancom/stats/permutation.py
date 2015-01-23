import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy
from scipy.stats import ttest_ind
import itertools

#from statsmodels.sandbox.stats.multicomp import multipletests 

def _init_perms(vec, permutations=1000):
    """
    Creates a permutation matrix
    
    vec: numpy.array
       Array of values to be permuted
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    c = len(vec)
    copy_vec = copy.deepcopy(vec)
    perms = np.array(np.zeros((c, permutations+1), dtype=vec.dtype))
    _samp_ones = np.array(np.ones(c), dtype=vec.dtype).transpose()
    for m in range(permutations+1):
        perms[:,m] = copy_vec
        np.random.shuffle(copy_vec)
    return perms

def _init_categorical_perms(cats, permutations=1000):
    """
    Creates a reciprocal permutation matrix
    
    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    c = len(cats)
    num_cats = len(np.unique(cats)) # Number of distinct categories
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)), dtype=cats.dtype))
    for m in range(permutations+1):
        for i in range(num_cats):
            perms[:,num_cats*m+i] = (copy_cats == i).astype(cats.dtype)
        np.random.shuffle(copy_cats)
    return perms


def _init_reciprocal_perms(cats, permutations=1000):
    """
    TODO: Make this function use _init_categorical_perms
    
    Creates a reciprocal permutation matrix
    
    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    num_cats = 2 #number of distinct categories
    c = len(cats)
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)), dtype=cats.dtype))
    _samp_ones = np.array(np.ones(c), dtype=cats.dtype).transpose()
    for m in range(permutations+1):
        #Perform division to make mean calculation easier
        perms[:,2*m] = copy_cats / float(copy_cats.sum())
        perms[:,2*m+1] = (_samp_ones - copy_cats) / float((_samp_ones - copy_cats).sum())
        np.random.shuffle(copy_cats)
    return perms

def _to_device(mat, perms):
    assert type(mat)==np.ndarray
    assert type(perms)==np.ndarray
    perms = perms.astype(mat.dtype)
    d_perms = pv.Matrix(perms)
    d_mat = pv.Matrix(mat)
    return d_mat, d_perms

def _init_device(mat, cats, permutations=1000):
    """
    TODO: This is deprecated in favor of _to_device()
    Creates a permutation matrix and loads it on device
    
    mat: numpy.ndarray
       Feature matrix
    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test
    """
    assert type(mat)==np.ndarray
    assert type(cats)==np.ndarray

    print "This is deprecated: Use _to_device() instead"
    perms = _init_reciprocal_perms(cats,permutations)
    perms = perms.astype(mat.dtype)
    d_perms = pv.Matrix(perms)
    d_mat = pv.Matrix(mat)
    return d_mat, d_perms

############################################################
## Mean permutation tests
############################################################

def _naive_mean_permutation_test(mat,cats,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now
    
    Returns
    =======
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

def _np_mean_permutation_test(mat, cats, permutations=1000):
    """
    mat: numpy.ndarray or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on
    permutations: int
         Number of permutations to calculate
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
    perms = _init_reciprocal_perms(cats, permutations)
    _mat = np.matrix(mat)
    _perms = np.matrix(perms)
    return _np_two_sample_mean_statistic(_mat, _perms)

def _np_two_sample_mean_statistic(mat, perms):
    """
    Calculates a permutative mean statistic just looking at binary classes

    mat: numpy.ndarray or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    perms: numpy.ndarray
         columns: permutations of samples
         rows: features    
         Permutative matrix

    Note: only works on binary classes now
    
    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    
    ## Create a permutation matrix
    num_cats = 2 #number of distinct categories
    n_otus, c = perms.shape
    permutations = (c-num_cats) / num_cats
        
    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    avgs = mat * perms
    ## Calculate the mean statistic
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    mean_stat = abs(avgs[:, idx+1] - avgs[:, idx])

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
        
    #_,pvalues,_,_ = multipletests(pvalues)
    return map(np.array,[mean_stat[:,0],pvalues])

#@profile
def _cl_two_sample_mean_statistic(d_mat, d_perms):
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
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    mean_stat = abs(avgs[:,idx+1] - avgs[:,idx])    

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = ( cmps.sum(axis=1) + 1.) / (permutations+1. )
    test_mean_stats = mean_stat[:,0]
    return map(np.array, [test_mean_stats,pvalues] )


def _cl_mean_permutation_test(mat, cats, permutations=1000):
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
    return _cl_two_sample_mean_statistic(d_mat, d_perms)


    
############################################################
## T-test permutation tests
############################################################
def _naive_t_permutation_test(mat,cats,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now
    
    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values
    
    This module will conduct a mean permutation test using
    the naive approach
    """
    def _t_test(values,cats):
        #calculates t statistic for binary categories
        T, _ =  ttest_ind(values[cats==0], values[cats==1], equal_var = False)
        return abs(T)
    
    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _t_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _t_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues

def _np_t_permutation_test(mat, cats, permutations=1000):
    """
    mat: numpy.ndarray or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on
    permutations: int
         Number of permutations to calculate
    Note: only works on binary classes now
    
    Return
    ------
    test_stats:
        List of t-test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    perms = _init_categorical_perms(cats, permutations)
    _mat = np.matrix(mat)
    _perms = np.matrix(perms)
    return _np_two_sample_t_statistic(_mat, _perms)

def _np_two_sample_t_statistic(mat, perms):
    """
    Calculates a permutative Welch's t-statistic

    mat: numpy.matrix or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    perms: numpy.matrix
         columns: permutations of samples
         rows: features    
         Permutative matrix

    Note: only works on binary classes now
    
    Returns
    =======
    test_stats:
        List of t-test statistics
    pvalues:
        List of p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    assert type(mat) == np.matrix
    assert type(perms) == np.matrix
    
    ## Create a permutation matrix
    num_cats = 2 # number of distinct categories
    n_otus, c = perms.shape
    permutations = (c-num_cats) / num_cats
    
    ## Perform matrix multiplication on data matrix
    ## and calculate sums and squared sums
    _sums  = mat * perms
    _sums2 = np.multiply(mat,mat) * perms

    ## Calculate means and sample variances
    tot =  perms.sum(axis=0)
    _avgs  = _sums / tot
    _avgs2 = _sums2 / tot
    _vars  = _avgs2 - np.multiply(_avgs, _avgs)
    _samp_vars =  np.multiply(tot,_vars) / (tot-1)
    
    ## Calculate the t statistic
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    denom  = np.sqrt(_samp_vars[:, idx+1] / tot[:,idx+1]  + _samp_vars[:, idx] / tot[:,idx])
    t_stat = np.divide(abs(_avgs[:, idx+1] - _avgs[:, idx]), denom)
    
    ## Calculate the p-values
    cmps =  t_stat[:,1:] >= t_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
        
    return map(np.array, map(np.ravel, [t_stat[:,0],pvalues]))


def _cl_two_sample_t_statistic(d_mat, d_perms):
    """
    TODO: Wait until element_prod bug is fixed
    
    Calculates a permutative Welch's t-statistic

    d_mat: pyviennacl.pycore.Matrix
        Device based feature matrix
    d_perm: pyviennacl.pycore.Matrix
        Device based permutation matrix

    Returns
    =======
    test_mean_stats: numpy.array
        Mean statistics for each feature
    pvalues: numpy.array
        Type I error calculations for each mean statistics

    This module will conduct a mean permutation test using
    pyviennacl matrix algebra
    """
    assert type(mat) == pv.Matrix
    assert type(perms) == pv.Matrix
    
    ## Create a permutation matrix
    num_cats = 2 # number of distinct categories
    n_samp, c = d_perms.shape
    permutations = (c-num_cats) / num_cats
    n_otus, _ = d_mat.shape
    ## Perform matrix multiplication on data matrix
    ## and calculate sums and squared sums

    d_mat2 = d_mat.element_prod(d_mat).result
    _sums  = (d_mat * d_perms).result
    _sums2 = (d_mat2 * d_perms).result

    ## Calculate means and sample variances
    _samp_ones = pv.Vector(np.ones(n_samp,dtype=d_mat.dtype))
    _otu_ones = pv.Vector(np.ones(n_otus,dtype=d_mat.dtype))
    
    tot =  (d_perms.T * _samp_ones).result
    
    _tot = pv.Matrix(_sums.shape, dtype=_sums.dtype, layout=_sums.layout)
    pv.Assign(_tot, tot.outer( _otu_ones )).execute()
    
    _avgs  = _sums.element_div( _tot )
    _avgs2 = _sums2.element_div( _tot )
    r, c = _avgs2.shape
    _vars  = _avgs2 - _avgs.element_prod(_avgs)[:r, :c]
    _samp_vars =  pv.ElementProd(_tot, _vars).element_div(tot-np.float32(1))
    
    ## Calculate the t statistic
    idx = np.arange(0, (permutations+1) * num_cats, num_cats)
    denom  = np.sqrt(_samp_vars[:, idx+1] / tot[:,idx+1]  + _samp_vars[:, idx] / tot[:,idx])
    t_stat = np.divide(abs(_avgs[:, idx+1] - _avgs[:, idx]), denom)
    
    ## Calculate the p-values
    cmps =  t_stat[:,1:] >= t_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
        
    return map(np.array, map(np.ravel, [t_stat[:,0],pvalues]))


############################################################
## F-test permutation tests
############################################################

"""
F = sum( MS_i for all i) /  MSE
"""
def _naive_f_permutation_test(mat,cats,permutations=1000):
    """
    Performs a 1-way ANOVA
    
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    
    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values
    
    This module will conduct a F permutation test using
    the naive approach
    """
    
    def _f_test(values,cats):
        #calculates t statistic for binary categories
        groups = []
        groups = [ values[cats==k] for k in set(cats) ]
        F, _ =  f_oneway(*groups)
        return abs(F)
    
    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _f_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _f_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    return test_stats, pvalues
    
def _np_k_sample_f_statistic(mat, perms):
    """
    Calculates a permutative one way F test

    mat: numpy.matrix or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    perms: numpy.matrix
         columns: permutations of samples
         rows: features    
         Permutative matrix

    Note: only works on binary classes now
    
    Returns
    =======
    test_stats:
        List of t-test statistics
    pvalues:
        List of p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    assert type(mat) == np.matrix
    assert type(perms) == np.matrix
    
    ## Create a permutation matrix
    num_cats = len(np.unique(cats)) # Number of distinct categories
    n_samp, c = perms.shape
    permutations = (c-num_cats) / num_cats
        
    mat2 = np.multiply(mat, mat)

    S = mat.sum(axis=1)
    SS = mat2.sum(axis=1)
    sstot = SS - np.multiply(S,S) / float(n_samp)
        
    #Create index to sum the ssE together
    _sum_idx = _init_categorical_perms(
        np.arange((permutations+1)*num_cats,dtype=np.int32)/num_cats,
        permutations=0)

    
    ## Perform matrix multiplication on data matrix
    ## and calculate sums and squared sums and sum of squares
    _sums  = np.dot(mat, perms)
    _sums2 = np.dot(np.multiply(mat,mat), perms)
    
    tot =  perms.sum(axis=0)
    ss = _sums2 - np.multiply(_sums,_sums)/tot
    sserr = np.dot(ss, _sum_idx)
    sstrt = sstot - sserr
    dftrt = num_cats-1
    dferr = np.dot(tot,_sum_idx) - num_cats
    
    f_stat = (sstrt / dftrt) / (sserr / dferr)

    cmps =  f_stat[:,1:] >= f_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.) / (permutations+1.)        
    return map(np.array, map(np.ravel, [f_stat[:, 0], pvalues]))
    
