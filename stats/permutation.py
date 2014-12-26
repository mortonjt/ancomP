import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy
#from statsmodels.sandbox.stats.multicomp import multipletests 

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

def _np_mean_permutation_test(mat,cats,permutations=1000):
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
    numpy matrix algebra
    """
    
    ## Create a permutation matrix
    num_cats = 2 #number of distinct categories
    r,c = mat.shape
    copy_cats = copy.deepcopy(cats)
    copy_cats = np.matrix(copy_cats).transpose() #convert to matrix
    perms = np.matrix(np.zeros((c,num_cats*(permutations+1))))
    _ones = np.matrix(np.ones(c)).transpose()
    for m in range(permutations+1):
        perms[:,2*m] = copy_cats 
        perms[:,2*m+1] = _ones - copy_cats 
        np.random.shuffle(copy_cats )

    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    sums = mat * perms
    avgs = sums / perms.sum(axis=0)
    ## Calculate the mean statistic
    idx = np.array([i for i in range(0,(permutations+1)*num_cats,2)])
    mean_stat = abs(avgs[:,idx+1] - avgs[:,idx])

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
        
    #_,pvalues,_,_ = multipletests(pvalues)
    return map(np.array,[mean_stat[:,0],pvalues])

def _two_sample_mean_stat(d_mat, d_perms, d_reps, permutations=1000):
    
    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    num_cats = 2
    n_otus, permuations = d_perms.shape
    d_sums = d_mat * d_perms
    avgs = np.matrix(((d_sums.T) * d_reps).value)
    #sums = np.matrix(d_sums.value)
    #avgs = sums / (np.matrix(perms).sum(axis=0))
    ## Calculate the mean statistic
    idx = np.array([i for i in range(0,(permutations+1)*num_cats, num_cats)])
    mean_stat = abs(avgs[:,idx+1] - avgs[:,idx])

    ## Calculate the p-values
    cmps =  mean_stat[:,1:] >= mean_stat[:,0]
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)
    return map(np.array,[mean_stat[:,0],pvalues])

def _init_device(mat,cats,permutations=1000):
    num_cats = 2 #number of distinct categories
    r,c = mat.shape
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c,num_cats*(permutations+1)),dtype=cats.dtype))
    _ones = np.array(np.ones(c),dtype=mat.dtype).transpose()
    for m in range(permutations+1):
        perms[:,2*m] = copy_cats
        perms[:,2*m+1] = _ones - copy_cats
        np.random.shuffle(copy_cats)
    d_perms = pv.Matrix(perms)
    d_mat = pv.Matrix(mat)
    d_reps = pv.Vector(_ones * (1./r))
    return d_mat, d_perms, d_reps

def _cl_mean_permutation_test(mat,cats,permutations=1000,num_cats=2):
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
    num_cats = 2 #number of distinct categories
    r,c = mat.shape
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c,num_cats*(permutations+1)),dtype=cats.dtype))
    _ones = np.array(np.ones(c),dtype=mat.dtype).transpose()
    for m in range(permutations+1):
        perms[:,2*m] = copy_cats
        perms[:,2*m+1] = _ones - copy_cats
        np.random.shuffle(copy_cats)

    #Now start copying stuff over to GPU
    d_perms = pv.Matrix(perms)
    d_mat = pv.Matrix(mat)
    d_reps = pv.Vector(_ones)/r
        
    #_,pvalues,_,_ = multipletests(pvalues)
    #return map(np.array,[mean_stat[:,0],pvalues])
    return _mean_stat(d_mat, d_perms, d_reps, num_cats, permutations)


    
