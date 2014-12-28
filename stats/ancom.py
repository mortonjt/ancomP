import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy

import os, sys, site
import pandas.rpy.common as com
import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from math import log
from stats.permutation import (_init_device,
                               _init_perms,
                               _np_two_sample_mean_statistic,
                               _cl_two_sample_mean_statistic,
                               _cl_mean_permutation_test,
                               _np_mean_permutation_test,
                               _naive_mean_permutation_test)

def Holm(p):
    """
    Performs Holm-Boniferroni correction for pvalues
    to account for multiple comparisons
    
    p: numpy.array

    Returns
    =======
    numpy.array

    corrected pvalues
    """
    K = len(p)
    sort_index = -np.ones(K,dtype=np.int64)
    sorted_p = np.sort(p)
    sorted_p_adj = sorted_p*(K-np.arange(K))
    for j in range(K):
        num_ties = len(sort_index[(p==sorted_p[j]) & (sort_index<0)])
        sort_index[(p==sorted_p[j]) & (sort_index<0)] = np.arange(j,(j+num_ties),dtype=np.int64)

    sorted_holm_p = [min([max(sorted_p_adj[:k]),1]) for k in range(1,K+1)]
    holm_p = [sorted_holm_p[sort_index[k]] for k in range(K)]
    return holm_p
 

def _stationary_log_compare(mat,cats,permutations=1000,gpu=False):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest

    This is an optimized version to minimize data transfer
    between the CPU and GPU.  Assumes a stationary set of permutations
    
    otu_table: numpy 2d matrix
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array
    Returns:
    --------
    log ratio: numpy.ndarray
        pvalue matrix
    """
    r,c = mat.shape
    log_mat = np.log(mat+(1./r))
    log_ratio = np.zeros((r,r),dtype=mat.dtype)
    if gpu:
        log_mat, perms = _init_device(log_mat, cats, permutations)
        _ones = pv.Matrix(np.ones((r-1,2),dtype=mat.dtype))[:,1] #hacky way to make 1-D matrices
    else:
        perms = _init_perms(cats, permutations)
        perms = perms.astype(mat.dtype)
        _ones = np.matrix(np.ones(r-1,dtype=mat.dtype)).transpose()
    
    for i in range(r-1):        
        ## Perform outer product to create copies of log_mat[i,:]
        ## similar to np.tile
        outer = _ones[i:] * log_mat[i,:]
        ratio =  log_mat[i+1:,:] - outer
        if gpu:
            m, p  = _cl_two_sample_mean_statistic(ratio.result, perms)
        else:
            m, p  = _np_two_sample_mean_statistic(ratio, perms)
            
        log_ratio[i,i+1:] = np.matrix(p).transpose()
        print "OTU: ", i
    return log_ratio

def _log_compare(mat, cats, stat_test=_np_mean_permutation_test, permutations=1000):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest
    
    otu_table: numpy 2d matrix
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array

    stat_stat: function
    statistical test to run
    
    Returns:
    --------
    log ratio pvalue matrix
    """    
    r,c = mat.shape
    log_mat = np.log(mat+(1./r))
    log_ratio = np.zeros((r,r),dtype=np.float32)
    for i in range(r-1):
        ratio =  np.array(np.matrix(log_mat[i+1:,:]) - np.matrix(log_mat[i,:]))
        m, p = stat_test(ratio,cats,permutations)
        log_ratio[i,i+1:] = np.squeeze(np.array(np.matrix(p).transpose()))
    return log_ratio

def ancom_cl(otu_table,cats,alpha,permutations=1000,multicorr = False,gpu=False):
    """
    Calculates pairwise log ratios between all otus
    and performs permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest
    
    otu_table: pandas.core.DataFrame
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array

    permutations: int
    Number of permutations to use in permutation test
    
    Returns:
    --------
    log ratio pvalue matrix
    """

    mat = otu_table.as_matrix().transpose()
    mat = mat.astype(np.float32)
    cats = cats.astype(np.float32)
    
    _logratio_mat = _stationary_log_compare(mat,cats,permutations,gpu)
    # _logratio_mat = _log_compare(mat, cats,
    #                              stat_test = _np_mean_permutation_test,
    #                              permutations = permutations)
    logratio_mat = _logratio_mat + _logratio_mat.transpose()
    np.savetxt("log_ratio.gz",logratio_mat)
    n_otu,n_samp = mat.shape
    ##Multiple comparisons
    if multicorr:
        for i in range(n_otu):
            pvalues = Holm(logratio_mat[i,:])
            logratio_mat[i,:] = pvalues

    W = np.zeros(n_otu)
    for i in range(n_otu):
        W[i] = sum(logratio_mat[i,:] < alpha)
    par = n_otu-1 #cutoff

    c_start = max(W)/par
    cutoff = c_start - np.linspace(0.05,0.25,5)
    D = 0.02 # Some arbituary constant
    dels = np.zeros(len(cutoff))
    prop_cut = np.zeros(len(cutoff),dtype=np.float32)
    for cut in range(len(cutoff)):
        prop_cut[cut] = sum(W > par*cutoff[cut])/float(len(W))
    for i in range(len(cutoff)-1):
        dels[i] = abs(prop_cut[i]-prop_cut[i+1])
        
    if (dels[1]<D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[1]
    elif (dels[1]>=D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[2]
    elif (dels[2]>=D) and (dels[3]<D) and (dels[4]<D):
        nu=cutoff[3]
    else:
        nu=cutoff[4]
    up_point = min(W[W>nu*par])
    results = otu_table.columns[W>=nu*par]
    return results

