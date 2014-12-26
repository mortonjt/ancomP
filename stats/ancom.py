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
from permutation import (_cl_mean_permutation_test,
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
 

def _opt_log_compare(mat,cats,permutations=1000):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest

    This is an optimized version to minimize data transfer
    between the CPU and GPU
    
    otu_table: numpy 2d matrix
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array
    Returns:
    --------
    log ratio pvalue matrix
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
        
    ##log ratio calculations
    log_mat = np.log(mat+(1./r))
    log_ratio = np.zeros((r,r),dtype=np.float32)
    
    d_perms = pv.Matrix(perms)
    d_log_mat = pv.Matrix(log_mat)
    for i in range(r-1):
        d_mat =  d_log_mat[i+1:,:] - d_log_mat[i,:]
        d_sums = d_mat * d_perms
        
        log_ratio[i,i+1:] = np.matrix(p).transpose()
    return log_ratio

def _log_compare(mat,cats,permutations=1000):
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
    Returns:
    --------
    log ratio pvalue matrix
    """    
    r,c = mat.shape
    log_mat = np.log(mat+(1./r))
    log_ratio = np.zeros((r,r),dtype=np.float32)
    for i in range(r-1):
        ratio =  np.array(np.matrix(log_mat[i+1:,:]) - np.matrix(log_mat[i,:]))
        m, p = _cl_mean_permutation_test(ratio,cats,permutations)
        log_ratio[i,i+1:] = np.matrix(p).transpose()
    return log_ratio

def ancom_cl(otu_table,cats,alpha,permutations=1000):
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
    
    _logratio_mat = _log_compare(mat,cats,permutations)
    logratio_mat = _logratio_mat + _logratio_mat.transpose()
    n_otu,n_samp = mat.shape
    ##Multiple comparisons
    for i in range(n_otu):
         pvalues = Holm(logratio_mat[i,:])
         logratio_mat[i,:] = pvalues
         print("OTU:",i)
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

