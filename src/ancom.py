import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy

import os, sys, site
import rpy2.robjects as robj
import pandas.rpy.common as com
import numpy as np
from numpy import random, array
from pandas import DataFrame, Series

from scipy.stats import mannwhitneyu
from math import log
from statsmodels.sandbox.stats.multicomp import multipletests 
from permutation import (_cl_mean_permutation_test,
                         _np_mean_permutation_test,
                         _naive_mean_permutation_test)

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
    log_ratio = np.zeros((r,r))
    for i in range(r-1):
        ratio =  np.array(np.matrix(log_mat[i+1:,:]) - np.matrix(log_mat[i,:]))
        m, p = _np_mean_permutation_test(ratio,cats,permutations)
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
    _logratio_mat = _log_compare(mat,cats,permutations)
    logratio_mat = _logratio_mat + _logratio_mat.transpose()
    n_otu,n_samp = mat.shape
    ##Multiple comparisons
    for i in range(n_otu):
         _,pvalues,_,_ = multipletests(logratio_mat[i,:])
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

def ancom_R(otu_table,sig,multcorr,wilcox):
    """
    otu_table: pandas.DataFrame
        rows = samples
        cols = otus
        A table of OTU abundances
        Last column will be group ids (e.g. Urban/Rural)
    sig: float
        significance value
    multcorr: int
        multiple corrections (e.g. 1, 2, 3)
        1: Very strict
        2: Not as strict
        3: No multiple hypotheses correction (very bad idea)
    wilcox: bool
       perform an exact wilcox test or not

    Returns:
    --------
    Names of signficantly correlated OTUs
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    rcode = open("%s/ANCOM_without_covariates.r"%base_path).read()
    otu_Rtable = com.convert_to_r_dataframe(otu_table)
    Rfunc = robj.r(rcode)
    sig_Rotus = Rfunc(otu_Rtable,sig,multcorr,wilcox)
    sig_otus = com.convert_robj(sig_Rotus)
    return sig_otus

if __name__=="__main__":
    ## 7 OTUs, 10 samples
    mat = np.array(
        np.matrix(np.vstack((
            np.array([0]*5+[1]*5),
            np.array([0]*10),
            np.array([0]*10),
            np.array([0]*10),
            np.array([0]*10),
            np.array([0]*10),
            np.random.random(10))),dtype=np.float32))
    
    cats = np.array([0]*5+[1]*5,dtype=np.float32)
    lr = _log_compare(mat,cats,permutations=10000)

    ## 7 OTUs, N samples
    N = 10
    mat = np.array(
        np.matrix(np.vstack((
            np.array([0]*(N/2)+[1]*(N/2)),
            np.array([0]*N),
            np.array([0]*N),
            np.array([0]*N),
            np.array([0]*N),
            np.array([0]*N),
            np.random.random(N))),dtype=np.float32))
    
    cats = np.array([0]*(N/2)+[1]*(N/2),dtype=np.float32)
    lr = _log_compare(mat,cats,permutations=1000)

