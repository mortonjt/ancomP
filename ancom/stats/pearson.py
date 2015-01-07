import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy
from scipy.stats.stats import pearsonr

def _naive_nonzero_pearson_test(mat,x,permutations=1000):
    """
    Calculates Pearson coefficient only using nonzero data
    between mat and x
    
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of pearson coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    the naive approach
    """
    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    _nnz_x = np.nonzero(x)
    for r in range(rows):
        values = np.squeeze(np.array(mat[r,:].transpose()))
        _nnz_values = np.nonzero(values)
        index = np.intersect1d(_nnz_x,_nnz_values)
        _x = x[index]
        _values = values[index]
        test_stat,_ = pearsonr(_values,_x)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(_x)
            test_stat,_ = pearsonr(_values,_x) 
        p_value = ((perm_stats > test_stat).sum() + 1) / (permutations + 1)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues


def _naive_pearson_test(mat,x,permutations=1000):
    """
    Calculate Pearson correlations between mat and x
    
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of pearson coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    the naive approach
    """
    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = np.squeeze(np.array(mat[r,:].transpose()))
        test_stat,_ = pearsonr(values,x) 
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(x)
            test_stat,_ = pearsonr(values,x) 
        p_value = ((perm_stats > test_stat).sum() + 1) / (permutations + 1)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues

def _np_pearson_test(mat,x,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    x: numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of pearson coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    numpy matrix algebra
    """
    
    r,c = mat.shape
    x= np.matrix(x).transpose()
    _x = copy.deepcopy(x)
    perms = np.matrix(np.zeros((c,permutations+1)))
    _ones = np.matrix(np.ones(c)).transpose()
    for m in range(permutations+1):
        perms[:,m] = _x
        np.random.shuffle(_x)
        
    ## Calculate the covariance
    avgX = x.mean()
    dX = perms-avgX
    avgM = mat.mean(axis=1)
    dM = mat - avgM
    cov = dM*dX / c
    avgX2 = np.matrix.diagonal(perms.transpose()*perms)/c
    avgM2 = np.matrix.diagonal(mat*mat.transpose())/c

    ##Calulate the variances
    stdX = np.sqrt(avgX2-avgX*avgX)
    stdM = np.sqrt(avgM2.transpose()-np.multiply(avgM,avgM))
    denom = stdM*stdX
    r_perms = cov / denom

    #Now calculate pvalues
    cmps =  abs(r_perms[:,1:]) >= abs(r_perms[:,0])
    pvalues = (cmps.sum(axis=1)+1)/(permutations+1)
    return map(np.array,[r_perms[:,0],pvalues])


def _np_nonzero_pearson_test(mat,x,permutations=1000):
    """
    Calculates Pearsons correlation coefficients
    treating all zeros as missing data
    
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    x: numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of pearson coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    numpy matrix algebra
    """
    
    r,c = mat.shape
    _x = copy.deepcopy(np.matrix(x).transpose())
    
    ## Create permutations
    perms = np.matrix(np.zeros((c,permutations+1)),dtype=mat.dtype)
    _ones = np.matrix(np.ones(c),dtype=mat.dtype)
    _onesT = np.matrix(np.ones(c)).transpose()
    
    for m in range(permutations+1):
        perms[:,m] = _x
        np.random.shuffle(_x)

    ## Calculate non-zero matrices
    mat_nnz = (mat != 0).astype(mat.dtype)
    perms_nnz = (perms != 0).astype(mat.dtype)
    nnz = mat_nnz * perms_nnz ## Correct divisor for average
    
    ## Calculate mean matrices
    avgP = (mat_nnz * perms) / nnz
    avgM = (mat * perms_nnz) / nnz 

    ## Calculate covariance matrix
    dM = np.multiply(mat - avgM, mat_nnz)
    dM = np.multiply(dM, perms_nnz.transpose())
    #cov = ( dM * perms - np.multiply(dM, avgP) * _onesT ) / nnz

    ## Cannot just do permsT * perms.  Doesn't account for mat non zeros
    #avgX2 = np.matrix.diagonal(perms.transpose()*perms)/c
    #avgM2 = np.matrix.diagonal(mat*mat.transpose())/c
    avgP2 = mat_nnz * np.multiply(perms, perms) / nnz
    avgM2 = np.multiply(mat, mat) * perms_nnz / nnz
                
    ## Calulate the variances
    stdP = np.sqrt(avgP2 - np.multiply(avgP,avgP))
    stdM = np.sqrt(avgM2.transpose() - np.multiply(avgM, avgM))
    denom = np.multiply(stdM,stdP)
    r_perms = cov / denom
    
    ## Now calculate pvalues
    cmps =  abs(r_perms[:,1:]) >= abs(r_perms[:,0])
    pvalues = (cmps.sum(axis=1)+1)/(permutations+1)
    return map(np.array,[r_perms[:,0],pvalues])


def _cl_pearson_test(mat,x,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    x: numpy array
         A feature to run correlations against the features in mat
    
    Return
    ------
    test_stats:
        Array of pearson coefficients
    pvalues: numpy array 
        Array of corrected p-values
    
    This module will conduct a permutation test using
    numpy matrix algebra
    """
    r,c = mat.shape
    x= np.matrix(x).transpose()
    _x = copy.deepcopy(x)
    perms = np.matrix(np.zeros((c,permutations+1)))
    _ones = np.matrix(np.ones(c)).transpose()
    for m in range(permutations+1):
        perms[:,m] = _x
        np.random.shuffle(_x)
        
    g_perms = pv.Matrix(perms)
    g_mat = pv.Matrix(mat)
    g_ones = pv.Vector(_ones)
    g_x = pv.Vector(x)
    
    ## Calculate the covariance
    avgX = g_x.dot(g_ones * (1./c))
    dX = perms - avgX
    avgM = mat.mean(axis=1)
    dM = mat - avgM
    cov = dM*dX / c
    avgX2 = np.matrix.diagonal(perms.transpose()*perms)/c
    avgM2 = np.matrix.diagonal(mat*mat.transpose())/c

    ##Calulate the variances
    stdX = np.sqrt(avgX2-avgX*avgX)
    stdM = np.sqrt(avgM2.transpose()-np.multiply(avgM,avgM))
    denom = stdM*stdX
    r_perms = cov/denom

    #Now calculate pvalues
    cmps = abs(r_perms[:,0]) > abs(r_perms[:,1:])
    pvalues = (cmps.sum(axis=1)+1)/(permutations+1)
    return map(np.array,[r_perms[:,0],pvalues])

    
