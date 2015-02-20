"""
Makes use of Aitchson geometry to analyze/manipulate compositional data

Reference
=========
http://www.sediment.uni-goettingen.de/staff/tolosana/extra/CoDa.pdf

"""

import numpy as np
import scipy.stats as ss

def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1

    mat: numpy.ndarray
       columns = features
       rows = samples
    """
    if mat.dtype != type(0.0):
        mat = mat.astype(np.float64)

    if len(mat.shape) == 1:
        num_samps = len(mat)
        total = mat.sum()
    else:
        num_samps, num_feats = mat.shape
        total = np.reshape(mat.sum(axis=1), (num_samps, 1))
    return mat / total

def zero_replacement(mat):
    """
    Performs multiplicative replacement strategy
    mat: numpy.ndarray
       columns = features
       rows = samples
    Returns:
    --------
    mat: numpy.ndarray
    """
    num_samps, num_feats = mat.shape
    delta = (1. / num_feats)**2
    z_mat = (mat == 0).astype(np.float32)
    zcnts = 1 - np.reshape(z_mat.sum(axis=1) * delta, (num_samps, 1) )
    #z_mat = np.multiply(z_mat, zcnts)
    mat = z_mat*delta + np.multiply((1-z_mat), np.multiply(zcnts,mat))
    return closure(mat)

def perturb(x, y):
    """
    Performs the perturbation operation
    x: numpy.ndarray
    y: numpy.ndarray
    """    
    mat = np.multiply(x, y)
    return closure(mat)

def perturb_inv(x, y):
    """
    Performs the inverse perturbation operation
    x: numpy.ndarray
    y: numpy.ndarray
    """
    _y = power(y,-1)
    mat = np.multiply(x, _y)
    return closure(mat)

def power(x, y):
    """
    Performs the perturbation operation
    x: numpy.ndarray
    y: numpy.ndarray
    """
    num_samps, num_feats = x.shape
    mat = np.multiply(np.log(x), y)
    return closure(np.exp(mat))

def clr_inv(mat):
    """
    Performs inverse centre log ratio transformation
    """
    return closure(np.exp(mat))

def clr(mat):
    """
    Performs centre log ratio transformation

    Returns
    =======
    clr: numpy.ndarray
    clr transformed matrix
    """
    lmat = np.log(mat) # Need to account for zeros
    if len(mat.shape) == 1:
        num_samps = len(mat)
        gm = lmat.mean()
    else:
        num_samps, num_feats = mat.shape
        gm = lmat.mean(axis = 1)
        gm = np.reshape(gm, (num_samps, 1))

    _clr = lmat - gm
    return _clr

def ilr(mat):
    """
    Performs isometric log ratio transformation
    """
    if len(mat.shape) == 1:
        c = len(mat)
    else:
        r,c = mat.shape
    basis = np.zeros((c, c-1))
    for j in range(c-1):
        i = float(j+1)
        e = np.array( [1 / np.sqrt(i*(i+1))]*(j)+[-np.sqrt(i/(i+1))]+[1]*(c-j-1))
        basis[j,:] = clr(closure(np.exp(e)))
    _ilr = np.dot(clr(mat), basis)
    return _ilr

def ilr_inv(mat):    
    """
    Performs inverse isometric log ratio transform
    """
    if len(mat.shape) == 1:
        c = len(mat)
    else:
        r,c = mat.shape
    k = c+1
    basis = np.zeros((c, k))
    for j in range(c):
        i = float(j+1)
        e = np.array( [1 / np.sqrt(i*(i+1))]*(j+1)+[-np.sqrt(i/(i+1))]+[1]*(k-j-2))
        basis[j,:] = clr(closure(np.exp(e)))
    return closure(np.dot(np.exp(mat), basis))

def inner(mat1, mat2):
    """
    Calculates the inner product
    mat1: numpy.ndarray 1 D
    mat2: numpy.ndarray 1 D
    """
    assert len(mat1)==len(mat2)
    D = len(mat1)
    _in = 0
    for i in range(D):
        for j in range(D):
            _in += np.log(mat1[i]/mat1[j])*np.log(mat2[i]/mat2[j])
    return _in / (2*D)
    

def centre(mat):
    """
    Performs a perturbation and centers the data around the center
    mat: numpy.ndarray
       columns = features
       rows = samples
    """
    r,c = mat.shape
    cen = sps.gmean(mat, axis=0)
    cen = np.tile(cen, (r,1))
    return perturb_inv(mat, cen)

def variation_matrix(mat):
    """
    Calculates the variation matrix
    """
    pass

def total_variation(mat):
    """
    Calculate total variation 
    """
    pass
