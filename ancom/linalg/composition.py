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
    return np.divide(mat, total)

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
    mat = np.multiply(np.log(x), y)
    return closure(np.exp(mat))


def inner(mat1, mat2):
    """
    Calculates the Aitchson inner product
    mat1: numpy.ndarray 
    mat2: numpy.ndarray
    """
    if len(mat1.shape) == 1:
        D1 = len(mat1)
    else:
        _, D1 = mat1.shape
    if len(mat2.shape) == 1:
        D2 = len(mat2)
    else:
        _, D2 = mat2.shape
    assert D1==D2
    D = D1
    
    # _in = 0
    # for i in range(D):
    #     for j in range(D):
    #         _in += np.log(mat1[i]/mat1[j])*np.log(mat2[i]/mat2[j])
    # return _in / (2*D)
    M = np.ones((D,D))*-1 + np.identity(D)*D
    a = clr(mat1)
    b = clr(mat2).T
    return np.dot(np.dot(a,M),b)/D


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

def clr_inv(mat):
    """
    Performs inverse centre log ratio transformation
    """
    return closure(np.exp(mat))

def ilr(mat, basis=None):
    """
    Performs isometric log ratio transformation
    mat: numpy.ndarray
    basis: numpy.ndarray
        orthonormal basis for Aitchison simplex
    """
    if len(mat.shape) == 1:
        c = len(mat)
    else:
        r,c = mat.shape
    if basis==None: # Default to J.J.Egozcue orthonormal basis
        basis = np.zeros((c, c-1))
        for j in range(c-1):
            i = float(j+1)
            e = np.array( [(1/i)]*int(i)+[-1]+[0]*int(c-i-1))*np.sqrt(i/(i+1))
            basis[:,j] = e
    _ilr = np.dot(clr(mat), basis)
    return _ilr


def ilr_inv(mat, basis=None):    
    """
    Performs inverse isometric log ratio transform
    """
    if len(mat.shape) == 1:
        c = len(mat)
    else:
        r,c = mat.shape
    c = c+1
    if basis==None: # Default to J.J.Egozue orthonormal basis
        basis = np.zeros((c, c-1))
        for j in range(c-1):
            i = float(j+1)
            e = np.array( [(1/i)]*int(i)+[-1]+[0]*int(c-i-1))*np.sqrt(i/(i+1))
            basis[:,j] = e    
    return clr_inv(np.dot(mat, basis.T))

    
def centre(mat):
    """
    Performs a perturbation and centers the data around the center
    mat: numpy.ndarray
       columns = features
       rows = samples
    """
    r,c = mat.shape
    cen = ss.gmean(mat, axis=0)
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
