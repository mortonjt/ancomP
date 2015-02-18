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
    num_samps, num_feats = mat.shape
    if mat.dtype != type(0.0):
        mat = mat.astype(np.float64)
        
    total = np.reshape(mat.sum(axis=1), (num_samps, 1))
    return mat / total

def zero_replacement(mat):
    """
    Performs multiplicative replacement strategy

    mat: numpy.ndarray
       columns = features
       rows = samples
    """
    num_samps, num_feats = mat.shape
    delta = 1. / num_feats
    z_mat = (mat == 0).astype(np.float32)
    zcnts = np.reshape(z_mat.sum(axis=1) * delta, (num_samps, 1) )
    #zcnts = z_mat.sum(axis=1) * delta
    z_mat = np.multiply(z_mat, zcnts)
    mat = mat + z_mat
    return mat


def perturb(x, y):
    """
    Performs the perturbation operation
    x: numpy.ndarray
    y: numpy.ndarray
    """
    num_samps, num_feats = x.shape
    assert num_feats == y.shape[0]
    mat = np.multiply(x, y)
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

def clr(mat):
    """
    Performs centre log ratio transformation

    Returns
    =======
    clr: numpy.ndarray
    clr transformed matrix
    """
    lmat = np.log(mat) # Need to account for zeros
    gm = lmat.mean(axis = 1)
    num_samps, num_feats = mat.shape
    gm = np.reshape(gm, (num_samps, 1))

    _clr = lmat - gm
    return _clr

def ilr(mat):
    """
    Performs isometric log ratio transformation
    """
    r,c = mat.shape
    basis = np.ones((r-1, r-1))
    basis = np.multiply(basis, np.diag(np.diag(basis)) * np.exp(1))
    for row in range(r):
        continue
    pass

def centre(mat):
    """
    Calculates the mean composition
    via geometric mean estimation
    """
    return ss.gmean(mat, axis=1)

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
