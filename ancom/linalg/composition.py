"""
Makes use of Aitchson geometry to analyze/manipulate compositional data

Reference
=========
http://www.sediment.uni-goettingen.de/staff/tolosana/extra/CoDa.pdf

"""

import numpy as np
import scipy.stats as sps
import numpy.linalg as nl


def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1

    mat: numpy.ndarray
       columns = features
       rows = samples
    """
    return mat / mat.sum(axis=1)

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
    zcnts = z_mat.sum(axis=1)
    z_mat = np.multiply(z_mat, zcnts * delta)
    mat = mat + z_mat
    return mat

class CompositionMatrix():

    def __init__(self, mat):
        """
        mat: numpy.matrix
            rows = samples
            columns = features            
        """
        self.mat = closure( zero_replacement(mat) )
    
    def __add__(self,vec):
        """
        Performs the perturbation operation
    
        vec: numpy.ndarray
           perturbation vector
        """
        num_samps, num_feats = mat.shape
        assert num_feats == len(vec)
        mat = numpy.multiply(self.mat, amat.mat)
        return CompositionMatrix(mat)
        
    def __mult__(self,alpha):
        """
        Performs the power perturbation operation

        alpha: numpy.float
        """
        mat = numpy.power(self.mat,alpha)
        return CompositionMatrix(mat)
    
    def dot(self,amat):
        """
        Performs inner product
        """
        pass
                  
    def __str__(self):
        return str(self.mat)
    
    def clr(self):
        """
        Performs centre log ratio transformation

        Returns
        =======
        clr: numpy.ndarray
        clr transformed matrix
        """
        gm = sps.gmean(self.mat, axis=1)
        lmat = np.log(self.mat) # Need to account for zeros
        clr = lmat-gm
        return clr

    def ilr(self):
        """
        TODO
        Performs isometric log ratio transformation
        """
        pass
    
    def svd(self):
        """
        Performs singular value decomposition on matrix
        """
        Z = self.clr()
        L, K, M = nl.svd(Z)
        return L, K, M
