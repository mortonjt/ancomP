"""
Construct biplot from singular value decomposition

Reference
=========
http://okomestudio.net/biboroku/?p=2292

"""
from ancom.linalg.composition import CompositionMatrix
from math import sqrt
import matplotlib.pyplot as plt
import numpy.linalg as nl
import numpy as np

def rank_2_approx(mat):
    """
    Rank 2 approximation
    
    mat: numpy.ndarray
       columns: features
       rows: samples
    """

    # Log ratios transformation
    amat = CompositionMatrix(mat)
    L, K, M = nl.svd(amat.clr())
    n, _ = L.shape
    # Now extract only the first 2 eigenvectors
    _k = K[:2]
    _x = L[:,:2] * sqrt(n-1)
    _y = np.multiply(M[:2,:],(_k.reshape(2,1) / sqrt(n-1)))    
    return _x, _y

def biplot(_x, _y):
    """
    Creates a biplot
    
    _x: numpy.ndarray
       n x 2 matrix
    _y: numpy.ndarray
       2 x n matrix
    """
    # Create figure
    # Points = row markers (samples)
    # Projections = column markers (features)
    fig = plt.figure()
    pca1 = np.ravel(_x[:,0])
    pca2 = np.ravel(_x[:,1])
    plt.scatter(pca1, pca2)
        
    _, feats = _y.shape
    for i in range(feats):
        pnt = _y[:,i]
        a, b = np.asscalar(pnt[0]), np.asscalar(pnt[1])
        #print a,b
        plt.arrow(0, 0, a, b, color='r',
                  width=0.002, head_width=0.05)

    ## Create some padding
    xmin, xmax = _x.min(), _x.max()
    ymin, ymax = _y.min(), _y.max()
    xpad = (xmax - xmin) * 0.1
    ypad = (ymax - ymin) * 0.1
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    return fig

