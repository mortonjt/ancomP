"""
Construct biplot from svd
"""
from aitchson import AitchsonMatrix
from math import sqrt
import matplotlib.pyplot as plt


def biplot(mat):
    """
    mat: numpy.ndarray
       columns: features
       rows: samples
    """

    # Log ratios transformation
    amat = AitchsonMatrix(mat)
    L, K, M = amat.svd()
    n, _ = L.shape
    # Now extract only the first 2 eigenvectors
    _k = K[:2]
    _x = L[:,:2] * sqrt(n-1)
    _y = M[:2,:] * (_k.reshape(2,1) / sqrt(n-1))

def 2d_biplot(x, y):
    """
    Constructs a 2 dimensional biplot
    
    x: 2D numpy.ndarray
       row markers
    y: 2D numpy.ndarray
       feature projections

    Returns
    =======
    fig: matplotlib.fig

    Reference
    =========
    http://okomestudio.net/biboroku/?p=2292
    """
    fig = plt.figure()
    plt.scatter(x)

    _, feats = y.shape
    for i in range(feats):
        pnt = y[:,i]
        _x, _y = pnt[0], pnt[1]
        plt.arrow(0, 0, _x, _y, color='r',
                  width=0.002, head_width=0.05)
    return fig
    
    
