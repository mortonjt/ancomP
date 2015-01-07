"""
Construct biplot from svd
"""
from aitchson import AitchsonMatrix
from math import sqrt
import matplotlib.pyplot as plt
import numpy.linalg as nl


def biplot(mat):
    """
    mat: numpy.ndarray
       columns: features
       rows: samples
    """

    # Log ratios transformation
    amat = AitchsonMatrix(mat)
    L, K, M = nl.svd(amat.clr())
    L, K, M = amat.svd()
    n, _ = L.shape
    # Now extract only the first 2 eigenvectors
    _k = K[:2]
    _x = L[:,:2] * sqrt(n-1)
    _y = M[:2,:] * (_k.reshape(2,1) / sqrt(n-1))

    # Create figure
    # Points = row markers (samples)
    # Projections = column markers (features)
    fig = plt.figure()
    plt.scatter(_x)

    _, feats = _y.shape
    for i in range(feats):
        pnt = _y[:,i]
        a, b = pnt[0], pnt[1]
        plt.arrow(0, 0, a, b, color='r',
                  width=0.002, head_width=0.05)
    return fig

