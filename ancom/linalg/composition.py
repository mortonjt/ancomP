"""
Makes use of Aitchson geometry to analyze/manipulate compositional data

Reference
=========
http://www.sediment.uni-goettingen.de/staff/tolosana/extra/CoDa.pdf

"""
import numpy as np
import scipy.stats as ss


def _closure(mat):
    """
    Performs closure to ensure that all elements add up to 1
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    """

    mat = np.asarray(mat, dtype=np.float64)

    if mat.ndim == 1:
        total = mat.sum()
    elif mat.ndim == 2:
        num_samps, num_feats = mat.shape
        total = np.reshape(mat.sum(axis=1), (num_samps, 1))
    else:
        raise ValueError("mat has too many dimensions")
    return np.divide(mat, total)


def multiplicative_replacement(mat, delta=None):
    """
    Performs multiplicative replacement strategy to replace
    all of the zeros with small non-zero values.  A closure
    operation is applied so that the compositions still
    add up to 1
    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import multiplicative_replacement
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> multiplicative_replacement(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])
    """
    mat = np.asarray(mat, dtype=np.float64)
    z_mat = (mat == 0)

    if mat.ndim == 1:
        num_feats = len(mat)
        num_samps = 1
        tot = z_mat.sum()
    elif mat.ndim == 2:
        num_samps, num_feats = mat.shape
        tot = z_mat.sum(axis=1)
    else:
        raise ValueError("mat has too many dimensions")

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - np.reshape(tot * delta, (num_samps, 1))
    mat_ = _closure(z_mat*delta + np.multiply((1-z_mat),
                                              np.multiply(zcnts, mat)))
    if mat.ndim == 1:
        mat_ = np.ravel(mat_)
    return mat_


def perturb(x, y):
    r"""
    Performs the perturbation operation
    This operation is defined as
    :math:`x \oplus y = C[x_1 y_1, ..., x_D y_D]`
    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.
    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Notes
    -----
    - Each row must add up to 1
    - All of the values in x and y must be greater than zero
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb(x,y)
    array([ 0.0625,  0.1875,  0.5   ,  0.25  ])
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _closure(np.multiply(x, y))


def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation
    This operation is defined as
    :math:`x \ominus y = C[x_1 y_1^{-1}, ..., x_D y_D^{-1}]`
    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.
    Parameters
    ----------
    x : numpy.ndarray
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : numpy.ndarray
        rows = compositions and
        columns = components
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Notes
    -----
    - Each row must add up to 1 for x.
    - y doesn't neccessary have to be a matrix of compositions
    - All of the values in x and y must be greater than zero
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb_inv
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb_inv(x,y)
    array([ 0.14285714,  0.42857143,  0.28571429,  0.14285714])
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _y = power(y, -1)
    return _closure(np.multiply(x, _y))


def power(x, a):
    r"""
    Performs the power operation
    This operation is defined as follows
    :math:`x \odot a = C[x_1^a, ..., x_D^a]`
    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.
    Parameters
    ----------
    x : numpy.ndarray, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    a : float
        a scalar float
    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Notes
    -----
    - Each row must add up to 1 for x
    - All of the values in x must be greater than zero
    >>> import numpy as np
    >>> from skbio.stats.composition import power
    >>> x = np.array([.1,.3,.4, .2])
    >>> power(x, .1)
    array([ 0.23059566,  0.25737316,  0.26488486,  0.24714631])
    """
    x = np.asarray(x, dtype=np.float64)
    mat = np.multiply(np.log(x), a)
    return _closure(np.exp(mat))




def centralize(mat):
    """
    This calculates the average sample and centers the data
    around this sample.
    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    Returns
    -------
    numpy.ndarray
         centered composition matrix
    Notes
    -----
    - Each row must add up to 1 for mat
    - All of the values must be greater than zero
    >>> import numpy as np
    >>> from skbio.stats.composition import centralize
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> centralize(X)
    array([[ 0.17445763,  0.30216948,  0.34891526,  0.17445763],
           [ 0.32495488,  0.18761279,  0.16247744,  0.32495488]])
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        raise ValueError("mat needs more than 1 row")
    if mat.ndim > 2:
        raise ValueError("mat has too many dimensions")
    r, c = mat.shape
    cen = ss.gmean(mat, axis=0)
    cen = np.tile(cen, (r, 1))
    return perturb_inv(mat, cen)


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

# ===================================================================
# LINEAR TRANSFORMATIONS
# ===================================================================
def alr(mat):
    """
    Performs additive log ratio transformation
    Always takes the community over the last
    component

    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    """
    if mat.ndim == 1:
        D = len(mat)
        mat = mat.reshape((1,D))
    else:
        _, D = mat.shape

    return np.log(mat[:,:-1]) -  np.log(mat[:,-1])


def alr_inv(mat):
    """
    Performs inverse additive log ratio transformation

    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    """
    if mat.ndim == 1:
        mat = mat.reshape((1,len(mat)))
        D = 1
    else:
        D, _ = mat.shape

    _mat = np.hstack((mat, np.zeros((D,1))))
    return _closure(np.exp(_mat))


def clr(mat):
    r"""
    Performs centre log ratio transformation that transforms
    compositions from Aitchison geometry to the real space.
    This transformation is an isometry, but not an isomorphism.
    This transformation is defined for a composition :math:`x` as follows
    :math:`clr(x) = ln[\frac{x_1}{g_m(x)}, ..., \frac{x_D}{g_m(x)}]`
    where :math:`g_m(x) = (\prod_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.
    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    Returns
    -------
    numpy.ndarray
         clr transformed matrix
    Notes
    -----
    - Each row must add up to 1
    - All of the values must be greater than zero for mat
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1,.3,.4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])
    """
    mat = np.asarray(mat, dtype=np.float64)
    lmat = np.log(mat)
    if mat.ndim == 1:
        num_samps = len(mat)
        gm = lmat.mean()
    elif mat.ndim == 2:
        num_samps, num_feats = mat.shape
        gm = lmat.mean(axis=1)
        gm = np.reshape(gm, (num_samps, 1))
    else:
        raise ValueError("mat has too many dimensions")
    return lmat - gm

def clr_inv(mat):
    """
    Performs inverse centre log ratio transformation

    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    """
    return _closure(np.exp(mat))


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


def variation_matrix(mat):
    """
    TODO
    Calculates the variation matrix
    """
    pass

def total_variation(mat):
    """
    TODO
    Calculate total variation
    """
    pass
