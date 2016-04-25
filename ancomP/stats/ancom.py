from __future__ import division
import numpy as np
import pandas as pd
from time import time
import copy

import os, sys, site
import numpy as np
from numpy import random, array
from pandas import DataFrame, Series
import scipy

from math import log
from ancomP.stats.permutation import (_init_categorical_perms,
                                      _np_k_sample_f_statistic)


def ancom(table, grouping,
          alpha=0.05,
          tau=0.02,
          theta=0.1,
          multiple_comparisons_correction=None,
          significance_test=None, permutations=1000):
    r""" Performs a differential abundance test using ANCOM.
    This is done by calculating pairwise log ratios between all features
    and performing a significance test to determine if there is a significant
    difference in feature ratios with respect to the variable of interest.
    In an experiment with only two treatments, this test tests the following
    hypothesis for feature :math:`i`
    .. math::
        H_{0i}: \mathbb{E}[\ln(u_i^{(1)})] = \mathbb{E}[\ln(u_i^{(2)})]
    where :math:`u_i^{(1)}` is the mean abundance for feature :math:`i` in the
    first group and :math:`u_i^{(2)}` is the mean abundance for feature
    :math:`i` in the second group.
    Parameters
    ----------
    table : pd.DataFrame
        A 2D matrix of strictly positive values (i.e. counts or proportions)
        where the rows correspond to samples and the columns correspond to
        features.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    alpha : float, optional
        Significance level for each of the statistical tests.
        This can can be anywhere between 0 and 1 exclusive.
    tau : float, optional
        A constant used to determine an appropriate cutoff.
        A value close to zero indicates a conservative cutoff.
        This can can be anywhere between 0 and 1 exclusive.
    theta : float, optional
        Lower bound for the proportion for the W-statistic.
        If all W-statistics are lower than theta, then no features
        will be detected to be differentially significant.
        This can can be anywhere between 0 and 1 exclusive.
    multiple_comparisons_correction : {None, 'holm-bonferroni'}, optional
        The multiple comparison correction procedure to run.  If None,
        then no multiple comparison correction procedure will be run.
        If 'holm-boniferroni' is specified, then the Holm-Boniferroni
        procedure [1]_ will be run.
    significance_test : function, optional
        A statistical significance function to test for significance between
        classes.  This function must be able to accept at least two 1D
        array_like arguments of floats and returns a test statistic and a
        p-value. By default ``scipy.stats.f_oneway`` is used.
    permutations : int
        The number of permutations run if a permutation test is specified.

    Returns
    -------
    pd.DataFrame
        A table of features, their W-statistics and whether the null hypothesis
        is rejected.
        `"W"` is the W-statistic, or number of features that a single feature
        is tested to be significantly different against.
        `"reject"` indicates if feature is significantly different or not.
    See Also
    --------
    multiplicative_replacement
    scipy.stats.ttest_ind
    scipy.stats.f_oneway
    scipy.stats.wilcoxon
    scipy.stats.kruskal
    Notes
    -----
    The developers of this method recommend the following significance tests
    ([2]_, Supplementary File 1, top of page 11): the standard parametric
    t-test (``scipy.stats.ttest_ind``) or one-way ANOVA
    (``scipy.stats.f_oneway``) if the number of groups is greater
    than 2, or non-parametric variants such as Wilcoxon rank sum
    (``scipy.stats.wilcoxon``) or Kruskal-Wallis (``scipy.stats.kruskal``)
    if the number of groups is greater than 2.  Because one-way ANOVA is
    equivalent to the standard t-test when the number of groups is two,
    we default to ``scipy.stats.f_oneway`` here, which can be used when
    there are two or more groups.  Users should refer to the documentation
    of these tests in SciPy to understand the assumptions made by each test.
    This method cannot handle any zero counts as input, since the logarithm
    of zero cannot be computed.  While this is an unsolved problem, many
    studies have shown promising results by replacing the zeros with pseudo
    counts. This can be also be done via the ``multiplicative_replacement``
    method.
    References
    ----------
    .. [1] Holm, S. "A simple sequentially rejective multiple test procedure".
       Scandinavian Journal of Statistics (1979), 6.
    .. [2] Mandal et al. "Analysis of composition of microbiomes: a novel
       method for studying microbial composition", Microbial Ecology in Health
       & Disease, (2015), 26.
    Examples
    --------
    First import all of the necessary modules:
    >>> from ancomP.stats.ancom import ancom
    >>> import pandas as pd
    Now let's load in a pd.DataFrame with 6 samples and 7 unknown bacteria:
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    Then create a grouping vector.  In this scenario, there
    are only two classes, and suppose these classes correspond to the
    treatment due to a drug and a control.  The first three samples
    are controls and the last three samples are treatments.
    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])
    Now run ``ancom`` and see if there are any features that have any
    significant differences between the treatment and the control.
    >>> results = ancom(table, grouping,
    ...                 significance_test='permutative_anova',
    ...                 permutations=100)
    >>> results['W']
    b1    0
    b2    4
    b3    1
    b4    1
    b5    1
    b6    0
    b7    1
    Name: W, dtype: int64
    The W-statistic is the number of features that a single feature is tested
    to be significantly different against.  In this scenario, `b2` was detected
    to have significantly different abundances compared to four of the other
    species. To summarize the results from the W-statistic, let's take a look
    at the results from the hypothesis test:
    >>> results['reject']
    b1    False
    b2     True
    b3    False
    b4    False
    b5    False
    b6    False
    b7    False
    Name: reject, dtype: bool
    From this we can conclude that only `b2` was significantly
    different between the treatment and the control.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError('`table` must be a `pd.DataFrame`, '
                        'not %r.' % type(table).__name__)
    if not isinstance(grouping, pd.Series):
        raise TypeError('`grouping` must be a `pd.Series`,'
                        ' not %r.' % type(grouping).__name__)

    if np.any(table <= 0):
        raise ValueError('Cannot handle zeros or negative values in `table`. '
                         'Use pseudo counts or ``multiplicative_replacement``.'
                         )

    if not 0 < alpha < 1:
        raise ValueError('`alpha`=%f is not within 0 and 1.' % alpha)

    if not 0 < tau < 1:
        raise ValueError('`tau`=%f is not within 0 and 1.' % tau)

    if not 0 < theta < 1:
        raise ValueError('`theta`=%f is not within 0 and 1.' % theta)

    if multiple_comparisons_correction is not None:
        if multiple_comparisons_correction != 'holm-bonferroni':
            raise ValueError('%r is not an available option for '
                             '`multiple_comparisons_correction`.'
                             % multiple_comparisons_correction)

    if (grouping.isnull()).any():
        raise ValueError('Cannot handle missing values in `grouping`.')

    if (table.isnull()).any().any():
        raise ValueError('Cannot handle missing values in `table`.')

    groups, _grouping = np.unique(grouping, return_inverse=True)
    grouping = pd.Series(_grouping, index=grouping.index)
    num_groups = len(groups)

    if num_groups == len(grouping):
        raise ValueError(
            "All values in `grouping` are unique. This method cannot "
            "operate on a grouping vector with only unique values (e.g., "
            "there are no 'within' variance because each group of samples "
            "contains only a single sample).")

    if num_groups == 1:
        raise ValueError(
            "All values the `grouping` are the same. This method cannot "
            "operate on a grouping vector with only a single group of samples"
            "(e.g., there are no 'between' variance because there is only a "
            "single group).")

    if significance_test is None:
        significance_test = scipy.stats.f_oneway

    table_index_len = len(table.index)
    grouping_index_len = len(grouping.index)
    mat, cats = table.align(grouping, axis=0, join='inner')
    if (len(mat) != table_index_len or len(cats) != grouping_index_len):
        raise ValueError('`table` index and `grouping` '
                         'index must be consistent.')

    n_feat = mat.shape[1]
    if significance_test == 'permutative-anova':
        _logratio_mat = _stationary_log_compare(mat.values, cats.values,
                                                permutations=permutations)
    else:
        _logratio_mat = _log_compare(mat.values, cats.values, significance_test)
    logratio_mat = _logratio_mat + _logratio_mat.T

    # Multiple comparisons
    if multiple_comparisons_correction == 'holm-bonferroni':
        logratio_mat = np.apply_along_axis(_holm_bonferroni,
                                           1, logratio_mat)
    np.fill_diagonal(logratio_mat, 1)
    W = (logratio_mat < alpha).sum(axis=1)
    c_start = W.max() / n_feat
    if c_start < theta:
        reject = np.zeros_like(W, dtype=bool)
    else:
        # Select appropriate cutoff
        cutoff = c_start - np.linspace(0.05, 0.25, 5)
        prop_cut = np.array([(W > n_feat*cut).mean() for cut in cutoff])
        dels = np.abs(prop_cut - np.roll(prop_cut, -1))
        dels[-1] = 0

        if (dels[0] < tau) and (dels[1] < tau) and (dels[2] < tau):
            nu = cutoff[1]
        elif (dels[0] >= tau) and (dels[1] < tau) and (dels[2] < tau):
            nu = cutoff[2]
        elif (dels[1] >= tau) and (dels[2] < tau) and (dels[3] < tau):
            nu = cutoff[3]
        else:
            nu = cutoff[4]
        reject = (W >= nu*n_feat)

    labs = mat.columns
    return pd.DataFrame({'W': pd.Series(W, index=labs),
                         'reject': pd.Series(reject, index=labs)})


def _holm_bonferroni(p):
    """ Performs Holm-Bonferroni correction for pvalues
    to account for multiple comparisons
    Parameters
    ---------
    p: numpy.array
        array of pvalues
    Returns
    -------
    numpy.array
        corrected pvalues
    """
    K = len(p)
    sort_index = -np.ones(K, dtype=np.int64)
    sorted_p = np.sort(p)
    sorted_p_adj = sorted_p*(K-np.arange(K))
    for j in range(K):
        idx = (p == sorted_p[j]) & (sort_index < 0)
        num_ties = len(sort_index[idx])
        sort_index[idx] = np.arange(j, (j+num_ties), dtype=np.int64)

    sorted_holm_p = [min([max(sorted_p_adj[:k]), 1])
                     for k in range(1, K+1)]
    holm_p = [sorted_holm_p[sort_index[k]] for k in range(K)]
    return holm_p


def _log_compare(mat, cats,
                 significance_test=scipy.stats.f_oneway):
    """ Calculates pairwise log ratios between all features and performs a
    significiance test (i.e. t-test) to determine if there is a significant
    difference in feature ratios with respect to the variable of interest.
    Parameters
    ----------
    mat: np.array
       rows correspond to samples and columns correspond to
       features (i.e. OTUs)
    cats: np.array, float
       Vector of categories
    significance_test: function
        statistical test to run
    Returns:
    --------
    log_ratio : np.array
        log ratio pvalue matrix
    """
    r, c = mat.shape
    log_ratio = np.zeros((c, c))
    log_mat = np.log(mat)
    cs = np.unique(cats)

    def func(x):
        return significance_test(*[x[cats == k] for k in cs])

    for i in range(c-1):
        ratio = (log_mat[:, i].T - log_mat[:, i+1:].T).T
        m, p = np.apply_along_axis(func,
                                   axis=0,
                                   arr=ratio)
        log_ratio[i, i+1:] = np.squeeze(np.array(p.T))
    return log_ratio


def _stationary_log_compare(mat,cats,permutations=1000):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest.  Assumes a stationary set of permutations

    otu_table: numpy 2d matrix
        Contingency table where rows correspond to samples and
        cols correspond to otus

    cats: numpy array float32
        Categorical array

    Returns:
    --------
    log ratio: numpy.ndarray
        pvalue matrix
    """
    r, c = mat.shape
    log_mat = np.log(mat)
    log_ratio = np.zeros((c,c),dtype=mat.dtype)
    perms = _init_categorical_perms(cats, permutations)
    perms = perms.astype(mat.dtype)

    _ones = np.matrix(np.ones(c,dtype=mat.dtype)).transpose()
    for i in range(c-1):
        ratio =  log_mat[:, i] - log_mat[:, i+1:].T
        m, p  = _np_k_sample_f_statistic(ratio, cats, perms)
        log_ratio[i,i+1:] = p
    return log_ratio


if __name__=="__main__":
    pass
