from __future__ import absolute_import, division, print_function

from unittest import TestCase, main
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt
from numpy.random import normal
import pandas as pd
import scipy
import copy
from ancomP.util import assert_data_frame_almost_equal
from ancomP.stats.ancom import _holm_bonferroni, _log_compare, _stationary_log_compare
from ancomP import ancom


class AncomTests(TestCase):
    def setUp(self):
        # Basic count data with 2 groupings
        self.table1 = pd.DataFrame([
            [10, 10, 10, 20, 20, 20],
            [11, 12, 11, 21, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 13, 10, 10, 10, 12]]).T
        self.cats1 = pd.Series([0, 0, 0, 1, 1, 1])
        # Real valued data with 2 groupings
        D, L = 40, 80
        np.random.seed(0)
        self.table2 = np.vstack((np.concatenate((normal(10, 1, D),
                                                 normal(200, 1, D))),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L)))
        self.table2 = np.absolute(self.table2)
        self.table2 = pd.DataFrame(self.table2.astype(np.int).T)
        self.cats2 = pd.Series([0]*D + [1]*D)

        # Real valued data with 2 groupings and no significant difference
        self.table3 = pd.DataFrame([
            [10, 10.5, 10, 10, 10.5, 10.3],
            [11, 11.5, 11, 11, 11.5, 11.3],
            [10, 10.5, 10, 10, 10.5, 10.2],
            [10, 10.5, 10, 10, 10.5, 10.3],
            [10, 10.5, 10, 10, 10.5, 10.1],
            [10, 10.5, 10, 10, 10.5, 10.6],
            [10, 10.5, 10, 10, 10.5, 10.4]]).T
        self.cats3 = pd.Series([0, 0, 0, 1, 1, 1])

        # Real valued data with 3 groupings
        D, L = 40, 120
        np.random.seed(0)
        self.table4 = np.vstack((np.concatenate((normal(10, 1, D),
                                                 normal(200, 1, D),
                                                 normal(400, 1, D))),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D),
                                                 normal(2000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D),
                                                 normal(2000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L)))
        self.table4 = np.absolute(self.table4)
        self.table4 = pd.DataFrame(self.table4.astype(np.int).T)
        self.cats4 = pd.Series([0]*D + [1]*D + [2]*D)

        # Noncontiguous case
        self.table5 = pd.DataFrame([
            [11, 12, 21, 11, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 10, 20, 9,  20, 20],
            [10, 13, 10, 10, 10, 12]]).T
        self.cats5 = pd.Series([0, 0, 1, 0, 1, 1])

        # Different number of classes case
        self.table6 = pd.DataFrame([
            [11, 12, 9, 11, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 10, 10, 9,  20, 20],
            [10, 13, 10, 10, 10, 12]]).T
        self.cats6 = pd.Series([0, 0, 0, 0, 1, 1])

        # Categories are letters
        self.table7 = pd.DataFrame([
            [11, 12, 9, 11, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 10, 10, 9,  20, 20],
            [10, 13, 10, 10, 10, 12]]).T
        self.cats7 = pd.Series(['a', 'a', 'a', 'a', 'b', 'b'])

        # Swap samples
        self.table8 = pd.DataFrame([
            [10, 10, 10, 20, 20, 20],
            [11, 12, 11, 21, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 13, 10, 10, 10, 12]]).T
        self.table8.index = ['a', 'b', 'c',
                             'd', 'e', 'f']
        self.cats8 = pd.Series([0, 0, 1, 0, 1, 1],
                               index=['a', 'b', 'd',
                                      'c', 'e', 'f'])

        # Real valued data with 3 groupings
        D, L = 40, 120
        np.random.seed(0)
        self.table9 = np.vstack((np.concatenate((normal(10, 1, D),
                                                 normal(200, 1, D),
                                                 normal(400, 1, D))),
                                 np.concatenate((normal(200000, 1, D),
                                                 normal(10, 1, D),
                                                 normal(2000, 1, D))),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 np.concatenate((normal(2000, 1, D),
                                                 normal(100000, 1, D),
                                                 normal(2000, 1, D))),
                                 normal(10000, 1000, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 normal(10000, 1000, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 np.concatenate((normal(2000, 1, D),
                                                 normal(100000, 1, D),
                                                 normal(2000, 1, D))),
                                 normal(10000, 1000, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L),
                                 normal(10, 10, L)))
        self.table9 = np.absolute(self.table9)+1
        self.table9 = pd.DataFrame(self.table9.astype(np.int).T)
        self.cats9 = pd.Series([0]*D + [1]*D + [2]*D)

        # Real valued data with 2 groupings
        D, L = 40, 80
        np.random.seed(0)
        self.table10 = np.vstack((np.concatenate((normal(10, 1, D),
                                                  normal(200, 1, D))),
                                  np.concatenate((normal(10, 1, D),
                                                  normal(200, 1, D))),
                                  np.concatenate((normal(20, 10, D),
                                                  normal(100, 10, D))),
                                  normal(10, 1, L),
                                  np.concatenate((normal(200, 100, D),
                                                  normal(100000, 100, D))),
                                  np.concatenate((normal(200000, 100, D),
                                                  normal(300, 100, D))),
                                  np.concatenate((normal(200000, 100, D),
                                                  normal(300, 100, D))),
                                  np.concatenate((normal(20, 20, D),
                                                  normal(40, 10, D))),
                                  np.concatenate((normal(20, 20, D),
                                                  normal(40, 10, D))),
                                  np.concatenate((normal(20, 20, D),
                                                  normal(40, 10, D))),
                                  normal(100, 10, L),
                                  normal(100, 10, L),
                                  normal(1000, 10, L),
                                  normal(1000, 10, L),
                                  normal(10, 10, L),
                                  normal(10, 10, L),
                                  normal(10, 10, L),
                                  normal(10, 10, L)))
        self.table10 = np.absolute(self.table10) + 1
        self.table10 = pd.DataFrame(self.table10.astype(np.int).T)
        self.cats10 = pd.Series([0]*D + [1]*D)

        # zero count
        self.bad1 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 0],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10]]).T)
        # negative count
        self.bad2 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, -1],
            [10, 10, 10, 10, 10, 10]]).T)

        # missing count
        self.bad3 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, np.nan],
            [10, 10, 10, 10, 10, 10]]).T)
        self.badcats1 = pd.Series([0, 0, 0, 1, np.nan, 1])
        self.badcats2 = pd.Series([0, 0, 0, 0, 0, 0])
        self.badcats3 = pd.Series([0, 0, 1, 1])
        self.badcats4 = pd.Series(range(len(self.table1)))
        self.badcats5 = pd.Series([1]*len(self.table1))

    def test_ancom_basic_counts(self):
        test_table = pd.DataFrame(self.table1)
        original_table = copy.deepcopy(test_table)
        test_cats = pd.Series(self.cats1)
        original_cats = copy.deepcopy(test_cats)
        result = ancom(test_table,
                       test_cats,
                       multiple_comparisons_correction=None)
        # Test to make sure that the input table hasn't be altered
        assert_data_frame_almost_equal(original_table, test_table)
        # Test to make sure that the input table hasn't be altered
        pdt.assert_series_equal(original_cats, test_cats)
        exp = pd.DataFrame({'W': np.array([5, 5, 2, 2, 2, 2, 2]),
                            'reject': np.array([True, True, False, False,
                                                False, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_basic_proportions(self):
        # Converts from counts to proportions
        test_table = pd.DataFrame(closure(self.table1))
        original_table = copy.deepcopy(test_table)
        test_cats = pd.Series(self.cats1)
        original_cats = copy.deepcopy(test_cats)
        result = ancom(test_table,
                       test_cats,
                       multiple_comparisons_correction=None)
        # Test to make sure that the input table hasn't be altered
        assert_data_frame_almost_equal(original_table, test_table)
        # Test to make sure that the input table hasn't be altered
        pdt.assert_series_equal(original_cats, test_cats)
        exp = pd.DataFrame({'W': np.array([5, 5, 2, 2, 2, 2, 2]),
                            'reject': np.array([True, True, False, False,
                                                False, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_multiple_groups(self):
        test_table = pd.DataFrame(self.table4)
        original_table = copy.deepcopy(test_table)
        test_cats = pd.Series(self.cats4)
        original_cats = copy.deepcopy(test_cats)
        result = ancom(test_table, test_cats)
        # Test to make sure that the input table hasn't be altered
        assert_data_frame_almost_equal(original_table, test_table)
        # Test to make sure that the input table hasn't be altered
        pdt.assert_series_equal(original_cats, test_cats)
        exp = pd.DataFrame({'W': np.array([8, 7, 3, 3, 7, 3, 3, 3, 3]),
                            'reject': np.array([True, True, False, False,
                                                True, False, False, False,
                                                False], dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_noncontiguous(self):
        result = ancom(self.table5,
                       self.cats5,
                       multiple_comparisons_correction=None)
        exp = pd.DataFrame({'W': np.array([6, 2, 2, 2, 2, 6, 2]),
                            'reject': np.array([True, False, False, False,
                                                False, True, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_unbalanced(self):
        result = ancom(self.table6,
                       self.cats6,
                       multiple_comparisons_correction=None)
        exp = pd.DataFrame({'W': np.array([5, 3, 3, 2, 2, 5, 2]),
                            'reject': np.array([True, False, False, False,
                                                False, True, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_letter_categories(self):
        result = ancom(self.table7,
                       self.cats7,
                       multiple_comparisons_correction=None)
        exp = pd.DataFrame({'W': np.array([5, 3, 3, 2, 2, 5, 2]),
                            'reject': np.array([True, False, False, False,
                                                False, True, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_multiple_comparisons(self):
        result = ancom(self.table1,
                       self.cats1,
                       multiple_comparisons_correction='holm-bonferroni',
                       significance_test=scipy.stats.mannwhitneyu)
        exp = pd.DataFrame({'W': np.array([0]*7),
                            'reject': np.array([False]*7, dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_alternative_test(self):
        result = ancom(self.table1,
                       self.cats1,
                       multiple_comparisons_correction=None,
                       significance_test=scipy.stats.ttest_ind)
        exp = pd.DataFrame({'W': np.array([5, 5, 2, 2, 2, 2, 2]),
                            'reject': np.array([True,  True, False, False,
                                                False, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_normal_data(self):
        result = ancom(self.table2,
                       self.cats2,
                       multiple_comparisons_correction=None,
                       significance_test=scipy.stats.ttest_ind)
        exp = pd.DataFrame({'W': np.array([8, 8, 3, 3,
                                           8, 3, 3, 3, 3]),
                            'reject': np.array([True, True, False, False,
                                                True, False, False,
                                                False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_basic_counts_swapped(self):
        result = ancom(self.table8, self.cats8)
        exp = pd.DataFrame({'W': np.array([5, 5, 2, 2, 2, 2, 2]),
                            'reject': np.array([True, True, False, False,
                                                False, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_no_signal(self):
        result = ancom(self.table3,
                       self.cats3,
                       multiple_comparisons_correction=None)
        exp = pd.DataFrame({'W': np.array([0]*7),
                            'reject': np.array([False]*7, dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_tau(self):
        exp1 = pd.DataFrame({'W': np.array([8, 7, 3, 3, 7, 3, 3, 3, 3]),
                            'reject': np.array([True, False, False, False,
                                                False, False, False, False,
                                                False], dtype=bool)})
        exp2 = pd.DataFrame({'W': np.array([17, 17, 5, 6, 16, 5, 7, 5,
                                            4, 5, 8, 4, 5, 16, 5, 11, 4, 6]),
                            'reject': np.array([True, True, False, False,
                                                True, False, False, False,
                                                False, False, False, False,
                                                False, True, False, False,
                                                False, False],  dtype=bool)})
        exp3 = pd.DataFrame({'W': np.array([16, 16, 17, 10, 17, 16, 16,
                                            15, 15, 15, 13, 10, 10, 10,
                                            9, 9, 9, 9]),
                            'reject': np.array([True, True, True, False,
                                                True, True, True, True,
                                                True, True, True, False,
                                                False, False, False, False,
                                                False, False],  dtype=bool)})

        result1 = ancom(self.table4, self.cats4, tau=0.25)
        result2 = ancom(self.table9, self.cats9, tau=0.02)
        result3 = ancom(self.table10, self.cats10, tau=0.02)

        assert_data_frame_almost_equal(result1, exp1)
        assert_data_frame_almost_equal(result2, exp2)
        assert_data_frame_almost_equal(result3, exp3)

    def test_ancom_theta(self):
        result = ancom(self.table1, self.cats1, theta=0.3)
        exp = pd.DataFrame({'W': np.array([5, 5, 2, 2, 2, 2, 2]),
                            'reject': np.array([True, True, False, False,
                                                False, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_alpha(self):
        result = ancom(self.table1, self.cats1, alpha=0.5)
        exp = pd.DataFrame({'W': np.array([6, 6, 4, 5, 5, 4, 2]),
                            'reject': np.array([True, True, False, True,
                                                True, False, False],
                                               dtype=bool)})
        assert_data_frame_almost_equal(result, exp)

    def test_ancom_fail_type(self):
        with self.assertRaises(TypeError):
            ancom(self.table1.values, self.cats1)
        with self.assertRaises(TypeError):
            ancom(self.table1, self.cats1.values)

    def test_ancom_fail_zeros(self):
        with self.assertRaises(ValueError):
            ancom(self.bad1, self.cats2, multiple_comparisons_correction=None)

    def test_ancom_fail_negative(self):
        with self.assertRaises(ValueError):
            ancom(self.bad2, self.cats2, multiple_comparisons_correction=None)

    def test_ancom_fail_not_implemented_multiple_comparisons_correction(self):
        with self.assertRaises(ValueError):
            ancom(self.table2, self.cats2,
                  multiple_comparisons_correction='fdr')

    def test_ancom_fail_missing(self):
        with self.assertRaises(ValueError):
            ancom(self.bad3, self.cats1)

        with self.assertRaises(ValueError):
            ancom(self.table1, self.badcats1)

    def test_ancom_fail_groups(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.badcats2)

    def test_ancom_fail_size_mismatch(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.badcats3)

    def test_ancom_fail_group_unique(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.badcats4)

    def test_ancom_fail_1_group(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.badcats5)

    def test_ancom_fail_tau(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, tau=-1)
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, tau=1.1)

    def test_ancom_fail_theta(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, theta=-1)
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, theta=1.1)

    def test_ancom_fail_alpha(self):
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, alpha=-1)
        with self.assertRaises(ValueError):
            ancom(self.table1, self.cats1, alpha=1.1)

    def test_ancom_fail_multiple_groups(self):
        with self.assertRaises(TypeError):
            ancom(self.table4, self.cats4,
                  significance_test=scipy.stats.ttest_ind)

    def test_holm_bonferroni(self):
        p = [0.005, 0.011, 0.02, 0.04, 0.13]
        corrected_p = p * np.arange(1, 6)[::-1]
        guessed_p = _holm_bonferroni(p)
        for a, b in zip(corrected_p, guessed_p):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    main()
