"""Tests for functions in calculate_weights"""

import pandas as pd
import numpy as np
import pytest
import bsrm.estimation.calculate_weights as calw
from bsrm.estimation.estimation_main import run_estimation
from pandas._testing import assert_frame_equal, assert_series_equal


class TestCalcLowerN:
    """Test for calc_lower_n with duplicate refs."""
    def test_calc_lower_n(self):
        """Test for calc_lower_n with duplicate refs."""
        input_cols = ["reference", "709"]
        data = [
            [1, "A"],
            [2, "B"],
            [2, "C"],
            [4, "D"],
            [1, "E"],
            [4, np.nan]
        ]
        input_df = pd.DataFrame(data=data, columns=input_cols)

        # Call calc_lower_n function
        actual_result = calw.calc_lower_n(input_df, "reference")
        # Defined expected result
        expected_result = 3
        assert actual_result == expected_result, "calc_lower_n not behaving as expected"


    def test_calc_lower_n_nan_ref(self):
        """Test for calc_lower_n with nan in reference."""
        input_cols = ["reference", "709"]
        data = [
            [1, "A"],
            [2, "B"],
            [np.nan, "C"],
            [4, "D"],
            [1, "E"],
        ]

        input_df = pd.DataFrame(data=data, columns=input_cols)

        # Call calc_lower_n function
        actual_result = calw.calc_lower_n(input_df, "reference")

        # Defined expected result
        expected_result = 3

        assert actual_result == expected_result, "calc_lower_n not behaving as expected"

class TestCalcLowerE:
    """Test for calc_lower_e with nan."""
    def create_input_df(self):
        """Creates input df for test"""

        input_cols = ["employment", "711"]
        data = [
            [1, 10],
            [2, 5],
            [2, np.nan],
            [4, np.nan],
            [1, 10],
            [4, np.nan],
        ]
        input_df = pd.DataFrame(data=data, columns=input_cols)
        return input_df

    def test_calc_lower_e(self):
        """Test for calc_lower_e with nan."""

        input_df = self.create_input_df()
        expected_result = 14

        # Call calc_lower_e function
        actual_result = calw.calc_lower_e(input_df, "employment")
        assert actual_result == expected_result, "calc_lower_e not behaving as expected"

class TestCalcLowerS:
    """Test for calc_lower_s"""
    def create_input_df(self):
        """Creates input dataframe"""
        cols =['reference', 'cellnumber', 'employment', 'outlier']
        data =[[1, 22, 100, False],
               [2, 22, 10, False],
               [3, 22, 5, False],
               [4, 8, 60, False],
               [5, 8, 45, True],
               [6, 8, 100, True]]
        input_df = pd.DataFrame(data=data, columns=cols)
        return input_df

    def test_calc_lower_s(self):
        """Test for lower_s calculation"""

        input_df = self.create_input_df()
        # Call calc_lower_s function
        actual_result = calw.calc_lower_s(input_df, "employment")
        # Define expected result
        expected_result = 145
        assert actual_result == expected_result, "calc_lower_s not behaving as expected"


class TestCalcLowerSNoOutliers:
    """Test to check if calc_lower_s returns 0 when there are no outliers"""
    def create_input_df(self):
        """Creates input dataframe"""
        cols = ['reference', 'cellnumber', 'employment', 'outlier']
        data = [[1, 22, 100, False],
                [2, 22, 10, False],
                [3, 22, 5, False],
                [4, 8, 60, False],
                [5, 8, 45, False],
                [6, 8, 100, False]]
        input_df = pd.DataFrame(data=data, columns=cols)
        return input_df

    def test_calc_lower_s_emptydf_(self):
        """Test for lower_s calculation"""

        input_df = self.create_input_df()
        # Call calc_lower_s function
        actual_result = calw.calc_lower_s(input_df, "employment")
        # Define expected result
        expected_result = 0
        assert actual_result == expected_result, "calc_lower_s not behaving as expected"



# Five tests for calculate_weights:
# testing calculate_weights where missing outlier col
# testing calculate_weights filter
# testing calculate_weights 709 to numeric with no missing vals
# testing calculate_weights 709 to numeric with missing vals
# testing calculate_weights with missing vals
class TestCalcWeightFactors:
    """Tests for calculate_weights function."""

    def create_input_df(self):
        """Creates input df for test"""
        input_cols = [
            "reference",
            "instance",
            "cellnumber",
            "709",
            "uni_count",
            "uni_employment",
            "employment",
            "outlier"
        ]

        data = [
            [1, 0, 1, "12", 20, 5000, 66, True],
            [2, 0, 2, 14, 4, 5000, 77, False],
            [3, 0, 1, 1, 20, 5000, 11, False],
            [4, 0, 4, 18, 3, 5000, 88, False],
            [5, 0, 2, 14, 4, 5000, 22, False],
            [6, 0, 1, 10, 20, 5000, 7, False],
            [7, 0, 5, 20, 50, 5000, 20, False],
            [8, 0, 2, np.nan, 4, 5000, 7, False],
            [9, 0, 1, 5, 20, 5000, 44, False],
            [10, 0, 1, 10, 20, 5000, 44, False],
            [11, 0, 5, 20, 50, 5000, 20, False],
        ]

        input_df = pd.DataFrame(data=data, columns=input_cols)
        return input_df

    def create_expected_output(self):
        """Creates expected df for test"""
        expected_cols = [
            "reference",
            "instance",
            "cellnumber",
            "709",
            "uni_count",
            "uni_employment",
            "employment",
            "outlier",
            "a_weight",
            "g_weight"
        ]

        data = [
            [1, 0, 1, "12", 20, 5000, 66, True, 1.0, 1.0],
            [2, 0, 2, 14, 4, 5000, 77, False, 1.3, 35.4],
            [3, 0, 1, 1, 20, 5000, 11, False, 4.8, 9.8],
            [4, 0, 4, 18, 3, 5000, 88, False, 3.0, 18.9],
            [5, 0, 2, 14, 4, 5000, 22, False, 1.3, 35.4],
            [6, 0, 1, 10, 20, 5000, 7, False, 4.8, 9.8],
            [7, 0, 5, 20, 50, 5000, 20, False, 25.0, 5.0],
            [8, 0, 2, np.nan, 4, 5000, 7, False, 1.3, 35.4],
            [9, 0, 1, 5, 20, 5000, 44, False, 4.8, 9.8],
            [10, 0, 1, 10, 20, 5000, 44, False, 4.8, 9.8],
            [11, 0, 5, 20, 50, 5000, 20, False, 25.0, 5.0],
        ]
        expected_df = pd.DataFrame(data=data, columns=expected_cols)
        return expected_df

    def create_expected_qa(self):
        """Creates expected qa df for test"""
        expected_qa_cols = [
            "Cell Number",
            "N - uni_count",
            "n - num clear records in cell",
            "o - num outliers in cell",
            "E - uni_employment",
            "e - sum of employment in cell",
            "s - sum of employment outliers in cell",
            "a_weight",
            "g_weight"
        ]

        data = [
            [1, 20, 5, 1, 5000, 172, 66, 4.8, 9.8],
            [2, 4, 3, 0, 5000, 106, 0, 1.3, 35.4],
            [4, 3, 1, 0, 5000, 88, 0, 3.0, 18.9],
            [5, 50, 2, 0, 5000, 40, 0, 25.0, 5.0],
        ]

        expected_qa_df = pd.DataFrame(data=data, columns=expected_qa_cols)
        return expected_qa_df

    def test_calculate_weights_g_weight_true(self):
        """Test for calculate_weights with g_weight set to True"""

        input_df = self.create_input_df()
        expected_df = self.create_expected_output()
        expected_qa_df = self.create_expected_qa()
        print(expected_qa_df.columns)
        result_df, result_qa_df = run_estimation(input_df, "cellnumber", "reference", "employment", incl_g_wts=True)

        for df in [result_qa_df, result_df]:
            df["a_weight"] = df["a_weight"].round(1)
            df["g_weight"] = df["g_weight"].round(1)

        assert_frame_equal(result_df, expected_df, check_exact=False, rtol=0.01)
        assert_frame_equal(result_qa_df, expected_qa_df, check_exact=False, rtol=0.01)

    def test_calculate_weights_g_weight_false(self):
        """Test for calculate_weights for filter
        and np.nan taken out of calculation"""

        input_df = self.create_input_df()
        expected_df = self.create_expected_output()
        expected_qa_df = self.create_expected_qa()

        expected_df = expected_df.drop(columns=["g_weight"])
        expected_qa_df = expected_qa_df.drop(
            columns=[
                "E - uni_employment",
                "e - sum of employment in cell",
                "s - sum of employment outliers in cell",
                "g_weight"
            ]
        )

        result_df, result_qa_df = run_estimation(input_df, "cellnumber", "reference", "employment", incl_g_wts=False)

        # Round specified columns in each DataFrame
        for df in [result_qa_df, result_df]:
            df["a_weight"] = df["a_weight"].round(1)

        # Ensure both DataFrames have the same data type for the "709" column
        result_df["709"] = result_df["709"].astype(float)
        expected_df["709"] = expected_df["709"].astype(float)

        assert_frame_equal(result_df, expected_df, check_exact=False, rtol=0.01, check_dtype=False)
        assert_frame_equal(result_qa_df, expected_qa_df, check_exact=False, rtol=0.01, check_dtype=False)

class TestOutlierWeight:
    """Test for outlier_weights."""

    def create_input_df(self):
        """Creates input df for test"""
        input_cols = ["reference","outlier"]
        data = [
            [1, True],
            [2, False],
            [2, True],
            [4, True],
            [1, False],
        ]

        input_df = pd.DataFrame(data=data, columns=input_cols)
        return input_df

    def create_expected_output(self):
        """Creates expected df for test"""
        expected_cols = [
            "reference",
            "outlier",
            "a_weight",
            "g_weight"
        ]

        data = [
            [1, True, 1.0, 1.0],
            [2, False, None, None],
            [2, True, 1.0, 1.0],
            [4, True, 1.0, 1.0],
            [1, False, None, None],
        ]

        expected_df = pd.DataFrame(data=data, columns=expected_cols)
        return expected_df

    def test_outlier_weights(self):
        """Test for outlier_weights."""
        input_df = self.create_input_df()
        expected_df = self.create_expected_output()

        result_df = calw.outlier_weights(input_df)
        assert_frame_equal(result_df, expected_df)
