"""Tests for function in apply_weights"""
import pandas as pd

from bsrm.estimation.apply_weights import apply_weights
from pandas._testing import assert_frame_equal


class TestApplyWeights:
    """Test for apply_weights()"""
    def create_input_df(self):
        """Create an input_ dataframe for the test based on the BERD survey."""
        input_columns = [
            "reference",
            "instance",
            "211",
            "218",
            "emp_researcher",
            "emp_technician",
            "701",
            "702",
            "709",
            "705",
            "706",
            "707",
            "711",
            "a_weight",
            "g_weight",
        ]

        data = [
            [111, 0, 0, 0, 0, 0, 100, 0, 100, 5, 3, 0, 6, 10, 1.5],
            [222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0],
            [333, 0, 0, 0, 0, 0, 900, 500, 1400, 7, 6, 3, 9, 4, 0.5],
        ]

        input_df = pd.DataFrame(data=data, columns=input_columns)
        return input_df

    # Create an expected dataframe for the test
    def create_expected_df(self):
        """Create an expected dataframe for the test."""
        expected_columns = [
            "reference",
            "instance",
            "211",
            "218",
            "emp_researcher",
            "emp_technician",
            "701",
            "702",
            "709",
            "705",
            "706",
            "707",
            "711",
            "a_weight",
            "g_weight",
        ]

        data = [
            [111, 0, 0, 0, 0, 0, 1000, 0, 1000, 75, 45, 0, 90, 10, 1.5],
            [222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0],
            [333, 0, 0, 0, 0, 0, 3600, 2000, 5600, 14, 12, 6, 18, 4, 0.5],
        ]

        expected_df = pd.DataFrame(data=data, columns=expected_columns)
        return expected_df

    def test_apply_weights(self):
        """Test for apply_weights()"""
        input_df = self.create_input_df()
        exp_output_df = self.create_expected_df()

        g_weight_cols = ["emp_researcher", "emp_technician", "705", "706", "707", "711"]

        a_weight_cols = g_weight_cols + ["211", "218", "701", "702", "709"]

        result_df = apply_weights(input_df, a_weight_cols, g_weight_cols, True, 2)

        assert_frame_equal(
            result_df, exp_output_df, check_like=True, check_exact=False, check_dtype=False
        )
