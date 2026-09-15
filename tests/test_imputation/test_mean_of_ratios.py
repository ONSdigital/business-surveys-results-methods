"""Tests for MoR.py."""

# Local Imports
import os

# Third Party Imports
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import numpy as np

# Local Imports
from bsrm.imputation.mean_of_ratios import (
    mor_preprocessing,
    get_imputation_lists,
    join_current_backdata,
    carry_forwards,
    calculate_growth_rates,
    group_calc_link,
    calculate_links
)

@pytest.fixture(scope="module")
def pre_processing_expected():
    """Data from pre-processing_expected.csv."""
    columns = [
        "reference",
        "Q10",
        "Q11",
        "Q20",
        "sic",
        "status",
        "is_clear",
        "to_impute",
    ]

    data = [
        ["1030", 20.0, 10.0, 6.0, "333", "Clear", True, False],
        ["1031", 10.0, 0.0, 9.0, "333", "Clear", True, False],
        ["1032", 0.0, 0.0, 0.0, "222", "Clear", True, False],
        ["1033", 550.0, 330.0, 110.0, "222", "Clear", True, False],
        ["1040", 86000.0, 7700.0, 130.0, "222", "Clear", True, False],
        ["1041", np.nan, np.nan, np.nan, "222", "Form sent out", False, True],
        ["1042", 8900.0, 2200.0, 64.0, "444", "Clear", True, False],
        ["1043", np.nan, np.nan, np.nan, "444", "Form sent out", False, True],
        ["1044", 36000.0, 30.0, 10.0, "444", "Clear", True, False],
        ["1045", 80500.0, 7000.0, 66.0, "444", "Clear", True, False],
        ["1046", 6000.0, 2500.0, 80.0, "444", "Clear", True, False],
        ["1047", 0.0, 0.0, 0.0, "444", "Clear", True, False],
        ["1048", 444.0, 330.0, 9.0, "555", "Clear", True, False],
        ["1049", 400.0, 20.0, np.nan, "555", "Clear", True, False],
        ["1050", 200.0, 10.0, 10.0, "555", "Clear", True, False],
        ["2060", np.nan, np.nan, np.nan, "555", "Form sent out", False, True],
        ["2061", np.nan, np.nan, np.nan, "555", "Form sent out", False, True],
    ]

    return pd.DataFrame(columns=columns, data=data)


class TestMorPreprocessing:
    """Tests for mor_preprocessing."""
    def current_data(self):
        """Data from current_data.csv."""
        columns = [
            "reference",
            "Q10",
            "Q11",
            "Q20",
            "sic",
            "status",
        ]

        data = [
            ["1030", 20.0, 10.0, 6.0, "333", "Clear"],
            ["1031", 10.0, 0.0, 9.0, "333", "Clear"],
            ["1032", 0.0, 0.0, 0.0, "222", "Clear"],
            ["1033", 550.0, 330.0, 110.0, "222", "Clear"],
            ["1040", 86000.0, 7700.0, 130.0, "222", "Clear"],
            ["1041", np.nan, np.nan, np.nan, "222", "Form sent out"],
            ["1042", 8900.0, 2200.0, 64.0, "444", "Clear"],
            ["1043", np.nan, np.nan, np.nan, "444", "Form sent out"],
            ["1044", 36000.0, 30.0, 10.0, "444", "Clear"],
            ["1045", 80500.0, 7000.0, 66.0, "444", "Clear"],
            ["1046", 6000.0, 2500.0, 80.0, "444", "Clear"],
            ["1047", 0.0, 0.0, 0.0, "444", "Clear"],
            ["1048", 444.0, 330.0, 9.0, "555", "Clear"],
            ["1049", 400.0, 20.0, np.nan, "555", "Clear"],
            ["1050", 200.0, 10.0, 10.0, "555", "Clear"],
            ["2060", np.nan, np.nan, np.nan, "555", "Form sent out"],
            ["2061", np.nan, np.nan, np.nan, "555", "Form sent out"],
        ]

        return pd.DataFrame(columns=columns, data=data)

    def test_mor_preprocessing(self, pre_processing_expected):
        """General tests for mor_preprocessing."""
        current_data = self.current_data()
        output = mor_preprocessing(current_data)
        assert_frame_equal(output, pre_processing_expected)


class TestGetImputedLists:
    """Tests for get_imputed_lists."""
    def test_get_imputed_lists(self):
        """Test the get_imputed_lists function."""
        input_dict = {
            "Q10": ["Q11", "Q12"],
            "Q20": ["Q21"],
        }
        expected_target_vars = ["Q10", "Q20"]
        expected_imputed_vars = ["Q10", "Q11", "Q12", "Q20", "Q21"]
        target_vars, imputed_vars = get_imputation_lists(input_dict)
        assert target_vars == expected_target_vars, "get_imputation_lists() not returning expected target vars"
        assert imputed_vars == expected_imputed_vars, "get_imputation_lists() not returning expected imputed vars"



    @pytest.fixture(scope="module")
    def backdata(self):
        """Data from backdata.csv."""
        columns = [
            "reference",
            "Q10",
            "Q11",
            "Q20",
            "sic",
        ]

        data = [
            ["1030", 20, 10, 5, "333"],
            ["1031", 11, 4, 8, "333"],
            ["1032", 99, 55, 6, "222"],
            ["1033", 500, 300, 100, "222"],
            ["1040", 86000, 7000, 120, "222"],
            ["1041", 440, 330, 50, "222"],
            ["1042", 9000, 2000, 60, "444"],
            ["1043", 77, 66, 9, "444"],
            ["1044", 36000, 30, 10, "444"],
            ["1045", 80000, 0, 66, "444"],
            ["1046", 5000, 3000, 100, "444"],
            ["1047", 0, 0, 0, "444"],
            ["1048", 444, 333, 22, "555"],
            ["1049", 400, 20, 12, "555"],
            ["1050", 230, 12, 11, "555"],
            ["2060", 600, 500, 30, "555"],
        ]

        return pd.DataFrame(columns=columns, data=data)

    @pytest.fixture(scope="module")
    def joined_expected(self):
        """Data from joined_expected.csv."""
        columns = [
            "reference",
            "Q10",
            "Q11",
            "Q20",
            "sic",
            "status",
            "is_clear",
            "to_impute",
            "Q10_prev",
            "Q11_prev",
            "Q20_prev",
            "_merge",
        ]

        data = [
            ["1030", 20.0, 10.0, 6.0, "333", "Clear", True, False, 20.0, 10.0, 5.0, "both"],
            ["1031", 10.0, 0.0, 9.0, "333", "Clear", True, False, 11.0, 4.0, 8.0, "both"],
            ["1032", 0.0, 0.0, 0.0, "222", "Clear", True, False, 99.0, 55.0, 6.0, "both"],
            ["1033", 550.0, 330.0, 110.0, "222", "Clear", True, False, 500.0, 300.0, 100.0, "both"],
            ["1040", 86000.0, 7700.0, 130.0, "222", "Clear", True, False, 86000.0, 7000.0, 120.0, "both"],
            ["1041", np.nan, np.nan, np.nan, "222", "Form sent out", False, True, 440.0, 330.0, 50.0, "both"],
            ["1042", 8900.0, 2200.0, 64.0, "444", "Clear", True, False, 9000.0, 2000.0, 60.0, "both"],
            ["1043", np.nan, np.nan, np.nan, "444", "Form sent out", False, True, 77.0, 66.0, 9.0, "both"],
            ["1044", 36000.0, 30.0, 10.0, "444", "Clear", True, False, 36000.0, 30.0, 10.0, "both"],
            ["1045", 80500.0, 7000.0, 66.0, "444", "Clear", True, False, 80000.0, 0.0, 66.0, "both"],
            ["1046", 6000.0, 2500.0, 80.0, "444", "Clear", True, False, 5000.0, 3000.0, 100.0, "both"],
            ["1047", 0.0, 0.0, 0.0, "444", "Clear", True, False, 0.0, 0.0, 0.0, "both"],
            ["1048", 444.0, 330.0, 9.0, "555", "Clear", True, False, 444.0, 333.0, 22.0, "both"],
            ["1049", 400.0, 20.0, np.nan, "555", "Clear", True, False, 400.0, 20.0, 12.0, "both"],
            ["1050", 200.0, 10.0, 10.0, "555", "Clear", True, False, 230.0, 12.0, 11.0, "both"],
            ["2060", np.nan, np.nan, np.nan, "555", "Form sent out", False, True, 600.0, 500.0, 30.0, "both"],
            ["2061", np.nan, np.nan, np.nan, "555", "Form sent out", False, True, np.nan, np.nan, np.nan, "left_only"],
        ]

        df = pd.DataFrame(columns=columns, data=data)
        df['_merge'] = pd.Categorical(df['_merge'], categories=['left_only', 'right_only', 'both'])
        return df

    def test_join_current_backdata(self, pre_processing_expected, backdata, joined_expected): # Please construct your function
        """General tests for join_current_backdata."""
        output = join_current_backdata(pre_processing_expected, backdata, ru_col="reference", imp_class_col="sic")
        assert_frame_equal(output, joined_expected, check_dtype=False)

# # pytestmark = pytest.mark.runwip
# class TestCarryForwards(object):


# class Test_calculate_growth_rates(object):
#     """Tests for calculate_growth_rates."""
#     def target_vars_list(self):
#         """A simple method that returns a list."""
#         return ["211", "emp_researcher", "emp_technician"]






#     def create_test_CGR_expected_df(self):
#         """Create an test_CGR_expected dataframe for the test."""
#         test_CGR_expected_columns = [
#         "reference",
#         "imp_class",
#         "211",
#         "emp_researcher",
#         "emp_technician",
#         "211_prev",
#         "emp_researcher_prev",
#         "emp_technician_prev",
#         "211_gr",
#         "emp_researcher_gr",
#         "emp_technician_gr",
#     ]

#         data = [
#         [1031, "C_AA", 20, 10, 10.0, 0.0, 10.0, 0.0, np.nan, 1.0, np.nan],
#         [1031, "D_AA", 10, 0, 10.0, 10.0, 10.0, 20.0, 1.0, np.nan, 0.5],
#         [1045, "C_AH", 80500, 20, 0.0, 10000.0, 0.0, 10.0, 8.05, np.nan, np.nan],
#         [1045, "D_AH", 36000, 30, 10.0, 10000.0, 10.0, 10.0, 3.6, 3.0, 1.0],
#         [1047, "C_BC", 400, 20, 0.0, 400.0, 20.0, 0.0, 1.0, 1.0, np.nan],
#         [1047, "D_BC", 200, 10, 10.0, 200.0, 10.0, 10.0, 1.0, 1.0, 1.0],
#     ]

#         test_CGR_expected_df = pd.DataFrame(data=data, columns=test_CGR_expected_columns)
#         return test_CGR_expected_df


#     def test_calculate_growth_rates(self):
#         """Test the calculate_growth_rates function."""
#         current_df = self.create_test_CGR_current_df()
#         backdata_df = self.create_test_CGR_backdata_df()
#         expected_df = self.create_test_CGR_expected_df()
#         target_vars = self.target_vars_list()

#         result_df = calculate_growth_rates(current_df, backdata_df, target_vars)

#         assert_frame_equal(result_df, expected_df, check_dtype=False, check_exact=False)


# class TestGroupCalcLink(object):
#     """Tests for the group_calc_links function."""
#     def create_input_df(self) -> pd.DataFrame:
#         """A dummy dataframe used for testing group_calc_links function."""
#         columns = [
#             "reference",
#             "imp_class",
#             "211",
#             "emp_researcher",
#             "211_gr",
#             "emp_researcher_gr",
#         ]
#         data = [
#             [1031, "C_AA", 20, 10, np.nan, 1.0],
#             [1031, "C_AA", 10, 0, 1.0, np.nan],
#             [1045, "C_AA", 80500, 20, 8.05, np.nan],
#             [1045, "C_AA", 36000, 30, 3.6, 3.0],
#             [1047, "C_AA", 400, 20, 1.0, 1.0],
#             [1047, "C_AA", 200, 10, 1.0, 1.0]]

#         input_df = pd.DataFrame(data=data, columns=columns)
#         return input_df

#     def dummy_config(self) -> dict:
#         """A dummy config for testing."""
#         config = {"imputation": {
#             "mor_threshold": 3,
#             "trim_threshold": 10,
#             "lower_trim_perc": 15,
#             "upper_trim_perc": 15,
#             "target_vars": ["211","emp_researcher"]},
#         }
#         return config

#     def expected_output_df(self) -> pd.DataFrame:
#         """Expected dataframe after running group_calc_links function.
#             'group_size' is calculated by the sum of valid values in the column.
#             'link' is calculated by the mean growth rate of the column.
#             'trim' is specified conditions in the config."""
#         columns = [
#             "reference",
#             "imp_class",
#             "211",
#             "emp_researcher",
#             "211_gr",
#             "emp_researcher_gr",
#             "211_gr_trim",
#             "211_group_size",
#             "211_link",
#             "emp_researcher_gr_trim",
#             "emp_researcher_group_size",
#             "emp_researcher_link",
#         ]
#         # Data has been sorted by growth rate (emp_researcher_gr) in descending order
#         data = [
#             [1047, "C_AA", 400, 20, 1.0, 1.0, False, 5, 2.93, False, 4, 1.5],
#             [1047, "C_AA", 200, 10, 1.0, 1.0, False, 5, 2.93, False, 4, 1.5],
#             [1031, "C_AA", 20, 10, np.nan, 1.0, False, 5, 2.93, False, 4, 1.5],
#             [1045, "C_AA", 36000, 30, 3.6, 3.0, False, 5, 2.93, False, 4, 1.5],
#             [1031, "C_AA", 10, 0, 1.0, np.nan, False, 5, 2.93, False, 4, 1.5],
#             [1045, "C_AA", 80500, 20, 8.05, np.nan, False, 5, 2.93, False, 4, 1.5],
#             ]

#         expected_output_df = pd.DataFrame(data=data, columns=columns)
#         return expected_output_df


#     def test_group_calc_link(self):
#         # Create the input and expected output dataframes
#         input_df = self.create_input_df()
#         config = self.dummy_config()
#         expected_output_df = self.expected_output_df()
#         target_vars = config["imputation"]["target_vars"]

#         # Run the function
#         result_df = group_calc_link(input_df, target_vars, config)

#         # Reset index for comparison
#         df_list = [expected_output_df, result_df]

#         for df in df_list:
#             df.reset_index(drop=True, inplace=True)

#         # Compare the results
#         assert_frame_equal(result_df, expected_output_df, check_dtype=False, check_exact=False), (
#             "group_calc_links() not calculating links as expected."
#         )
# class TestCalculateLinks(object):
#     """Tests to check the function is ordering the data correctly"""

#     def config_dict(self):
#         """Dummy config for testing."""
#         config = {"imputation": {
#             "mor_threshold": 3,
#             "trim_threshold": 10,
#             "lower_trim_perc": 15,
#             "upper_trim_perc": 15,
#             "target_vars": ["211", "emp_researcher", "emp_technician"]},
#         }
#         return config

#     def create_input_df(self) -> pd.DataFrame:
#         """A dummy dataframe used for testing calculate_links function."""
#         columns = [
#             "reference",
#             "imp_class",
#             "211",
#             "emp_researcher",
#             "emp_technician",
#             "211_prev",
#             "emp_researcher_prev",
#             "emp_technician_prev",
#             "211_gr",
#             "emp_researcher_gr",
#             "emp_technician_gr",
#         ]

#         data = [
#             [1031, "C_AA", 20, 10, 10.0, 0.0, 10.0, 0.0, np.nan, 1.0, np.nan],
#             [1031, "D_AA", 10, 0, 10.0, 10.0, 10.0, 20.0, 1.0, np.nan, 0.5],
#             [1045, "C_AH", 80500, 20, 0.0, 10000.0, 0.0, 10.0, 8.05, np.nan, np.nan],
#             [1045, "D_AH", 36000, 30, 10.0, 10000.0, 10.0, 10.0, 3.6, 3.0, 1.0],
#             [1047, "C_BC", 400, 20, 0.0, 400.0, 20.0, 0.0, 1.0, 1.0, np.nan],
#             [1047, "D_BC", 200, 10, 10.0, 200.0, 10.0, 10.0, 1.0, 1.0, 1.0],
#         ]
#         input_df = pd.DataFrame(data, columns=columns)
#         return input_df

#     def expected_output(self) -> pd.DataFrame:
#         """Expected dataframe if 'is_current' is set to false.
#        Returns filtered data of both previous and current period data"""
#         columns = [
#             "imp_class",
#             "reference",
#             "211",
#             "211_prev",
#             "211_group_size",
#             "211_gr",
#             "211_gr_trim",
#             "211_link",
#             "emp_researcher",
#             "emp_researcher_prev",
#             "emp_researcher_group_size",
#             "emp_researcher_gr",
#             "emp_researcher_gr_trim",
#             "emp_researcher_link",
#             "emp_technician",
#             "emp_technician_prev",
#             "emp_technician_group_size",
#             "emp_technician_gr",
#             "emp_technician_gr_trim",
#             "emp_technician_link",
#         ]

#         data = [
#             ["C_AA", 1031, 20, 0.0, 0, np.nan, False, 1.0, 10, 10.0, 1, 1.0, False, 1.0, 10.0, 0.0, 0, np.nan, False, 1.0],
#             ["D_AA", 1031, 10, 10.0, 1, 1.0, False, 1.0, 0, 10.0, 0, np.nan, False, 1.0, 10.0, 20.0, 1, 0.5, False, 1.0],
#             ["C_AH", 1045, 80500, 10000.0, 1, 8.05, False, 1.0, 20, 0.0, 0, np.nan, False, 1.0, 0.0, 10.0, 0, np.nan, False, 1.0],
#             ["D_AH", 1045, 36000, 10000.0, 1, 3.6, False, 1.0, 30, 10.0, 1, 3.0, False, 1.0, 10.0, 10.0, 1, 1.0, False, 1.0],
#             ["C_BC", 1047, 400, 400.0, 1, 1.0, False, 1.0, 20, 20.0, 1, 1.0, False, 1.0, 0.0, 0.0, 0, np.nan, False, 1.0],
#             ["D_BC", 1047, 200, 200.0, 1, 1.0, False, 1.0, 10, 10.0, 1, 1.0, False, 1.0, 10.0, 10.0, 1, 1.0, False, 1.0],
#         ]

#         exp_df = pd.DataFrame(data, columns=columns)
#         return exp_df

#     def test_calculate_links(self):
#         # Create the input and expected output dataframes
#         input_df = self.create_input_df()
#         exp_df = self.expected_output()
#         config = self.config_dict()
#         target_vars = config["imputation"]["target_vars"]

#         # Run the function
#         result = calculate_links(input_df, target_vars, config)

#         # Compare the results
#         assert_frame_equal(result, exp_df, check_dtype=False, check_exact=False), (
#             "calculate_links() not ordering data as expected."
#         )
