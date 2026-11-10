
import pytest
from pandas.testing import assert_frame_equal
from bsrm.utils.helpers import create_test_dataframe

import pandas as pd

from bsrm.imputation.imputation_methods import (
    impute_capex,
    impute_missing_value
)


@pytest.fixture
def input_df():
    """Create dataframe with data to be imputed."""
    return create_test_dataframe(
        [
            # Growth rates should be ~0.8 and ~1.2 but not exactly
            ("ruref", "cell_no", "turnover", "empment", "Q311", "Q399", "Q600", "Q601"),
            (1,          "class1",          0,       101,     98,    123,    100,  None),
            (2,          "class1",          0,       101,    365,    456,    118,   147),
            (3,          "class1",          0,       101,    631,    789,    206,   258),
            (4,          "class1",          0,       101,   None,    100,    295,   369),

            # Growth rates should be ~0.75 and ~1.333 but not exactly
            (5,          "class2",          0,       101,      0,      0,    100,  None),
            (6,          "class2",          0,       101,    926,   1234,   5559,  7412),
            (7,          "class2",          0,       101,   4109,   5478,   6392,  8523),
            (8,          "class2",          0,       101,   None,    100,   7223,  9631),
        ]
    )


@pytest.fixture
def expected_df():
    """Create dataframe with data that has been imputed."""
    return create_test_dataframe(
        [
            # Growth rates should be ~0.8 and ~1.2 but not exactly
            ("ruref", "cell_no", "turnover", "empment", "Q311", "Q399", "Q600", "Q601", "Q311_growth_factor", "Q601_growth_factor", "Q311_median_growth_factor", "Q601_median_growth_factor", "Q311_imputed_estimate", "Q601_imputed_estimate"),
            (1,          "class1",          0,       101,     98,    123,    100,  None, 0.796747967479674796,                 None,        0.799746514575411913,        1.250847457627118644,                    None,    125.0847457627118644),
            (2,          "class1",          0,       101,    365,    456,    118,   147, 0.800438596491228070, 1.245762711864406779,        0.799746514575411913,        1.250847457627118644,                    None,                    None),
            (3,          "class1",          0,       101,    631,    789,    206,   258, 0.799746514575411913, 1.252427184466019417,        0.799746514575411913,        1.250847457627118644,                    None,                    None),
            (4,          "class1",          0,       101,   None,    100,    295,   369,                 None, 1.250847457627118644,        0.799746514575411913,        1.250847457627118644,     79.9746514575411913,                    None),

            # Growth rates should be ~0.75 and ~1.333 but not exactly
            # Note that the raw Q311 median is 0.749032671129486266 but ruref 5 is 0, so the adjusted mean is 2/3 that
            (5,          "class2",          0,       101,      0,      0,    100,  None,                 None,                 None,        0.500165486857799056,        1.333379482209608196,                    None,    133.3379482209608196),
            (6,          "class2",          0,       101,    926,   1234,   5559,  7412, 0.750405186385737439, 1.333333333333333333,        0.500165486857799056,        1.333379482209608196,                    None,                    None),
            (7,          "class2",          0,       101,   4109,   5478,   6392,  8523, 0.750091274187659729, 1.333385481852315394,        0.500165486857799056,        1.333379482209608196,                    None,                    None),
            (8,          "class2",          0,       101,   None,    100,   7223,  9631,                 None, 1.333379482209608196,        0.500165486857799056,        1.333379482209608196,     50.0165486857799056,                    None),
        ]
    )


def test_impute_capex(
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Test the CAPEX imputation function."""
    # The function is intended take an imputation class, not the whole dataframe, so we split the
    # input by imputation class to run them through separately, then concatenate after
    input_class1 = input_df[input_df["cell_no"] == "class1"].copy()
    result_class1 = impute_capex(input_class1, sector="acp")

    input_class2 = input_df[input_df["cell_no"] == "class2"].copy()
    result_class2 = impute_capex(input_class2, sector="acp")

    result = pd.concat([result_class1, result_class2], axis="index").reset_index()

    # Reorder the columns since the output of the function doesn't preserve column order
    result = result[expected_df.columns]

    assert_frame_equal(result, expected_df)


@pytest.fixture
def input_prev_df():
    """Create dataframe with data from previous year to be imputed."""
    return create_test_dataframe(
        [
            ("ruref",   "cell_no", "question", "value", "prev_value",),
            (    "1",           1,       "Q1",     123,           83,),
            (    "1",           1,       "Q2",     456,          356,),
            (    "1",           1,       "Q3",     789,          689,),

            (    "2",           1,       "Q1",    None,          887,),
            (    "2",           1,       "Q2",     654,          554,),
            (    "2",           1,       "Q3",     321,          121,),

            (    "3",           1,       "Q1",     147,           47,),
            (    "3",           1,       "Q2",    None,          158,),
            (    "3",           1,       "Q3",     369,          269,),

            (    "4",           1,       "Q1",     147,           47,),
            (    "4",           1,       "Q2",     753,          653,),
            (    "4",           1,       "Q3",    None,          169,),


            (    "5",           2,       "Q1",     789,          689,),
            (    "5",           2,       "Q2",     951,          851,),
            (    "5",           2,       "Q3",     123,           23,),

            (    "6",           2,       "Q1",    None,          641,),
            (    "6",           2,       "Q2",     852,          752,),
            (    "6",           2,       "Q3",     963,          863,),

            (    "7",           2,       "Q1",     321,          121,),
            (    "7",           2,       "Q2",    None,          554,),
            (    "7",           2,       "Q3",     987,          887,),

            (    "8",           2,       "Q1",     789,          689,),
            (    "8",           2,       "Q2",     456,          356,),
            (    "8",           2,       "Q3",    None,           23,),

            # This group does not have enough members with both periods for valid imputation so
            # imputed_value should be None
            (    "9",           3,       "Q1",     123,          321,),
            (   "10",           3,       "Q1",     456,          654,),
            (   "11",           3,       "Q1",    None,          987,),
        ]
    )


@pytest.fixture
def expected_prev_df():
    """Create dataframe with data imputed from previous year values."""
    return create_test_dataframe(
        [
            ("ruref",   "cell_no", "question", "value", "prev_value", "imputed_value",),
            (    "1",           1,       "Q1",     123,           83,            None,),
            (    "1",           1,       "Q2",     456,          356,            None,),
            (    "1",           1,       "Q3",     789,          689,            None,),

            (    "2",           1,       "Q1",    None,          887,     2774.234043,),
            (    "2",           1,       "Q2",     654,          554,            None,),
            (    "2",           1,       "Q3",     321,          121,            None,),

            (    "3",           1,       "Q1",     147,           47,            None,),
            (    "3",           1,       "Q2",    None,          158,     186.5198556,),
            (    "3",           1,       "Q3",     369,          269,            None,),

            (    "4",           1,       "Q1",     147,           47,            None,),
            (    "4",           1,       "Q2",     753,          653,            None,),
            (    "4",           1,       "Q3",    None,          169,     231.8252788,),


            (    "5",           2,       "Q1",     789,          689,            None,),
            (    "5",           2,       "Q2",     951,          851,            None,),
            (    "5",           2,       "Q3",     123,           23,            None,),

            (    "6",           2,       "Q1",    None,          641,     734.0333817,),
            (    "6",           2,       "Q2",     852,          752,            None,),
            (    "6",           2,       "Q3",     963,          863,            None,),

            (    "7",           2,       "Q1",     321,          121,            None,),
            (    "7",           2,       "Q2",    None,          554,     627.6702128,),
            (    "7",           2,       "Q3",     987,          887,            None,),

            (    "8",           2,       "Q1",     789,          689,            None,),
            (    "8",           2,       "Q2",     456,          356,            None,),
            (    "8",           2,       "Q3",    None,           23,     25.66512167,),

            (    "9",           3,       "Q1",     123,          321,            None,),
            (   "10",           3,       "Q1",     456,          654,            None,),
            (   "11",           3,       "Q1",    None,          987,            None,),
        ]
    )

def test_impute_prev(
    input_prev_df: pd.DataFrame,
    expected_prev_df: pd.DataFrame,
):
    """Test the previous year imputation function."""

    result_df = impute_missing_value(
        df=input_prev_df.copy(),
        current_value="value",
        prev_value="prev_value",
        adjusted = False,
        keep_qa_cols = False,
    )

    assert_frame_equal(result_df, expected_prev_df)
