
import pytest
from pandas.testing import assert_frame_equal
from bsrm.utils.helpers import create_test_dataframe

import pandas as pd

from bsrm.imputation.imputation_methods import (
    impute_capex
)


@pytest.fixture
def input_df():
    """Create dataframe with data to be imputed."""
    return create_test_dataframe(
        [
            # Growth rates should be ~0.8 and ~1.2 but not exactly
            ("ruref", "imp_class", "turnover", "empment", "Q311", "Q399", "Q600", "Q601"),
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
            ("ruref", "imp_class", "turnover", "empment", "Q311", "Q399", "Q600", "Q601", "Q311_growth_factor", "Q601_growth_factor", "Q311_median_growth_factor", "Q601_median_growth_factor", "Q311_imputed_estimate", "Q601_imputed_estimate"),
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
    input_class1 = input_df[input_df["imp_class"] == "class1"].copy()
    result_class1 = impute_capex(input_class1, sector="acp")

    input_class2 = input_df[input_df["imp_class"] == "class2"].copy()
    result_class2 = impute_capex(input_class2, sector="acp")

    result = pd.concat([result_class1, result_class2], axis="index").reset_index()

    # Reorder the columns since the output of the function doesn't preserve column order
    result = result[expected_df.columns]

    with pd.option_context('display.max_columns', None):
        print("===EXPECTED===")
        print(expected_df)
        print("====RESULT====")
        print(result)

    assert_frame_equal(result, expected_df)
