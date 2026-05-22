"""Tests for functions in calculate_weights"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from bsrm.estimation.calculate_weights import (
    calc_aux_col_sum,
    calc_lower_n,
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


@pytest.fixture
def input_data() -> pd.DataFrame:
    """Input for calculate_weights tests."""
    return pd.DataFrame(
        {
            "ruref": [17, 1, 16, 8, 9, 13, 14, 19, 18, 26, 34, 29, 23, 25],
            "cell_no": [1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 7],
            "k": [1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 5],
            "nh": [3, 3, 3, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 3],
            "nk": [3, 3, 3, 4, 4, 4, 4, 1, 1, 5, 5, 5, 5, 5],
            "rusic2007": [
                11010,
                11010,
                11010,
                11010,
                11010,
                11010,
                11010,
                11010,
                11010,
                11020,
                11020,
                11020,
                11020,
                11020,
            ],
            "region": [
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
                "AA",
            ],
            "y": [
                10.2,
                30.0,
                30.6,
                265.2,
                306.0,
                1100.0,
                632.4,
                3111.0,
                10230.6,
                30.6,
                40.8,
                295.8,
                306.0,
                500.0,
            ],
            "x": [1, 2, 3, 26, 30, 48, 62, 305, 1003, 3, 4, 29, 30, 40],
            "Nh": [8, 8, 8, 9, 9, 9, 1, 1, 1, 4, 4, 10, 10, 10],
            "Nk": [8, 8, 8, 10, 10, 10, 10, 1, 1, 14, 14, 14, 14, 14],
            "sum_y": [
                416.4,
                416.4,
                416.4,
                4030.0,
                4030.0,
                4030.0,
                4030.0,
                3111.0,
                10230.6,
                2876.6,
                2876.6,
                2876.6,
                2876.6,
                2876.6,
            ],
            "sum_x": [
                40,
                40,
                40,
                278,
                278,
                278,
                278,
                305,
                1003,
                273,
                273,
                273,
                273,
                273,
            ],
        }
    )


@pytest.fixture
def expected_a_weights() -> pd.DataFrame:
    """Expected output for a_Weight value by cell"""
    return pd.DataFrame(
        {
            "cell_no": [1, 2, 3, 4, 5, 6, 7],
            "a_weight": [2.666667, 3.0, 1.0, 1.0, 1.0, 2.0, 3.333333],
        }
    )


@pytest.fixture
def expected_g_weights() -> pd.DataFrame:
    """Expected g_weight value by calibration group k"""
    return pd.DataFrame(
        {
            "k": [1, 2, 3, 4, 5],
            "g_weight": [2.5, 0.743316, 1.0, 1.0, 0.793605],
        }
    )


@pytest.fixture
def expected_qa_df() -> pd.DataFrame:
    """Expected output for QA dataframe."""
    return pd.DataFrame(
        {
            "k": [1, 2, 3, 4, 5],
            "N": [8, 9, 1, 1, 4],
            "n": [3, 3, 1, 1, 2],
            "univ_aux_sum": [40.0, 278.0, 305.0, 1003.0, 273.0],
            "aux_col_sum": [6.0, 166.0, 305.0, 1003.0, 106.0],
            "g_weight": [2.5, 0.743316, 1.0, 1.0, 0.793605],
        }
    )


def test_calculation_a_weights(
    input_data: pd.DataFrame, expected_a_weights: pd.DataFrame
) -> None:
    """Test a_weight =Nh/nh per stratum."""
    result = calculate_a_weights(
        input_data,
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
    )
    actual = result.groupby("cell_no")["a_weight"].first().reset_index()
    actual = actual.sort_values("cell_no").reset_index(drop=True)
    expected = expected_a_weights.sort_values("cell_no").reset_index(drop=True)
    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-6)


def test_calc_lower_n_counts_unique_ruref(input_data: pd.DataFrame) -> None:
    """Test calc_lower_n counts unique reporting units."""
    cell_one = input_data[input_data["cell_no"] == 1]
    actual = calc_lower_n(cell_one, "ruref")
    assert actual == 3


def test_calc_aux_col_sum(input_data: pd.DataFrame) -> None:
    """Test calc_aux_col_sum returns the sum of x in a subset."""
    k_two = input_data[input_data["k"] == 2]
    actual = calc_aux_col_sum(k_two, "x")
    assert actual == 166


def test_calculate_g_weights(
    input_data: pd.DataFrame, expected_g_weights: pd.DataFrame
) -> None:
    """Test g_weight = sum_x / sum(a * x) per calibration group k."""
    df_with_a = calculate_a_weights(
        input_data,
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
    )
    result = calculate_g_weights(
        df_with_a,
        strata_col="k",
        aux_col="x",
        univ_aux_col="sum_x",
    )

    actual = result.groupby("k")["g_weight"].first().reset_index()
    actual = actual.sort_values("k").reset_index(drop=True)
    expected = expected_g_weights.sort_values("k").reset_index(drop=True)

    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-5)


def test_g_weight_same_within_group(input_data: pd.DataFrame) -> None:
    """All rows in same k should have the same g_weight even with mixed a values."""
    df_with_a = calculate_a_weights(
        input_data, strata_col="cell_no", ru_col="ruref", univ_count_col="Nh"
    )
    result = calculate_g_weights(
        df_with_a, strata_col="k", aux_col="x", univ_aux_col="sum_x"
    )

    k2 = result[result["k"] == 2]
    assert k2["g_weight"].nunique() == 1, "All rows in k=2 should have same g_weight"
    assert k2["g_weight"].iloc[0] == pytest.approx(278 / 374, rel=1e-5)


def test_create_weights_qa_df(
    input_data: pd.DataFrame, expected_qa_df: pd.DataFrame
) -> None:
    """Test QA dataframe values with g weights included."""
    df_with_a = calculate_a_weights(
        input_data, strata_col="cell_no", ru_col="ruref", univ_count_col="Nh"
    )
    df_with_g = calculate_g_weights(
        df_with_a, strata_col="k", aux_col="x", univ_aux_col="sum_x"
    )
    qa = create_weights_qa_df(df_with_g, strata_col="k", incl_g_wts=True)

    actual = qa[["k", "N", "n", "univ_aux_sum", "aux_col_sum", "g_weight"]]
    actual = actual.sort_values("k").reset_index(drop=True)
    expected = expected_qa_df.sort_values("k").reset_index(drop=True)

    assert_frame_equal(
        actual, expected, check_exact=False, rtol=1e-5, check_dtype=False
    )


def test_create_weights_qa_df_no_g_weights(input_data: pd.DataFrame) -> None:
    """Test QA dataframe columns when g weights are not included."""
    df_with_a = calculate_a_weights(
        input_data, strata_col="cell_no", ru_col="ruref", univ_count_col="Nh"
    )
    qa = create_weights_qa_df(df_with_a, strata_col="cell_no", incl_g_wts=False)

    assert list(qa.columns) == ["cell_no", "N", "n", "a_weight"]
