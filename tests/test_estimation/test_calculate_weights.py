"""Tests for functions in calculate_weights."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from bsrm.estimation.calculate_weights import (
    calc_lower_n,
    a_weight,
    g_weight,
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


@pytest.fixture
def input_data():
    """Small sample data used across calculate_weights tests."""
    col_name = ["ruref", "cell_no", "k", "n", "nk", "y", "x", "N", "Nk", "sum_x"]
    data = [
        ["A", 1, 1, 3, 5, 10.0, 2.0, 6, 20, 40.0],
        ["B", 1, 1, 3, 5, 20.0, 4.0, 6, 20, 40.0],
        ["C", 1, 1, 3, 5, 30.0, 0.0, 6, 20, 40.0],
        ["D", 2, 2, 2, 4, 15.0, 10.0, 6, 18, 60.0],
        ["E", 2, 2, 2, 4, 25.0, 20.0, 6, 18, 60.0],
    ]
    return pd.DataFrame(columns=col_name, data=data)


@pytest.fixture
def expected_a_weights_df():
    """Expected output after calculating a_weights."""
    cols = [
        "ruref",
        "cell_no",
        "k",
        "n",
        "nk",
        "y",
        "x",
        "N",
        "Nk",
        "sum_x",
        "a_weight",
    ]
    data = [
        ["A", 1, 1, 3, 5, 10.0, 2.0, 6, 20, 40.0, 2.0],
        ["B", 1, 1, 3, 5, 20.0, 4.0, 6, 20, 40.0, 2.0],
        ["C", 1, 1, 3, 5, 30.0, 0.0, 6, 20, 40.0, 2.0],
        ["D", 2, 2, 2, 4, 15.0, 10.0, 6, 18, 60.0, 3.0],
        ["E", 2, 2, 2, 4, 25.0, 20.0, 6, 18, 60.0, 3.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


@pytest.fixture
def expected_g_weights_df():
    """Expected output after calculating g_weights."""
    cols = [
        "ruref",
        "cell_no",
        "k",
        "n",
        "nk",
        "y",
        "x",
        "N",
        "Nk",
        "sum_x",
        "a_weight",
        "g_weight",
        "univ_aux_sum",
        "aux_col_sum",
    ]
    data = [
        ["A", 1, 1, 3, 5, 10.0, 2.0, 6, 20, 40.0, 2.0, 3.333, 40.0, 6.0],
        ["B", 1, 1, 3, 5, 20.0, 4.0, 6, 20, 40.0, 2.0, 3.333, 40.0, 6.0],
        ["C", 1, 1, 3, 5, 30.0, 0.0, 6, 20, 40.0, 2.0, 3.333, 40.0, 6.0],
        ["D", 2, 2, 2, 4, 15.0, 10.0, 6, 18, 60.0, 3.0, 0.667, 60.0, 30.0],
        ["E", 2, 2, 2, 4, 25.0, 20.0, 6, 18, 60.0, 3.0, 0.667, 60.0, 30.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


@pytest.fixture
def expected_qa_df():
    """Expected QA dataframe output."""
    cols = [
        "cell_no",
        "N",
        "n",
        "a_weight",
        "univ_aux_sum",
        "aux_col_sum",
        "g_weight",
    ]
    data = [
        [1, 6, 3, 2.0, 40.0, 6.0, 3.333],
        [2, 6, 2, 3.0, 60.0, 30.0, 0.667],
    ]
    return pd.DataFrame(data=data, columns=cols)


def test_calc_lower_n(input_data):
    """Test for lower n (Sample) calculation."""
    cell_1 = input_data[input_data["cell_no"] == 1]
    result = calc_lower_n(cell_1, "ruref")
    assert result == 3


def test_a_weight(input_data):
    """Test for a weight calculation."""
    cell_1 = input_data[input_data["cell_no"] == 1]
    result = a_weight(cell_1, "ruref", "N")
    assert result["a_weight"].iloc[0] == 2.0


def test_g_weight(expected_a_weights_df):
    """Test for g weight calculation."""
    cell_1 = expected_a_weights_df[expected_a_weights_df["cell_no"] == 1]
    result = g_weight(cell_1, "x", "sum_x")
    assert result["g_weight"].iloc[0] == pytest.approx(40.0 / 12.0, abs=1e-4)


def test_calculate_a_weights(input_data, expected_a_weights_df):
    """Test that the a weights are calculated correctly."""
    result = calculate_a_weights(df=input_data, strata_col="cell_no", ru_col="ruref", univ_count_col="N")
    assert_frame_equal(result, expected_a_weights_df, check_dtype=False, rtol=1e-6)


def test_calculate_g_weights(expected_a_weights_df, expected_g_weights_df):
    """Test that the g weights are calculated correctly."""
    result = calculate_g_weights(expected_a_weights_df, "k", "x", "sum_x")
    assert_frame_equal(result.round(3), expected_g_weights_df, check_dtype=False, rtol=1e-6)


def test_create_weights_qa_df(expected_g_weights_df, expected_qa_df):
    """Test that the QA dataframe is created correctly."""
    result = create_weights_qa_df(expected_g_weights_df, "cell_no", incl_g_wts=True)
    assert_frame_equal(result.round(3), expected_qa_df, check_dtype=False, rtol=1e-6)
