"""Tests for functions in calculate_weights."""

import pytest
from pandas.testing import assert_frame_equal
from bsrm.utils.helpers import create_test_dataframe
from bsrm.estimation.calculate_weights import (
    calc_lower_n,
    a_weight,
    g_weight,
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


@pytest.fixture(scope="function")
def input_data():
    """Small sample data used across calculate_weights tests."""
    df = create_test_dataframe([
        ("ruref", "a_wt_band", "g_wt_band", "question", "value", "aux_var", "aux_var_sum", "N", "g_wgt_type"),
        ("A",     1,           1,           "Q1",       100,     10,        20,            6,   "x"),
        ("B",     1,           1,           "Q1",       100,     20,        20,            6,   "x"),
        ("C",     1,           1,           "Q1",       100,     30,        20,            6,   "x"),
        ("D",     2,           2,           "Q1",       100,     15,        18,            6,   "x"),
        ("E",     2,           2,           "Q1",       100,     25,        18,            6,   "x"),
        ("A",     1,           1,           "Q2",       200,     2,         20,            6,   "y"),
        ("B",     1,           1,           "Q2",       200,     4,         20,            6,   "y"),
        ("C",     1,           1,           "Q2",       200,     0,         20,            6,   "y"),
        ("D",     2,           2,           "Q2",       200,     10,        18,            6,   "y"),
        ("E",     2,           2,           "Q2",       200,     20,        18,            6,   "y"),
    ])
    return df


@pytest.fixture(scope="function")
def expected_a_weights_df():
    """Expected output after calculating a_weights."""
    df = create_test_dataframe([
        ("ruref", "a_wt_band", "g_wt_band", "question", "value", "aux_var", "aux_var_sum", "N", "g_wgt_type", "n", "a_weight"),
        ("A",     1,           1,           "Q1",       100,     10,        20,            6,   "x",          3,   2.0),
        ("B",     1,           1,           "Q1",       100,     20,        20,            6,   "x",          3,   2.0),
        ("C",     1,           1,           "Q1",       100,     30,        20,            6,   "x",          3,   2.0),
        ("D",     2,           2,           "Q1",       100,     15,        18,            6,   "x",          2,   3.0),
        ("E",     2,           2,           "Q1",       100,     25,        18,            6,   "x",          2,   3.0),
        ("A",     1,           1,           "Q2",       200,     2,         20,            6,   "y",          3,   2.0),
        ("B",     1,           1,           "Q2",       200,     4,         20,            6,   "y",          3,   2.0),
        ("C",     1,           1,           "Q2",       200,     0,         20,            6,   "y",          3,   2.0),
        ("D",     2,           2,           "Q2",       200,     10,        18,            6,   "y",          2,   3.0),
        ("E",     2,           2,           "Q2",       200,     20,        18,            6,   "y",          2,   3.0),
    ])
    return df


@pytest.fixture(scope="function")
def expected_g_weights_df():
    """Expected output after calculating g_weights."""
    df = create_test_dataframe([
        ("ruref", "a_wt_band", "g_wt_band", "question", "value", "aux_var", "aux_var_sum", "N", "g_wgt_type", "n", "a_weight", "g_weight"),
        ("A",     1,           1,           "Q1",       100,     10,        60,            6,   "x",          3,   2.0,        0.167),
        ("B",     1,           1,           "Q1",       100,     20,        60,            6,   "x",          3,   2.0,        0.167),
        ("C",     1,           1,           "Q1",       100,     30,        60,            6,   "x",          3,   2.0,        0.167),
        ("D",     2,           2,           "Q1",       100,     15,        40,            6,   "x",          2,   3.0,        0.15),
        ("E",     2,           2,           "Q1",       100,     25,        40,            6,   "x",          2,   3.0,        0.15),
        ("A",     1,           1,           "Q2",       200,     2,         6,             6,   "y",          3,   2.0,        1.667),
        ("B",     1,           1,           "Q2",       200,     4,         6,             6,   "y",          3,   2.0,        1.667),
        ("C",     1,           1,           "Q2",       200,     0,         6,             6,   "y",          3,   2.0,        1.667),
        ("D",     2,           2,           "Q2",       200,     10,        30,            6,   "y",          2,   3.0,        0.2),
        ("E",     2,           2,           "Q2",       200,     20,        30,            6,   "y",          2,   3.0,        0.2),
    ])
    return df


@pytest.fixture(scope="function")
def expected_qa_df():
    """Expected QA dataframe output."""
    df = create_test_dataframe([
        ("a_wt_band", "g_wt_band", "N", "n", "a_weight", "g_weight_x", "g_weight_y"),
        (1,           1,           6,   3,   2.0,        0.167,        1.667),
        (2,           2,           6,   2,   3.0,        0.15,         0.2),
    ])
    return df


def test_calc_lower_n(input_data):
    """Test for lower n (Sample) calculation."""
    band_1 = input_data[input_data["a_wt_band"] == 1]
    result = calc_lower_n(band_1, "ruref")
    assert result == 3


def test_a_weight(input_data):
    """Test for a weight calculation."""
    band_1 = input_data[input_data["a_wt_band"] == 1].copy()
    band_1.name = 1
    result = a_weight(band_1, "ruref", "N")
    assert result["a_weight"].iloc[0] == 2.0


def test_a_weight_with_a_weight_col(input_data):
    """Test that a_wgt_band_col is preserved when passed."""
    band_1 = input_data[input_data["a_wt_band"] == 1].copy()
    band_1.name = 1
    result = a_weight(band_1, "ruref", "N")
    assert result["a_weight"].iloc[0] == 2.0
    assert result["a_wt_band"].iloc[0] == 1


def test_calculate_a_weights(input_data, expected_a_weights_df):
    """Test that the a weights are calculated correctly."""
    result = calculate_a_weights(
        df=input_data,
        a_wgt_band_col="a_wt_band",
        ru_col="ruref",
        univ_count_col="N",
        question_col="question",
    )
    assert_frame_equal(result, expected_a_weights_df, check_like=True, check_dtype=False, rtol=1e-6)


def test_calculate_g_weights(expected_a_weights_df, expected_g_weights_df):
    """Test that the g weights are calculated correctly."""
    result = calculate_g_weights(
        expected_a_weights_df, "g_wt_band", "aux_var", "aux_var_sum", "question"
    )
    assert_frame_equal(result.round(3), expected_g_weights_df, check_like=True, check_dtype=False, rtol=1e-6)


def test_create_weights_qa_df(expected_g_weights_df, expected_qa_df):
    """Test that the QA dataframe is created correctly."""
    result = create_weights_qa_df(
        expected_g_weights_df,
        univ_count_col="N",
        a_wgt_band_col="a_wt_band",
        g_wgt_band_col="g_wt_band",
        incl_g_wts=True,
        g_wgt_type_col="g_wgt_type",
    )
    assert_frame_equal(result.round(3), expected_qa_df, check_dtype=False, rtol=1e-6)
