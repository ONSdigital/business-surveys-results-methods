"""Tests for functions in apply_weights."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from bsrm.estimation.apply_weights import apply_weights


@pytest.fixture
def input_df():
    """Small sample data used across apply_weights tests."""
    cols = ["ruref", "cell_no", "question", "value", "a_weight", "g_weight_turnover"]
    data = [
        ["A", 1, "Q1", 10.0, 2.0, 3.3333],
        ["B", 1, "Q1", 20.0, 2.0, 3.3333],
        ["C", 1, "Q1", 30.0, 2.0, 3.3333],
        ["D", 2, "Q1", 15.0, 2.0, 1.0],
        ["E", 2, "Q1", 25.0, 2.0, 1.0],
        ["A", 1, "Q2", 100.0, 2.0, 3.3333],
        ["B", 1, "Q2", 200.0, 2.0, 3.3333],
        ["C", 1, "Q2", 300.0, 2.0, 3.3333],
        ["D", 2, "Q2", 150.0, 2.0, 1.0],
        ["E", 2, "Q2", 250.0, 2.0, 1.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


@pytest.fixture
def expected_a_and_g_df():
    """Expected output after applying both a_weight and g_weight to y.

    for ruref A (cell_no=1):
       input:    y=10.0, a_weight=2.0, g_weight=3.3333
       y_new = y * a_weight * g_weight
              = 10.0  * 2.0      * 3.3333
              = 66.666 (rounded to 4dp)
    """
    cols = ["ruref", "cell_no", "question", "value", "a_weight", "g_weight_turnover", "value_grossed"]
    data = [
        ["A", 1, "Q1", 10.0, 2.0, 3.3333, 66.666],
        ["B", 1, "Q1", 20.0, 2.0, 3.3333, 133.332],
        ["C", 1, "Q1", 30.0, 2.0, 3.3333, 199.998],
        ["D", 2, "Q1", 15.0, 2.0, 1.0, 30.0],
        ["E", 2, "Q1", 25.0, 2.0, 1.0, 50.0],
        ["A", 1, "Q2", 100.0, 2.0, 3.3333, 666.66],
        ["B", 1, "Q2", 200.0, 2.0, 3.3333, 1333.32],
        ["C", 1, "Q2", 300.0, 2.0, 3.3333, 1999.98],
        ["D", 2, "Q2", 150.0, 2.0, 1.0, 300.0],
        ["E", 2, "Q2", 250.0, 2.0, 1.0, 500.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


@pytest.fixture
def expected_a_only_df():
    """Expected output after applying only a_weight to the value column.

    Example for ruref A (cell_no=1):
        input:    y=10.0, a_weight=2.0, g_weight not applied
        y_new = y * a_weight
               = 10.0  * 2.0
               = 20.0 (rounded to 4dp)
    """
    cols = ["ruref", "cell_no", "question", "value", "a_weight", "value_grossed"]

    data = [
        ["A", 1, "Q1", 10.0, 2.0, 20.0],
        ["B", 1, "Q1", 20.0, 2.0, 40.0],
        ["C", 1, "Q1", 30.0, 2.0, 60.0],
        ["D", 2, "Q1", 15.0, 2.0, 30.0],
        ["E", 2, "Q1", 25.0, 2.0, 50.0],
        ["A", 1, "Q2", 100.0, 2.0, 200.0],
        ["B", 1, "Q2", 200.0, 2.0, 400.0],
        ["C", 1, "Q2", 300.0, 2.0, 600.0],
        ["D", 2, "Q2", 150.0, 2.0, 300.0],
        ["E", 2, "Q2", 250.0, 2.0, 500.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


def test_apply_a_and_g_weights(input_df, expected_a_and_g_df):
    """Test applying both a_weight and g_weight to y."""
    result = apply_weights(
        df=input_df.copy(),
        a_weight_columns=["Q1", "Q2"],
        g_weight_columns={"turnover": ["Q1", "Q2"]},
        aux_cols=["turnover"],
        calc_g_weight=True,
        round_val=4,
    )
    assert_frame_equal(result, expected_a_and_g_df, check_dtype=False, atol=1e-4)


def test_apply_a_weight_only(input_df, expected_a_only_df):
    """Test applying only a_weight when calc_g_weight is False."""
    result = apply_weights(
        df=input_df.copy().drop(columns=["g_weight_turnover"]),
        a_weight_columns=["Q1", "Q2"],
        calc_g_weight=False,
        round_val=4,
    )
    assert_frame_equal(result, expected_a_only_df, check_dtype=False, atol=1e-4)


def test_apply_weights_invalid_column(input_df):
    """Raise an error for a column that does not exist."""
    with pytest.raises(
        Exception,
        match=r"Specified column\(s\): invalid_column must exist in the question column.",
    ):
        apply_weights(
            df=input_df.copy(),
            a_weight_columns=["invalid_column"],
            calc_g_weight=False,
            round_val=4,
        )
