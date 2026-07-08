"""Tests for functions in apply_weights."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from business_surveys_results_methods.estimation.apply_weights import apply_weights


@pytest.fixture
def input_df():
    """Small sample data used across apply_weights tests."""
    cols = ["ruref", "cell_no", "y", "a_weight", "g_weight"]
    data = [
        ["A", 1, 10.0, 2.0, 3.3333],
        ["B", 1, 20.0, 2.0, 3.3333],
        ["C", 1, 30.0, 2.0, 3.3333],
        ["D", 2, 15.0, 2.0, 1.0],
        ["E", 2, 25.0, 2.0, 1.0],
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
    cols = ["ruref", "cell_no", "y", "a_weight", "g_weight"]
    data = [
        ["A", 1, 66.666, 2.0, 3.3333],
        ["B", 1, 133.332, 2.0, 3.3333],
        ["C", 1, 199.998, 2.0, 3.3333],
        ["D", 2, 30.0, 2.0, 1.0],
        ["E", 2, 50.0, 2.0, 1.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


@pytest.fixture
def expected_a_only_df():
    """Expected output after applying only a_weight to y.

    Example for ruref A (cell_no=1):
        input:    y=10.0, a_weight=2.0, g_weight not applied
        y_new = y * a_weight
               = 10.0  * 2.0
               = 20.0 (rounded to 4dp)
    """
    cols = ["ruref", "cell_no", "y", "a_weight"]

    data = [
        ["A", 1, 20.0, 2.0],
        ["B", 1, 40.0, 2.0],
        ["C", 1, 60.0, 2.0],
        ["D", 2, 30.0, 2.0],
        ["E", 2, 50.0, 2.0],
    ]
    return pd.DataFrame(data=data, columns=cols)


def test_apply_a_and_g_weights(input_df, expected_a_and_g_df):
    """Test applying both a_weight and g_weight to y."""
    result = apply_weights(
        df=input_df.copy(),
        a_weight_cols=["y"],
        g_weight_cols=["y"],
        calc_g_weight=True,
        round_val=4,
    )
    assert_frame_equal(result, expected_a_and_g_df, check_dtype=False, atol=1e-4)


def test_apply_a_weight_only(input_df, expected_a_only_df):
    """Test applying only a_weight when calc_g_weight is False."""
    result = apply_weights(
        df=input_df.copy().drop(columns=["g_weight"]),
        a_weight_cols=["y"],
        calc_g_weight=False,
        round_val=4,
    )
    assert_frame_equal(result, expected_a_only_df, check_dtype=False, atol=1e-4)


def test_apply_weights_invalid_column(input_df):
    """Raise KeyError for a column that does not exist."""
    with pytest.raises(KeyError):
        apply_weights(
            df=input_df.copy(),
            a_weight_cols=["invalid_column"],
            calc_g_weight=False,
            round_val=4,
        )
