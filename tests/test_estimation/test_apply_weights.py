"""Tests for functions in apply_weights."""

import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from bsrm.estimation.apply_weights import apply_weights
from bsrm.estimation.calculate_weights import calculate_a_weights, calculate_g_weights


@pytest.fixture
def sample_data():
    """Small sample data used across apply_weights tests."""
    col_name = ["ruref", "cell_no", "k", "nh", "nk", "y", "x", "Nh", "Nk", "sum_x"]
    data = [
        ["A", 1, 1, 3, 5, 10.0, 2.0, 6, 20, 40.0],
        ["B", 1, 1, 3, 5, 20.0, 4.0, 6, 20, 40.0],
        ["C", 1, 1, 3, 5, 30.0, 0.0, 6, 20, 40.0],
        ["D", 2, 2, 2, 4, 15.0, 10.0, 4, 18, 60.0],
        ["E", 2, 2, 2, 4, 25.0, 20.0, 4, 18, 60.0],
    ]
    return pd.DataFrame(columns=col_name, data=data)


@pytest.fixture
def weighted_input(sample_data):
    """Input data with calculated a_weight and g_weight columns."""
    df_a = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    return calculate_g_weights(df_a, "k", "x", "sum_x")


def test_apply_weights(weighted_input):
    """Apply both a and g weights to column y"""
    input_df = weighted_input.copy()
    expected_y = (input_df["y"] * input_df["a_weight"] * input_df["g_weight"]).round(4)

    result = apply_weights(
        df=input_df,
        a_weight_cols=["y"],
        g_weight_cols=["y"],
        calc_g_weight=True,
        round_val=4,
    )

    assert_series_equal(
        result["y"].reset_index(drop=True),
        expected_y.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_apply_weights_a_only(weighted_input):
    """Apply only a weight when calc_g_weight is False."""
    input_df = weighted_input.copy()
    expected_y = (input_df["y"] * input_df["a_weight"]).round(4)
    result = apply_weights(
        df=input_df,
        a_weight_cols=["y"],
        calc_g_weight=False,
        round_val=4,
    )

    actual = result["y"].reset_index(drop=True)
    expected = expected_y.reset_index(drop=True)
    assert actual.equals(expected)


def test_apply_weights_invalid_column(weighted_input):
    """Raise KeyError for non-existent column."""
    input_df = weighted_input.copy()

    with pytest.raises(KeyError):
        apply_weights(
            df=input_df,
            a_weight_cols=["invalid_columns"],
            g_weight_cols=None,
            calc_g_weight=False,
            round_val=4,
        )
