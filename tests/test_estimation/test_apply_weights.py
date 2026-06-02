"""Tests for functions in apply_weights."""

import pytest
from pandas.testing import assert_frame_equal
from bsrm.estimation.apply_weights import apply_weights
from bsrm.estimation.calculate_weights import calculate_a_weights, calculate_g_weights


@pytest.fixture(scope="function")
def weighted_input(estimation_input):
    """Input data with calculated a_weight and g_weight columns."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    return calculate_g_weights(df_a, "cell_no", "x", "sum_x")


def test_apply_weights_expected(weighted_input):
    """Apply both a and g weights and verify the weighted y values."""
    input_df = weighted_input.copy()
    expected = input_df.copy()
    expected["y"] = round(
        input_df["y"] * input_df["a_weight"] * input_df["g_weight"], 4
    )

    result = apply_weights(input_df, ["y"], ["y"], True, 4)

    assert_frame_equal(
        result[["y"]].reset_index(drop=True),
        expected[["y"]].reset_index(drop=True),
        check_dtype=False,
    )
