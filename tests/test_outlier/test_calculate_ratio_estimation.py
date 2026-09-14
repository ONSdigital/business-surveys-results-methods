"""Tests for functions in calculate_ratio_estimation."""

import pytest
from pandas.testing import assert_frame_equal
from bsrm.utils.helpers import create_test_dataframe
from bsrm.outliering.calculate_ratio_estimation import (
    calculate_ag_product,
    calculate_ratio_threshold,
    apply_non_winsorisable_mask,
    calculate_ratio_estimation_threshold,
)


@pytest.fixture
def input_data():
    """Small sample data to test ratio estimation functions."""
    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "g_weight",
                "l_value",
                "predicted_unit_value",
                "non_winsorisable_marker",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, 10, False),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, 20, False),
            (1, 1, 12, 15, 30, 2.5, 0.8, 10, 30, False),
            (2, 2, 20, 20, 40, 2.0, 0.5, 10, 40, False),
            (2, 2, 21, 25, 50, 2.0, 0.5, 10, 50, True),
        ]
    )


@pytest.fixture
def expected_ag_product_df():
    """Expected output after calculating weight_product (ag_product = a_weight * g_weight)."""
    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "g_weight",
                "l_value",
                "predicted_unit_value",
                "non_winsorisable_marker",
                "ag_product",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, 10, False, 2.0),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, 20, False, 2.0),
            (1, 1, 12, 15, 30, 2.5, 0.8, 10, 30, False, 2.0),
            (2, 2, 20, 20, 40, 2.0, 0.5, 10, 40, False, 1.0),
            (2, 2, 21, 25, 50, 2.0, 0.5, 10, 50, True, 1.0),
        ]
    )


@pytest.fixture
def expected_ratio_threshold_df():
    """Expected output after calculating ratio_threshold (predicted + l_value / (ag_product - 1))."""

    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "g_weight",
                "l_value",
                "predicted_unit_value",
                "non_winsorisable_marker",
                "ag_product",
                "denominator_k_i",
                "ratio_threshold",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, 10, False, 2.0, 1.0, 20),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, 20, False, 2.0, 1.0, 30),
            (1, 1, 12, 15, 30, 2.5, 0.8, 10, 30, False, 2.0, 1.0, 40),
            (2, 2, 20, 20, 40, 2.0, 0.5, 10, 40, False, 1.0, None, None),
            (2, 2, 21, 25, 50, 2.0, 0.5, 10, 50, True, 1.0, None, None),
        ]
    )


@pytest.fixture
def expected_masked_ratio_threshold_df():
    """Expected output after masking non-winsorisable units to None."""
    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "g_weight",
                "l_value",
                "predicted_unit_value",
                "non_winsorisable_marker",
                "ag_product",
                "denominator_k_i",
                "ratio_threshold",
                "masked_ratio_threshold",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, 10, False, 2.0, 1.0, 20, 20),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, 20, False, 2.0, 1.0, 30, 30),
            (1, 1, 12, 15, 30, 2.5, 0.8, 10, 30, False, 2.0, 1.0, 40, 40),
            (2, 2, 20, 20, 40, 2.0, 0.5, 10, 40, False, 1.0, None, None, None),
            (2, 2, 21, 25, 50, 2.0, 0.5, 10, 50, True, 1.0, None, None, None),
        ]
    )


@pytest.fixture
def expected_ratio_estimation_threshold_df():
    """Expected output from the complete ratio_estimation_threshold pipeline."""
    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "g_weight",
                "l_value",
                "predicted_unit_value",
                "non_winsorisable_marker",
                "ratio_estimation_threshold",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, 10, False, 20),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, 20, False, 30),
            (1, 1, 12, 15, 30, 2.5, 0.8, 10, 30, False, 40),
            (2, 2, 20, 20, 40, 2.0, 0.5, 10, 40, False, None),
            (2, 2, 21, 25, 50, 2.0, 0.5, 10, 50, True, None),
        ]
    )


def test_calculate_ag_product(input_data, expected_ag_product_df):
    """Test that weight_product (ag_product = a_weight * g_weight) is calculated correctly."""
    result = calculate_ag_product(input_data.copy(), "a_weight", "g_weight")
    assert_frame_equal(result, expected_ag_product_df, check_dtype=False, rtol=1e-5)


def test_expected_ratio_threshold_df(expected_ag_product_df, expected_ratio_threshold_df):
    """Test that ratio_threshold is calculated from predicted, l_value, and ag_product."""
    result = calculate_ratio_threshold(
        expected_ag_product_df.copy(),
        "predicted_unit_value",
        "l_value",
        "ag_product",
    )
    assert_frame_equal(result, expected_ratio_threshold_df, check_dtype=False, rtol=1e-5)


def test_apply_non_winsorisable_mask(expected_ratio_threshold_df, expected_masked_ratio_threshold_df):
    """Test that exclude_wins units are masked to NaN."""
    result = apply_non_winsorisable_mask(
        expected_ratio_threshold_df.copy(),
        "ratio_threshold",
        "non_winsorisable_marker",
    )
    assert_frame_equal(result, expected_masked_ratio_threshold_df, check_dtype=False)


def test_calculate_ratio_estimation_threshold(input_data, expected_ratio_estimation_threshold_df):
    """Test the complete pipeline using a_weight and g_weight."""
    result = calculate_ratio_estimation_threshold(
        input_data,
        "a_weight",
        "g_weight",
        "predicted_unit_value",
        "l_value",
        "non_winsorisable_marker",
    )

    assert_frame_equal(
        result[["ratio_estimation_threshold"]],
        expected_ratio_estimation_threshold_df[["ratio_estimation_threshold"]],
        check_dtype=False,
        rtol=1e-4,
    )
