"""Tests for functions in calculate_ratio_estimation."""

import pytest
from pandas.testing import assert_frame_equal
from bsrm.utils.helpers import create_test_dataframe
from bsrm.outlier.calculate_ratio_estimation import (
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
            ("cell", "k", "ruref", "y", "x", "ah", "gk", "L", "mu_i"),
            (1, 1, 17, 14, 10.2, 2.666667, 2.205508, 15, 10.31657),
            (1, 1, 1, 28, 30, 2.666667, 2.205508, 15, 30.34286),
            (1, 1, 16, 28, 30.6, 2.666667, 2.205508, 15, 30.94971),
            (2, 2, 8, 42, 265.2, 3, 0.713779, 15, 3517.478),
            (2, 2, 9, 42, 306, 3, 0.713779, 15, 4058.629),
        ]
    )


@pytest.fixture
def expected_ag_product_df():
    """Expected output after calculating ag_product (ah*gk)."""
    return create_test_dataframe(
        [
            ("cell", "k", "ruref", "y", "x", "ah", "gk", "L", "mu_i", "ag_product"),
            (1, 1, 17, 14, 10.2, 2.666667, 2.205508, 15, 10.31657, 5.881355401836),
            (1, 1, 1, 28, 30, 2.666667, 2.205508, 15, 30.34286, 5.881355401836),
            (1, 1, 16, 28, 30.6, 2.666667, 2.205508, 15, 30.94971, 5.881355401836),
            (2, 2, 8, 42, 265.2, 3, 0.713779, 15, 3517.478, 2.141337),
            (2, 2, 9, 42, 306, 3, 0.713779, 15, 4058.629, 2.141337),
        ]
    )


@pytest.fixture
def expected_ratio_threshold_df():
    """Expected output after calculating ratio_threshold (ki = mu_i + L / (ag_product - 1))."""
    return create_test_dataframe(
        [
            ("cell", "k", "ruref", "y", "x", "ah", "gk", "L", "mu_i", "ag_product", "ratio_threshold"),
            (1, 1, 17, 14, 10.2, 2.666667, 2.205508, 15, 10.31657, 5.881355401836, 13.389487253508),
            (1, 1, 1, 28, 30, 2.666667, 2.205508, 15, 30.34286, 5.881355401836, 33.415777253508),
            (1, 1, 16, 28, 30.6, 2.666667, 2.205508, 15, 30.94971, 5.881355401836, 34.022627253508),
            (2, 2, 8, 42, 265.2, 3, 0.713779, 15, 3517.478, 2.141337, 3530.620481),
            (2, 2, 9, 42, 306, 3, 0.713779, 15, 4058.629, 2.141337, 4071.771481),
        ]
    )


def test_calculate_ag_product(input_data, expected_ag_product_df):
    """Test that ag_product (ah*gk) is calculated correctly."""
    result = calculate_ag_product(input_data.copy(), "ah", "gk")
    assert_frame_equal(result, expected_ag_product_df, check_dtype=False, rtol=1e-5)


def test_expected_ratio_threshold_df(expected_ag_product_df, expected_ratio_threshold_df):
    """Test that ratio_threshold (ki = mu_i + L / (ag_product - 1)) is calculated correctly."""
    result = calculate_ratio_threshold(expected_ag_product_df.copy(), "mu_i", "L", "ag_product")
    assert_frame_equal(result, expected_ratio_threshold_df, check_dtype=False, rtol=1e-5)
