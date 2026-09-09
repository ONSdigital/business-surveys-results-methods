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
                "strat",
                "unit_ref",
                "y_targ_val",
                "x_aux",
                "strat_wgt",
                "unit_wgt",
                "tuning_param",
                "predicted",
                "exclude_wins",
            ),
            (1, 1, 10, 5, 20, 3, 2, 10, 10, False),
            (1, 1, 11, 10, 25, 3, 2, 10, 20, False),
            (1, 1, 12, 15, 30, 3, 2, 10, 30, False),
            (2, 2, 20, 20, 40, 3, 2, 10, 40, False),
            (2, 2, 21, 25, 50, 3, 2, 10, 50, True),
        ]
    )


@pytest.fixture
def expected_ag_product_df():
    """Expected output after calculating weight_product (wgt_product = strat_wgt * unit_wgt)."""
    return create_test_dataframe(
        [
            (
                "cell",
                "strat",
                "unit_ref",
                "y_targ_val",
                "x_aux",
                "strat_wgt",
                "unit_wgt",
                "tuning_param",
                "predicted",
                "exclude_wins",
                "ag_product",
            ),
            (1, 1, 10, 5, 20, 3, 2, 10, 10, False, 6),
            (1, 1, 11, 10, 25, 3, 2, 10, 20, False, 6),
            (1, 1, 12, 15, 30, 3, 2, 10, 30, False, 6),
            (2, 2, 20, 20, 40, 3, 2, 10, 40, False, 6),
            (2, 2, 21, 25, 50, 3, 2, 10, 50, True, 6),
        ]
    )


@pytest.fixture
def expected_ratio_threshold_df():
    """Expected output after calculating ratio_threshold (wgt_product = strat_wgt * unit_wgt)."""

    return create_test_dataframe(
        [
            (
                "cell",
                "strat",
                "unit_ref",
                "y_targ_val",
                "x_aux",
                "strat_wgt",
                "unit_wgt",
                "tuning_param",
                "predicted",
                "exclude_wins",
                "ag_product",
                "ratio_threshold",
            ),
            (1, 1, 10, 5, 20, 3, 2, 10, 10, False, 6, 12),
            (1, 1, 11, 10, 25, 3, 2, 10, 20, False, 6, 22),
            (1, 1, 12, 15, 30, 3, 2, 10, 30, False, 6, 32),
            (2, 2, 20, 20, 40, 3, 2, 10, 40, False, 6, 42),
            (2, 2, 21, 25, 50, 3, 2, 10, 50, True, 6, 52),
        ]
    )


@pytest.fixture
def expected_masked_ratio_threshold_df():
    """Expected output after applying exclude_winsorisation mask.

    Units with exclude_winsorisation=True get ratio_thresh_masked=None.
    """
    return create_test_dataframe(
        [
            (
                "cell",
                "strat",
                "unit_ref",
                "y_targ_val",
                "x_aux",
                "strat_wgt",
                "unit_wgt",
                "tuning_param",
                "predicted",
                "exclude_wins",
                "ag_product",
                "ratio_threshold",
                "masked_ratio_threshold",
            ),
            (1, 1, 10, 5, 20, 3, 2, 10, 10, False, 6, 12, 12),
            (1, 1, 11, 10, 25, 3, 2, 10, 20, False, 6, 22, 22),
            (1, 1, 12, 15, 30, 3, 2, 10, 30, False, 6, 32, 32),
            (2, 2, 20, 20, 40, 3, 2, 10, 40, False, 6, 42, 42),
            (2, 2, 21, 25, 50, 3, 2, 10, 50, True, 6, 52, None),
        ]
    )


@pytest.fixture
def expected_ratio_estimation_threshold_df():
    """Expected output from complete pipeline.

    Final output with ratio_thresh_final (masked for exclude_winsorisation units).
    """
    return create_test_dataframe(
        [
            (
                "cell",
                "strat",
                "unit_ref",
                "y_targ_val",
                "x_aux",
                "strat_wgt",
                "unit_wgt",
                "tuning_param",
                "predicted",
                "exclude_wins",
                "ratio_estimation_threshold",
            ),
            (1, 1, 10, 5, 20, 3, 2, 10, 10, False, 12),
            (1, 1, 11, 10, 25, 3, 2, 10, 20, False, 22),
            (1, 1, 12, 15, 30, 3, 2, 10, 30, False, 32),
            (2, 2, 20, 20, 40, 3, 2, 10, 40, False, 42),
            (2, 2, 21, 25, 50, 3, 2, 10, 50, True, None),
        ]
    )


def test_calculate_ag_product(input_data, expected_ag_product_df):
    """Test that weight_product (ag_product = strat_wgt * unit_wgt) is calculated correctly."""
    result = calculate_ag_product(input_data.copy(), "strat_wgt", "unit_wgt")
    assert_frame_equal(result, expected_ag_product_df, check_dtype=False, rtol=1e-5)


def test_expected_ratio_threshold_df(expected_ag_product_df, expected_ratio_threshold_df):
    """Test that ratio_threshold (ki = predicted + tuning_param / (ag_product - 1)) is calculated correctly."""
    result = calculate_ratio_threshold(expected_ag_product_df.copy(), "predicted", "tuning_param", "ag_product")
    assert_frame_equal(result, expected_ratio_threshold_df, check_dtype=False, rtol=1e-5)


def test_apply_non_winsorisable_mask(expected_ratio_threshold_df, expected_masked_ratio_threshold_df):
    """Test that exclude_winsorisation units are masked to NaN."""
    result = apply_non_winsorisable_mask(expected_ratio_threshold_df.copy(), "ratio_threshold", "exclude_wins")
    assert_frame_equal(result, expected_masked_ratio_threshold_df, check_dtype=False)


def test_calculate_ratio_estimation_threshold(input_data, expected_ratio_estimation_threshold_df):
    """Test the complete ratio_estimation_threshold calculation pipeline with stratum_weight, unit_weight, predicted_value, tuning_parameter, and exclude_winsorisation."""
    result = calculate_ratio_estimation_threshold(
        input_data,
        "strat_wgt",
        "unit_wgt",
        "predicted",
        "tuning_param",
        "exclude_wins",
    )

    assert_frame_equal(
        result[["ratio_estimation_threshold"]],
        expected_ratio_estimation_threshold_df[["ratio_estimation_threshold"]],
        check_dtype=False,
        rtol=1e-4,
    )
