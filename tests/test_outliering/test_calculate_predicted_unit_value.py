"""Tests for predicted unit value calculations."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from bsrm.outliering.calculate_predicted_unit_value import (
    calculate_group_sums,
    calculate_predicted_unit_values,
    calculate_predicted_unit_values_from_group_sums,
)
from bsrm.utils.helpers import create_test_dataframe


@pytest.fixture
def input_data():
    """Small sample data for predicted unit value tests."""
    return create_test_dataframe(
        [
            (
                "cell",
                "calibration_group",
                "unit_ref",
                "target",
                "aux",
                "a_weight",
                "non_winsorisable_marker",
            ),
            (1, 1, 10, 10.0, 2.0, 2.0, False),
            (1, 1, 11, 100.0, 4.0, 3.0, True),
            (2, 2, 12, 40.0, 3.0, 2.0, True),
            (2, 2, 13, 30.0, 6.0, 3.0, False),
        ]
    )


@pytest.fixture
def expected_group_sums():
    """Expected weighted totals for each calibration group."""
    return create_test_dataframe(
        [
            (
                "calibration_group",
                "sum_weighted_target_values",
                "sum_weighted_auxiliary_values",
            ),
            (1, 20.0, 4.0),
            (2, 90.0, 18.0),
        ]
    )


def test_calculate_group_sums_excludes_non_winsorisable_rows(input_data, expected_group_sums):
    """Group totals should include only winsorisable rows."""
    result = calculate_group_sums(
        input_data,
        "calibration_group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable_marker",
    )

    assert_frame_equal(result, expected_group_sums)


def test_calculate_predicted_unit_values_from_group_sums_returns_predictions():
    """Predictions should be added to the input dataframe."""
    merged_df = create_test_dataframe(
        [
            ("aux", "sum_weighted_target_values", "sum_weighted_auxiliary_values"),
            (2.0, 20.0, 4.0),
            (6.0, 90.0, 18.0),
        ]
    )

    result = calculate_predicted_unit_values_from_group_sums(merged_df, "aux")
    expected = create_test_dataframe(
        [
            (
                "aux",
                "sum_weighted_target_values",
                "sum_weighted_auxiliary_values",
                "predicted_unit_value",
            ),
            (2.0, 20.0, 4.0, 10.0),
            (6.0, 90.0, 18.0, 30.0),
        ]
    )

    assert_frame_equal(result, expected)


def test_calculate_predicted_unit_values_handles_non_default_index(input_data):
    """Predictions and markers should remain matched to their input rows."""
    input_data.index = [10, 11, 12, 13]

    result = calculate_predicted_unit_values(
        input_data,
        "calibration_group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable_marker",
    )
    expected = pd.Series(
        [10.0, np.nan, np.nan, 30.0],
        name="predicted_unit_value",
    )

    assert_series_equal(result["predicted_unit_value"], expected)


def test_calculate_predicted_unit_values_returns_nan_for_zero_auxiliary_total():
    """A zero weighted auxiliary total should not cause division by zero."""
    merged_df = create_test_dataframe(
        [
            ("aux", "sum_weighted_target_values", "sum_weighted_auxiliary_values"),
            (5.0, 20.0, 0.0),
        ]
    )

    result = calculate_predicted_unit_values_from_group_sums(merged_df, "aux")

    assert np.isnan(result["predicted_unit_value"].iloc[0])
