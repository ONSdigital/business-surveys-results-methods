"""Tests for predicted unit value calculations."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from bsrm.utils.helpers import create_test_dataframe
from bsrm.outliering.calculate_predicted_unit_value import (
    calculate_group_sums,
    calculate_predicted_unit_value,
    calculate_predicted_unit_values_from_group_sums,
)


@pytest.fixture
def input_data() -> pd.DataFrame:
    """Return input data containing winsorisable and non-winsorisable rows."""
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
            (1, 1, 10, 5, 20, 2.5, False),
            (1, 1, 11, 10, 25, 2.5, False),
            (1, 1, 12, 15, 30, 2.5, False),
            (2, 2, 20, 20, 40, 2.0, False),
            (2, 2, 21, 25, 50, 2.0, True),
        ]
    )


def test_calculate_group_sums_excludes_non_winsorisable_rows(input_data: pd.DataFrame) -> None:
    """Group sums should only include winsorisable rows."""
    result = calculate_group_sums(
        input_data,
        "calibration_group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable_marker",
    )
    expected = create_test_dataframe(
        [
            (
                "calibration_group",
                "sum_weighted_target_values",
                "sum_weighted_auxiliary_values",
            ),
            (1, 75.0, 187.5),
            (2, 40.0, 80.0),
        ]
    )

    assert_frame_equal(result, expected)


def test_calculate_predicted_unit_values_from_group_sums_returns_prediction_column() -> None:
    """Predictions should apply the group ratio to each auxiliary value."""
    merged_df = create_test_dataframe(
        [
            (
                "calibration_group",
                "unit_ref",
                "aux",
                "sum_weighted_target_values",
                "sum_weighted_auxiliary_values",
            ),
            (1, 10, 20.0, 75.0, 187.5),
            (2, 20, 40.0, 40.0, 80.0),
        ]
    )

    result = calculate_predicted_unit_values_from_group_sums(merged_df, "aux")
    expected = merged_df.assign(predicted_unit_value=[8.0, 20.0])

    assert_frame_equal(result, expected)


def test_calculate_predicted_unit_value_masks_non_winsorisable_rows(
    input_data: pd.DataFrame,
) -> None:
    """The main calculation should return NaN for non-winsorisable rows."""
    result = calculate_predicted_unit_value(
        input_data,
        "calibration_group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable_marker",
    )
    expected = input_data.assign(predicted_unit_value=[8.0, 10.0, 12.0, 20.0, None])

    assert_frame_equal(result, expected)
