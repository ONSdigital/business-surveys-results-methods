"""Tests for predicted unit value calculations."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from bsrm.outliering.calculate_predicted_unit_value import (
    calculate_group_sums,
    calculate_predicted_unit_value,
    calculate_predicted_unit_values_from_group_sums,
)


def create_input_data() -> pd.DataFrame:
    """Return input data containing winsorisable and non-winsorisable rows."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "aux": [2.0, 4.0, 3.0, 6.0],
            "a_weight": [2.0, 3.0, 2.0, 3.0],
            "target": [10.0, 100.0, 40.0, 30.0],
            "non_winsorisable": [False, True, True, False],
        }
    )


def test_calculate_group_sums_excludes_non_winsorisable_rows() -> None:
    """Group sums should only include winsorisable rows."""
    input_df = create_input_data()
    result = calculate_group_sums(
        input_df,
        "group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable",
    )
    expected = pd.DataFrame(
        {
            "group": ["A", "B"],
            "sum_weighted_target_values": [20.0, 90.0],
            "sum_weighted_auxiliary_values": [4.0, 18.0],
        }
    )

    assert_frame_equal(result, expected)


def test_calculate_predicted_unit_values_from_group_sums_returns_prediction_column() -> None:
    """Predictions should apply the group ratio to each auxiliary value."""
    merged_df = pd.DataFrame(
        {
            "aux": [2.0, 6.0],
            "sum_weighted_target_values": [20.0, 90.0],
            "sum_weighted_auxiliary_values": [4.0, 18.0],
        }
    )

    result = calculate_predicted_unit_values_from_group_sums(merged_df, "aux")
    expected = merged_df.assign(predicted_unit_value=[10.0, 30.0])

    assert_frame_equal(result, expected)


def test_calculate_predicted_unit_value_masks_non_winsorisable_rows() -> None:
    """The main calculation should return NaN for non-winsorisable rows."""
    input_df = create_input_data()
    result = calculate_predicted_unit_value(
        input_df,
        "group",
        "aux",
        "a_weight",
        "target",
        "non_winsorisable",
    )
    expected = pd.Series(
        [10.0, np.nan, np.nan, 30.0],
        name="predicted_unit_value",
    )

    assert_series_equal(result["predicted_unit_value"], expected)


