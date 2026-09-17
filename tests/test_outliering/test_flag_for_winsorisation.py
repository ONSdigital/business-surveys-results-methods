"""Tests for flagging non-winsorisable units."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from bsrm.outliering.flag_for_winsorisation import winsorisation_flag


def test_winsorisation_flag_marks_only_rows_where_both_weights_equal_one() -> None:
    """Cover every combination of weights equal and not equal to one."""
    input_df = pd.DataFrame(
        {
            "a_weight": [1, 1, 2, 2],
            "g_weight": [1, 2, 1, 2],
        }
    )

    result = winsorisation_flag(input_df, "a_weight", "g_weight")

    expected = pd.Series(
        [True, False, False, False],
        name="non_winsorisable_marker",
    )
    assert_series_equal(result["non_winsorisable_marker"], expected)


def test_winsorisation_flag_handles_float_and_missing_weights() -> None:
    """Numeric ones should match, while missing weights should not be marked."""
    input_df = pd.DataFrame(
        {
            "a_weight": [1.0, np.nan, 1.0],
            "g_weight": [1.0, 1.0, np.nan],
        }
    )

    result = winsorisation_flag(input_df, "a_weight", "g_weight")

    expected = pd.Series(
        [True, False, False],
        name="non_winsorisable_marker",
    )
    assert_series_equal(result["non_winsorisable_marker"], expected)


def test_winsorisation_flag_uses_custom_columns_without_mutating_input() -> None:
    """Preserve source data and recalculate an existing marker in the result."""
    input_df = pd.DataFrame(
        {
            "design": [1, 2],
            "calibration": [1, 2],
            "reference": ["unit-1", "unit-2"],
            "non_winsorisable_marker": [False, True],
        },
        index=[10, 20],
    )
    original = input_df.copy(deep=True)

    result = winsorisation_flag(input_df, "design", "calibration")

    assert_frame_equal(input_df, original)
    expected = original.assign(non_winsorisable_marker=[True, False])
    assert_frame_equal(result, expected)


def test_winsorisation_flag_handles_empty_dataframe() -> None:
    """Return an empty boolean marker when there are no units."""
    input_df = pd.DataFrame(
        {
            "a_weight": pd.Series(dtype="float64"),
            "g_weight": pd.Series(dtype="float64"),
        }
    )

    result = winsorisation_flag(input_df, "a_weight", "g_weight")

    expected = input_df.assign(non_winsorisable_marker=pd.Series(dtype="bool"))
    assert_frame_equal(result, expected)


def test_winsorisation_flag_raises_key_error_for_missing_weight_columns() -> None:
    """Reject calls where either requested weight column is absent."""
    complete_df = pd.DataFrame({"a_weight": [1], "g_weight": [1]})

    for missing_column in ("a_weight", "g_weight"):
        input_df = complete_df.drop(columns=missing_column)

        with pytest.raises(KeyError, match=missing_column):
            winsorisation_flag(input_df, "a_weight", "g_weight")