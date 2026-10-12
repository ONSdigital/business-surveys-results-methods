"""Tests for functions in winsorisation."""

import pytest
from pandas.testing import assert_frame_equal

from bsrm.outliering.winsorisation import winsorise
from bsrm.utils.helpers import create_test_dataframe


@pytest.fixture
def input_data():
    """small sample  data to test the Winsorisation function."""
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
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10),
            (1, 1, 12, 100, 30, 2.5, 0.8, 10),
            (2, 2, 20, 20, 40, 2.0, 1.0, 10),
            (2, 2, 21, 25, 50, 1.0, 1.0, 10),
        ]
    )


@pytest.fixture
def expected_winsorised_df():
    """Expected output from the winsorisation function."""
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
                "non_winsorisable_marker",
                "predicted_unit_value",
                "ratio_threshold",
                "ratio_estimation_threshold",
                "adjusted_return",
                "outlier_flag",
                "outlier_weight",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 10, False, 30.666667, 40.666667, 40.666667, 5, 0, 1.0),
            (1, 1, 11, 10, 25, 2.5, 0.8, 10, False, 38.333333, 48.333333, 48.333333, 10, 0, 1.0),
            (1, 1, 12, 100, 30, 2.5, 0.8, 10, False, 46.0, 56.0, 56.0, 128.0, 1, 1.28),
            (2, 2, 20, 20, 40, 2.0, 1.0, 10, False, 20.0, 30.0, 30.0, 20, 0, 1.0),
            (2, 2, 21, 25, 50, 1.0, 1.0, 10, True, None, None, None, 0, 0, 1.0),
        ]
    )


def test_winsorise(input_data, expected_winsorised_df):
    """Test the Winsorisation function."""
    result = winsorise(
        input_data.copy(),
        "calibration_group",
        "aux",
        "a_weight",
        "g_weight",
        "target",
        "l_value",
    )

    assert_frame_equal(result, expected_winsorised_df, check_dtype=False, rtol=1e-5)
