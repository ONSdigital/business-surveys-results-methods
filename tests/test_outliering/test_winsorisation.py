"""Tests for functions in winsorisation."""

import pytest
from pandas.testing import assert_frame_equal

from bsrm.outliering.winsorisation import winsorise_ratio
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
                "mu_i",
                "mu_factor",
                "ki",
                "yi*",
                "outlier_flag",
                "oi",
            ),
            (1, 1, 10, 5, 20, 2.5, 0.8, 30.666667, 1.5333, 40.666667, 5.0, 0, 1.0),
            (1, 1, 11, 10, 25, 2.5, 0.8, 38.333333, 1.5333, 48.333333, 10.0, 0, 1.0),
            (1, 1, 12, 100, 30, 2.5, 0.8, 46.0, 1.5333, 56.0, 78.0, 1, 0.78),
            (2, 2, 20, 20, 40, 2.0, 1.0, 20.0, 0.5, 30.0, 20.0, 0, 1.0),
            (2, 2, 21, 25, 50, 1.0, 1.0, None, None, None, 25.0, 0, 1.0),
        ]
    )


def test_winsorise(input_data, expected_winsorised_df):
    """Test the Winsorisation function."""
    result = winsorise_ratio(
        input_data.copy(),
        "calibration_group",
        "aux",
        "a_weight",
        "g_weight",
        "target",
        "l_value",
    )

    assert_frame_equal(
        result[expected_winsorised_df.columns].round(4),
        expected_winsorised_df.round(4),
        check_dtype=False,
        atol=1e-4,
    )
