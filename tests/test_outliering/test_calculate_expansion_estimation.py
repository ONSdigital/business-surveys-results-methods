"""Tests for expansion estimation calculations."""

import numpy as np
from pandas.testing import assert_frame_equal

from bsrm.outliering.calculate_expansion_estimation import calculate_stratum_mean
from bsrm.utils.helpers import create_test_dataframe


def test_calculate_stratum_mean() -> None:
    """Test stratum and weight-band means with missing responses."""
    df = create_test_dataframe(
        [
            ("cell", "a_wt_band", "response"),
            (1, 1, 10.0),
            (1, 2, 100.0),
            (2, 1, 40.0),
            (1, 1, 20.0),
            (2, 1, np.nan),
            (1, 2, 140.0),
            (3, 1, np.nan),
        ],
        index=[9, 3, 7, 1, 8, 2, 6],
    )
    expected = df.assign(
        stratum_mean=[15.0, 120.0, 40.0, 15.0, 40.0, 120.0, np.nan],
    )

    result = calculate_stratum_mean(df, strata_col="cell", target_col="response")

    assert_frame_equal(result, expected)
