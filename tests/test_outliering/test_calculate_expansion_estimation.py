"""Tests for expansion estimation calculations."""

import numpy as np
from pandas.testing import assert_frame_equal

from bsrm.outliering.calculate_expansion_estimation import calculate_stratum_mean
from bsrm.utils.helpers import create_test_dataframe


def test_calculate_stratum_mean() -> None:
    """Test stratum and weight-band means with missing responses."""
    input_df = create_test_dataframe(
        [
            ("ruref", "a_wt_band", "response"),
            (100, 1, 10.0),
            (101, 2, 100.0),
            (201, 1, 40.0),
            (102, 1, 20.0),
            (202, 1, np.nan),
            (103, 2, 140.0),
            (203, 1, np.nan),
        ],
    )
    exected_df = create_test_dataframe(
        [
            ("ruref", "a_wt_band", "response", "stratum_mean"),
            (100, 1, 10.0, 23.3333),
            (101, 2, 100.0, 120.0),
            (201, 1, 40.0, 23.3333),
            (102, 1, 20.0, 23.3333),
            (202, 1, np.nan, 23.3333),
            (103, 2, 140.0, 120.0),
            (203, 1, np.nan, 23.3333),
        ],
    )


    result = calculate_stratum_mean(input_df, strata_col="a_wt_band", target_col="response")

    assert_frame_equal(result, exected_df, rtol=1e-4)
