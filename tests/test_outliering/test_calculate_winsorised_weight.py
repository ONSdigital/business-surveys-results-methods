"""Tests for adjusted returns and outlier weights."""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from bsrm.outliering.calculate_winsorised_weight import calculate_ratio_winsorised_weight
from bsrm.utils.helpers import create_test_dataframe


@pytest.mark.parametrize(
    "target,a_weight,g_weight,threshold,non_winsorisable,adjusted_return,outlier_weight,outlier_flag",
    [
        (20, 2, 1, 10, False, 15, 0.75, 1),
        (5, 2, 1, 10, False, 5, 1, 0),
        (10, 2, 1, 10, False, 10, 1, 0),
        (14, 1, 1, np.nan, True, 14, 1, 0),
        (28, 1, 1, np.nan, True, 28, 1, 0),
        (0, 1, 1, np.nan, True, 0, 1, 0),
        (0, 2, 1, np.nan, False, 0, 1, 0),
    ],
)
def test_calculate_ratio_winsorised_weight(
    target,
    a_weight,
    g_weight,
    threshold,
    non_winsorisable,
    adjusted_return,
    outlier_weight,
    outlier_flag,
):
    """Preserve excluded returns and adjust only units above the threshold."""
    data = create_test_dataframe(
        [
            ("target", "a_weight", "g_weight", "threshold", "non_winsorisable"),
            (target, a_weight, g_weight, threshold, non_winsorisable),
        ]
    )
    expected = create_test_dataframe(
        [
            ("adjusted_return", "outlier_weight", "outlier_flag"),
            (adjusted_return, outlier_weight, outlier_flag),
        ]
    )

    result = calculate_ratio_winsorised_weight(
        data, "a_weight", "g_weight", "target", "threshold", "non_winsorisable"
    )

    assert_frame_equal(result[expected.columns], expected, check_dtype=False)
