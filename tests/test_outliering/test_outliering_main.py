import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from bsrm.outliering.winsorisation import winsorise_ratio, winsorise_expansion
from bsrm.outliering.outliering_main import (
    apply_winsorise_expansion,
    apply_winsorise_ratio,
    detect_outlier,
)
from bsrm.utils.helpers import create_test_dataframe


@pytest.fixture(scope="function")
def winsorise_ratio_component_test_input():
    """Input data for ratio estimation winsorisation - Q1 (turnover)."""
    return create_test_dataframe(
        [
            (
                "calibration_group_j",
                "ruref",
                "a_wt_band",
                "g_wt_band",
                "question",
                "value",
                "a_weight",
                "g_weight",
                "aux_var",
                "g_wgt_type",
                "L_yi",
            ),
            (1, 17, 1, 1, "Q1", 100, 2.666667, 2.205508, 10.2, "turnover", 100),
            (1, 1, 1, 1, "Q1", 200, 2.666667, 2.205508, 30.0, "turnover", 100),
            (1, 16, 1, 1, "Q1", 200, 2.666667, 0.677677, 30.6, "turnover", 100),
            (2, 8, 2, 2, "Q1", 300, 3.0, 0.677677, 265.2, "turnover", 100),
            (2, 9, 2, 2, "Q1", 300, 3.0, 0.677677, 306.0, "turnover", 100),
            (2, 13, 2, 2, "Q1", 300, 3.0, 0.677677, 1100.0, "turnover", 100),
            (3, 19, 4, 3, "Q1", 100, 1.0, 1.0, 3111.0, "turnover", 100),
            (4, 18, 5, 4, "Q1", 200, 1.0, 1.0, 10230.6, "turnover", 100),
            (5, 26, 6, 5, "Q1", 200, 2.0, 0.753931, 30.6, "turnover", 100),
            (5, 34, 6, 5, "Q1", 300, 2.0, 0.753931, 40.8, "turnover", 100),
            (5, 29, 7, 5, "Q1", 300, 3.333333, 0.753931, 295.8, "turnover", 100),
            (5, 23, 7, 5, "Q1", 300, 3.333333, 0.753931, 306.0, "turnover", 100),
            (5, 25, 7, 5, "Q1", 300, 3.333333, 0.753931, 500.0, "turnover", 100),
            (6, 33, 8, 6, "Q1", 100, 1.666667, 0.838091, 1100.0, "turnover", 100),
            (6, 21, 8, 6, "Q1", 200, 1.666667, 0.838091, 846.6, "turnover", 100),
            (6, 28, 8, 6, "Q1", 200, 1.666667, 0.838091, 1020.0, "turnover", 100),
            (7, 41, 9, 7, "Q1", 300, 1.0, 1.0, 6000.0, "turnover", 100),
            (7, 27, 9, 7, "Q1", 300, 1.0, 1.0, 5018.4, "turnover", 100),
            (8, 60, 10, 8, "Q1", 300, 2.0, 1.344054, 200.0, "turnover", 100),
            (8, 59, 11, 8, "Q1", 300, 3.0, 1.344054, 200.0, "turnover", 100),
            (8, 48, 11, 8, "Q1", 100, 3.0, 1.344054, 359.0, "turnover", 100),
            (9, 46, 12, 9, "Q1", 200, 3.0, 1.104874, 2000.0, "turnover", 100),
            (9, 50, 12, 9, "Q1", 200, 3.0, 1.104874, 4000.0, "turnover", 100),
            (9, 45, 12, 9, "Q1", 300, 3.0, 1.104874, 744.6, "turnover", 100),
            (10, 55, 13, 10, "Q1", 300, 1.0, 1.0, 10000.0, "turnover", 100),
            (10, 56, 13, 10, "Q1", 300, 1.0, 1.0, 100000.0, "turnover", 100),
            (11, 95, 14, 11, "Q1", 300, 10.0, 1.8425, 80.0, "turnover", 100),
            (12, 93, 15, 12, "Q1", 100, 8.5, 0.495819, 130.0, "turnover", 100),
            (12, 65, 15, 12, "Q1", 200, 8.5, 0.495819, 5000.0, "turnover", 100),
            (13, 100, 16, 13, "Q1", 200, 1.666667, 0.787571, 601.8, "turnover", 100),
            (13, 99, 16, 13, "Q1", 300, 1.666667, 0.787571, 744.6, "turnover", 100),
            (13, 87, 16, 13, "Q1", 300, 1.666667, 0.787571, 877.2, "turnover", 100),
            (13, 88, 16, 13, "Q1", 300, 1.666667, 0.787571, 1300.0, "turnover", 100),
            (13, 84, 16, 13, "Q1", 300, 1.666667, 0.787571, 1020.0, "turnover", 100),
            (13, 75, 16, 13, "Q1", 200, 1.666667, 0.787571, 10000.0, "turnover", 100),
            (14, 97, 17, 14, "Q1", 200, 1.0, 1.0, 4947.0, "turnover", 100),
            (14, 83, 17, 14, "Q1", 200, 1.0, 1.0, 5018.4, "turnover", 100),
            (14, 74, 17, 14, "Q1", 200, 1.0, 1.0, 50000.0, "turnover", 100),
        ]
    )

@pytest.fixture(scope="function")
def winsorise_ratio_component_test_output():
    """Expected output for ratio estimation winsorisation."""
    return create_test_dataframe(
        [
            (
                "calibration_group_j",
                "ruref",
                "a_wt_band",
                "g_wt_band",
                "question",
                "value",
                "a_weight",
                "g_weight",
                "aux_var",
                "g_wgt_type",
                "L_yi",
                "mu_i",
                "mu_factor",
                "ki",
                "yi*",
                "outlier_flag",
                "oi",
            ),
            (1, 17, 1, 1, "Q1", 100, 2.666667, 2.205508, 10.2, "turnover", 100, 72.033898, 7.0621468, 92.520009, 93.791823, 1, 0.9379182337023092),
            (1, 1, 1, 1, "Q1", 200, 2.666667, 2.205508, 30.0, "turnover", 100, 211.86440, 7.0621468, 232.35051, 200.0, 0, 1.0),
            (1, 16, 1, 1, "Q1", 200, 2.666667, 0.677677, 30.6, "turnover", 100, 216.10169, 7.0621468, 339.99625, 200.0, 0, 1.0),
            (2, 8, 2, 2, "Q1", 300, 3.0, 0.677677, 265.2, "turnover", 100, 142.81953, 0.5385351, 239.62212, 269.32059, 1, 0.8977353016928521),
            (2, 9, 2, 2, "Q1", 300, 3.0, 0.677677, 306.0, "turnover", 100, 164.79176, 0.5385351, 261.59436, 280.48519, 1, 0.934950657),
            (2, 13, 2, 2, "Q1", 300, 3.0, 0.677677, 1100.0, "turnover", 100, 592.38870, 0.5385351, 689.19129, 300.0, 0, 1.0),
            (3, 19, 4, 3, "Q1", 100, 1.0, 1.0, 3111.0, "turnover", 100, None, None, None, 100.0, 0, 1.0),
            (4, 18, 5, 4, "Q1", 200, 1.0, 1.0, 10230.6, "turnover", 100, None, None, None, 200.0, 0, 1.0),
            (5, 26, 6, 5, "Q1", 200, 2.0, 0.753931, 30.6, "turnover", 100, 32.079955, 1.0483645, 228.98355, 200.0, 0, 1.0),
            (5, 34, 6, 5, "Q1", 300, 2.0, 0.753931, 40.8, "turnover", 100, 42.77327, 1.0483645, 239.67687, 279.68258, 1, 0.9322752931707703),
            (5, 29, 7, 5, "Q1", 300, 3.333333, 0.753931, 295.8, "turnover", 100, 310.10623, 1.0483645, 376.19552, 300.0, 0, 1.0),
            (5, 23, 7, 5, "Q1", 300, 3.333333, 0.753931, 306.0, "turnover", 100, 320.79955, 1.0483645, 386.88883, 300.0, 0, 1.0),
            (5, 25, 7, 5, "Q1", 300, 3.333333, 0.753931, 500.0, "turnover", 100, 524.18227, 1.0483645, 590.27156, 300.0, 0, 1.0),
            (6, 33, 8, 6, "Q1", 100, 1.666667, 0.838091, 1100.0, "turnover", 100, 185.39742, 0.1685431, 437.4022, 100.0, 0, 1.0),
            (6, 21, 8, 6, "Q1", 200, 1.666667, 0.838091, 846.6, "turnover", 100, 142.68859, 0.1685431, 394.69335, 200.0, 0, 1.0),
            (6, 28, 8, 6, "Q1", 200, 1.666667, 0.838091, 1020.0, "turnover", 100, 171.91397, 0.1685431, 423.91873, 200.0, 0, 1.0),
            (7, 41, 9, 7, "Q1", 300, 1.0, 1.0, 6000.0, "turnover", 100, None, None, None, 300.0, 0, 1.0),
            (7, 27, 9, 7, "Q1", 300, 1.0, 1.0, 5018.4, "turnover", 100, None, None, None, 300.0, 0, 1.0),
            (8, 60, 10, 8, "Q1", 300, 2.0, 1.344054, 200.0, "turnover", 100, 173.32691, 0.8666345, 232.56483, 257.65131, 1, 0.8588377186343168),
            (8, 59, 11, 8, "Q1", 300, 3.0, 1.344054, 200.0, "turnover", 100, 173.32691, 0.8666345, 206.30668, 229.54318, 1, 0.765143938873298),
            (8, 48, 11, 8, "Q1", 100, 3.0, 1.344054, 359.0, "turnover", 100, 311.12181, 0.8666345, 344.10158, 100.0, 0, 1.0),
            (9, 46, 12, 9, "Q1", 200, 3.0, 1.104874, 2000.0, "turnover", 100, 207.57346, 0.1037867, 250.77706, 200.0, 0, 1.0),
            (9, 50, 12, 9, "Q1", 200, 3.0, 1.104874, 4000.0, "turnover", 100, 415.14693, 0.1037867, 458.35052, 200.0, 0, 1.0),
            (9, 45, 12, 9, "Q1", 300, 3.0, 1.104874, 744.6, "turnover", 100, 77.279601, 0.1037867, 120.48319, 174.64225, 1, 0.5821408484085395),
            (10, 55, 13, 10, "Q1", 300, 1.0, 1.0, 10000.0, "turnover", 100, None, None, None, 300.0, 0, 1.0),
            (10, 56, 13, 10, "Q1", 300, 1.0, 1.0, 100000.0, "turnover", 100, None, None, None, 300.0, 0, 1.0),
            (11, 95, 14, 11, "Q1", 300, 10.0, 1.8425, 80.0, "turnover", 100, 300.0, 3.75, 305.73888, 300.0, 0, 1.0),
            (12, 93, 15, 12, "Q1", 100, 8.5, 0.495819, 130.0, "turnover", 100, 7.6023391, 0.05848, 38.711725, 53.254090, 1, 0.5325409076360982),
            (12, 65, 15, 12, "Q1", 200, 8.5, 0.495819, 5000.0, "turnover", 100, 292.39766, 0.05848, 323.50704, 200.0, 0, 1.0),
            (13, 100, 16, 13, "Q1", 200, 1.666667, 0.787571, 601.8, "turnover", 100, 66.206441, 0.1100140, 386.08503, 200.0, 0, 1.0),
            (13, 99, 16, 13, "Q1", 300, 1.666667, 0.787571, 744.6, "turnover", 100, 81.916444, 0.1100140, 401.79503, 300.0, 0, 1.0),
            (13, 87, 16, 13, "Q1", 300, 1.666667, 0.787571, 877.2, "turnover", 100, 96.504304, 0.1100140, 416.38289, 300.0, 0, 1.0),
            (13, 88, 16, 13, "Q1", 300, 1.666667, 0.787571, 1300.0, "turnover", 100, 143.01823, 0.1100140, 462.89682, 300.0, 0, 1.0),
            (13, 84, 16, 13, "Q1", 300, 1.666667, 0.787571, 1020.0, "turnover", 100, 112.21430, 0.1100140, 432.09289, 300.0, 0, 1.0),
            (13, 75, 16, 13, "Q1", 200, 1.666667, 0.787571, 10000.0, "turnover", 100, 1100.1402, 0.1100140, 1420.0188, 200.0, 0, 1.0),
            (14, 97, 17, 14, "Q1", 200, 1.0, 1.0, 4947.0, "turnover", 100, None, None, None, 200.0, 0, 1.0),
            (14, 83, 17, 14, "Q1", 200, 1.0, 1.0, 5018.4, "turnover", 100, None, None, None, 200.0, 0, 1.0),
            (14, 74, 17, 14, "Q1", 200, 1.0, 1.0, 50000.0, "turnover", 100, None, None, None, 200.0, 0, 1.0),
        ],
    )

def test_winsorise_component_ratio(
    winsorise_ratio_component_test_input,
    winsorise_ratio_component_test_output,
):
    result = winsorise_ratio(
        df=winsorise_ratio_component_test_input,
        calibration_group_col="calibration_group_j",
        aux_col="aux_var",
        a_weight_col="a_weight",
        g_weight_col="g_weight",
        target_col="value",
        l_values_col="L_yi",
    )
    expected = winsorise_ratio_component_test_output
    assert_frame_equal(
        result[expected.columns].round(4),
        expected.round(4),
        check_dtype=False,
        atol=1e-4,
    )

@pytest.fixture(scope="function")
def winsorise_expansion_component_test_input():
    """Input data for expansion winsorisation - Q3 (no g_weight)."""
    return create_test_dataframe(
        [
            (
                "calibration_group_j",
                "ruref",
                "a_wt_band",
                "g_wt_band",
                "question",
                "value",
                "a_weight",
                "g_weight",
                "aux_var",
                "g_wgt_type",
                "L_yi",
            ),
            (1, 17, 1, 1, "Q3", 14, 2.666667, 1.0, 10.2, "no_g_wt", 10),
            (1, 1, 1, 1, "Q3", 28, 2.666667, 1.0, 30.0, "no_g_wt", 10),
            (1, 16, 1, 1, "Q3", 28, 2.666667, 1.0, 30.6, "no_g_wt", 10),
            (2, 8, 2, 2, "Q3", 42, 3.0, 1.0, 265.2, "no_g_wt", 10),
            (2, 9, 2, 2, "Q3", 42, 3.0, 1.0, 306.0, "no_g_wt", 10),
            (2, 13, 2, 2, "Q3", 42, 3.0, 1.0, 1100.0, "no_g_wt", 10),
            (3, 19, 4, 3, "Q3", 14, 1.0, 1.0, 3111.0, "no_g_wt", 10),
            (4, 18, 5, 4, "Q3", 28, 1.0, 1.0, 10230.6, "no_g_wt", 10),
            (5, 26, 6, 5, "Q3", 28, 2.0, 1.0, 30.6, "no_g_wt", 10),
            (5, 34, 6, 5, "Q3", 42, 2.0, 1.0, 40.8, "no_g_wt", 10),
            (5, 29, 7, 5, "Q3", 42, 3.333333, 1.0, 295.8, "no_g_wt", 10),
            (5, 23, 7, 5, "Q3", 42, 3.333333, 1.0, 306.0, "no_g_wt", 10),
            (5, 25, 7, 5, "Q3", 42, 3.333333, 1.0, 500.0, "no_g_wt", 10),
            (6, 33, 8, 6, "Q3", 14, 1.666667, 1.0, 1100.0, "no_g_wt", 10),
            (6, 21, 8, 6, "Q3", 28, 1.666667, 1.0, 846.6, "no_g_wt", 10),
            (6, 28, 8, 6, "Q3", 28, 1.666667, 1.0, 1020.0, "no_g_wt", 10),
            (7, 41, 9, 7, "Q3", 42, 1.0, 1.0, 6000.0, "no_g_wt", 10),
            (7, 27, 9, 7, "Q3", 42, 1.0, 1.0, 5018.4, "no_g_wt", 10),
            (8, 60, 10, 8, "Q3", 42, 2.0, 1.0, 200.0, "no_g_wt", 10),
            (8, 59, 11, 8, "Q3", 42, 3.0, 1.0, 200.0, "no_g_wt", 10),
            (8, 48, 11, 8, "Q3", 14, 3.0, 1.0, 359.0, "no_g_wt", 10),
            (9, 46, 12, 9, "Q3", 28, 3.0, 1.0, 2000.0, "no_g_wt", 10),
            (9, 50, 12, 9, "Q3", 28, 3.0, 1.0, 4000.0, "no_g_wt", 10),
            (9, 45, 12, 9, "Q3", 42, 3.0, 1.0, 744.6, "no_g_wt", 10),
            (10, 55, 13, 10, "Q3", 42, 1.0, 1.0, 10000.0, "no_g_wt", 10),
            (10, 56, 13, 10, "Q3", 42, 1.0, 1.0, 100000.0, "no_g_wt", 10),
            (11, 95, 14, 11, "Q3", 42, 10.0, 1.0, 80.0, "no_g_wt", 10),
            (12, 93, 15, 12, "Q3", 14, 8.5, 1.0, 130.0, "no_g_wt", 10),
            (12, 65, 15, 12, "Q3", 28, 8.5, 1.0, 5000.0, "no_g_wt", 10),
            (13, 100, 16, 13, "Q3", 28, 1.666667, 1.0, 601.8, "no_g_wt", 10),
            (13, 99, 16, 13, "Q3", 42, 1.666667, 1.0, 744.6, "no_g_wt", 10),
            (13, 87, 16, 13, "Q3", 42, 1.666667, 1.0, 877.2, "no_g_wt", 10),
            (13, 88, 16, 13, "Q3", 42, 1.666667, 1.0, 1300.0, "no_g_wt", 10),
            (13, 84, 16, 13, "Q3", 42, 1.666667, 1.0, 1020.0, "no_g_wt", 10),
            (13, 75, 16, 13, "Q3", 28, 1.666667, 1.0, 10000.0, "no_g_wt", 10),
            (14, 97, 17, 14, "Q3", 28, 1.0, 1.0, 4947.0, "no_g_wt", 10),
            (14, 83, 17, 14, "Q3", 28, 1.0, 1.0, 5018.4, "no_g_wt", 10),
            (14, 74, 17, 14, "Q3", 28, 1.0, 1.0, 50000.0, "no_g_wt", 10),
        ]
    )


@pytest.fixture(scope="function")
def winsorise_expansion_component_test_output():
    """Expected output for expansion winsorisation."""
    return create_test_dataframe(
        [
            (
                "calibration_group_j",
                "ruref",
                "a_wt_band",
                "g_wt_band",
                "question",
                "value",
                "a_weight",
                "g_weight",
                "aux_var",
                "g_wgt_type",
                "L_yi",
                "ki",
                "yi*",
                "outlier_flag",
                "oi",
                "mean_yi",
            ),
            (1, 17, 1, 1, "Q3", 14, 2.666667, 1.0, 10.2, "no_g_wt", 10, 29.333333, 14.0, 0, 1.0, 23.333333333333332),
            (1, 1, 1, 1, "Q3", 28, 2.666667, 1.0, 30.0, "no_g_wt", 10, 29.333333, 28.0, 0, 1.0, 23.333333333333332),
            (1, 16, 1, 1, "Q3", 28, 2.666667, 1.0, 30.6, "no_g_wt", 10, 29.333333, 28.0, 0, 1.0, 23.333333333333332),
            (2, 8, 2, 2, "Q3", 42, 3.0, 1.0, 265.2, "no_g_wt", 10, 47.0, 42.0, 0, 1.0, 42.0),
            (2, 9, 2, 2, "Q3", 42, 3.0, 1.0, 306.0, "no_g_wt", 10, 47.0, 42.0, 0, 1.0, 42.0),
            (2, 13, 2, 2, "Q3", 42, 3.0, 1.0, 1100.0, "no_g_wt", 10, 47.0, 42.0, 0, 1.0, 42.0),
            (3, 19, 4, 3, "Q3", 14, 1.0, 1.0, 3111.0, "no_g_wt", 10, None, 14.0, 0, 1.0, None),
            (4, 18, 5, 4, "Q3", 28, 1.0, 1.0, 10230.6, "no_g_wt", 10, None, 28.0, 0, 1.0, None),
            (5, 26, 6, 5, "Q3", 28, 2.0, 1.0, 30.6, "no_g_wt", 10, 45.0, 28.0, 0, 1.0, 35.0),
            (5, 34, 6, 5, "Q3", 42, 2.0, 1.0, 40.8, "no_g_wt", 10, 45.0, 42.0, 0, 1.0, 35.0),
            (5, 29, 7, 5, "Q3", 42, 3.333333, 1.0, 295.8, "no_g_wt", 10, 46.285714, 42.0, 0, 1.0, 42.0),
            (5, 23, 7, 5, "Q3", 42, 3.333333, 1.0, 306.0, "no_g_wt", 10, 46.285714, 42.0, 0, 1.0, 42.0),
            (5, 25, 7, 5, "Q3", 42, 3.333333, 1.0, 500.0, "no_g_wt", 10, 46.285714, 42.0, 0, 1.0, 42.0),
            (6, 33, 8, 6, "Q3", 14, 1.666667, 1.0, 1100.0, "no_g_wt", 10, 38.333333, 14.0, 0, 1.0, 23.333333333333332),
            (6, 21, 8, 6, "Q3", 28, 1.666667, 1.0, 846.6, "no_g_wt", 10, 38.333333, 28.0, 0, 1.0, 23.333333333333332),
            (6, 28, 8, 6, "Q3", 28, 1.666667, 1.0, 1020.0, "no_g_wt", 10, 38.333333, 28.0, 0, 1.0, 23.333333333333332),
            (7, 41, 9, 7, "Q3", 42, 1.0, 1.0, 6000.0, "no_g_wt", 10, None, 42.0, 0, 1.0, None),
            (7, 27, 9, 7, "Q3", 42, 1.0, 1.0, 5018.4, "no_g_wt", 10, None, 42.0, 0, 1.0, None),
            (8, 60, 10, 8, "Q3", 42, 2.0, 1.0, 200.0, "no_g_wt", 10, 52.0, 42.0, 0, 1.0, 42.0),
            (8, 59, 11, 8, "Q3", 42, 3.0, 1.0, 200.0, "no_g_wt", 10, 33.0, 36.0, 1, 0.8571428571428571, 28.0),
            (8, 48, 11, 8, "Q3", 14, 3.0, 1.0, 359.0, "no_g_wt", 10, 33.0, 14.0, 0, 1.0, 28.0),
            (9, 46, 12, 9, "Q3", 28, 3.0, 1.0, 2000.0, "no_g_wt", 10, 37.666666, 28.0, 0, 1.0, 32.666666666666664),
            (9, 50, 12, 9, "Q3", 28, 3.0, 1.0, 4000.0, "no_g_wt", 10, 37.666666, 28.0, 0, 1.0, 32.666666666666664),
            (9, 45, 12, 9, "Q3", 42, 3.0, 1.0, 744.6, "no_g_wt", 10, 37.666666, 39.111111, 1, 0.9312169312169312, 32.666666666666664),
            (10, 55, 13, 10, "Q3", 42, 1.0, 1.0, 10000.0, "no_g_wt", 10, None, 42.0, 0, 1.0, None),
            (10, 56, 13, 10, "Q3", 42, 1.0, 1.0, 100000.0, "no_g_wt", 10, None, 42.0, 0, 1.0, None),
            (11, 95, 14, 11, "Q3", 42, 10.0, 1.0, 80.0, "no_g_wt", 10, 43.111111, 42.0, 0, 1.0, 42.0),
            (12, 93, 15, 12, "Q3", 14, 8.5, 1.0, 130.0, "no_g_wt", 10, 22.333333, 14.0, 0, 1.0, 21.0),
            (12, 65, 15, 12, "Q3", 28, 8.5, 1.0, 5000.0, "no_g_wt", 10, 22.333333, 23.0, 1, 0.8214285714285714, 21.0),
            (13, 100, 16, 13, "Q3", 28, 1.666667, 1.0, 601.8, "no_g_wt", 10, 52.333333, 28.0, 0, 1.0, 37.333333333333336),
            (13, 99, 16, 13, "Q3", 42, 1.666667, 1.0, 744.6, "no_g_wt", 10, 52.333333, 42.0, 0, 1.0, 37.333333333333336),
            (13, 87, 16, 13, "Q3", 42, 1.666667, 1.0, 877.2, "no_g_wt", 10, 52.333333, 42.0, 0, 1.0, 37.333333333333336),
            (13, 88, 16, 13, "Q3", 42, 1.666667, 1.0, 1300.0, "no_g_wt", 10, 52.333333, 42.0, 0, 1.0, 37.333333333333336),
            (13, 84, 16, 13, "Q3", 42, 1.666667, 1.0, 1020.0, "no_g_wt", 10, 52.333333, 42.0, 0, 1.0, 37.333333333333336),
            (13, 75, 16, 13, "Q3", 28, 1.666667, 1.0, 10000.0, "no_g_wt", 10, 52.333333, 28.0, 0, 1.0, 37.333333333333336),
            (14, 97, 17, 14, "Q3", 28, 1.0, 1.0, 4947.0, "no_g_wt", 10, None, 28.0, 0, 1.0, None),
            (14, 83, 17, 14, "Q3", 28, 1.0, 1.0, 5018.4, "no_g_wt", 10, None, 28.0, 0, 1.0, None),
            (14, 74, 17, 14, "Q3", 28, 1.0, 1.0, 50000.0, "no_g_wt", 10, None, 28.0, 0, 1.0, None),
        ],
    )

def test_winsorise_component_expansion(
    winsorise_expansion_component_test_input,
    winsorise_expansion_component_test_output,
):
    result = winsorise_expansion(
        df=winsorise_expansion_component_test_input,
        strata_col="calibration_group_j",
        target_col="value",
        a_weight_col="a_weight",
        l_values_col="L_yi",
    )
    expected = winsorise_expansion_component_test_output
    assert_frame_equal(
        result[expected.columns].round(4),
        expected.round(4),
        check_dtype=False,
        atol=1e-4,
    )

@pytest.mark.parametrize("include_mean_yi", [False, True])
def test_apply_winsorise_ratio(
    winsorise_ratio_component_test_input,
    winsorise_ratio_component_test_output,
    include_mean_yi,
):
    """Return ratio Winsorisation results without the expansion-only mean."""
    if include_mean_yi:
        winsorise_ratio_component_test_input["mean_yi"] = 150.0

    result = apply_winsorise_ratio(
        winsorise_ratio_component_test_input,
        calibration_group_col="calibration_group_j",
        aux_col="aux_var",
        a_weight_col="a_weight",
        g_weight_col="g_weight",
        target_col="value",
        l_values_col="L_yi",
    )

    expected = winsorise_ratio_component_test_output
    assert_frame_equal(
        result.round(4),
        expected.round(4),
        check_dtype=False,
        atol=1e-4,
    )

@pytest.mark.parametrize("include_mu_factor", [False, True])
def test_apply_winsorise_expansion(
    winsorise_expansion_component_test_input,
    winsorise_expansion_component_test_output,
    include_mu_factor,
):
    """Return expansion results without the ratio-only factor."""
    if include_mu_factor:
        winsorise_expansion_component_test_input["mu_factor"] = 1.5

    result = apply_winsorise_expansion(
        winsorise_expansion_component_test_input,
        strata_col="a_wt_band",
        target_col="value",
        a_weight_col="a_weight",
        l_values_col="L_yi",
    )

    expected = winsorise_expansion_component_test_output
    assert_frame_equal(
        result.round(4),
        expected.round(4),
        check_dtype=False,
        check_like=True,
        atol=1e-4,
    )


def test_detect_outlier(
    winsorise_ratio_component_test_input,
    winsorise_ratio_component_test_output,
    winsorise_expansion_component_test_input,
    winsorise_expansion_component_test_output,
):
    """Test ratio and expansion Winsorisation for their respective questions."""
    df = pd.concat(
        [
            winsorise_ratio_component_test_input,
            winsorise_expansion_component_test_input,
        ],
        ignore_index=True,
    )

    result = detect_outlier(
        df,
        question_no_col="question",
        strata_col="a_wt_band",
        expansion_methods=["Q3"],
        calibration_group_col="calibration_group_j",
        aux_col="aux_var",
        a_weight_col="a_weight",
        g_weight_col="g_weight",
        target_col="value",
        l_values_col="L_yi",
    )

    expected = pd.concat(
        [
            winsorise_ratio_component_test_output,
            winsorise_expansion_component_test_output,
        ],
        ignore_index=True,
        sort=False,
    )
    assert_frame_equal(
        result.round(4),
        expected.round(4),
        check_dtype=False,
        check_like=True,
        atol=1e-4,
    )
