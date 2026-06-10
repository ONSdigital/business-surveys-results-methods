"""Tests for run_estimation in estimation_main."""

import pandas.testing as pdt
from bsrm.estimation.estimation_main import run_estimation
import pytest
import pandas as pd


@pytest.fixture(scope="function")
def estimation_input():
    """Data from Estimation_input from the methodology paper."""
    columns = [
        "ruref",
        "cell_no",
        "k",
        "nh",
        "nk",
        "rusic2007",
        "region",
        "y",
        "x",
        "Nh",
        "Nk",
        "sum_y",
        "sum_x",
    ]
    data = [
        [17, "1", 1, 3, 3, 11010, "AA", 10.2, 1.0, 8, 8, 416.4, 40.0],
        [1, "1", 1, 3, 3, 11010, "AA", 30.0, 2.0, 8, 8, 416.4, 40.0],
        [16, "1", 1, 3, 3, 11010, "AA", 30.6, 3.0, 8, 8, 416.4, 40.0],
        [8, "2", 2, 3, 4, 11010, "AA", 265.2, 26.0, 9, 10, 4030.0, 278.0],
        [9, "2", 2, 3, 4, 11010, "AA", 306.0, 30.0, 9, 10, 4030.0, 278.0],
        [13, "2", 2, 3, 4, 11010, "AA", 1100.0, 48.0, 9, 10, 4030.0, 278.0],
        [14, "3", 2, 1, 4, 11010, "AA", 632.4, 62.0, 1, 10, 4030.0, 278.0],
        [19, "4", 3, 1, 1, 11010, "AA", 3111.0, 305.0, 1, 1, 3111.0, 305.0],
        [18, "5", 4, 1, 1, 11010, "AA", 10230.6, 1003.0, 1, 1, 10230.6, 1003.0],
        [26, "6", 5, 2, 5, 11020, "AA", 30.6, 3.0, 4, 14, 2876.6, 273.0],
        [34, "6", 5, 2, 5, 11020, "AA", 40.8, 4.0, 4, 14, 2876.6, 273.0],
        [29, "7", 5, 3, 5, 11020, "AA", 295.8, 29.0, 10, 14, 2876.6, 273.0],
        [23, "7", 5, 3, 5, 11020, "AA", 306.0, 30.0, 10, 14, 2876.6, 273.0],
        [25, "7", 5, 3, 5, 11020, "AA", 500.0, 40.0, 10, 14, 2876.6, 273.0],
        [33, "8", 6, 3, 3, 11020, "AA", 1100.0, 82.0, 5, 5, 4143.8, 443.0],
        [21, "8", 6, 3, 3, 11020, "AA", 846.6, 83.0, 5, 5, 4143.8, 443.0],
        [28, "8", 6, 3, 3, 11020, "AA", 1020.0, 100.0, 5, 5, 4143.8, 443.0],
        [41, "9", 7, 2, 2, 11020, "AA", 6000.0, 485.0, 2, 2, 11018.4, 977.0],
        [27, "9", 7, 2, 2, 11020, "AA", 5018.4, 492.0, 2, 2, 11018.4, 977.0],
        [60, "10", 8, 1, 3, 11030, "AA", 200.0, 8.0, 2, 8, 2791.6, 159.0],
        [59, "11", 8, 2, 3, 11030, "AA", 200.0, 16.0, 6, 8, 2791.6, 159.0],
        [48, "11", 8, 2, 3, 11030, "AA", 359.0, 29.0, 6, 8, 2791.6, 159.0],
        [46, "12", 9, 3, 3, 11030, "AA", 2000.0, 59.0, 9, 9, 22355.8, 728.0],
        [50, "12", 9, 3, 3, 11030, "AA", 4000.0, 60.0, 9, 9, 22355.8, 728.0],
        [45, "12", 9, 3, 3, 11030, "AA", 744.6, 73.0, 9, 9, 22355.8, 728.0],
        [55, "13", 10, 2, 2, 11030, "AA", 10000.0, 290.0, 2, 2, 110000.0, 5290.0],
        [56, "13", 10, 2, 2, 11030, "AA", 100000.0, 5000.0, 2, 2, 110000.0, 5290.0],
        [95, "14", 11, 1, 1, 11040, "AA", 80.0, 7.0, 10, 10, 1474.0, 53.0],
        [93, "15", 12, 2, 2, 11040, "AA", 130.0, 11.0, 17, 17, 21620.2, 439.0],
        [65, "15", 12, 2, 2, 11040, "AA", 5000.0, 30.0, 17, 17, 21620.2, 439.0],
        [100, "16", 13, 6, 6, 11040, "AA", 601.8, 59.0, 10, 10, 19090.2, 1027.0],
        [99, "16", 13, 6, 6, 11040, "AA", 744.6, 73.0, 10, 10, 19090.2, 1027.0],
        [87, "16", 13, 6, 6, 11040, "AA", 877.2, 86.0, 10, 10, 19090.2, 1027.0],
        [88, "16", 13, 6, 6, 11040, "AA", 1300.0, 92.0, 10, 10, 19090.2, 1027.0],
        [84, "16", 13, 6, 6, 11040, "AA", 1020.0, 100.0, 10, 10, 19090.2, 1027.0],
        [75, "16", 13, 6, 6, 11040, "AA", 10000.0, 305.0, 10, 10, 19090.2, 1027.0],
        [97, "17", 14, 3, 3, 11040, "AA", 4947.0, 485.0, 3, 3, 59965.4, 1980.0],
        [83, "17", 14, 3, 3, 11040, "AA", 5018.4, 492.0, 3, 3, 59965.4, 1980.0],
        [74, "17", 14, 3, 3, 11040, "AA", 50000.0, 1003.0, 3, 3, 59965.4, 1980.0],
    ]
    return pd.DataFrame(columns=columns, data=data)


@pytest.fixture(scope="function")
def estimation_output():
    """Data from Estimation_output from the methodology paper."""
    columns = [
        "k",
        "cell_no",
        "sum_y",
        "sum_x",
        "sum_ay",
        "sum_ax",
        "a_wt",
        "g_wt_y",
        "g_wt_x",
    ]
    data = [
        [1, "1", 416.4, 40.0, 188.8, 16.0, 2.6667, 2.2055, 2.5],
        [2, "2", 4030.0, 278.0, 5646.0, 374.0, 3.0, 0.7138, 0.7433],
        [2, "3", 4030.0, 278.0, 5646.0, 374.0, 1.0, 0.7138, 0.7433],
        [3, "4", 3111.0, 305.0, 3111.0, 305.0, 1.0, 1.0, 1.0],
        [4, "5", 10230.6, 1003.0, 10230.6, 1003.0, 1.0, 1.0, 1.0],
        [5, "6", 2876.6, 273.0, 3815.4667, 344.0, 2.0, 0.7539, 0.7936],
        [5, "7", 2876.6, 273.0, 3815.4667, 344.0, 3.3333, 0.7539, 0.7936],
        [6, "8", 4143.8, 443.0, 4944.3333, 441.6667, 1.6667, 0.8381, 1.003],
        [7, "9", 11018.4, 977.0, 11018.4, 977.0, 1.0, 1.0, 1.0],
        [8, "10", 2791.6, 159.0, 2077.0, 151.0, 2.0, 1.3441, 1.053],
        [8, "11", 2791.6, 159.0, 2077.0, 151.0, 3.0, 1.3441, 1.053],
        [9, "12", 22355.8, 728.0, 20233.8, 576.0, 3.0, 1.1049, 1.2639],
        [10, "13", 110000.0, 5290.0, 110000.0, 5290.0, 1.0, 1.0, 1.0],
        [11, "14", 1474.0, 53.0, 800.0, 70.0, 10.0, 1.8425, 0.7571],
        [12, "15", 21620.2, 439.0, 43605.0, 348.5, 8.5, 0.4958, 1.2597],
        [13, "16", 19090.2, 1027.0, 24239.3333, 1191.6667, 1.6667, 0.7876, 0.8618],
        [14, "17", 59965.4, 1980.0, 59965.4, 1980.0, 1.0, 1.0, 1.0],
    ]
    return pd.DataFrame(columns=columns, data=data)


# Test for the main estimation functon
def test_run_estimation(estimation_input):
    """test the run_estimation function with the estimation_input data."""
    weighted_df, qa_df = run_estimation(
        df=estimation_input,
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
        aux_col="x",
        univ_aux_col="sum_x",
        incl_g_wts=True,
    )

    assert isinstance(weighted_df, pd.DataFrame)
    assert isinstance(qa_df, pd.DataFrame)

    assert weighted_df.shape[0] > 0
    assert qa_df.shape[0] > 0


# test 2 without g-weights
def test_run_estimation_no_g_weights(estimation_input):
    """test without g-weights."""
    weighted_df, qa_df = run_estimation(
        df=estimation_input,
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
        aux_col="",
        univ_aux_col="",
        incl_g_wts=False,
    )
    assert isinstance(weighted_df, pd.DataFrame)
    assert isinstance(qa_df, pd.DataFrame)


def build_actual_table(weighted_df):
    """Transform weighted_df to match methodology output shape."""
    weighted_df["ay"] = weighted_df["y"] * weighted_df["a_weight"]
    weighted_df["ax"] = weighted_df["x"] * weighted_df["a_weight"]

    weighted_df["sum_ay"] = weighted_df.groupby("k")["ay"].transform("sum")
    weighted_df["sum_ax"] = weighted_df.groupby("k")["ax"].transform("sum")

    weighted_df["g_wt_y"] = weighted_df["sum_y"] / weighted_df["sum_ay"]
    weighted_df["g_wt_x"] = weighted_df["sum_x"] / weighted_df["sum_ax"]

    cols = [
        "k",
        "cell_no",
        "sum_y",
        "sum_x",
        "sum_ay",
        "sum_ax",
        "a_wt",
        "g_wt_y",
        "g_wt_x",
    ]
    actual = weighted_df.groupby(["k", "cell_no"], as_index=False).agg(
        sum_y=("sum_y", "first"),
        sum_x=("sum_x", "first"),
        sum_ay=("sum_ay", "first"),
        sum_ax=("sum_ax", "first"),
        a_wt=("a_weight", "first"),
        g_wt_y=("g_wt_y", "first"),
        g_wt_x=("g_wt_x", "first"),
    )
    return actual[cols].round(4).sort_values(["k", "cell_no"]).reset_index(drop=True)


def test_run_estimation_output_matches_expected(estimation_input, estimation_output):
    """Test that run_estimation output matches the methodology paper results."""
    weighted_df, qa_df = run_estimation(
        df=estimation_input,
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
        aux_col="x",
        univ_aux_col="sum_x",
        incl_g_wts=True,
    )

    actual = build_actual_table(weighted_df)
    expected = (
        estimation_output.round(4).sort_values(["k", "cell_no"]).reset_index(drop=True)
    )

    pdt.assert_frame_equal(actual, expected, check_dtype=False)
    assert isinstance(qa_df, pd.DataFrame)
    assert not qa_df.empty
