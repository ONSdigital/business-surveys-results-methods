"""Tests for functions in calculate_weights."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from bsrm.estimation.calculate_weights import (
    calc_aux_col_sum,
    calc_lower_n,
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


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
        [1, "1", 416.4, 40.0, 188.8, 16.0, 2.666666667, 2.205508475, 2.5],
        [2, "2", 4030.0, 278.0, 5646.0, 374.0, 3.0, 0.713779667, 0.743315508],
        [2, "3", 4030.0, 278.0, 5646.0, 374.0, 1.0, 0.713779667, 0.743315508],
        [3, "4", 3111.0, 305.0, 3111.0, 305.0, 1.0, 1.0, 1.0],
        [4, "5", 10230.6, 1003.0, 10230.6, 1003.0, 1.0, 1.0, 1.0],
        [5, "6", 2876.6, 273.0, 3815.466667, 344.0, 2.0, 0.753931367, 0.793604651],
        [
            5,
            "7",
            2876.6,
            273.0,
            3815.466667,
            344.0,
            3.333333333,
            0.753931367,
            0.793604651,
        ],
        [
            6,
            "8",
            4143.8,
            443.0,
            4944.333333,
            441.6666667,
            1.666666667,
            0.838090744,
            1.003018868,
        ],
        [7, "9", 11018.4, 977.0, 11018.4, 977.0, 1.0, 1.0, 1.0],
        [8, "10", 2791.6, 159.0, 2077.0, 151.0, 2.0, 1.344053924, 1.052980132],
        [8, "11", 2791.6, 159.0, 2077.0, 151.0, 3.0, 1.344053924, 1.052980132],
        [9, "12", 22355.8, 728.0, 20233.8, 576.0, 3.0, 1.104874023, 1.263888889],
        [10, "13", 110000.0, 5290.0, 110000.0, 5290.0, 1.0, 1.0, 1.0],
        [11, "14", 1474.0, 53.0, 800.0, 70.0, 10.0, 1.8425, 0.757142857],
        [12, "15", 21620.2, 439.0, 43605.0, 348.5, 8.5, 0.495819287, 1.259684362],
        [
            13,
            "16",
            19090.2,
            1027.0,
            24239.33333,
            1191.666667,
            1.666666667,
            0.787571165,
            0.861818182,
        ],
        [14, "17", 59965.4, 1980.0, 59965.4, 1980.0, 1.0, 1.0, 1.0],
    ]
    return pd.DataFrame(columns=columns, data=data)


def test_calc_aux_col_sum_expected(estimation_input):
    """Expected output for calc_aux_col_sum."""
    k1 = estimation_input[estimation_input["k"] == 1]
    result = calc_aux_col_sum(k1, "x")
    assert result == 6.0


def test_calc_lower_n_expected(estimation_input):
    """Expected output for calc_lower_n."""
    cell_1 = estimation_input[estimation_input["cell_no"] == "1"]
    result = calc_lower_n(cell_1, "ruref")
    assert result == 3


def test_calc_lower_n_single_row(estimation_input):
    """Expected output for calc_lower_n with single row."""
    cell_4 = estimation_input[estimation_input["cell_no"] == "4"]
    result = calc_lower_n(cell_4, "ruref")
    assert result == 1


def test_calc_lower_n_invalid_column(estimation_input):
    """Test calc_lower_n with invalid column."""
    with pytest.raises(KeyError):
        calc_lower_n(estimation_input, "invalid_column")


def test_calculate_a_weights_expected(estimation_input, estimation_output):
    """test calculate_a_weights with expected output."""
    result = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    result_qa = (
        result[["cell_no", "a_weight"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values("cell_no")
        .reset_index(drop=True)
    )
    expected = (
        estimation_output[["cell_no", "a_wt"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"a_wt": "a_weight"})
        .sort_values("cell_no")
        .reset_index(drop=True)
    )
    assert_frame_equal(result_qa, expected, check_dtype=False, atol=1e-6)


def test_calculate_a_weight_invalid_column(estimation_input):
    """Test calculate_a_weights with invalid column."""
    with pytest.raises(KeyError):
        calculate_a_weights(estimation_input, "cell_no", "invalid_column", "Nh")


def test_g_weight_expected(estimation_input, estimation_output):
    """Test calculate_g_weights with expected output."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    result = calculate_g_weights(df_a, "k", "x", "sum_x")
    # Extract only k and g_weight, one row per k
    result_qa = (
        result[["k", "g_weight"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values("k")
        .reset_index(drop=True)
    )
    # Get expected from estimation_output
    expected = (
        estimation_output[["k", "g_wt_x"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"g_wt_x": "g_weight"})
        .sort_values("k")
        .reset_index(drop=True)
    )
    assert_frame_equal(result_qa, expected, check_dtype=False, atol=1e-6)


def test_calculate_g_weight_invalid_column(estimation_input):
    """Test calculate_g_weights with invalid column."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    with pytest.raises(KeyError):
        calculate_g_weights(df_a, "k", "invalid_column", "sum_x")


def test_create_weights_qa_df_expected(estimation_input):
    """Test create_weights_qa_df with expected output."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "k", "x", "sum_x")
    result = create_weights_qa_df(df_g, "cell_no", True)
    result = result.rename(
        columns={
            "N": "Nh",
            "n": "nh",
            "univ_aux_sum": "sum_x",
            "aux_col_sum": "x_sum",
            "a_weight": "a_wt",
            "g_weight": "g_wt_x",
        }
    )
    # Convert cell_no to int for proper numeric sorting
    result["cell_no"] = result["cell_no"].astype(int)
    result = result.sort_values("cell_no").reset_index(drop=True)

    expected_columns = [
        "cell_no",
        "Nh",
        "nh",
        "sum_x",
        "x_sum",
        "a_wt",
        "g_wt_x",
    ]
    expected_data = [
        [1, 8, 3, 40.0, 6.0, 2.666666667, 2.5],
        [2, 9, 3, 278.0, 166.0, 3.0, 0.743315508],
        [3, 1, 1, 278.0, 166.0, 1.0, 0.743315508],
        [4, 1, 1, 305.0, 305.0, 1.0, 1.0],
        [5, 1, 1, 1003.0, 1003.0, 1.0, 1.0],
        [6, 4, 2, 273.0, 106.0, 2.0, 0.793604651],
        [7, 10, 3, 273.0, 106.0, 3.333333333, 0.793604651],
        [8, 5, 3, 443.0, 265.0, 1.666666667, 1.003018868],
        [9, 2, 2, 977.0, 977.0, 1.0, 1.0],
        [10, 2, 1, 159.0, 53.0, 2.0, 1.052980132],
        [11, 6, 2, 159.0, 53.0, 3.0, 1.052980132],
        [12, 9, 3, 728.0, 192.0, 3.0, 1.263888889],
        [13, 2, 2, 5290.0, 5290.0, 1.0, 1.0],
        [14, 10, 1, 53.0, 7.0, 10.0, 0.757142857],
        [15, 17, 2, 439.0, 41.0, 8.5, 1.259684362],
        [16, 10, 6, 1027.0, 715.0, 1.666666667, 0.861818182],
        [17, 3, 3, 1980.0, 1980.0, 1.0, 1.0],
    ]
    expected = pd.DataFrame(columns=expected_columns, data=expected_data)
    assert_frame_equal(result, expected, check_dtype=False, atol=1e-6)


def test_create_weights_qa_df_invalid_column(estimation_input):
    """Test create_weights_qa_df with invalid column."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "k", "x", "sum_x")
    with pytest.raises(KeyError):
        create_weights_qa_df(df_g, "invalid_column", True)
