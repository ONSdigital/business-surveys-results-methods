"""Tests for functions in calculate_weights."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from bsrm.estimation.calculate_weights import (
    calc_lower_n,
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


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


@pytest.mark.skip(reason="WIP - Testing in new branch")
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


@pytest.mark.skip(reason="WIP - Testing in new branch")
def test_calculate_a_weight_invalid_column(estimation_input):
    """Test calculate_a_weights with invalid column."""
    with pytest.raises(KeyError):
        calculate_a_weights(estimation_input, "cell_no", "invalid_column", "Nh")


@pytest.mark.skip(reason="WIP - Testing in new branch")
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


@pytest.mark.skip(reason="WIP - Testing in new branch")
def test_calculate_g_weight_invalid_column(estimation_input):
    """Test calculate_g_weights with invalid column."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    with pytest.raises(KeyError):
        calculate_g_weights(df_a, "k", "invalid_column", "sum_x")


@pytest.mark.skip(reason="WIP - Testing in new branch")
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


@pytest.mark.skip(reason="WIP - Testing in new branch")
def test_create_weights_qa_df_invalid_column(estimation_input):
    """Test create_weights_qa_df with invalid column."""
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "k", "x", "sum_x")
    with pytest.raises(KeyError):
        create_weights_qa_df(df_g, "invalid_column", True)
