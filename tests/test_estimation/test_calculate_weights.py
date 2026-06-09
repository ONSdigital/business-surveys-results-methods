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


@pytest.fixture
def sample_data():
    """
    Sample data for testing calculate weight functions.
    """
    col_name = ["ruref", "cell_no", "k", "nh", "nk", "y", "x", "Nh", "Nk", "sum_x"]
    data = [
        ["A", 1, 1, 3, 5, 10.0, 2.0, 6, 20, 40.0],
        ["B", 1, 1, 3, 5, 20.0, 4.0, 6, 20, 40.0],
        ["C", 1, 1, 3, 5, 30.0, 0.0, 6, 20, 40.0],
        ["D", 2, 2, 2, 4, 15.0, 10.0, 4, 18, 60.0],
        ["E", 2, 2, 2, 4, 25.0, 20.0, 4, 18, 60.0],
    ]
    return pd.DataFrame(columns=col_name, data=data)


def test_calc_lower_n_for_stratum_1(sample_data):
    """Test for lower n unique business in stratum 1."""
    cell_1 = sample_data[sample_data["cell_no"] == 1]
    result = calc_lower_n(cell_1, "ruref")
    assert result == 3


def test_calc_lower_n_invalid_column(sample_data):
    """Test for lower n unique business with invalid column."""
    with pytest.raises(KeyError):
        calc_lower_n(sample_data, "invalid_column")


def test_calculate_a_weights(sample_data):
    """Test a weight calculation a= Nh/nh or a = N/n.
    For stratum (k)_weight= 6/3 = 2.0,
    for stratum (k)_weight=4/2=2.0."""
    result = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    actual = (
        result[["cell_no", "a_weight"]]
        .drop_duplicates()
        .sort_values("cell_no")
        .reset_index(drop=True)
    )

    expected = pd.DataFrame({"cell_no": [1, 2], "a_weight": [2.0, 2.0]})

    assert_frame_equal(actual, expected, check_dtype=False)


def test_calculate_a_weight_invalid_column(sample_data):
    """Test calculate_a_weights with invalid column."""
    with pytest.raises(KeyError):
        calculate_a_weights(sample_data, "cell_no", "invalid_column", "Nh")


def test_g_weight_calculation(sample_data):
    """Test g_weight calculation: g = sum_x/sum_ax.
    where sum_ax = sum(a_weight*x)
    For stratum (k)1 sum_ax = 2(2+4+0) 12, g_weight = 40/12 = 3.333333333"""

    df_a = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    result = calculate_g_weights(df_a, "k", "x", "sum_x")
    # Extract unique g_weights per calibration group.
    actual = (
        result[["k", "g_weight"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values("k")
        .reset_index(drop=True)
    )
    # check g_weight for k=1 is 3.333333333
    assert actual.loc[0, "g_weight"] == pytest.approx(40.0 / 12.0, abs=1e-6)


def test_calculate_g_weight_invalid_column(sample_data):
    """Test calculate_g_weights with invalid column."""
    df_a = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    with pytest.raises(KeyError):
        calculate_g_weights(df_a, "k", "invalid_column", "sum_x")


def test_create_weights_qa_df(sample_data):
    """Test to check if it returns correct QA dataframe with expected columns and values."""
    df_a = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "k", "x", "sum_x")
    result = create_weights_qa_df(df_g, "cell_no", incl_g_wts=True).rename(
        columns={
            "N": "Nh",
            "n": "nh",
            "univ_aux_sum": "sum_x",
            "aux_col_sum": "x_sum",
            "a_weight": "a_wt",
            "g_weight": "g_wt_x",
        }
    )
    expected_columns = {"cell_no", "Nh", "nh", "sum_x", "x_sum", "a_wt", "g_wt_x"}

    expected_row_count = sample_data["cell_no"].nunique()
    actual_row_count = result.shape[0]
    assert actual_row_count == expected_row_count

    actual_columns = set(result.columns)
    assert (
        actual_columns == expected_columns
    ), f"Expected columns {expected_columns}, got {actual_columns}"


def test_create_weights_qa_df_without_g_weights(sample_data):
    """Exclude g_weight column when incl_g_wts=False."""
    df_a = calculate_a_weights(sample_data, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "k", "x", "sum_x")
    result = create_weights_qa_df(df_g, "cell_no", incl_g_wts=False)

    expected_columns = {"cell_no", "N", "n", "a_weight"}

    assert set(result.columns) == expected_columns
