"""Tests for run_estimation in estimation_main."""

import pytest
from pandas.testing import assert_frame_equal

from business_surveys_results_methods.estimation.estimation_main import run_estimation
from business_surveys_results_methods.estimation.calculate_weights import (
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)


@pytest.fixture(scope="function")
def estimation_result(estimation_input):
    """Run estimation on the methodology paper input and return weighted_df and qa_df."""
    return run_estimation(
        estimation_input.copy(),
        strata_col="cell_no",
        ru_col="ruref",
        univ_count_col="Nh",
        aux_col="x",
        univ_aux_col="sum_x",
        incl_g_wts=True,
    )


def test_run_estimation_expected(estimation_input, estimation_result):
    """run_estimation returns a weighted df with correct a_weight and g_weight values."""
    weighted_df, qa_df = estimation_result

    # build expected weights independently using calculate functions
    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    expected = calculate_g_weights(df_a, "cell_no", "x", "sum_x")

    assert_frame_equal(
        weighted_df[["a_weight", "g_weight"]].reset_index(drop=True),
        expected[["a_weight", "g_weight"]].reset_index(drop=True),
        check_dtype=False,
        atol=1e-6,
    )


def test_run_estimation_qa_df_expected(estimation_input, estimation_result):
    """run_estimation returns a QA df matching create_weights_qa_df output."""
    weighted_df, qa_df = estimation_result

    df_a = calculate_a_weights(estimation_input, "cell_no", "ruref", "Nh")
    df_g = calculate_g_weights(df_a, "cell_no", "x", "sum_x")
    expected_qa = create_weights_qa_df(df_g, "cell_no", True)

    assert_frame_equal(qa_df, expected_qa, check_dtype=False, atol=1e-6)


def test_run_estimation_invalid_strata_col(estimation_input):
    """run_estimation raises KeyError when strata_col does not exist."""
    with pytest.raises(KeyError):
        run_estimation(
            estimation_input.copy(),
            strata_col="invalid_col",
            ru_col="ruref",
            univ_count_col="Nh",
            incl_g_wts=False,
        )
