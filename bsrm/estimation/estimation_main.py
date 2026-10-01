"""Main file for the estimation module."""

import logging
import pandas as pd

from bsrm.estimation.calculate_weights import (
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)
from bsrm.estimation.apply_weights import apply_weights

EstMainLogger = logging.getLogger(__name__)


def run_estimation(
    df: pd.DataFrame,
    strata_col: str,
    ru_col: str,
    univ_count_col: str,
    aux_col: str = "",
    univ_aux_col: str = "",
    incl_g_wts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the estimation module.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data were estimation will be applied.
    strata_col : str
        The column representing the strata.
    ru_col : str
        The column representing the reference unit.
    aux_col: str
        The column representing the auxiliary variable.
    univ_count_col : str
        The column representing the universe count.
    univ_aux_col : str
        The column representing the universe auxiliary variable.
    incl_g_wts : bool
        Whether to include g weights in the calculation.
    round_val :int
        The number of decimal places to round the final results to

    Returns
    -------
    pd.DataFrame
        The main dataset after the application of estimation.
    """
    EstMainLogger.info("Starting estimation weights calculation...")

    # calculate the weights
    weighted_df = calculate_a_weights(df, strata_col, ru_col, univ_count_col)

    # if required also calculate g weights
    if incl_g_wts:
        weighted_df = calculate_g_weights(weighted_df, strata_col, aux_col, univ_aux_col)

    # Create a QA dataframe
    qa_frame = create_weights_qa_df(weighted_df, strata_col, incl_g_wts)

    # drop intermediate calculation columns
    if incl_g_wts:
        weighted_df = weighted_df.drop(columns=["univ_aux_sum", "aux_col_sum"], axis=1)

    return weighted_df, qa_frame


# example usage
if __name__ == "__main__":
    input_path = "path/to/input.csv"

    df = pd.read_csv(input_path)

    a_weight_cols = ["question1", "question2"]
    g_weight_cols = ["question3"]
    incl_g_wts = True
    round_val = 2

    ru_col = "reference"
    univ_count_col = "uni_count"
    aux_col = "employment"
    univ_aux_col = "uni_employment"
    strata_col = "cellnumber"

    # call the method to return the dataframe with new weights columns, and qa dataframe
    weighted_df, qa_df = run_estimation(
        df,
        strata_col,
        ru_col,
        univ_count_col,
        aux_col,
        univ_aux_col,
        incl_g_wts,
    )

    # call the method to return the dataframe with the new weights applied
    # to the specified columns, and qa dataframe
    final_weighted_df = apply_weights(
        weighted_df, a_weight_cols, g_weight_cols, incl_g_wts, round_val
    )
