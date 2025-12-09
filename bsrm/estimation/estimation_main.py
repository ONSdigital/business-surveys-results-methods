"""Main file for the estimation module."""

import logging
import pandas as pd

from bsrm.estimation.calculate_weights import calculate_weights
from bsrm.estimation.apply_weights import apply_weights

EstMainLogger = logging.getLogger(__name__)


def run_estimation(
    df: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str] | None = None,
    incl_g_wts: bool = True,
    round_val: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the estimation module.

    Parameters
    ----------
        df (pd.DataFrame): The survey data were estimation will be applied.
        a_weight_cols (list[str]): List of columns to apply a_weight to.
        g_weight_cols (list[str]): List of columns to apply g_weight to.
        incl_g_wts (bool): Whether to include g weights in the calculation.
        round_val (int): The number of decimal places to round the final results to

    Returns
    -------
        pd.DataFrame: The main dataset after the application of estimation.
    """
    EstMainLogger.info("Starting estimation weights calculation...")

    # calculate the weights
    weighted_df, qa_df = calculate_weights(df, incl_g_wts)

    # apply the weights to the dataframe and apply the specified rounding
    final_weighted_df = apply_weights(
        weighted_df, a_weight_cols, g_weight_cols, incl_g_wts, round_val
    )

    return final_weighted_df, qa_df


# example usage
if __name__ == "__main__":
    input_path = "path/to/input.csv"

    df = pd.read_csv(input_path)
    a_weight_cols = ["question1", "question2"]
    g_weight_cols = ["question3"]
    # call the method to return the weighted dataframe and qa dataframe
    weighted_df, qa_df = run_estimation(
        df,
        a_weight_cols,
        g_weight_cols,
        incl_g_wts=True,
    )
