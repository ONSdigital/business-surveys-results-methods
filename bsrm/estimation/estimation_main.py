"""Main file for the estimation module.

In general pipeline use, individual functions from this module could be imported and used
separately with appropriate filtering and preprocessing steps.

This script acts as a runner for the estimation functions to be run in sequence.

To use this script and run it directly, provide the appropriate input and output file paths
and variable names in the section under `if __name__ == "__main__":`.
"""

import logging
import pandas as pd

from dataclasses import dataclass

from bsrm.estimation.calculate_weights import (
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)
from bsrm.estimation.apply_weights import apply_weights

EstMainLogger = logging.getLogger(__name__)


@dataclass
class EstimationConfig:
    """Configuration for the estimation module.

    Attributes
    ----------
    a_weight_columns: list[str]
        The columns the a_weights should be applied to.
    g_weight_columns: list[str]
        The columns the g_weights should be applied to (if applicable).
    incl_g_wts: bool
        Whether to include g weights in the calculation.
    round_val: int
        The number of decimal places to round the final results to.
    ru_col: str
        The column representing the reference unit.
    univ_count_col: str
        The column representing the universe count.
    aux_cols: list[str]
        The columns representing the auxiliary variables.
    univ_aux_cols: list[str]
        The columns representing the universe sums for the auxiliary variables.
    a_weight_col: str
        The column representing the a weight band (strata).
    g_weight_col: str
        The column representing the g weight band (if applicable).
    """

    a_weight_columns: list[str]
    g_weight_columns: list[str] | None
    incl_g_wts: bool
    round_val: int
    ru_col: str
    univ_count_col: str
    aux_cols: list[str] | None
    univ_aux_cols: list[str] | None
    a_weight_col: str
    g_weight_col: str | None


def run_estimation(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str | None,
    ru_col: str,
    univ_count_col: str,
    aux_cols: list[str] | None = None,
    univ_aux_cols: list[str] | None = None,
    incl_g_wts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the estimation module.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data were estimation will be applied.
    a_weight_col: str
        The column representing the a weight (strata).
    g_weight_col: str
        The column representing the g weight (if applicable).
    ru_col: str
        The column representing the reference unit.
    univ_count_col: str
        The column representing the universe count.
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.
    univ_aux_cols: list[str] | None
        The columns representing the universe auxiliary variables.
    incl_g_wts: bool
        Whether to include g weights in the calculation.
    round_val: int
        The number of decimal places to round the final results to

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The first element is the main dataset after the application of estimation.
        The second element is the QA dataframe.
    """
    EstMainLogger.info("Starting estimation weights calculation...")

    # calculate the weights
    weighted_df = calculate_a_weights(df, a_weight_col, ru_col, univ_count_col)

    # if required also calculate g weights
    if incl_g_wts:
        if g_weight_col is None or aux_cols is None or univ_aux_cols is None:
            msg = "G weights cannot be calculated due to missing columns."
            EstMainLogger.error(msg)
            raise ValueError(msg)

        if len(aux_cols) != len(univ_aux_cols):
            msg = "aux_cols and univ_aux_cols must be the same length."
            EstMainLogger.error(msg)
            raise ValueError(msg)

        for aux_col, univ_aux_col in zip(aux_cols, univ_aux_cols, strict=False):
            weighted_df = calculate_g_weights(weighted_df, g_weight_col, aux_col, univ_aux_col)

    # Create a QA dataframe
    qa_frame = create_weights_qa_df(weighted_df, a_weight_col, incl_g_wts)

    # drop intermediate calculation columns
    if incl_g_wts:
        weighted_df = weighted_df.drop(columns=["univ_aux_sum", "aux_col_sum"], axis=1)

    return weighted_df, qa_frame


# example usage
if __name__ == "__main__":
    input_path = "path/to/input.csv"
    qa_output_path = "path/to/qa_output.csv"
    data_with_weights_output_path = "path/to/output.csv"
    data_with_weights_applied_output_path = "path/to/final_output.csv"

    df = pd.read_csv(input_path)

    config = EstimationConfig(
        a_weight_columns=["question1", "question2"],
        g_weight_columns=["question3"],
        incl_g_wts=True,
        round_val=2,
        ru_col="reference",
        univ_count_col="uni_count",
        aux_cols=["turnover", "employment"],
        univ_aux_cols=["uni_turnover", "uni_employment"],
        a_weight_col="cellnumber",
        g_weight_col="cellnumber",
    )

    # call the method to return the dataframe with new weights columns, and qa dataframe
    weighted_df, qa_df = run_estimation(
        df=df,
        a_weight_col=config.a_weight_col,
        g_weight_col=config.g_weight_col,
        ru_col=config.ru_col,
        univ_count_col=config.univ_count_col,
        aux_cols=config.aux_cols,
        univ_aux_cols=config.univ_aux_cols,
        incl_g_wts=config.incl_g_wts,
    )

    # call the method to return the dataframe with the new weights applied
    # to the specified columns, and qa dataframe
    final_weighted_df = apply_weights(
        weighted_df,
        a_weight_columns=config.a_weight_columns,
        g_weight_columns=config.g_weight_columns,
        calc_g_weight=config.incl_g_wts,
        round_val=config.round_val,
    )

    # save the intermediate and final outputs
    weighted_df.to_csv(data_with_weights_output_path, index=False)
    qa_df.to_csv(qa_output_path, index=False)
    final_weighted_df.to_csv(data_with_weights_applied_output_path, index=False)
