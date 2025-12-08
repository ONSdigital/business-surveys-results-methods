"""Apply the estimation weights to questions."""

import logging

import pandas as pd


AppWeights_Logger = logging.getLogger(__name__)


def apply_weights(
    df: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str],
    for_qa: bool = False,
    round_val: int = 4,
) -> pd.DataFrame:
    """Apply the estimation weights to survey questions.

    Args:
        df (pd.DataFrame): The survey dataframe weights are calculated for.
        a_weight_cols (list[str]): List of columns to apply a_weight to.
        g_weight_cols (list[str]): List of columns to apply g_weight to.
        for_qa (bool): If True, keep the values before and after weights are applied.
        round_val (int): The number of dec places we round to

    Returns
    -------
        pd.DataFrame: The dataframe with the estimated values.
    """
    # if the dataframe is for QA output, create new columns with the weights applied.
    if for_qa:
        for col in a_weight_cols:
            df[f"{col}_estimated"] = round(df[col] * df["a_weight"], round_val)
        for col in g_weight_cols:
            df[f"{col}_estimated"] = round(
                df[f"{col}_estimated"] * df["g_weight"], round_val
            )

    # otherwise, apply the weights directly to the existing columns
    else:
        for col in a_weight_cols:
            df[col] = round(df[col] * df["a_weight"], round_val)
        for col in g_weight_cols:
            df[col] = round(df[col] * df["g_weight"], round_val)

    return df
