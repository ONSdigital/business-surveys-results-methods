"""Apply the estimation weights to questions."""

import logging
import pandas as pd


AppWeights_Logger = logging.getLogger(__name__)


def apply_weights(
    df: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str] | None = None,
    calc_g_weight: bool = True,
    round_val: int = 4,
) -> pd.DataFrame:
    """Apply the estimation weights to survey questions.

    Parameters
    ----------
        df (pd.DataFrame): The survey dataframe weights are calculated for.
        a_weight_cols (list[str]): List of columns to apply a_weight to.
        g_weight_cols (list[str]): List of columns to apply g_weight to.
        for_qa (bool): If True, keep the values before and after weights are applied.
        round_val (int): The number of dec places we round to
        calc_g_weight (bool): Whether g weights are to be applied.

    Returns
    -------
        pd.DataFrame: The dataframe with the estimated values.
    """
    for col in a_weight_cols:
        df[col] = round(df[col] * df["a_weight"], round_val)
    if calc_g_weight and g_weight_cols is not None:
        for col in g_weight_cols:
            df[col] = round(df[col] * df["g_weight"], round_val)

    return df
