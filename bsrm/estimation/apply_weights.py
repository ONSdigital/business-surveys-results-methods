"""Apply the estimation weights to questions."""

import logging
import pandas as pd

from bsrm.utils.validate_user_input import validate_apply_weights_input

AppWeights_Logger = logging.getLogger(__name__)


def apply_weights(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: list[str] | None = None,
    question_col: str = "question",
    value_col: str = "value",
    calc_g_weight: bool = False,
    round_val: int = 4,
) -> pd.DataFrame:
    """Apply the estimation weights to survey questions.

    Parameters
    ----------
        df (pd.DataFrame): The survey dataframe weights are calculated for.
        a_weight_columns (list[str]): List of columns to apply a_weight to.
        g_weight_columns (list[str] | None): List of columns to apply g_weight to.
        round_val (int): The number of dec places we round to
        calc_g_weight (bool): Whether g weights are to be applied.

    Returns
    -------
        pd.DataFrame: The dataframe with the estimated values.
    """
    validate_apply_weights_input(
        df,
        a_weight_columns,
        g_weight_columns,
        calc_g_weight,
        round_val,
        question_col,
    )
    # create a new column for the grossed (weighted) values
    df[f"{value_col}_grossed"] = df[value_col].astype(float)

    # apply a_weights to the questions specified for a_weights
    a_weight_cond = df[question_col].isin(a_weight_columns)
    df.loc[a_weight_cond, f"{value_col}_grossed"] = (
        df.loc[a_weight_cond, value_col] * df.loc[a_weight_cond, "a_weight"]
    )

    # update the grossed values with g_weights for the questions specified for g_weights (if any)
    if calc_g_weight and g_weight_columns is not None:
        g_weight_cond = df[question_col].isin(g_weight_columns)
        df.loc[g_weight_cond, f"{value_col}_grossed"] = (
            df.loc[g_weight_cond, f"{value_col}_grossed"] * df.loc[g_weight_cond, "g_weight"]
        )

    # round the grossed values to the specified number of decimal places
    df[f"{value_col}_grossed"] = df[f"{value_col}_grossed"].round(round_val)

    return df
