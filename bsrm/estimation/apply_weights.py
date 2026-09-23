"""Apply the estimation weights to questions."""

import logging
import pandas as pd

from bsrm.utils.validate_user_input import validate_apply_weights_input

AppWeights_Logger = logging.getLogger(__name__)


def apply_weights(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: dict[str, list[str]] | None = None,
    aux_cols: list[str] | None = None,
    calc_g_weight: bool = True,
    round_val: int = 4,
) -> pd.DataFrame:
    """Apply the estimation weights to survey questions.

    Parameters
    ----------
        df (pd.DataFrame): The survey dataframe weights are calculated for.
        a_weight_columns (list[str]): List of columns to apply a_weight to.
        g_weight_columns (dict[str, list[str]] | None): Dictionary of columns to
            apply g_weight to, organized by variable.
        aux_cols (list[str] | None): List of auxiliary columns related to g weights.
        round_val (int): The number of dec places we round to
        calc_g_weight (bool): Whether g weights are to be applied.

    Returns
    -------
        pd.DataFrame: The dataframe with the estimated values.
    """
    validate_apply_weights_input(
        data=df,
        a_weight_columns=a_weight_columns,
        g_weight_columns=g_weight_columns,
        aux_cols=aux_cols,
        calc_g_weight=calc_g_weight,
        round_val=round_val,
    )

    # apply a weights to the columns specified for a weights
    for col in a_weight_columns:
        df[col] = round(df[col] * df["a_weight"], round_val)
    if calc_g_weight and g_weight_columns is not None and aux_cols is not None:
        # for each type of g weight, apply it to the corresponding columns
        for g_wt_type, cols in g_weight_columns.items():
            for col in cols:
                df[col] = round(df[col] * df[f"g_weight_{g_wt_type}"], round_val)

    return df
