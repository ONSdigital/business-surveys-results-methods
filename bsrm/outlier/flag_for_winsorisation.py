"""Mark rows for winsorisation based on a_weight and g_weight."""

import pandas as pd


def winsorisation_flag(df: pd.DataFrame, a_weight_col: str, g_weight_col: str) -> pd.DataFrame:
    """Flag units where Winsorisation should not be applied.

    a_i * g_i = 1 cannot be outliers and must receive outlier_weight = 1.

    Mapping to paper notation:

        a_i   = a_weight_col   (design weight)
        g_i   = g_weight_col   (calibration weight)

    Parameters
    ----------
    df (pd.DataFrame) : Input dataframe.
    a_weight_col (str) : Name of the design weight column.
    g_weight_col (str) : Name of the calibration weight column.

    Returns
    -------
    pd.DataFrame: Dataframe with an added boolean column non_winsorisable_marker.
    """
    df = df.copy()
    df["ag_product"] = df[a_weight_col] * df[g_weight_col]
    df["non_winsorisable_marker"] = df["ag_product"] <= 1
    df = df.drop("ag_product", axis=1)

    return df
