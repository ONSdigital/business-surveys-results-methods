"""Mark rows for winsorisation based on a_weight and g_weight."""

import pandas as pd


def winsorisation_flag(df: pd.DataFrame, a_weight_col: str, g_weight_col: str) -> pd.DataFrame:
    """Flag units where Winsorisation should not be applied.

    Units with both design and calibration weights equal to 1 represent only
    themselves, cannot be outliers, and are excluded from Winsorisation.

    Mapping to paper notation:

        a_i   = a_weight_col   (design weight)
        g_i   = g_weight_col   (calibration weight)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    a_weight_col : str
        Name of the design weight column.
    g_weight_col : str
        Name of the calibration weight column.

    Returns
    -------
    pd.DataFrame
    Dataframe with an added boolean column non_winsorisable_marker.
    """
    df = df.copy()
    df["non_winsorisable_marker"] = (df[a_weight_col] == 1) & (df[g_weight_col] == 1)

    return df
