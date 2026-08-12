"""Calculate the ratio estimation threshold for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_ratio_estimation(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str,
    predicted_unit_value_col: str,
    l_values_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the ratio estimation threshold (k_i) for each unit.

    Formula from the spec:

        k_i = mu_i + L / (a_i * g_i - 1)

    Units marked as non-winsorisable (a_i * g_i <= 1) receive NaN.

    Mapping to paper notation:

        k_i   = ratio_estimation_threshold (output column)
        mu_i  = predicted_unit_value_col   (expected unit value)
        L     = l_values_col               (tuning parameter)
        a_i   = a_weight_col               (design weight)
        g_i   = g_weight_col               (calibration weight)

    Parameters
    ----------
    df (pd.DataFrame) : Input dataframe.
    a_weight_col (str) : Name of the design weight column.
    g_weight_col (str) : Name of the calibration weight column.
    predicted_unit_value_col (str) : Name of the predicted unit value column.
    l_values_col (str) : Name of the tuning parameter column.
    non_winsorisable_marker_col (str) : Name of the column marking units where a_i * g_i <= 1.

    Returns
    -------
    pd.DataFrame: Dataframe with an added ratio_estimation_threshold column (k_i in the paper).
    """
    df = df.copy()
    df["ag_product"] = df[a_weight_col] * df[g_weight_col]
    df["ratio_estimation_threshold"] = df[predicted_unit_value_col] + (
        df[l_values_col] / (df["ag_product"] - 1)
    )
    df = df.drop("ag_product", axis=1)

    df["ratio_estimation_threshold"] = df["ratio_estimation_threshold"].mask(
        df[non_winsorisable_marker_col], np.nan
    )

    return df
