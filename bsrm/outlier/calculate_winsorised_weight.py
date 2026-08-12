"""Calculate adjusted returns and outlier weights for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_winsorised_weight(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str,
    target_col: str,
    ratio_estimation_threshold_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the adjusted return (y*_i) and outlier weight (o_i) for each unit.

    Formula provided in the paper (provided by methodology):

        y*_i = y_i + (a_i * g_i - 1) * k_i / (a_i * g_i)
        o_i  = y*_i / y_i

    Calculation based on the appendix example (Cell 1, Unit 1):

        y_i = 14, k_i = 13.389, a_i * g_i = 2.667 * 2.206 = 5.882
        y*_i = 14 + (5.882 - 1) * 13.389 / 5.882 = 25.11

    Units marked as non-winsorisable (a_i * g_i <= 1) receive outlier_weight = 1.

    Mapping to paper notation:

        y*_i  = adjusted_return                  (output column)
        o_i   = outlier_weight                   (output column)
        y_i   = target_col                       (survey return)
        k_i   = ratio_estimation_threshold_col   (ratio threshold)
        a_i   = a_weight_col                     (design weight)
        g_i   = g_weight_col                     (calibration weight)

    Parameters
    ----------
    df (pd.DataFrame) : Input dataframe.
    a_weight_col (str) : Name of the design weight column (a_i in the spec).
    g_weight_col (str) : Name of the calibration weight column (g_i in the spec).
    target_col (str) : Name of the target variable column (y_i in the spec).
    ratio_estimation_threshold_col (str) : Name of the ratio threshold column.
    non_winsorisable_marker_col (str) : Name of the column marking units where a_i * g_i <= 1.

    Returns
    -------
    pd.DataFrame: Dataframe with added adjusted_return (y*_i) and outlier_weight (o_i) columns.
    """
    df["ag_product"] = df[a_weight_col] * df[g_weight_col]

    df["adjusted_value"] = df[target_col] + (
        (df["ag_product"] - 1) * df[ratio_estimation_threshold_col] / df["ag_product"]
    )

    mask = df[target_col] <= df[ratio_estimation_threshold_col]
    df["adjusted_return"] = np.where(mask, df[target_col], df["adjusted_value"])

    df["outlier_weight"] = df["adjusted_return"] / df[target_col]

    df = df.drop(["ag_product", "adjusted_value"], axis=1)

    non_winsorisable = df[non_winsorisable_marker_col]
    division_with_0 = ~non_winsorisable & (df[target_col] == 0)

    df["outlier_weight"] = df["outlier_weight"].mask(non_winsorisable | division_with_0, 1)
    df["adjusted_return"] = df["adjusted_return"].mask(non_winsorisable | division_with_0, np.nan)

    return df
