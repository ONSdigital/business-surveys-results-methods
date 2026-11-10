"""Calculate adjusted returns and outlier weights for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_ratio_winsorised_weight(
    df: pd.DataFrame,
    target_col: str,
    ratio_estimation_threshold_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the adjusted return (y*_i) and outlier weight (o_i) for each unit.

    This function implements the ratio estimation method for Winsorisation.
    Formula provided in the paper (provided by methodology):

        y*_i = [y_i + (a_i * g_i - 1) * k_i] / (a_i * g_i)
        o_i  = y*_i / y_i

    Calculation based on the paper example (Cell 1, Unit 1):

        y_i = 14, k_i = 13.15766243
        a_i = 2.666666667, g_i = 2.205508475
        y*_i = [14 + (2.666666667 * 2.205508475 - 1) * 13.15766243]
               / (2.666666667 * 2.205508475)
         y*_i =  13.30088409
        o_i = 13.30088409 / 14
        o_i =  0.950063149

    Units with both a_i == 1 and g_i == 1 are non-winsorisable.
    They retain their original return, receive outlier_weight = 1 and
    receive outlier_flag = 0.

    Mapping to paper notation:

        y*_i  = adjusted_return                  (output column)
        o_i   = outlier_weight                   (output column)
        y_i   = target_col                       (survey return)
        k_i   = ratio_estimation_threshold_col   (ratio threshold)
        a_i   = a_weight_col                     (design weight)
        g_i   = g_weight_col                     (calibration weight)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target variable column (y_i in the spec).
    ratio_estimation_threshold_col : str
        Name of the ratio threshold column (k_i in the spec).
    non_winsorisable_marker_col : str
        Name of the column marking units with both design and calibration
        weights equal to 1.

    Returns
    -------
    pd.DataFrame
         Dataframe with added adjusted_return (y*_i) and outlier_weight (o_i) columns.

    """
    non_winsorisable = df[non_winsorisable_marker_col]

    df["adjusted_value"] = (
        df[target_col] + (df["ag_product"] - 1) * df[ratio_estimation_threshold_col]
    ) / df["ag_product"]

    df["mask"] = df[target_col] <= df[ratio_estimation_threshold_col]
    df["adjusted_return"] = np.where(df["mask"], df[target_col], df["adjusted_value"])
    df["outlier_flag"] = (~df["mask"]).astype(int)

    df["outlier_weight"] = df["adjusted_return"] / df[target_col].replace(0, np.nan)

    df = df.drop(columns=["mask", "adjusted_value"])

    division_with_0 = ~non_winsorisable & (df[target_col] == 0)

    df["outlier_weight"] = df["outlier_weight"].mask(non_winsorisable | division_with_0, 1)
    df["adjusted_return"] = df["adjusted_return"].mask(
        non_winsorisable | division_with_0,
        df[target_col],
    )
    df["outlier_flag"] = df["outlier_flag"].mask(non_winsorisable | division_with_0, 0)

    return df
