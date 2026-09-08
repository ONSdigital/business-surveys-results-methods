"""Calculate ratio estimation thresholds for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_ag_product(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str,
) -> pd.DataFrame:
    """Calculate the combined design and calibration weight.

    In the paper this is denoted as a_i * g_i for each unit.
    Adds 'ag_product' column to dataframe.
    """
    df["ag_product"] = df[a_weight_col] * df[g_weight_col]
    return df


def calculate_ratio_threshold(
    df: pd.DataFrame,
    predicted_unit_col: str,
    l_values_col: str,
    ag_product_col: str,
) -> pd.DataFrame:
    """Calculate the ratio estimation threshold before masking.

    Formula from the paper:

        k_i = mu_i + L / (a_i * g_i - 1)

    In the paper k_i is the ratio estimation threshold, mu_i is the
    predicted unit value, L is the tuning parameter and a_i * g_i is the
    combined design and calibration weight.

    Adds 'ratio_threshold' column to dataframe.
    """
    denominator = df[ag_product_col] - 1
    # Replace zero denominators with NaN to avoid division by zero
    denominator = denominator.mask(denominator == 0)
    df["ratio_threshold"] = df[predicted_unit_col] + (df[l_values_col] / denominator)
    return df


def apply_non_winsorisable_mask(
    df: pd.DataFrame,
    ratio_threshold_col: str,
    non_winsorisable_col: str,
) -> pd.DataFrame:
    """Set ratio thresholds to NaN for non-winsorisable units."""
    df["masked_ratio_threshold"] = df[ratio_threshold_col].mask(df[non_winsorisable_col], np.nan)
    return df


def calculate_ratio_estimation_threshold(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str,
    predicted_unit_value_col: str,
    l_values_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the ratio estimation threshold for each unit.

    Adds the ratio_estimation_threshold column used by the Winsorisation
    pipeline.

    Formula from the spec:

        k_i = mu_i + L / (a_i * g_i - 1)

    Units marked as non-winsorisable receive NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    a_weight_col : str
        Name of the column containing design weight values.
    g_weight_col : str
        Name of the column containing calibration weight values.
    predicted_unit_value_col : str
        Name of the column containing predicted unit values.
    l_values_col : str
        Name of the column containing tuning parameter values.
    non_winsorisable_marker_col : str
        Name of the column marking units where Winsorisation should not be applied.

    Returns
    -------
    pd.DataFrame
        Dataframe with an added ratio_estimation_threshold column.
    """
    df = df.copy()

    df = calculate_ag_product(
        df,
        a_weight_col,
        g_weight_col,
    )

    df = calculate_ratio_threshold(
        df,
        predicted_unit_value_col,
        l_values_col,
        "ag_product",
    )

    df = apply_non_winsorisable_mask(
        df,
        "ratio_threshold",
        non_winsorisable_marker_col,
    )

    df["ratio_estimation_threshold"] = df["masked_ratio_threshold"]

    return df
