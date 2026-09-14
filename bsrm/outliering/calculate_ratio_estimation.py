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
    """Calculate the intermediate ratio threshold (k_i) for each unit.

    This function applies the ratio-estimation formula before the
    non-winsorisable mask is applied. The complete calculation, including
    masking, is performed by calculate_ratio_estimation_threshold().

    Formula from the paper:

        k_i = mu_i + L / (a_i * g_i - 1)

    In the paper k_i is the ratio estimation threshold, mu_i is the
    predicted unit value, L is the tuning parameter and a_i * g_i is the
    combined design and calibration weight.

    Adds denominator_k_i and ratio_threshold columns to dataframe.
    """
    df["denominator_k_i"] = df[ag_product_col] - 1
    # Replace zero denominators with NaN to avoid division by zero
    df["denominator_k_i"] = df["denominator_k_i"].mask(df["denominator_k_i"] == 0)
    df["ratio_threshold"] = df[predicted_unit_col] + (df[l_values_col] / df["denominator_k_i"])
    return df


def apply_non_winsorisable_mask(
    df: pd.DataFrame,
    ratio_threshold_col: str,
    non_winsorisable_col: str,
) -> pd.DataFrame:
    """Set the ratio threshold to NaN for non-winsorisable units.

    This keeps those units in the dataframe while marking their threshold as
    not applicable because they represent themselves and cannot be outliers.
    """
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
    """Calculate the final ratio-estimation threshold for each unit.

    This function runs the complete process: it calculates the combined
    weight, calculates the intermediate ratio threshold (k_i), and masks the
    threshold for non-winsorisable units.

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

    df = df.rename(columns={"masked_ratio_threshold": "ratio_estimation_threshold"})

    return df
