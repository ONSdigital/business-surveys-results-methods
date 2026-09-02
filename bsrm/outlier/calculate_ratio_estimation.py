"""Calculate ratio estimation thresholds for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_ag_product(
    df: pd.DataFrame,
    a_weight_col: str,
    g_weight_col: str,
) -> pd.Series:
    """Calculate the combined design and calibration weight.

    In the paper this is denoted as a_i * g_i for each unit.
    """
    ag_product = df[a_weight_col] * df[g_weight_col]

    return ag_product


def calculate_ratio_threshold(
    predicted_unit_values: pd.Series,
    l_values: pd.Series,
    ag_product: pd.Series,
) -> pd.Series:
    """Calculate the ratio estimation threshold before masking.

    Formula from the paper:

        k_i = mu_i + L / (a_i * g_i - 1)
     In the paper k_i is the ratio estimation threshold, mu_i is the
    predicted unit value, L is the tuning parameter and a_i * g_i is the
    combined design and calibration weight.
    """
    denominator = ag_product - 1
    ratio_threshold = predicted_unit_values + (l_values / denominator)

    return ratio_threshold


def apply_non_winsorisable_mask(
    ratio_threshold: pd.Series,
    non_winsorisable_marker: pd.Series,
) -> pd.Series:
    """Set ratio thresholds to NaN for non-winsorisable units."""
    masked_ratio_threshold = ratio_threshold.mask(non_winsorisable_marker, np.nan)

    return masked_ratio_threshold


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

    ag_product = calculate_ag_product(
        df,
        a_weight_col,
        g_weight_col,
    )

    ratio_threshold = calculate_ratio_threshold(
        df[predicted_unit_value_col],
        df[l_values_col],
        ag_product,
    )

    masked_ratio_threshold = apply_non_winsorisable_mask(
        ratio_threshold,
        df[non_winsorisable_marker_col],
    )

    df["ratio_estimation_threshold"] = masked_ratio_threshold

    return df
