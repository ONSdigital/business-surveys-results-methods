"""Calculate the expansion estimation threshold for Winsorisation."""

import numpy as np
import pandas as pd


def calculate_stratum_mean(
    df: pd.DataFrame,
    strata_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Calculate the mean response for each stratum and adds stratum_mean to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strata_col : str
        The column name for the strata which is cell.
    target_col : str
        The column name for the target variable which is y.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column for the stratum mean (stratum_mean).
    """
    df = df.copy()
    df["stratum_mean"] = df.groupby(strata_col)[target_col].transform("mean")
    return df


def calculate_expansion_estimation_threshold(
    df: pd.DataFrame,
    a_weight_col: str,
    stratum_mean_col: str,
    l_values_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the expansion estimation threshold(k_h) for each unit.

    Formula from the paper:
    k_h = ȳ_h + L / (a_h - 1)
    mapping to the paper notation:
    k_h = expansion_estimation_threshold
    ȳ_h = stratum_mean_col
    L = l_values_col
    a_h = a_weight_col

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    a_weight_col : str
        The column name for the a_h weights.
    stratum_mean_col : str
        The column name for the stratum means (ȳ_h).
    l_values_col : str
        The column name for the L values.
    non_winsorisable_marker_col : str
        The column name indicating non-winsorisable units.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column for the expansion estimation threshold (k_h).
    """
    df = df.copy()
    denominator = df[a_weight_col] - 1
    denominator = denominator.mask(denominator == 0)

    df["expansion_estimation_threshold"] = df[stratum_mean_col] + df[l_values_col] / denominator

    # non-winsorisable are marekd as NaN
    df["expansion_estimation_threshold"] = df["expansion_estimation_threshold"].mask(
        df[non_winsorisable_marker_col], np.nan
    )

    return df
