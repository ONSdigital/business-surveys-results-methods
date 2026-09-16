"""Calculate predicted unit value for each row in a DataFrame."""

import numpy as np
import pandas as pd


def calculate_group_sums(
    df: pd.DataFrame,
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    target_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Filter non winsorisable rows and calculate group sums.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    calibration_group_col : str
        Name of the calibration group column.
    aux_col : str
        Name of the auxiliary variable column.
    a_weight_col : str
        Name of the design weight column.
    target_col : str
        Name of the target variable column.
    non_winsorisable_marker_col : str
        Name of the column marking non-winsorisable units.

    Returns
    -------
    pd.DataFrame
        Weighted target and auxiliary sums for each calibration group.
    """
    filtered_df = df.loc[~df[non_winsorisable_marker_col]].copy()
    filtered_df["weighted_target_values"] = (
        filtered_df[a_weight_col] * filtered_df[target_col]
    )
    filtered_df["weighted_auxiliary_values"] = (
        filtered_df[a_weight_col] * filtered_df[aux_col]
    )

    group_sums = (
        filtered_df.groupby(calibration_group_col, as_index=False)
        .agg(
            sum_weighted_target_values=("weighted_target_values", "sum"),
            sum_weighted_auxiliary_values=("weighted_auxiliary_values", "sum"),
        )
    )
    return group_sums


def calculate_predicted_unit_values_from_group_sums(
    merged_df: pd.DataFrame, aux_col: str
) -> pd.Series:
    """Calculate predicted unit values from the merged dataframe.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Dataframe containing auxiliary values and weighted group sums.
    aux_col : str
        Name of the auxiliary variable column.

    Returns
    -------
    pd.Series
        Predicted unit values.
    """
    predicted_unit_values = merged_df[aux_col] * (
        merged_df["sum_weighted_target_values"]
        / merged_df["sum_weighted_auxiliary_values"]
    )
    predicted_unit_values.name = "predicted_unit_value"
    return predicted_unit_values


def calculate_predicted_unit_values(
    df: pd.DataFrame,
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    target_col: str,
    non_winsorisable_marker_col: str,
) -> pd.DataFrame:
    """Calculate the expected unit value (mu_i) for each unit.

    Formula from the paper ( provided by methodology):

        mu_i = x_i * sum(a_i * y_i) / sum(a_i * x_i)

    where sums are taken over calibration group j. Units where both
    a_i == 1 and g_i == 1 (non-winsorisable) receive NaN.

    Mapping to paper notation:

        mu_i  = predicted_unit_value (output column)
        x_i   = aux_col             (auxiliary variable)
        a_i   = a_weight_col        (design weight)
        y_i   = target_col          (survey return)
        j     = calibration_group_col

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    calibration_group_col : str
        Name of the calibration group column.
    aux_col : str
        Name of the auxiliary variable column.
    a_weight_col : str
        Name of the design weight column.
    target_col : str
        Name of the target variable column.
    non_winsorisable_marker_col : str
        Name of the column marking non-winsorisable units.

    Returns
    -------
    pd.DataFrame
        Dataframe with an added predicted_unit_value column.
    """
    group_sums = calculate_group_sums(
        df,
        calibration_group_col,
        aux_col,
        a_weight_col,
        target_col,
        non_winsorisable_marker_col,
    )
    final_df = df.merge(group_sums, on=calibration_group_col, how="left")
    final_df["predicted_unit_value"] = calculate_predicted_unit_values_from_group_sums(
        final_df,
        aux_col,
    )

    final_df = final_df.drop(
        ["sum_weighted_target_values", "sum_weighted_auxiliary_values"],
        axis=1,
    )

    final_df["predicted_unit_value"] = final_df["predicted_unit_value"].mask(
        df[non_winsorisable_marker_col], np.nan
    )

    return final_df
