"""Calculate predicted unit value for each row in a DataFrame."""

import numpy as np
import pandas as pd


def calculate_predicted_unit_value(
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

    where sums are taken over calibration group j. Units marked as
    non-winsorisable (a_i * g_i <= 1) receive NaN.

    Mapping to paper notation:

        mu_i  = predicted_unit_value (output column)
        x_i   = aux_col             (auxiliary variable)
        a_i   = a_weight_col        (design weight)
        y_i   = target_col          (survey return)
        j     = calibration_group_col

    Parameters
    ----------
    df:pd.DataFrame
        Input dataframe.
    calibration_group_col: str
      Name of the calibration group column.
    aux_col : str
      Name of the auxiliary variable column.
    a_weight_col : str
      Name of the design weight column.
    target_col : str
      Name of the target variable column.
    non_winsorisable_marker_col : str
      Name of the column marking units where a_i * g_i <= 1.

    Returns
    -------
    pd.DataFrame
      Dataframe with an added predicted_unit_value column.
    """
    filtered_df = df.copy().loc[~df[non_winsorisable_marker_col]]

    filtered_df["weighted_target_values"] = filtered_df[a_weight_col] * filtered_df[target_col]
    filtered_df["weighted_auxiliary_values"] = filtered_df[a_weight_col] * filtered_df[aux_col]

    sum_weighted_target_values = (
        filtered_df.groupby(calibration_group_col)["weighted_target_values"]
        .sum()
        .reset_index(name="sum_weighted_target_values")
    )
    sum_weighted_auxiliary_values = (
        filtered_df.groupby(calibration_group_col)["weighted_auxiliary_values"]
        .sum()
        .reset_index(name="sum_weighted_auxiliary_values")
    )

    total_sum_weighted = sum_weighted_target_values.merge(
        sum_weighted_auxiliary_values,
        on=calibration_group_col,
        how="left",
    )

    final_df = df.merge(total_sum_weighted, on=calibration_group_col, how="left")

    final_df["predicted_unit_value"] = final_df[aux_col] * (
        final_df["sum_weighted_target_values"] / final_df["sum_weighted_auxiliary_values"]
    )

    final_df = final_df.drop(
        ["sum_weighted_target_values", "sum_weighted_auxiliary_values"],
        axis=1,
    )

    final_df["predicted_unit_value"] = final_df["predicted_unit_value"].mask(
        df[non_winsorisable_marker_col], np.nan
    )

    return final_df
