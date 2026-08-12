"""Run one-sided Winsorisation for a dataframe."""

import pandas as pd

from bsrm.outlier.calculate_predicted_unit_value import (
    calculate_predicted_unit_value,
)
from bsrm.outlier.calculate_ratio_estimation import (
    calculate_ratio_estimation,
)
from bsrm.outlier.calculate_winsorised_weight import (
    calculate_winsorised_weight,
)
from bsrm.outlier.flag_for_winsorisation import winsorisation_flag


def winsorise(
    df: pd.DataFrame,
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    g_weight_col: str,
    target_col: str,
    l_values_col: str,
) -> pd.DataFrame:
    """Apply one-sided Winsorisation to the input dataframe.

    Pipes the dataframe through four steps: flag non-winsorisable units,
    calculate the predicted unit value (mu_i), calculate the ratio threshold
    (k_i), then compute adjusted returns (y*_i) and outlier weights (o_i).

    Mapping to  paper notation:

        j     = calibration_group_col   (calibration group)
        x_i   = aux_col                 (auxiliary variable)
        a_i   = a_weight_col            (design weight)
        g_i   = g_weight_col            (calibration weight)
        y_i   = target_col              (survey return)
        L     = l_values_col            (tuning parameter)

    Parameters
    ----------
    df (pd.DataFrame) : Input dataframe.
    calibration_group_col (str) : Name of the calibration group column (j in the spec).
    aux_col (str) : Name of the auxiliary variable column (x_i in the spec).
    a_weight_col (str) : Name of the design weight column (a_i in the spec).
    g_weight_col (str) : Name of the calibration weight column (g_i in the spec).
    target_col (str) : Name of the target variable column (y_i in the spec).
    l_values_col (str) : Name of the tuning parameter column (L in the spec).

    Returns
    -------
    pd.DataFrame: Dataframe with added outlier_weight (o_i) and
        adjusted_return (y*_i) columns.
    """
    winsorised_df = (
        df.pipe(winsorisation_flag, a_weight_col, g_weight_col)
        .pipe(
            calculate_predicted_unit_value,
            calibration_group_col,
            aux_col,
            a_weight_col,
            target_col,
            "non_winsorisable_marker",
        )
        .pipe(
            calculate_ratio_estimation,
            a_weight_col,
            g_weight_col,
            "predicted_unit_value",
            l_values_col,
            "non_winsorisable_marker",
        )
        .pipe(
            calculate_winsorised_weight,
            a_weight_col,
            g_weight_col,
            target_col,
            "ratio_estimation_threshold",
            "non_winsorisable_marker",
        )
    )
    return winsorised_df
