"""Run one-sided Winsorisation for a dataframe."""

import pandas as pd

from bsrm.outliering.calculate_predicted_unit_value import (
    calculate_predicted_unit_value,
)
from bsrm.outliering.calculate_ratio_estimation import (
    calculate_ratio_estimation_threshold,
)
from bsrm.outliering.calculate_winsorised_weight import (
    calculate_ratio_winsorised_weight,
)
from bsrm.outliering.flag_for_winsorisation import winsorisation_flag

from bsrm.outliering.calculate_expansion_estimation import (
    calculate_stratum_mean,
    calculate_expansion_estimation_threshold,
    calculate_expansion_winsorised_weight,
)


def winsorise_ratio(
    df: pd.DataFrame,
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    g_weight_col: str,
    target_col: str,
    l_values_col: str,
) -> pd.DataFrame:
    """Apply one-sided ratio Winsorisation using supplied estimation weights.

    The pipeline calculates predicted unit values and thresholds within
    calibration groups. Calibration weights are calculated by calibration
    group, not by stratum.

    Non-winsorisable units have both design(a_i) and calibration(g_i) weights equal to
    1. They represent themselves only and are therefore not treated as
    outliers.

    Processes the dataframe in four steps: flag non-winsorisable units,
    calculate the predicted unit value (mu_i), calculate the ratio threshold
    (k_i), then compute adjusted returns (y*_i) and outlier weights (o_i).

    Process one question at a time so that predicted unit values are
    calculated using responses to that question only.

    Mapping to paper notation:

        j     = calibration_group_col   (calibration group)
        x_i   = aux_col                 (auxiliary variable)
        a_i   = a_weight_col            (design weight)
        g_i   = g_weight_col            (calibration weight)
        y_i   = target_col              (survey return)
        L     = l_values_col            (tuning parameter)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    calibration_group_col : str
        Name of the calibration group column (j in the spec).
        Design-weighted totals for predicted unit values are calculated
        within this group.
    aux_col : str
        Name of the auxiliary variable column (x_i in the spec).
    a_weight_col : str
        Name of the design weight column (a_i in the spec).
    g_weight_col : str
        Name of the calibration weight column (g_i in the spec).
    target_col : str
        Name of the target variable column (y_i in the spec).
    l_values_col : str
        Name of the tuning parameter column (L in the spec).

    Returns
    -------
    pd.DataFrame
        Dataframe with added calculation columns, outlier_weight (o_i),
        outlier_flag and adjusted_return (y*_i).
    """
    df = winsorisation_flag(df, a_weight_col, g_weight_col)
    df = calculate_predicted_unit_value(
        df,
        calibration_group_col,
        aux_col,
        a_weight_col,
        target_col,
        "non_winsorisable_marker",
    )
    df = calculate_ratio_estimation_threshold(
        df,
        a_weight_col,
        g_weight_col,
        "predicted_unit_value",
        l_values_col,
        "non_winsorisable_marker",
    )
    df = calculate_ratio_winsorised_weight(
        df,
        target_col,
        "ratio_estimation_threshold",
        "non_winsorisable_marker",
    )
    return df


def winsorise_expansion(
    df: pd.DataFrame,
    strata_col: str,
    target_col: str,
    a_weight_col: str,
    l_values_col: str,
) -> pd.DataFrame:
    """Apply one-sided expansion Winsorisation to the input dataframe.

    The pipeline calculates mean returns and expansion thresholds within
    strata, using supplied design weights. It does not use auxiliary
    values or calibration weights.

    Non-winsorisable units have a design weight (a_i) equal to 1.
    They represent themselves only and are therefore not treated as
    outliers.

    Processes the dataframe in four steps: flag non-winsorisable units,
    calculate the stratum mean return, calculate the expansion threshold
    (k_h), then compute adjusted returns (y*_i) and outlier weights (o_i).

    Call separately for each question so that stratum means are not
    calculated across different target variables.

    Mapping to paper notation:

        h     = strata_col              (stratum)
        y_i   = target_col              (survey return)
        a_i   = a_weight_col            (design weight, equal to a_h in stratum h)
        L     = l_values_col            (tuning parameter)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing returns, design weights and tuning values
        for one question.
    strata_col : str
        Name of the stratum column (h in the spec). Mean returns are
        calculated within each stratum.
    target_col : str
        Name of the target variable column (y_i in the spec).
    a_weight_col : str
        Name of the design weight column (a_i in the spec). The design
        weight is assumed to be constant within each stratum.
    l_values_col : str
        Name of the tuning parameter column (L in the spec). Tuning values
        are assumed to be constant within each stratum so that its
        expansion threshold is shared by its units.

    Returns
    -------
    pd.DataFrame
        Dataframe with added stratum_mean, expansion_estimation_threshold,
        non_winsorisable_marker, adjusted_return (y*_i), outlier_flag and
        outlier_weight (o_i) columns.
    """
    df = df.copy()
    df["non_winsorisable_marker"] = df[a_weight_col] == 1

    df = calculate_stratum_mean(df, strata_col, target_col)
    df = calculate_expansion_estimation_threshold(
        df,
        a_weight_col,
        "stratum_mean",
        l_values_col,
        "non_winsorisable_marker",
    )

    winsorised_df = calculate_expansion_winsorised_weight(
        df,
        target_col,
        "expansion_estimation_threshold",
        a_weight_col,
        "non_winsorisable_marker",
    )
    return winsorised_df
