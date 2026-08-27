"""Detect and weight outliers using one-sided Winsorisation."""

import pandas as pd

from bsrm.outlier.winsorisation import winsorise


def detect_outlier(
    df: pd.DataFrame,
    question_no_col: str,
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    g_weight_col: str,
    target_col: str,
    l_value: float,
) -> pd.DataFrame:
    """Apply Winsorisation to identify and weight outliers.

    Winsorisation is applied separately per question number to ensure
    calibration groups are scoped within each question.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    question_no_col : str
        Name of the question number column.
    calibration_group_col : str
        Name of the calibration group column.
    aux_col : str
        Name of the auxiliary variable column.
    a_weight_col : str
        Name of the design weight column.
    g_weight_col : str
        Name of the calibration factor column.
    target_col : str
        Name of the target variable column.
    l_value : float
        Tuning parameter for Winsorisation.

    Returns
    -------
    pd.DataFrame
        Dataframe with added outlier_weight and outlier_flag columns.
    """

    def winsorise_group(group: pd.DataFrame) -> pd.DataFrame:
        """Apply Winsorisation to a single question group."""
        winsorised = winsorise(
            group,
            calibration_group_col,
            aux_col,
            a_weight_col,
            g_weight_col,
            target_col,
            "l_value",
        )
        return winsorised

    df = df.copy()
    df["l_value"] = l_value

    post_win = df.groupby(question_no_col).apply(winsorise_group)
    post_win = post_win.reset_index(drop=True)

    return post_win
