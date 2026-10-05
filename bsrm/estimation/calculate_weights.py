"""Functions to calculate estimation weights for survey data."""

import pandas as pd
import logging

CalcWeights_Logger = logging.getLogger(__name__)


def calc_lower_n(df: pd.DataFrame, ru_column: str) -> int:
    """Calculate the number of unique reporting units (RUs) ("n") in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe which contains survey data, including expenditure data
    ru_column : str
        The name of the column containing reporting unit identifiers.

    Returns
    -------
    int
        The number of unique reporting units (RUs).
    """
    n = df[ru_column].nunique()

    return n


def a_weight(
    stratum_group: pd.DataFrame,
    ru_column: str,
    univ_count_col: str,
    a_wgt_band_col: str,
) -> pd.DataFrame:
    """Calculate the a-weights for a stratum group.

    The calculation here is:

    a_weight = N/n

    Where:
        N is population or universe count for the stratum
        n is the number of valid returns for the stratum

    Parameters
    ----------
    stratum_group : pd.DataFrame
        The dataframe grouped by the a-weight band column.
    ru_column : str
        The name of the column containing reporting unit identifiers.
    univ_count_col : str
        The name of the column containing the total number of reporting units in the stratum.

    Returns
    -------
    pd.DataFrame
        The dataframe with the a-weights calculated.
    """
    if stratum_group.empty:
        return stratum_group

    if stratum_group is not None:
        stratum_group[a_wgt_band_col] = stratum_group.name

    N = stratum_group[univ_count_col].iloc[0]  # noqa: N806 (allow capitals for vars)
    n = calc_lower_n(stratum_group, ru_column)

    stratum_group["n"] = n

    # Calculate 'a' for this group
    if n > 0:
        a_weight = N / n
    else:
        a_weight = 1.0

    stratum_group["a_weight"] = a_weight

    return stratum_group


def g_weight(
    calibration_group: pd.DataFrame,
    aux_col: str,
    univ_aux_col: str,
    g_wgt_band_col: str,
) -> pd.DataFrame:
    """Calculate the g-weights for a calibration group.

    The calibration group is the main dataframe grouped by the g_weight band column.

    The calculation for the g-weight is:

    g =  univ_aux_sum / sum_ax
    sum_ax = Σ a_i * x_i

    Where:
    - univ_aux_sum is the sum of the auxiliary value for the universe over the calibration group.
    - sum_ax is the sum of the auxiliary value multiplied by its a_weight for responders in the
        calibration group.
    - x_i represents each auxiliary value.
    - a_i represents the a_weight corresponding to x_i.

    Parameters
    ----------
    calibration_group : pd.DataFrame
        The dataframe grouped by calibration group.
    aux_col : str
        The name of the column containing auxiliary employment data.
    univ_aux_col : str
        The name of the column containing the total auxiliary employment in the calibration group.
    g_wgt_band_col : str
        The name of the column the data is grouped by. Pass this to keep that column in the result.

    Return
    -------
    pd.DataFrame
        The dataframe with the g-weights calculated.
    """
    if calibration_group.empty:
        return calibration_group

    if g_wgt_band_col is not None:
        calibration_group[g_wgt_band_col] = calibration_group.name

    univ_aux_sum = calibration_group[univ_aux_col].iloc[0]
    aux_col_sum = calibration_group[aux_col].sum()
    # sum_ax must be computed row-by-row (a_i * x_i) before summing
    sum_ax = (calibration_group["a_weight"] * calibration_group[aux_col]).sum()

    # Calculate g-weight for this group
    if aux_col_sum > 0 and sum_ax > 0:
        g_weight = univ_aux_sum / sum_ax
    else:
        g_weight = 1.0

    calibration_group[f"{univ_aux_col}_sum"] = univ_aux_sum
    calibration_group[f"{aux_col}_sum"] = aux_col_sum

    calibration_group[f"g_weight_{aux_col}"] = g_weight

    return calibration_group


def create_weights_qa_df(
    df: pd.DataFrame,
    a_wgt_band_col: str,
    univ_count_col: str,
    incl_g_wts: bool = True,
    g_wgt_band_col: str | None = None,
    aux_cols: list[str] | None = None,
    univ_aux_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create a QA dataframe for the weight calculation.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing survey data.
    a_wgt_band_col : str
        The name of the column containing stratum identifiers.
    univ_count_col : str
        The name of the column containing the total number of reporting units in the stratum.
    incl_g_wts : bool, optional
        Whether to include g-weights in the QA dataframe, by default True.
    g_wgt_band_col : str | None, optional
        The name of the column containing calibration group identifiers, by default None.
    aux_cols : list[str] | None, optional
        The list of columns containing auxiliary employment data, by default None.
    univ_aux_cols : list[str] | None, optional
        The list of columns containing the total auxiliary employment in the calibration group.

    Returns
    -------
    pd.DataFrame
        A QA dataframe summarizing the weight calculation for each stratum and calibration group.
    """
    qa_cols_list = [a_wgt_band_col, univ_count_col, "n", "a_weight"]

    if incl_g_wts and aux_cols is not None and univ_aux_cols is not None:
        if g_wgt_band_col is not None and g_wgt_band_col != a_wgt_band_col:
            qa_cols_list.append(g_wgt_band_col)

        for aux_col, univ_aux_col in zip(aux_cols, univ_aux_cols, strict=False):
            qa_cols_list += [f"{aux_col}_sum", f"{univ_aux_col}_sum", f"g_weight_{aux_col}"]

    qa_df = df[qa_cols_list].groupby(a_wgt_band_col).first()
    qa_df = qa_df.reset_index()

    return qa_df


def calculate_a_weights(
    df: pd.DataFrame,
    a_wgt_band_col: str,
    ru_col: str,
    univ_count_col: str,
) -> pd.DataFrame:
    """Calculate the 'a' weight for each stratum in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input df containing survey data.
    a_wgt_band_col : str
        The name of the column containing stratum identifiers.
    ru_col : str
        The name of the column containing reference unit data.
    univ_count_col : str
        The name of the column containing the total number of reporting units in the stratum.


    Returns
    -------
        pd.DataFrame: The full dataframe with the added new column "a_weight".
    """
    df = df.copy()
    df["a_weight"] = 1.0
    df = df.groupby(a_wgt_band_col, group_keys=False).apply(
        a_weight, ru_col, univ_count_col, a_wgt_band_col
    )

    return df


def calculate_g_weights(
    df: pd.DataFrame,
    g_wgt_col: str,
    aux_col: str,
    univ_aux_col: str,
) -> pd.DataFrame:
    """Calculate the g-weight for each calibration group in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input df containing survey data
    g_wgt_col : str
        The name of the column containing calibration group identifiers.
    aux_col : str
        The name of the column containing auxiliary employment data.
    univ_aux_col : str
        The name of the column containing the population auxiliary total for each calibration group.

    Returns
    -------
    pd.DataFrame
        The full dataframe with the added new column "g_weight_{aux_col}".
    """
    df = df.copy()

    df[f"g_weight_{aux_col}"] = 1.0
    df = df.groupby(g_wgt_col, group_keys=False).apply(g_weight, aux_col, univ_aux_col, g_wgt_col)

    return df
