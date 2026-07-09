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
    strata_group: pd.DataFrame,
    ru_column: str,
    univ_count_col: str,
) -> pd.DataFrame:
    """Calculate the a-weights for a stratum group.

    The calculation here is:

    a_weight = N/n

    Where:
        N is population or universe count for the stratum
        n is the number of valid returns for the stratum

    Parameters
    ----------
    strata_group : pd.DataFrame
        The dataframe grouped by strata.
    ru_column : str
        The name of the column containing reporting unit identifiers.
    univ_count_col : str
        The name of the column containing the total number of reporting units in the stratum.

    Returns
    -------
    pd.DataFrame
        The dataframe with the a-weights calculated.
    """
    if strata_group.empty:
        return strata_group

    N = strata_group[univ_count_col].iloc[0]
    n = calc_lower_n(strata_group, ru_column)

    # Calculate 'a' for this group
    if n > 0:
        a_weight = N / n
    else:
        a_weight = 1.0

    strata_group["a_weight"] = a_weight

    return strata_group


def g_weight(
    strata_group: pd.DataFrame,
    aux_col: str,
    univ_aux_col: str,
) -> pd.DataFrame:
    """Calculate the g-weights for a calibration group.

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
    strata_group : pd.DataFrame
        The dataframe grouped by calibration group.
    aux_col : str
        The name of the column containing auxiliary employment data.
    univ_aux_col : str
        The name of the column containing the total auxiliary employment in the calibration group.

    Returns
    -------
    pd.DataFrame
        The dataframe with the g-weights calculated.
    """
    if strata_group.empty:
        return strata_group

    univ_aux_sum = strata_group[univ_aux_col].iloc[0]
    aux_col_sum = strata_group[aux_col].sum()
    # sum_ax must be computed row-by-row (a_i * x_i) before summing
    sum_ax = (strata_group["a_weight"] * strata_group[aux_col]).sum()

    # Calculate g-weight for this group
    if aux_col_sum > 0 and sum_ax > 0:
        g_weight = univ_aux_sum / sum_ax
    else:
        g_weight = 1.0

    strata_group["univ_aux_sum"] = univ_aux_sum
    strata_group["aux_col_sum"] = aux_col_sum

    strata_group["g_weight"] = g_weight

    return strata_group


def create_weights_qa_df(
    df: pd.DataFrame,
    strata_col: str,
    incl_g_wts: bool = True,
) -> pd.DataFrame:
    """Create a QA dataframe for the weight calculation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the weights columns.
    strata_col : str
        The name of the column containing stratum identifiers.
    incl_g_wts : bool, optional
        Whether g weights were calculated.

    Returns
    -------
    pd.DataFrame
        The QA dataframe.
    """
    qa_cols_list = [strata_col, "N", "n", "a_weight"]
    if incl_g_wts:
        qa_cols_list += ["univ_aux_sum", "aux_col_sum", "g_weight"]

    qa_df = df[qa_cols_list].groupby(strata_col).first()
    qa_df = qa_df.reset_index()

    return qa_df


def calculate_a_weights(
    df: pd.DataFrame,
    strata_col: str,
    ru_col: str,
    univ_count_col: str,
) -> pd.DataFrame:
    """Calculate the 'a' weight for each stratum in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input df containing survey data.
    strata_col : str
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
    df = df.groupby(strata_col, group_keys=False).apply(a_weight, ru_col, univ_count_col)

    return df


def calculate_g_weights(
    df: pd.DataFrame,
    strata_col: str,
    aux_col: str,
    univ_aux_col: str,
) -> pd.DataFrame:
    """Calculate the g-weight for each calibration group in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input df containing survey data
    strata_col : str
        The name of the column containing calibration group identifiers (k).
    aux_col : str
        The name of the column containing auxiliary employment data.
    univ_aux_col : str
        The name of the column containing the population auxiliary total for each calibration group.

    Returns
    -------
    pd.DataFrame
        The full dataframe with the added new column "g_weight".
    """
    df = df.copy()

    df["g_weight"] = 1.0
    df = df.groupby(strata_col, group_keys=False).apply(g_weight, aux_col, univ_aux_col)

    return df
