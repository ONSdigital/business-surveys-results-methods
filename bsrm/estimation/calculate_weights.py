"""Functions to calculate estimation weights for survey data."""

import pandas as pd
import logging

CalcWeights_Logger = logging.getLogger(__name__)


def calc_lower_n(df: pd.DataFrame, ru_column: str) -> int:
    """Calculate 'n' which is a number of unique reporting units (RUs) in the dataset.

    Parameters
    ----------
        df (pd.DataFrame): The input dataframe which contains survey data,
            including expenditure data
        ru_column (str): The name of the column containing reporting unit identifiers.

    Returns
    -------
        int: The number of unique reporting units (RUs).
    """
    n = df[ru_column].nunique()

    return n


def calc_lower_e(df: pd.DataFrame, col_name: str) -> int:
    """Calculate 'e' which is a sum of IDBR employment data in the filtered dataset.

    Parameters
    ----------
        df (pd.DataFrame): The input dataframe which contains survey data,
            including IDBR employment data.
        col_name (str): The name of the column for this calculation.

    Returns
    -------
        int: The sum of IDBR employment of sampled.
    """
    e = df[col_name].sum()

    return e


def calc_lower_s(df: pd.DataFrame, col_name: str) -> int:
    """Calculate 's' which identifies the sum of outliers for a cell group.

    Parameters
    ----------
        df (pd.DataFrame): The input dataframe which contains survey data.
        col_name (str): The name of the column for this calculation.

    Returns
    -------
        int: Calculated value of s.
    """
    # Filter where outliers bool = true
    df = df.loc[df.outlier]

    # If there are no outliers, return 0
    if df.empty:
        s = 0
    else:
        # Sum the specified column
        s = df[col_name].sum()

    return s


def a_weight(cell_group: pd.DataFrame, ru_column: str) -> pd.DataFrame:
    """Calculate the 'a' weighting factor for a cell group.

    The calculation here is:

    a = (N-o) / (n-o)

    Where:
        - N is the total number of businesses in the cell
        - n is the number of businesses in sample for that cell
        - o is the number of outliers in the cell

    'o' is calculated in this function by summing all the `True` values
        because `True` == 1

    Parameters
    ----------
        cell_group (pd.DataFrame): The dataframe grouped by cellnumber.
        ru_column (str): The name of the column containing reporting unit identifiers.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'a' weighting factor calculated.
    """
    if cell_group.empty:
        return cell_group

    N = cell_group["uni_count"].iloc[0]  # noqa: N806 (allow capitals for variables)
    n = calc_lower_n(cell_group, ru_column)

    # Count the outliers for this group (will count all the `True` values)
    outlier_count = cell_group["outlier"].sum()

    # Calculate 'a' for this group
    if n > 0:
        a_weight = (N - outlier_count) / (n - outlier_count)
    else:
        a_weight = 1.0

    cell_group["N"] = N
    cell_group["n"] = n
    cell_group["o"] = outlier_count

    cell_group["a_weight"] = a_weight

    return cell_group


def calc_g_weight(cell_group: pd.DataFrame, aux_col_name: str) -> pd.DataFrame:
    """Calculate the 'g' weighting factor for a cell group.

    The calculation for the g-weight is:

    g = (E - s) / a * (e - s)

    TODO: this needs to be made more general, currently this is for R&D
    Where:
        - E is the sum of IDBR employment for all businesses in a cell
        - e is the sum of IDBR employment for all sampled, valid responses in the cell
        - s is the sum of IDBR employment for all outliered sampled, valid responses
        - a is the 'a' weighting factor for the cell

    Parameters
    ----------
        cell_group (pd.DataFrame): The dataframe grouped by cellnumber.
        aux_col_name (str): The name of the column containing auxiliary employment data.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'g' weighting factor calculated.
    """
    if cell_group.empty:
        return cell_group

    E = cell_group["uni_employment"].iloc[0]  # noqa: N806
    a = cell_group["a_weight"].iloc[0]
    e = calc_lower_e(cell_group, aux_col_name)
    s = calc_lower_s(cell_group, aux_col_name)

    # Calculate 'g' for this group
    if (e - s) > 0:
        g_weight = (E - s) / (a * (e - s))
    else:
        g_weight = 1.0

    cell_group["E"] = E
    cell_group["e"] = e
    cell_group["s"] = s

    cell_group["g_weight"] = g_weight

    return cell_group


def create_weights_qa_df(
    df: pd.DataFrame, strata_col: str, incl_g_wts: bool = True
) -> pd.DataFrame:
    """Create a QA dataframe for the weight calculation.

    Parameters
    ----------
        df (pd.DataFrame): The dataframe containing the weights columns.
        strata_col (str): The name of the column containing stratum identifiers.
        incl_g_wts (bool, optional): Whether g weights were calculated.

    Returns
    -------
        pd.DataFrame: The QA dataframe.
    """
    qa_cols_list = [strata_col, "N", "n", "o"]
    if incl_g_wts:
        qa_cols_list += ["E", "e", "s", "a_weight", "g_weight"]
    else:
        qa_cols_list += ["a_weight"]

    qa_frame = df[qa_cols_list].groupby(strata_col).first()
    qa_frame = qa_frame.reset_index()

    return qa_frame


def outlier_weights(df: pd.DataFrame, incl_g_wts: bool = True) -> pd.DataFrame:
    """Calculate weights for outliers.

    If a reference has been flagged as an outlier,
    the 'a weight' value is set to 1.0

    Parameters
    ----------
        df (pd.DataFrame): The dataframe weights are calculated for.
        incl_g_wts (bool, optional): Whether g weights were calculated.

    Returns
    -------
        pd.DataFrame: The dataframe with the a_weights set to 1.0 for outliers.
    """
    df.loc[df["outlier"], "a_weight"] = 1.0
    if incl_g_wts:
        df.loc[df["outlier"], "g_weight"] = 1.0
    return df


def calculate_a_weights(
    df: pd.DataFrame,
    strata_col: str,
    ru_col: str,
) -> pd.DataFrame:
    """Calculate the 'a' weight for each stratum in the data.

    Parameters
    ----------
        df (pd.DataFrame): The input df containing survey data.
        strata_col (str): The name of the column containing stratum identifiers.
        ru_col (str): The name of the column containing reference unit data.

    Returns
    -------
        pd.DataFrame: The full dataframe with the added new column "a_weight".
    """
    df = df.copy()
    df["a_weight"] = 1.0
    df = df.groupby(strata_col, group_keys=False).apply(a_weight, ru_col)

    return df


def calculate_g_weights(
    df: pd.DataFrame, strata_col: str, aux_col: str
) -> pd.DataFrame:
    """Calculate the 'g' weight for each stratum in the data.

    Parameters
    ----------
        df (pd.DataFrame): The input df containing survey data
        strata_col (str): The name of the column containing stratum identifiers.
        aux_col (str): The name of the column containing auxiliary employment data.

    Returns
    -------
        pd.DataFrame: The full dataframe with the added new column "g_weight".
    """
    df = df.copy()

    df["g_weight"] = 1.0
    df = df.groupby(strata_col, group_keys=False).apply(calc_g_weight, aux_col)

    return df
