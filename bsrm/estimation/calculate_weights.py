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


def calc_aux_col_sum(df: pd.DataFrame, aux_col: str) -> int:
    """Calculate aux_col_sum: sum of auxiliary data for sampled units.

    Parameters
    ----------
        df (pd.DataFrame): The input dataframe which contains survey data,
            including auxiliary data.
        aux_col (str): The name of the auxiliary column for this calculation.

    Returns
    -------
        int: The sum of auxiliary data for sampled units.
    """
    aux_col_sum = df[aux_col].sum()

    return aux_col_sum


def calc_lower_s(df: pd.DataFrame, col_name: str, outlier_col: str) -> int:
    """Calculate 's' which identifies the sum of outliers for a stratum group.

    Parameters
    ----------
        df (pd.DataFrame): The input dataframe which contains survey data.
        col_name (str): The name of the column for this calculation.
        outlier_col (str): The name of the column containing outlier indicators.

    Returns
    -------
        int: Calculated value of s.
    """
    # Filter where outliers bool = true
    df = df.loc[df[outlier_col]]

    # If there are no outliers, return 0
    if df.empty:
        s = 0
    else:
        # Sum the specified column
        s = df[col_name].sum()

    return s


def a_weight(
    strata_group: pd.DataFrame, ru_column: str, univ_count_col: str, outlier_col: str
) -> pd.DataFrame:
    """Calculate the 'a' weighting factor for a stratum group.

    The calculation here is:

    a = (N-o) / (n-o)

    Where:
        - N is the total number of reporting units in the stratum (universe)
        - n is the number of reporting units in sample for the stratum
        - o is the number of outliers in the stratum

    'o' is calculated in this function by summing all the `True` values

    Parameters
    ----------
        strata_group (pd.DataFrame): The dataframe grouped by strata.
        ru_column (str): The name of the column containing reporting unit identifiers.
        univ_count_col (str): The name of the column containing the total number of
            reporting units in the stratum.
        outlier_col (str): The name of the column containing outlier indicators.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'a' weighting factor calculated.
    """
    if strata_group.empty:
        return strata_group

    N = strata_group[univ_count_col].iloc[0]  # noqa: N806 (allow capitals for vars)
    n = calc_lower_n(strata_group, ru_column)

    # Count the outliers for this group (will count all the `True` values)
    outlier_count = strata_group[outlier_col].sum()

    # Calculate 'a' for this group
    if n > 0:
        a_weight = (N - outlier_count) / (n - outlier_count)
    else:
        a_weight = 1.0

    strata_group["N"] = N
    strata_group["n"] = n
    strata_group["o"] = outlier_count

    strata_group["a_weight"] = a_weight

    return strata_group


def calc_g_weight(
    strata_group: pd.DataFrame, aux_col: str, univ_aux_col: str, outlier_col: str
) -> pd.DataFrame:
    """Calculate the 'g' weighting factor for a stratum group.

    The calculation for the g-weight is:

    g = (univ_aux_col - s) / a * (aux_col - s)

    Where:
        - univ_aux_col is the universe auxiliary total for all reporting
            units in the stratum
        - aux_col is the sample auxiliary total for all sampled valid
            responses in the stratum
        - s is the sum of auxiliary data for all outliered sampled, valid
            responses in the stratum
        - a is the 'a' weighting factor for the stratum

    Parameters
    ----------
        strata_group (pd.DataFrame): The dataframe grouped by strata.
        aux_col (str): The name of the column containing auxiliary employment data.
        univ_aux_col (str): The name of the column containing the total auxiliary
            employment in the stratum.
        outlier_col (str): The name of the column containing outlier indicators.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'g' weighting factor calculated.
    """
    if strata_group.empty:
        return strata_group

    univ_aux_col_value = strata_group[univ_aux_col].iloc[0]  # noqa: N806
    a_weight_value = strata_group["a_weight"].iloc[0]
    aux_col_sum = calc_aux_col_sum(strata_group, aux_col)
    outlier_aux_col_sum = calc_lower_s(strata_group, aux_col, outlier_col)

    # Calculate 'g' for this group
    if (aux_col_sum - outlier_aux_col_sum) > 0:
        g_weight = (univ_aux_col_value - outlier_aux_col_sum) / (
            a_weight_value * (aux_col_sum - outlier_aux_col_sum)
        )
    else:
        g_weight = 1.0

    strata_group["univ_aux_col_value"] = univ_aux_col_value
    strata_group["aux_col_sum"] = aux_col_sum
    strata_group["s"] = outlier_aux_col_sum

    strata_group["g_weight"] = g_weight

    return strata_group


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
        qa_cols_list += [
            "univ_aux_col_value",
            "aux_col_sum",
            "s",
            "a_weight",
            "g_weight",
        ]
    else:
        qa_cols_list += ["a_weight"]

    qa_frame = df[qa_cols_list].groupby(strata_col).first()
    qa_frame = qa_frame.reset_index()

    return qa_frame


def outlier_weights(
    df: pd.DataFrame, outlier_col: str, incl_g_wts: bool = True
) -> pd.DataFrame:
    """Calculate weights for outliers.

    If a reference has been flagged as an outlier,
    the 'a weight' value is set to 1.0

    Parameters
    ----------
        df (pd.DataFrame): The dataframe weights are calculated for.
        outlier_col (str): The name of the column containing outlier indicators.
        incl_g_wts (bool, optional): Whether g weights were calculated.

    Returns
    -------
        pd.DataFrame: The dataframe with the a_weights set to 1.0 for outliers.
    """
    df.loc[df[outlier_col], "a_weight"] = 1.0
    if incl_g_wts:
        df.loc[df[outlier_col], "g_weight"] = 1.0
    return df


def calculate_a_weights(
    df: pd.DataFrame,
    strata_col: str,
    ru_col: str,
    univ_count_col: str,
    outlier_col: str,
) -> pd.DataFrame:
    """Calculate the 'a' weight for each stratum in the data.

    Parameters
    ----------
        df (pd.DataFrame): The input df containing survey data.
        strata_col (str): The name of the column containing stratum identifiers.
        ru_col (str): The name of the column containing reference unit data.
        univ_count_col (str): The name of the column containing the total number of
            reporting units in the stratum.
        outlier_col (str): The name of the column containing outlier indicators.

    Returns
    -------
        pd.DataFrame: The full dataframe with the added new column "a_weight".
    """
    df = df.copy()
    df["a_weight"] = 1.0
    df = df.groupby(strata_col, group_keys=False).apply(
        a_weight, ru_col, univ_count_col, outlier_col
    )

    return df


def calculate_g_weights(
    df: pd.DataFrame, strata_col: str, aux_col: str, univ_aux_col: str, outlier_col: str
) -> pd.DataFrame:
    """Calculate the 'g' weight for each stratum in the data.

    Parameters
    ----------
        df (pd.DataFrame): The input df containing survey data
        strata_col (str): The name of the column containing stratum identifiers.
        aux_col (str): The name of the column containing auxiliary employment data.
        univ_aux_col (str): The name of the column containing the total number of
            reporting units in the stratum for auxiliary data.
        outlier_col (str): The name of the column containing outlier indicators.

    Returns
    -------
        pd.DataFrame: The full dataframe with the added new column "g_weight".
    """
    df = df.copy()

    df["g_weight"] = 1.0
    df = df.groupby(strata_col, group_keys=False).apply(
        calc_g_weight, aux_col, univ_aux_col, outlier_col
    )

    return df
