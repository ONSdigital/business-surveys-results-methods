"""Functions to calculate estimation weights for survey data."""

import pandas as pd
import logging

CalcWeights_Logger = logging.getLogger(__name__)


def calc_lower_n(df: pd.DataFrame) -> int:
    """Calculate 'n' which is a number of unique RU references in the filtered dataset.

    Args:
        df (pd.DataFrame): The input dataframe which contains survey data,
            including expenditure data

    Returns
    -------
        int: The number of unique references.
    """
    n = df["reference"].nunique()

    return n


def calc_lower_e(df: pd.DataFrame) -> int:
    """Calculate 'e' which is a sum of IDBR employment data in the filtered dataset.

    Args:
        df (pd.DatatFrame): The input dataframe which contains survey data,
            including IDBR employment data.

    Returns
    -------
        int: The sum of IDBR employment of sampled.
    """
    e = df["employment"].sum()

    return e


def calc_lower_s(df: pd.DataFrame) -> int:
    """Calculate 's' which identifies the sum of outliers for a cell group.

    Args:
        df (pd.DataFrame): The input dataframe which contains survey data.

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
        # Sum the employment column
        s = df["employment"].sum()

    return s


def calc_a_weight(cell_group: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 'a' weighting factor for a cell group.

    The calculation here is:

    a = (N-o) / (n-o)

    Where:
        - N is the total number of businesses in the cell
        - n is the number of businesses in sample for that cell
        - o is the number of outliers in the cell

    'o' is calculated in this function by summing all the `True` values
        because `True` == 1

    Args:
        cell_group (pd.DataFrame): The dataframe grouped by cellnumber.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'a' weighting factor calculated.
    """
    if cell_group.empty:
        return cell_group

    N = cell_group["uni_count"].iloc[0]  # noqa: N806 (allow capitals for variables)
    n = calc_lower_n(cell_group)

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


def calc_g_weight(cell_group: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 'g' weighting factor for a cell group.

    The calculation for the g-weight is:

    g = (E - s) / a * (e - s)

    Where:
        - E is the sum of IDBR employment for all businesses in a cell
        - e is the sum of IDBR employment for all sampled, valid responses in the cell
        - s is the sum of IDBR employment for all outliered sampled, valid responses
        - a is the 'a' weighting factor for the cell

    Args:
        cell_group (pd.DataFrame): The dataframe grouped by cellnumber.

    Returns
    -------
        pd.DataFrame: The dataframe with the 'a' weighting factor calculated.
    """
    if cell_group.empty:
        return cell_group

    E = cell_group["uni_employment"].iloc[0]  # noqa: N806
    a = cell_group["a_weight"].iloc[0]
    e = calc_lower_e(cell_group)
    s = calc_lower_s(cell_group)

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


def create_weights_qa_df(df: pd.DataFrame, calc_g_weights: bool = True) -> pd.DataFrame:
    """Create a QA dataframe for the weight calculation.

    Args:
        df (pd.DataFrame): The dataframe containing the weights columns.
        calc_g_weights (bool, optional): Whether g weights were calculated.

    Returns
    -------
        pd.DataFrame: The QA dataframe.
    """
    if calc_g_weights:
        qa_cols_list = [
            "cellnumber",
            "N",
            "n",
            "o",
            "E",
            "e",
            "s",
            "a_weight",
            "g_weight",
        ]
    else:
        qa_cols_list = ["cellnumber", "N", "n", "o", "a_weight"]
    qa_frame = df[qa_cols_list].groupby("cellnumber").first()
    qa_frame = qa_frame.reset_index()
    qa_frame = qa_frame.rename(
        columns={
            "cellnumber": "Cell Number",
            "N": "N - uni_count",
            "n": "n - num clear records in cell",
            "o": "o - num outliers in cell",
            "E": "E - uni_employment",
            "e": "e - sum of employment in cell",
            "s": "s - sum of employment outliers in cell",
        }
    )

    return qa_frame


def outlier_weights(df: pd.DataFrame, calc_g_weights: bool = True) -> pd.DataFrame:
    """Calculate weights for outliers.

    If a reference has been flagged as an outlier,
    the 'a weight' value is set to 1.0

    Args:
        df (pd.DataFrame): The dataframe weights are calculated for.

    Returns
    -------
        pd.DataFrame: The dataframe with the a_weights set to 1.0 for outliers.
    """
    df.loc[df["outlier"], "a_weight"] = 1.0
    if calc_g_weights:
        df.loc[df["outlier"], "g_weight"] = 1.0
    return df


def calculate_weighting_factors(
    df: pd.DataFrame, calc_g_weights: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the 'a' weight and optionally 'g' weightfor each cell in the data.

    Args:
        df (pd.DataFrame): The input df containing survey data
        calc_g_weights (bool, optional): Whether to calculate g weights

    Returns
    -------
        tuple[pd.DataFrame, pd.DataFrame]:
        1) Returns the full dataframe with the added
        new column "a_weight".
        2) Returns a QA dataframe of all variables used in the calculation
    """
    df["a_weight"] = 1.0
    df = df.groupby("cellnumber", group_keys=False).apply(calc_a_weight)

    if calc_g_weights:
        df["g_weight"] = 1.0
        df = df.groupby("cellnumber", group_keys=False).apply(calc_g_weight)

    # Create a QA dataframe
    qa_frame = create_weights_qa_df(df, calc_g_weights=calc_g_weights)

    # Apply the outlier weights
    df = outlier_weights(df, calc_g_weights=calc_g_weights)

    # drop intermediate calculation columns
    drop_cols = ["N", "n", "o"]
    if calc_g_weights:
        drop_cols.extend(["E", "e", "s"])

    df = df.drop(columns=drop_cols)

    return df, qa_frame
