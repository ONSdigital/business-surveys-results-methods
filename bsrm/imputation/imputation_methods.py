"""Define imputation methods for the IABS pipeline."""

import pandas as pd


# This variable is used to assign a default value in get_imputed_estimate to a pd.Series where every
# value is True, following ruff's B008 pattern
# See: https://docs.astral.sh/ruff/rules/function-call-in-default-argument for more info
SERIES_ALL_TRUE = pd.Series(True)


def impute_capex(
    group: pd.DataFrame,
    sector: str,
) -> pd.DataFrame:
    """Impute missing responses for CAPEX variables.

    Q601 is Capex - assets under financial leasing
    Q311 is Capex - sales of merchanted goods

    Q601 is calculated only for the following sectors:
        production and construction (acp)
        motor trade (mt)
        wholesale (wa)
    Q311 is calculated only for production and construction (acp)
    """
    # This should "IDBR employment" according to spec, is that a different col?
    cond = group["empment"] > 99

    if sector in ["acp", "mt", "wa"]:
        group = get_imputed_estimate(group, "Q601", "Q600", cond, adjusted=True)

    if sector == "acp":
        group = get_imputed_estimate(group, "Q311", "Q399", cond, adjusted=True)

    return group


def impute_missing_value(
    df: pd.DataFrame,
    current_value: str,
    prev_value: str,
    adjusted: bool = False,
    keep_qa_cols: bool = False,
) -> pd.DataFrame:
    """Impute missing responses from previous year data via the median of ratios.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing data to be imputed (must be in long format).
    current_value : str
        The name of the column containing the current period value.
    prev_value : str
        The name of the column containing the previous period value.
    adjusted : bool
        Whether to adjust the median for the number of zeroes.
    keep_qa_cols : bool
        Whether to keep the columns used in computing the median in the output for QA or not.

    Returns
    -------
    pd.DataFrame
        The input dataframe with missing values imputed where possible.
    """
    # Calculate growth rate, then the median per group of question and imputation class ("cell_no")
    df["growth_rate"] = df[current_value] / df[prev_value]
    df["median"] = df.groupby(["cell_no", "question"])["growth_rate"].transform("median")

    # Calculate adjustment to account for zeroes if needed
    if adjusted:
        df["adjustment_numerator"] = df.groupby(["cell_no", "question"])[current_value].transform(
            lambda values: (values > 0).sum()
        )
        df["adjustment_denominator"] = df.groupby(["cell_no", "question"])[current_value].transform(
            lambda values: (values >= 0).sum()
        )
        df["median_adjustment"] = df["adjustment_numerator"] / df["adjustment_denominator"]
    else:
        df["median_adjustment"] = 1.0

    # Calculate the adjusted median
    df["median"] = df["median"] * df["median_adjustment"]

    # Use the adjusted median to impute missing values
    df.loc[df[current_value].isna(), "imputed_value"] = df[prev_value] * df["median"]

    # Count number of rows usable for imputation in each group, if less than 3 we can't impute
    group_count = df.groupby(["cell_no", "question"])[current_value].transform("count")
    df.loc[group_count < 3, "imputed_value"] = None

    # Drop columns used for calculation unless we need them for QA
    if not keep_qa_cols:
        df = df.drop(
            columns=[
                "growth_rate",
                "median",
                "adjustment_numerator",
                "adjustment_denominator",
                "median_adjustment",
            ],
            errors="ignore",
        )

    return df


def get_imputed_estimate(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    cond: pd.Series = SERIES_ALL_TRUE,
    adjusted: bool = False,
) -> pd.DataFrame:
    """Calculate the imputed estimate of a column via the median of ratios.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the column to impute.
    col1 : str
        The name of the column which will be imputed.
    col2 : str
        The name of the column to use to calculate growth factors for imputation.
    cond : bool
        A column condition for which rows are eligible to be used to calculate the median.
    adjusted : bool
        Whether to adjust the median for the inclusion of zeros.

    Returns
    -------
    pd.DataFrame
        The input dataframe with an additional column containing the imputed estimate, of the form
        {COL1}_imputed_estimate.
    """
    # Calculate the growth factor for rows where the condition is true
    growth_factor = f"{col1}_growth_factor"
    df.loc[cond, growth_factor] = df[col1] / df[col2]

    # Calculate the median
    median = df.loc[cond][growth_factor].median()

    # If needed, adjust the median to account for the inclusion of zeroes
    if adjusted:
        numerator = len(df[df[col1] > 0])
        denominator = len(df[df[col1] >= 0])
        median = (numerator / denominator) * median

    # Add the median into the dataframe
    df[f"{col1}_median_growth_factor"] = median

    # Impute the value using the median
    df.loc[df[col1].isna(), f"{col1}_imputed_estimate"] = df[col2] * median

    return df
