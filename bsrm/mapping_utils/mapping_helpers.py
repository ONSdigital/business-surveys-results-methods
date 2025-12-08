"""Specific functions applied in mapping_main.py.

Public Functions:
    * mapper_null_checks
    * join_with_null_check
    * check_mapping_unique

Private Functions:
    * None
"""

import pandas as pd
import logging

MappingLogger = logging.getLogger(__name__)


def mapper_null_checks(
    mapper_df: pd.DataFrame,
    mapper_name: str,
    validate_cols: list | None = None,
    warn: bool = False,
) -> None:
    """Perform null checks on selected columns of a mapper DataFrame.

    Parameters
    ----------
        mapper_df (pd.DataFrame): The mapper DataFrame to check.
        mapper_name (str): The name of the mapper being validated.
        validate_cols (list, optional): List of columns to validate.
            If None, all columns are validated.
        warn (bool,optional): Whether to warn instead of raising an error.

    Raises
    ------
        ValueError: Raised if nulls are found in the specified columns
            and 'warn' bool is False.

    """
    if validate_cols is None:
        validate_cols = mapper_df.columns.tolist()

    # List to store columns with null values
    null_cols = []

    # Check for null values of all columns in the list
    for col in validate_cols:
        if mapper_df[col].isna().any():
            # Append col to list
            null_cols.append(col)
    # If null cols is not empty, raise a warning that prints list of columns
    if null_cols:
        msg = f"Mapper {mapper_name} contains nulls values in {null_cols}"
        if warn:
            MappingLogger.warning(msg)
        else:
            raise ValueError(msg)


def join_with_null_check(
    df: pd.DataFrame,
    mapper_df: pd.DataFrame,
    mapper_name: str,
    join_col: str,
    warn: bool = False,
) -> pd.DataFrame:
    """Perform a left join on two DataFrames and check for nulls on the join.

    Parameters
    ----------
        df (pd.DataFrame): The main DataFrame.
        mapper_df (pd.DataFrame): The mapper DataFrame.
        mapper_name (str): The name of the mapper being validated.
        join_col (str): The column to join on.
        warn (bool, optional): Whether to warn instead of raising an error.

    Returns
    -------
        pd.DataFrame: The merged DataFrame.

    Raises
    ------
        ValueError: Raised if nulls are found in the join and 'warn' bool is False.

    """
    merged = df.merge(
        mapper_df,
        how="left",
        on=join_col,
        indicator=True,
        suffixes=("", "_drop"),
    )

    df = merged.drop(columns=[col for col in merged.columns if col.endswith("_drop")])

    # Check for nulls in the join. Either warn or raise an error.
    missing = df.loc[
        df[join_col].notna() & df["_merge"].eq("left_only"),
        join_col,
    ].unique()
    if len(missing) > 0:
        msg = (
            f"Nulls found in the join on {join_col} of {mapper_name} mapper."
            f"Missing values: {missing} "
        )
        if warn:
            MappingLogger.warning(msg)
        else:
            raise ValueError(msg)

    df = df.drop("_merge", axis=1)

    return df


def check_mapping_unique(
    mapper_df: pd.DataFrame,
    col_to_check: str,
    mapper_name: str,
) -> None:
    """
    Check that the values of column in a mapper DataFrame are all different (unique).

    This will ensure that they can be uniquely mapped to values in another column.

    Parameters
    ----------
        mapper_df (pd.DataFrame): The mapper DataFrame to check.
        col_to_check (str): The name of the column to check.
        mapper_name (str): The name of the mapper being validated.

    Raises
    ------
        ValueError: If the column does not contain unique values.
    """
    if not mapper_df[col_to_check].is_unique:
        e = f"{mapper_name} mapper has non-unique keys in {col_to_check}."
        raise ValueError(e)
