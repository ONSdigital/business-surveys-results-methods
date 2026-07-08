"""
Validate user input to run_estimation and apply_weights functions.

To catch user errors before processing and return meaningful messages

Public Functions:
    * validate_run_estimation_input
    * validate_apply_weights_input
"""

import pandas as pd
from warnings import warn


def validate_run_estimation_input(
    data: pd.DataFrame,
    strata_col: str,
    ru_col: str,
    univ_count_col: str,
    aux_col: str,
    univ_aux_col: str,
    outlier_col: str,
    incl_g_wts: bool = True,
) -> None:
    """
    Call functions to validate input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - strata_col (str),
    - ru_col (str),
    - univ_count_col (str),
    - aux_col (str),
    - univ_aux_col (str),
    - outlier_col (str),
    - incl_g_wts (bool)

    Raises
    ------
    - TypeError if data type validation fails
    - Exception if columns not in data
    """
    if not isinstance(data, pd.DataFrame):
        msg = "Specified value for data must be a Pandas DataFrame."
        raise TypeError(msg)

    if not isinstance(incl_g_wts, bool):
        msg = "Specified value for incl_g_wts must be a Boolean."
        raise TypeError(msg)

    # Check that column names are strings and in the data
    _check_columns(data, [strata_col, ru_col, univ_count_col, aux_col, univ_aux_col, outlier_col])


def validate_apply_weights_input(
    data: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str],
    calc_g_weight: bool,
    round_val: int,
) -> None:
    """
    Call functions to validate input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_cols (list[str])
    - g_weight_cols (list[str])
    - calc_g_weight (bool)
    - round_val (int)

    Raises
    ------
    - TypeError if data type validation fails
    """
    if not isinstance(data, pd.DataFrame):
        msg = "Specified value for data must be a Pandas DataFrame."
        raise TypeError(msg)

    if not isinstance(calc_g_weight, bool):
        msg = "Specified value for calc_g_weight must be a Boolean."
        raise TypeError(msg)

    if not isinstance(round_val, int):
        msg = "Specified value for round_val must be an integer."
        raise TypeError(msg)

    # Raise exception if no g_weights specified when calc_g_weight is True or
    # if g_weights specified when calc_g_weight is False
    if calc_g_weight and (len(g_weight_cols) == 0):
        msg = (
            "No g_weight columns have been specified but calc_g_weight is True. "
            "Please check whether g_weights are required."
        )
        raise Exception(msg)
    if (not calc_g_weight) and (len(g_weight_cols) > 0):
        msg = (
            "g_weight columns have been specified but calc_g_weight is False. "
            "Please check whether g_weights are required."
        )
        raise Exception(msg)

    # Validate weights specified
    _validate_weights(data, a_weight_cols, g_weight_cols)


def _validate_weights(
    data: pd.DataFrame, a_weight_cols: list[str], g_weight_cols: list[str]
) -> None:
    """
    Validate weights specified by user.

    Validate
    --------
    - a_weights and g_weights are lists of strings which are columns in the data
    - at least one column is specified for either a_weights or g_weights

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_cols (list[str])
    - g_weight_cols (list[str])

    Raises
    ------
    - Exception if specified columns are not in data or if no weights are specified
    - Warning if only a_weights or only g_weights will be applied
    """
    _check_is_list("a_weight_cols", a_weight_cols)
    _check_is_list("g_weight_cols", g_weight_cols)

    _check_columns(data, a_weight_cols + g_weight_cols)

    # Raise error if no weights specified
    if not (a_weight_cols or g_weight_cols):
        msg = "No a_weights or g_weights have been specified. Cannot apply Estimation."
        raise Exception(msg)

    # Warning if only a_weights or only g_weights
    elif not a_weight_cols:
        warn("No a_weights have been specified. Applying g_weights only.", stacklevel=3)
    elif not g_weight_cols:
        warn("No g_weights have been specified. Applying a_weights only.", stacklevel=3)


def _check_is_list(param_name: str, param_cols: list[str]) -> None:
    """
    Check specified parameter is a list.

    Parameters
    ----------
    - param_name: str
    - param_cols: list[str]

    Raises
    ------
    - TypeError if not list.
    """
    if not isinstance(param_cols, list):
        msg = (
            f"Expected {param_name} to be a list (defined using []), but got "
            f"{type(param_cols).__name__}!"
        )
        raise TypeError(msg)


def _check_columns(data: pd.DataFrame, cols_list: list[str]) -> None:
    """
    Check specified column names are strings and in the data.

    Validate
    --------
    - that column names are strings
    - that data contains specified columns

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - cols_list (list[str])

    Raises
    ------
    - TypeError if specified column name(s) not string.
    - Exception if specified column(s) not in data
    """
    not_strings = [col_name for col_name in cols_list if not isinstance(col_name, str)]
    if not_strings:
        msg = f"Specified column name(s): {', '.join(map(str, not_strings))} must be string(s)."
        raise TypeError(msg)

    cols_list = [c for c in cols_list if len(c) > 0]  # remove empty strings

    missing = [col_name for col_name in cols_list if col_name not in data.columns]
    if missing:
        msg = f"Specified column(s): {', '.join(missing)} must be column(s) in the data."
        raise Exception(msg)
