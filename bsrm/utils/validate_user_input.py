"""
Validate user input to run_estimation function.

To catch user errors before processing and return meaningful messages

Public Functions:
    * validate_user_input
"""

import pandas as pd
from warnings import warn


def validate_user_input(
    data: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str],
    round_val: int,
) -> None:
    """
    Call functions to validate input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_cols (list[str])
    - g_weight_cols (list[str])
    - round_val (int)
    """
    _validate_type(data, a_weight_cols, g_weight_cols, round_val)
    _validate_weights(data, a_weight_cols, g_weight_cols)


def _validate_type(
    data: pd.DataFrame,
    a_weight_cols: list[str],
    g_weight_cols: list[str],
    round_val: int,
) -> None:
    """
    Type validation on input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_cols (list[str])
    - g_weight_cols (list[str])
    - round_val (int)

    Raises
    ------
    - TypeError if validation fails
    """
    if not isinstance(data, pd.DataFrame):
        msg = "Specified value for data must be a Pandas DataFrame."
        raise TypeError(msg)

    # Check that a_weight_cols and g_weight_cols, are lists of strings
    _check_list_of_strings("a_weight_cols", a_weight_cols)
    _check_list_of_strings("g_weight_cols", g_weight_cols)

    # Check round_val is an integer
    if not isinstance(round_val, int):
        msg = "Specified value for round_val must be an integer."
        raise TypeError(msg)


def _check_list_of_strings(weights_type: str, weight_cols: list[str]) -> None:
    """
    Type validation on weights.

    Parameters
    ----------
    - weights_type: str
    - weight_cols: (list[str])

    Raises
    ------
    - TypeError if validation fails.
    """
    if not isinstance(weight_cols, list):
        msg = "Expected ", weights_type, "to be a list (defined using []), "
        f"but got '{type(weight_cols).__name__}'!"
        raise TypeError(msg)
    if not all(isinstance(item, str) for item in weight_cols):
        msg = "All items in ", weights_type, " list must be strings!"
        raise TypeError(msg)


def _validate_weights(
    data: pd.DataFrame, a_weight_cols: list[str], g_weight_cols: list[str]
) -> None:
    """
    Validate weights specified by user.

    Validate
    --------
    - if a_weight_cols are specified then data contains specified columns
    - if g_weight_cols are specified then data contains specified columns
    - at least one column specified for either a_weights or g_weights

    Returns message when only a_weights or only g_weights specified

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_cols (list[str])
    - g_weight_cols (list[str])

    Raises
    ------
    - Exception if validation fails
    - Warning if only a_weights or only g_weights specified
    """
    # Check a_weight_cols and g_weight_cols are columns in data
    if a_weight_cols and not all(item in data.columns for item in a_weight_cols):
        msg = "Specified value(s) for a_weight_cols must be column(s) in data."
        raise Exception(msg)

    if g_weight_cols and not all(item in data.columns for item in g_weight_cols):
        msg = "Specified value(s) for g_weight_cols must be column(s) in data."
        raise Exception(msg)

    # Raise error if no weights specified
    if not (a_weight_cols or g_weight_cols):
        msg = "No a_weights or g_weights have been specified. Cannot apply Estimation."
        raise Exception(msg)

    # Warning if only a_weights or only g_weights
    elif not a_weight_cols:
        warn("No a_weights have been specified. Applying g_weights only.", stacklevel=2)
    elif not g_weight_cols:
        warn("No g_weights have been specified. Applying a_weights only.", stacklevel=2)
