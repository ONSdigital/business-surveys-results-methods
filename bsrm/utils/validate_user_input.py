"""
Validate user input to run_estimation and apply_weights functions.

To catch user errors before processing and return meaningful messages

Public Functions:
    * validate_estimation_config_dict
    * validate_run_estimation_input
    * validate_g_weighting_input
    * validate_apply_weights_input
"""

import pandas as pd
from warnings import warn


def validate_estimation_config_dict(config_dict: dict) -> dict:
    """Reshape the estimation YAML dictionary into flat config fields."""
    g_weight_dict = config_dict.get("g_weight_columns")

    g_weight_columns = None
    aux_cols = None
    univ_aux_cols = None

    if g_weight_dict is not None:
        g_weight_columns = {}
        aux_cols = []
        univ_aux_cols = []

        for variable_config in g_weight_dict.values():
            aux_col = variable_config["aux_col"]
            g_weight_columns[aux_col] = variable_config["apply_cols"]
            aux_cols.append(aux_col)
            univ_aux_cols.append(variable_config["univ_aux_col"])

    config_dict = {
        "a_wgt_band_col": config_dict["a_wgt_band"],
        "g_wgt_band_col": config_dict["g_wgt_band"],
        "a_weight_columns": config_dict["a_weight_columns"]["apply_cols"],
        "g_weight_columns": g_weight_columns,
        "incl_g_wts": config_dict["incl_g_wts"],
        "round_val": config_dict["round_val"],
        "ru_col": config_dict["ru_col"],
        "univ_count_col": config_dict["univ_count_col"],
        "aux_cols": aux_cols,
        "univ_aux_cols": univ_aux_cols,
    }
    return config_dict


def validate_run_estimation_input(
    data: pd.DataFrame,
    a_weight_col: str,
    ru_col: str,
    univ_count_col: str,
    g_weight_col: str | None = None,
    aux_cols: list[str] | None = None,
    univ_aux_cols: list[str] | None = None,
    incl_g_wts: bool = True,
) -> None:
    """
    Call functions to validate input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_col (str),
    - ru_col (str),
    - univ_count_col (str),
    - g_weight_col (str | None),
    - aux_cols (list[str] | None),
    - univ_aux_cols (list[str] | None),
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

    _check_columns(data, [a_weight_col, ru_col, univ_count_col])

    if not incl_g_wts:
        return

    if g_weight_col is None or aux_cols is None or univ_aux_cols is None:
        msg = "G weights cannot be calculated due to missing columns."
        raise ValueError(msg)

    _check_is_list("aux_cols", aux_cols)
    _check_is_list("univ_aux_cols", univ_aux_cols)

    if len(aux_cols) != len(univ_aux_cols):
        msg = "aux_cols and univ_aux_cols must be the same length."
        raise ValueError(msg)

    _check_columns(data, [g_weight_col, *aux_cols, *univ_aux_cols])


def validate_g_weighting_input(
    g_weight_col: str | None,
    aux_cols: list[str] | None,
    univ_aux_cols: list[str] | None,
) -> tuple[str, list[str], list[str]]:
    """Return validated g-weight inputs as concrete values."""
    if g_weight_col is None or aux_cols is None or univ_aux_cols is None:
        msg = "G weights cannot be calculated due to missing columns."
        raise ValueError(msg)

    return g_weight_col, aux_cols, univ_aux_cols


def validate_apply_weights_input(
    data: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: dict[str, list[str]] | None,
    aux_cols: list[str] | None,
    calc_g_weight: bool,
    round_val: int,
) -> None:
    """
    Call functions to validate input data and parameters.

    Parameters
    ----------
    - data (pd.DataFrame): The main dataset
    - a_weight_columns (list[str])
    - g_weight_columns (dict[str, list[str]] | None)
    - aux_cols (list[str] | None)
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
    if calc_g_weight and not g_weight_columns:
        msg = (
            "No g_weight columns have been specified but calc_g_weight is True. "
            "Please check whether g_weights are required."
        )
        raise Exception(msg)
    if (not calc_g_weight) and g_weight_columns:
        msg = (
            "g_weight columns have been specified but calc_g_weight is False. "
            "Please check whether g_weights are required."
        )
        raise Exception(msg)

    # Validate weights specified
    _validate_weights(data, a_weight_columns, g_weight_columns, aux_cols)


def _validate_weights(
    data: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: dict[str, list[str]] | None,
    aux_cols: list[str] | None,
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
    - a_weight_columns (list[str])
    - g_weight_columns (dict[str, list[str]] | None)
    - aux_cols (list[str] | None)

    Raises
    ------
    - Exception if specified columns are not in data or if no weights are specified
    - Warning if only a_weights or only g_weights will be applied
    """
    _check_is_list("a_weight_columns", a_weight_columns)

    g_weight_columns = g_weight_columns or {}
    aux_cols = aux_cols or []

    if not isinstance(g_weight_columns, dict):
        msg = (
            "Expected g_weight_columns to be a dictionary keyed by auxiliary "
            f"column, but got {type(g_weight_columns).__name__}!"
        )
        raise TypeError(msg)

    _check_is_list("aux_cols", aux_cols)

    flattened_g_weight_columns: list[str] = []
    for aux_col, apply_cols in g_weight_columns.items():
        if not isinstance(aux_col, str):
            msg = f"Specified g weight key {aux_col} must be a string."
            raise TypeError(msg)

        _check_is_list(f"g_weight_columns[{aux_col}]", apply_cols)
        flattened_g_weight_columns.extend(apply_cols)

    missing_aux_cols = [aux_col for aux_col in g_weight_columns if aux_col not in aux_cols]
    if missing_aux_cols:
        msg = (
            "Configured g weight column(s): "
            f"{', '.join(missing_aux_cols)} were not found in aux_cols."
        )
        raise ValueError(msg)

    _check_columns(data, a_weight_columns + flattened_g_weight_columns)

    # Raise error if no weights specified
    if not (a_weight_columns or flattened_g_weight_columns):
        msg = "No a_weights or g_weights have been specified. Cannot apply Estimation."
        raise Exception(msg)

    # Warning if only a_weights or only g_weights
    elif not a_weight_columns:
        warn("No a_weights have been specified. Applying g_weights only.", stacklevel=3)
    elif not flattened_g_weight_columns:
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
