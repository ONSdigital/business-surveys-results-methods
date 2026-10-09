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
    """
    Reshape the estimation YAML dictionary into flat config fields.

    Parameters
    ----------
    config_dict: dict
        The estimation configuration dictionary loaded from YAML.

    Raises
    ------
    ValueError
        If g-weight inputs are missing or inconsistent when g weights are enabled.

    Returns
    -------
    dict
        A flattened configuration dictionary ready for run_estimation and
        apply_weights.
    """
    g_weight_dict = config_dict.get("g_weight_columns")

    g_weight_columns = None
    aux_cols = None
    univ_aux_cols = None

    if g_weight_dict is not None:
        if not config_dict["incl_g_wts"]:
            msg = "G weights are specified in the configuration, but incl_g_wts is set to False."
            warn(msg, UserWarning, stacklevel=2)

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
    df: pd.DataFrame,
    a_wgt_band_col: str,
    ru_col: str,
    univ_count_col: str,
    g_wgt_band_col: str | None = None,
    aux_cols: list[str] | None = None,
    univ_aux_cols: list[str] | None = None,
    incl_g_wts: bool = True,
) -> None:
    """
    Validate the inputs for run_estimation.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where estimation will be applied.
    a_wgt_band_col: str
        The column representing the a weight band (strata).
    ru_col: str
        The column representing the reference unit.
    univ_count_col: str
        The column representing the universe count.
    g_wgt_band_col: str | None
        The column representing the g weight band (if applicable).
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.
    univ_aux_cols: list[str] | None
        The columns representing the universe auxiliary variables.
    incl_g_wts: bool
        Whether to include g weights in the calculation.

    Raises
    ------
    TypeError
        If the input data or parameter types are invalid.
    ValueError
        If required input columns contain missing values, or g-weight inputs are
        missing or inconsistent when g weights are enabled.
    Exception
        If one or more specified columns are not present in the dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        msg = "Specified value for data must be a Pandas DataFrame."
        raise TypeError(msg)

    if not isinstance(incl_g_wts, bool):
        msg = "Specified value for incl_g_wts must be a Boolean."
        raise TypeError(msg)

    required_cols = [a_wgt_band_col, ru_col, univ_count_col]

    if incl_g_wts:
        if g_wgt_band_col is None or aux_cols is None or univ_aux_cols is None:
            msg = "G weights cannot be calculated due to missing columns."
            raise ValueError(msg)

        _check_is_list("aux_cols", aux_cols)
        _check_is_list("univ_aux_cols", univ_aux_cols)

        if len(aux_cols) != len(univ_aux_cols):
            msg = "aux_cols and univ_aux_cols must be the same length."
            raise ValueError(msg)

        required_cols.extend([g_wgt_band_col, *aux_cols, *univ_aux_cols])

    _check_columns(df, required_cols)
    _check_missing_values(df, required_cols)


def validate_g_weighting_input(
    g_wgt_band_col: str | None,
    aux_cols: list[str] | None,
    univ_aux_cols: list[str] | None,
) -> tuple[str, list[str], list[str]]:
    """
    Validate and return the g-weight inputs for run_estimation.

    Parameters
    ----------
    g_wgt_band_col: str | None
        The column representing the g weight band (if applicable).
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.
    univ_aux_cols: list[str] | None
        The columns representing the universe auxiliary variables.

    Returns
    -------
    tuple[str, list[str], list[str]]
        The validated g weight band, auxiliary columns, and universe
        auxiliary columns.

    Raises
    ------
    ValueError
        If any required g-weight input is missing.
    """
    if g_wgt_band_col is None or aux_cols is None or univ_aux_cols is None:
        msg = "G weights cannot be calculated due to missing columns."
        raise ValueError(msg)

    return g_wgt_band_col, aux_cols, univ_aux_cols


def validate_apply_weights_input(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: dict[str, list[str]] | None,
    aux_cols: list[str] | None,
    calc_g_weight: bool,
    round_val: int,
) -> None:
    """
    Validate the inputs for apply_weights.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where weights will be applied.
    a_weight_columns: list[str]
        The columns to which a weights will be applied.
    g_weight_columns: dict[str, list[str]] | None
        The columns to which g weights will be applied, keyed by auxiliary
        variable.
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.
    calc_g_weight: bool
        Whether to include g weights in the calculation.
    round_val: int
        The number of decimal places to use when rounding weighted values.

    Raises
    ------
    TypeError
        If the input data or parameter types are invalid.
    ValueError
        If required input columns contain missing values or g-weight inputs are
        inconsistent.
    Exception
        If the g-weight configuration is inconsistent with calc_g_weight.
    """
    if not isinstance(df, pd.DataFrame):
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
    _validate_weights(df, a_weight_columns, g_weight_columns, aux_cols)


def _validate_weights(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: dict[str, list[str]] | None,
    aux_cols: list[str] | None,
) -> None:
    """
    Validate weights specified by user.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where weights will be applied.
    a_weight_columns: list[str]
        The columns to which a weights will be applied.
    g_weight_columns: dict[str, list[str]] | None
        The columns to which g weights will be applied, keyed by auxiliary
        variable.
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.

    Raises
    ------
    TypeError
        If the weight specifications are not the expected collection types.
    ValueError
        If configured g-weight columns do not align with aux_cols.
    Exception
        If specified columns are not in the data or if no weights are specified.
    Warning
        If only a weights or only g weights will be applied.
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

    _check_columns(df, a_weight_columns + flattened_g_weight_columns)
    _check_missing_values(df, a_weight_columns + flattened_g_weight_columns)

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
    param_name: str
        The name of the parameter being validated.
    param_cols: list[str]
        The parameter value expected to be provided as a list.

    Raises
    ------
    TypeError
        If the specified parameter is not a list.
    """
    if not isinstance(param_cols, list):
        msg = (
            f"Expected {param_name} to be a list (defined using []), but got "
            f"{type(param_cols).__name__}!"
        )
        raise TypeError(msg)


def _check_columns(df: pd.DataFrame, cols_list: list[str]) -> None:
    """
    Check specified column names are strings and in the data.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe being validated.
    cols_list: list[str]
        The column names expected to exist in the dataframe.

    Raises
    ------
    TypeError
        If one or more specified column names are not strings.
    Exception
        If one or more specified columns are not present in the dataframe.
    """
    not_strings = [col_name for col_name in cols_list if not isinstance(col_name, str)]
    if not_strings:
        msg = f"Specified column name(s): {', '.join(map(str, not_strings))} must be string(s)."
        raise TypeError(msg)

    cols_list = [c for c in cols_list if len(c) > 0]  # remove empty strings

    missing = [col_name for col_name in cols_list if col_name not in df.columns]
    if missing:
        msg = f"Specified column(s): {', '.join(missing)} must be column(s) in the data."
        raise Exception(msg)


def _check_missing_values(df: pd.DataFrame, cols_list: list[str]) -> None:
    """Raise an error when required input columns contain missing values."""
    missing_counts = df[cols_list].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column} ({count})" for column, count in missing_counts.items())
        msg = f"Missing values found in required input columns: {details}."
        raise ValueError(msg)
