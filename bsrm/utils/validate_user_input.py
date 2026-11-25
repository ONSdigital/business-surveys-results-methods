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
    # create dictionaries or lists that should default to empty if not specified.
    outliering_config = config_dict.get("outliering", {})
    aux_col = config_dict.get("aux_col", None)
    univ_aux_col = config_dict.get("univ_aux_col", None)
    g_wgt_type_col = config_dict.get("g_wgt_type_col", None)

    g_weight_apply_cols = []

    if g_weight_dict is not None:
        if not config_dict["incl_g_wts"]:
            msg = "G weights are specified in the configuration, but incl_g_wts is set to False."
            warn(msg, UserWarning, stacklevel=2)

        for variable_config in g_weight_dict.values():
            g_weight_apply_cols = g_weight_apply_cols + variable_config["apply_cols"]

    config_dict = {
        "a_wgt_band_col": config_dict["a_wgt_band"],
        "g_wgt_band_col": config_dict["g_wgt_band"],
        "a_weight_columns": config_dict["a_weight_columns"]["apply_cols"],
        "g_weight_columns": g_weight_apply_cols,
        "question_col": config_dict["question_col"],
        "value_col": config_dict["value_col"],
        "incl_g_wts": config_dict["incl_g_wts"],
        "round_val": config_dict["round_val"],
        "ru_col": config_dict["ru_col"],
        "univ_count_col": config_dict["univ_count_col"],
        "aux_col": aux_col,
        "univ_aux_col": univ_aux_col,
        "outliering": outliering_config,
        "g_wgt_type_col": g_wgt_type_col,
    }
    return config_dict


def validate_run_estimation_input(
    df: pd.DataFrame,
    a_wgt_band_col: str,
    ru_col: str,
    univ_count_col: str,
    g_wgt_band_col: str | None = None,
    aux_col: str | None = None,
    univ_aux_col: str | None = None,
    incl_g_wts: bool = True,
    g_wgt_type_col: str | None = None,
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
    aux_col: str | None
        The column representing the auxiliary variable.
    univ_aux_col: str | None
        The column representing the universe auxiliary variable.
    incl_g_wts: bool
        Whether to include g weights in the calculation.
    g_wgt_type_col: str | None
        The column specifying the type of g weight for each reference unit.

    Raises
    ------
    TypeError
        If the input data or parameter types are invalid.
    ValueError
        If g-weight inputs are missing or inconsistent when g weights are enabled.
    Exception
        If one or more specified columns are not present in the dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        msg = "Specified value for data must be a Pandas DataFrame."
        raise TypeError(msg)

    if not isinstance(incl_g_wts, bool):
        msg = "Specified value for incl_g_wts must be a Boolean."
        raise TypeError(msg)

    _check_columns(df, [a_wgt_band_col, ru_col, univ_count_col])

    if not incl_g_wts:
        return

    if g_wgt_band_col is None or aux_col is None or univ_aux_col is None or g_wgt_type_col is None:
        msg = "G weights cannot be calculated due to missing columns."
        raise ValueError(msg)

    _check_columns(df, [g_wgt_band_col, aux_col, univ_aux_col, g_wgt_type_col])


def validate_apply_weights_input(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: list[str] | None,
    calc_g_weight: bool = False,
    round_val: int = 4,
) -> None:
    """
    Validate the inputs for apply_weights.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where weights will be applied.
    a_weight_columns: list[str]
        The columns to which a weights will be applied.
    g_weight_columns: list[str] | None
        The columns to which g weights will be applied.
    calc_g_weight: bool
        Whether to include g weights in the calculation.
    round_val: int
        The number of decimal places to use when rounding weighted values.

    Raises
    ------
    TypeError
        If the input data or parameter types are invalid.
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
    _validate_weights(df, a_weight_columns, g_weight_columns)


def _validate_weights(
    df: pd.DataFrame,
    a_weight_columns: list[str],
    g_weight_columns: list[str] | None,
    aux_col: str | None = None,
) -> None:
    """
    Validate weights specified by user.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where weights will be applied.
    a_weight_columns: list[str]
        The columns to which a weights will be applied.
    g_weight_columns: list[str] | None
        The columns to which g weights will be applied.
    aux_col: str | None
        The column representing the auxiliary variable.

    Raises
    ------
    TypeError
        If the weight specifications are not the expected collection types.
    ValueError
        If configured g-weight columns do not align with aux_col.
    Exception
        If specified columns are not in the data or if no weights are specified.
    Warning
        If only a weights or only g weights will be applied.
    """
    _check_is_list("a_weight_columns", a_weight_columns)

    g_weight_columns = g_weight_columns or []
    aux_col = aux_col or ""

    if not isinstance(g_weight_columns, list):
        msg = f"Expected g_weight_columns to be a list, but got {type(g_weight_columns).__name__}!"
        raise TypeError(msg)


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


def _check_columns(df: pd.DataFrame, cols_list: list[str], question_col: str = "question") -> None:
    """Check specified column names are strings and in the data.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe being validated.
    cols_list: list[str]
        The column names expected to exist in the dataframe.
    question_col: str
        The name of the question column expected to exist in the dataframe.

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
        msg = f"Specified column(s): {', '.join(missing)} must exist in the dataframe."
        raise Exception(msg)


def _check_question_columns(
    df: pd.DataFrame, cols_list: list[str], question_col: str = "question"
) -> None:
    """Check specified column names are strings and in the data.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe being validated.
    cols_list: list[str]
        The column names expected to exist in the dataframe.
    question_col: str
        The name of the question column expected to exist in the dataframe.

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

    question_names = df[question_col].unique()
    missing = [col_name for col_name in cols_list if col_name not in question_names]
    if missing:
        msg = f"Specified column(s): {', '.join(missing)} must exist in the question column."
        raise Exception(msg)
