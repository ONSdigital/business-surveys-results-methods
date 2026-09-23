"""Main file for the estimation module.

In general pipeline use, individual functions from this module could be imported and used
separately with appropriate filtering and preprocessing steps.

This script acts as a runner for the estimation functions to be run in sequence.

NOTE: It might be necessary to use different auxiliary columns for different variables/questions
in calculating the g_weights. For this reason, the code allows for a list.

To use this script and run it directly, provide the appropriate input and output file paths
and variable names in the section under `if __name__ == "__main__":`.
"""

import logging
from pathlib import Path
import pandas as pd

from bsrm.estimation.calculate_weights import (
    calculate_a_weights,
    calculate_g_weights,
    create_weights_qa_df,
)
from bsrm.estimation.apply_weights import apply_weights
from bsrm.utils.io_mods import safeload_yaml
from bsrm.utils.validate_user_input import (
    validate_estimation_config_dict,
    validate_g_weighting_input,
    validate_run_estimation_input,
)

EstMainLogger = logging.getLogger(__name__)


def run_estimation(
    df: pd.DataFrame,
    a_wgt_band_col: str,
    g_wgt_band_col: str | None,
    ru_col: str,
    univ_count_col: str,
    aux_cols: list[str] | None = None,
    univ_aux_cols: list[str] | None = None,
    incl_g_wts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the estimation module.

    Parameters
    ----------
    df: pd.DataFrame
        The survey data where estimation will be applied.
    a_wgt_band_col: str
        The column representing the a weight band (strata).
    g_wgt_band_col: str | None
        The column representing the g weight band (if applicable).
    ru_col: str
        The column representing the reference unit.
    univ_count_col: str
        The column representing the universe count.
    aux_cols: list[str] | None
        The columns representing the auxiliary variables.
    univ_aux_cols: list[str] | None
        The columns representing the universe auxiliary variables.
    incl_g_wts: bool
        Whether to include g weights in the calculation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The first element is the main dataset after the application of estimation.
        The second element is the QA dataframe.
    """
    EstMainLogger.info("Starting estimation weights calculation...")

    validate_run_estimation_input(
        data=df,
        a_weight_col=a_wgt_band_col,
        ru_col=ru_col,
        univ_count_col=univ_count_col,
        g_weight_col=g_wgt_band_col,
        aux_cols=aux_cols,
        univ_aux_cols=univ_aux_cols,
        incl_g_wts=incl_g_wts,
    )

    # calculate the weights
    weighted_df = calculate_a_weights(df, a_wgt_band_col, ru_col, univ_count_col)

    # if required also calculate g weights
    if incl_g_wts:
        g_weight_band_col, g_weight_aux_cols, g_weight_univ_aux_cols = validate_g_weighting_input(
            g_wgt_band_col,
            aux_cols,
            univ_aux_cols,
        )

        for aux_col, univ_aux_col in zip(g_weight_aux_cols, g_weight_univ_aux_cols, strict=False):
            weighted_df = calculate_g_weights(weighted_df, g_weight_band_col, aux_col, univ_aux_col)

    # Create a QA dataframe
    qa_frame = create_weights_qa_df(
        weighted_df,
        a_wgt_band_col,
        univ_count_col,
        incl_g_wts,
        g_wgt_band_col,
        aux_cols,
        univ_aux_cols,
    )

    # drop intermediate calculation columns
    if incl_g_wts and aux_cols is not None and univ_aux_cols is not None:
        intermediate_cols: list[str] = []
        for aux_col, univ_aux_col in zip(aux_cols, univ_aux_cols, strict=False):
            intermediate_cols.extend([f"{univ_aux_col}_sum", f"{aux_col}_sum"])

        weighted_df = weighted_df.drop(columns=list(dict.fromkeys(intermediate_cols)), axis=1)
    return weighted_df, qa_frame


# example usage
if __name__ == "__main__":
    config_path = "test_estimation_config.yaml"

    root_path = "Q:/IABS project/Test data/estimation_tests/"
    input_path = root_path + "estimation_component_test_input.csv"
    qa_output_path = root_path + "estimation_component_test_qa_output.csv"
    data_with_weights_output_path = root_path + "estimation_component_test_output.csv"
    data_with_weights_applied_output_path = root_path + "estimation_component_test_final_output.csv"

    df = pd.read_csv(input_path)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).with_name(config_path.name)

    config_dict = safeload_yaml(str(config_path))
    config = validate_estimation_config_dict(config_dict)

    # call the method to return the dataframe with new weights columns, and qa dataframe
    weighted_df, qa_df = run_estimation(
        df=df,
        a_wgt_band_col=config["a_wgt_band_col"],
        g_wgt_band_col=config["g_wgt_band_col"],
        ru_col=config["ru_col"],
        univ_count_col=config["univ_count_col"],
        aux_cols=config["aux_cols"],
        univ_aux_cols=config["univ_aux_cols"],
        incl_g_wts=config["incl_g_wts"],
    )

    # call the method to return the dataframe with the new weights applied
    # to the specified columns, and qa dataframe
    final_weighted_df = apply_weights(
        weighted_df.copy(),
        aux_cols=config["aux_cols"],
        a_weight_columns=config["a_weight_columns"],
        g_weight_columns=config["g_weight_columns"],
        calc_g_weight=config["incl_g_wts"],
        round_val=config["round_val"],
    )

    # save the intermediate and final outputs
    weighted_df.to_csv(data_with_weights_output_path, index=False)
    qa_df.to_csv(qa_output_path, index=False)
    final_weighted_df.to_csv(data_with_weights_applied_output_path, index=False)
