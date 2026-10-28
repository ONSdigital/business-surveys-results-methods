"""Detect and weight outliers using one-sided Winsorisation."""

import pandas as pd
from pathlib import Path
from bsrm.outliering.winsorisation import winsorise_ratio, winsorise_expansion
from bsrm.utils.io_mods import safeload_yaml


def detect_outlier(
    df: pd.DataFrame,
    question_no_col: str,
    strata_col: str,
    expansion_methods: list[str],
    calibration_group_col: str,
    aux_col: str,
    a_weight_col: str,
    g_weight_col: str,
    target_col: str,
    expansion_l_value: float,
    ratio_l_value: float,
) -> pd.DataFrame:
    """Apply Winsorisation to identify and weight outliers.

    Winsorisation is applied separately per question number to ensure
    calibration groups are scoped within each question.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    question_no_col : str
        Name of the question number column.
    strata_col : str
        Name of the stratum column.
    expansion_methods : list[str]
        Question numbers specified in YAML to use  expansion Winsorisation.
        All other questions use ratio Winsorisation.
    calibration_group_col : str
        Name of the calibration group column.
    aux_col : str
        Name of the auxiliary variable column.
    a_weight_col : str
        Name of the design weight column.
    g_weight_col : str
        Name of the calibration factor column.
    target_col : str
        Name of the target variable column.
    expansion_l_value : float
        Tuning parameter for expansion Winsorisation.
    ratio_l_value : float
        Tuning parameter for ratio Winsorisation.

    Returns
    -------
    pd.DataFrame
        Dataframe with added outlier_weight and outlier_flag columns.
    """

    def winsorise_group(group: pd.DataFrame) -> pd.DataFrame:
        """Apply expansion or ratio method Winsorisation to a single question group."""
        question = group[question_no_col].iloc[0]
        group = group.copy()

        if question in expansion_methods:
            group["l_value"] = expansion_l_value
            winsorised = winsorise_expansion(
                group,
                strata_col,
                target_col,
                a_weight_col,
                "l_value",
            )
            winsorised["approach"] = "expansion"
        else:
            group["l_value"] = ratio_l_value
            winsorised = winsorise_ratio(
                group,
                calibration_group_col,
                aux_col,
                a_weight_col,
                g_weight_col,
                target_col,
                "l_value",
            )
            winsorised["approach"] = "ratio"
        return winsorised

    df = df.copy()

    post_win = df.groupby(question_no_col).apply(winsorise_group)
    post_win = post_win.reset_index(drop=True)

    return post_win


if __name__ == "__main__":
    config_path = r"bsrm\estimation\example_estimation_config.yaml"

    config_dict = safeload_yaml(config_path)
    config = config_dict["outliering"]

    root_path = Path(r"Q:\IABS project\Test data\outliering_tests")
    input_path = root_path / "outliering_component_test_input.csv"
    output_path = root_path / "outliering_component_test_output.csv"

    expansion_l_value = config["expansion_l_value"]
    ratio_l_value = config["ratio_l_value"]

    df = pd.read_csv(input_path)

    expansion_methods = config["expansion_methods"]

    # call the method to return the dataframe with outlier weights
    outliered_df = detect_outlier(
        df=df,
        question_no_col="question",
        strata_col="cell_no",
        expansion_methods=expansion_methods,
        calibration_group_col="calibration_group_j",
        aux_col="xi",
        a_weight_col="ai",
        g_weight_col="gi",
        target_col="response",
        expansion_l_value=expansion_l_value,
        ratio_l_value=ratio_l_value,
    )

    # save the output
    outliered_df.to_csv(output_path, index=False)
