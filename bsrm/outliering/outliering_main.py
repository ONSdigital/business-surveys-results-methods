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
    l_values_col: str,
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
        Question numbers specified in YAML to use expansion Winsorisation.
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
    l_values_col : str
        Name of the column containing tuning parameters for Winsorisation.

    Returns
    -------
    pd.DataFrame
        Dataframe with added outlier_weight and outlier_flag columns.
    """
    df = df.drop(columns=["mu_i", "mean_yi", "ki", "yi*", "oi"], errors="ignore").copy()
    df[l_values_col] = pd.to_numeric(df[l_values_col], errors="raise")

    ratio_method = [
        question for question in df[question_no_col].unique() if question not in expansion_methods
    ]

    ratio_df = df[df[question_no_col].isin(ratio_method)].copy()
    expansion_df = df[df[question_no_col].isin(expansion_methods)].copy()

    ratio_df[aux_col] = pd.to_numeric(ratio_df[aux_col], errors="raise")

    ratio_results = []
    for _, group_df in ratio_df.groupby(question_no_col):
        result = winsorise_ratio(
            group_df,
            calibration_group_col=calibration_group_col,
            aux_col=aux_col,
            a_weight_col=a_weight_col,
            g_weight_col=g_weight_col,
            target_col=target_col,
            l_values_col=l_values_col,
        )
        # Drop expansion-only columns from ratio method
        result = result.drop(columns=["mean_yi"], errors="ignore")
        ratio_results.append(result)

    expansion_results = []
    for _, group_df in expansion_df.groupby(question_no_col):
        result = winsorise_expansion(
            group_df,
            strata_col=strata_col,
            target_col=target_col,
            a_weight_col=a_weight_col,
            l_values_col=l_values_col,
        )
        # Drop ratio-only columns from expansion method
        result = result.drop(columns=["mu_factor"], errors="ignore")
        expansion_results.append(result)

    # Concatenate all results
    ratio_concat = pd.concat(ratio_results, ignore_index=True) if ratio_results else pd.DataFrame()
    expansion_concat = (
        pd.concat(expansion_results, ignore_index=True) if expansion_results else pd.DataFrame()
    )

    post_winsorisation_df = pd.concat(
        [ratio_concat, expansion_concat], ignore_index=True, sort=False
    )

    return post_winsorisation_df


if __name__ == "__main__":
    config_path = r"bsrm\estimation\example_estimation_config.yaml"

    config_dict = safeload_yaml(config_path)
    config = config_dict["outliering"]

    root_path = Path(r"Q:\IABS project\Test data\outliering_tests")
    input_path = root_path / "outliering_component_test_input.csv"
    output_path = root_path / "outliering_component_test_output.csv"

    df = pd.read_csv(input_path)

    expansion_methods = config["expansion_methods"]

    # Call the method to return the dataframe with outlier weights
    outliered_df = detect_outlier(
        df=df,
        question_no_col="question",
        strata_col="a_wt_band",
        expansion_methods=expansion_methods,
        calibration_group_col="calibration_group_j",
        aux_col="aux_var",
        a_weight_col="a_weight",
        g_weight_col="g_weight",
        target_col="value",
        l_values_col="L_yi",
    )

    # Save the output
    outliered_df.to_csv(output_path, index=False)
