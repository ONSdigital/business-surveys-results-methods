"""Functions for the Mean of Ratios (MoR) methods."""

import itertools
import pandas as pd


good_statuses = ["Clear", "Clear - overridden"]
bad_statuses = ["Form sent out", "Check needed"]


def mor_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for the MoR method.

    This includes filtering the data to only include the relevant rows and columns,
    and creating the imputation class column.

    Args:
        df (pd.DataFrame): Processed full responses DataFrame.

    Returns
    -------
        pd.DataFrame: DataFrame of rows to be imputed.
    """
    # create a boolean column for whether the data is to be imputed based on status
    df["is_clear"] = df["status"].isin(good_statuses)
    df["to_impute"] = df["status"].isin(bad_statuses)
    return df


def get_imputation_lists(
    imputed_vars_dict: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Get the list of variables to be imputed and the target variables.

    Args:
        imputed_vars_dict (dict[str, list[str]]): Dictionary of variables to be imputed.

    Returns
    -------
        list[str]: List of target variables.
        list[str]: List of variables to be imputed.
    """
    target_vars = list(imputed_vars_dict.keys())
    imputed_vars = target_vars.copy()
    for var in imputed_vars_dict.values():
        imputed_vars += var
    # sort the imputed vars list to make sure the order is consistent for testing
    imputed_vars = sorted(set(imputed_vars))
    return target_vars, imputed_vars


def join_current_backdata(
    df: pd.DataFrame, backdata: pd.DataFrame, ru_col: str, imp_class_col: str
) -> pd.DataFrame:
    """Join the current data with the backdata.

    Args:
        df (pd.DataFrame): Processed full responses DataFrame.
        backdata (pd.DataFrame): One period of backdata.
        ru_col (str): Column name for the reference unit.
        imp_class_col (str): Column name for the imputation class.

    Returns
    -------
        pd.DataFrame: DataFrame with current and backdata merged.
    """
    merged_df = df.merge(
        backdata,
        how="left",
        on=[ru_col, imp_class_col],
        suffixes=("", "_prev"),
        indicator=True,
    )
    return merged_df


def carry_forwards(
    merged_df: pd.DataFrame,
    impute_vars: list[str],
    ru_col: str,
    imp_class_col: str,
) -> pd.DataFrame:
    """Carry forwards matching `backdata` values for references to be imputed.

    Based on the BERD example, we assume that the current data dataframe has just one
    row per reference, but the backdata can have multiple rows per reference. Hence
    we do a left merge of the current data with the backdata.

    Args:
        merged_df (pd.DataFrame): DataFrame with current and previous data merged.
        impute_vars (list[str]): Variables to be imputed.
        ru_col (str): Column name for the reference unit.
        imp_class_col (str): Column name for the imputation class.

    Returns
    -------
        pd.DataFrame: df with values carried forwards
    """
    # imputation condition: data is "to_impute" and we have a match in the backdata
    # including "indicator=True" in the merge adds a column "_merge" which shows whether
    # we have a match in the backdata or not
    to_impute_cond = merged_df["to_impute"] & (merged_df["_merge"] == "both")

    # Update the varibles to be imputed by the corresponding previous values
    for var in impute_vars:
        merged_df.loc[to_impute_cond, f"{var}_imputed"] = merged_df.loc[
            to_impute_cond, f"{var}_prev"
        ].fillna(0)

    merged_df.loc[to_impute_cond, "imp_marker"] = "CF"

    # Drop merge related column
    to_drop = ["_merge"]
    merged_df = merged_df.drop(to_drop, axis=1)
    return merged_df


def calculate_growth_rates(
    merged_df: pd.DataFrame, target_vars: list[str]
) -> pd.DataFrame:
    """Calculate the growth rates between previous and current data.

    Growth rates are caclucated for "matched pairs": where the reference and imp_class
    are the same in both the current and previous data. This is done for clear
    responders only.

    Args:
        merged_df (pd.DataFrame): DataFrame with current and previous data merged.
        target_vars (list[str]): target vars to impute.

    Returns
    -------
        pd.DataFrame: DataFrame of growth rates for each target variable.
    """
    # Select rows where we have a matched pair and the current period is clear
    growth_df = merged_df.loc[merged_df["is_clear"], :].copy()

    # Calculate the ratios for the relevant variables
    for target in target_vars:
        # Calculate a growth rate if both the current and previous values are non-zero
        valid_mask = (
            growth_df[f"{target}_prev"].notna()
            & growth_df[target].notna()
            & (growth_df[f"{target}_prev"] != 0)
            & (growth_df[target] != 0)
        )
        growth_df.loc[valid_mask, f"{target}_gr"] = (
            growth_df.loc[valid_mask, target]
            / growth_df.loc[valid_mask, f"{target}_prev"]
        )
    return growth_df


def calculate_links(
    growth_df: pd.DataFrame,
    target_vars: list[str],
    imp_class_col: str,
    threshold_num: int,
) -> pd.DataFrame:
    """Calculate the Means of Ratios (links) for each imp_class.

    Args:
        growth_df (pd.DataFrame): DataFrame of growth rates for each target variable
        target_vars (list[str]): list of target variables to use.
        imp_class_col (str): Column name for the imputation class.
        threshold_num (int): Minimum number of valid values required to calculate link.

    Returns
    -------
        pd.DataFrame: DataFrame with calculated links for each imp_class
    """
    # Apply trimming and calculate means for each imp class
    gr_df = growth_df.groupby(imp_class_col)
    gr_df = gr_df.apply(group_calc_link, target_vars, threshold_num)

    # Reorder columns to make QA easier
    column_order = ["imp_class", "reference"] + list(
        itertools.chain(
            *[
                (
                    var,
                    f"{var}_prev",
                    f"{var}_group_size",
                    f"{var}_gr",
                    f"{var}_gr_trim",
                    f"{var}_link",
                )
                for var in target_vars
            ]
        )
    )
    gr_df = gr_df[column_order].reset_index(drop=True)
    return gr_df


def group_calc_link(
    group: pd.DataFrame, target_vars: list[str], threshold_num: int
) -> pd.DataFrame:
    """Apply the MoR method to each group as part of the groupby operation.

    The calling function `calculate_links` groups the data by imp_class and then calls
    this function for each group using the .apply() method.

    This function calculates the
    mean growth rate for each variable, which is also called the "link".

    However, if the group is not of a valid size, the link is set to 1.

    Args:
        group (pd.DataFrame): Imputation class group
        target_vars (list[str]): list of the linked variables.
        threshold_num (int): Minimum number of valid values required to calculate link.

    Returns
    -------
        pd.core.groupby.DataFrameGroupBy: Group with calculated links.
    """
    for var in target_vars:
        # Create mask to not use 0s in mean calculation
        non_null_mask = pd.notna(group[f"{var}_gr"])

        num_valid_vars = sum(non_null_mask)

        group[f"{var}_group_size"] = num_valid_vars

        # If the group is a valid size, and there are non-null, non-zero values for this
        # 'var', then calculate the mean
        if num_valid_vars >= threshold_num:
            group[f"{var}_link"] = group.loc[
                ~group[f"{var}_gr_trim"] & non_null_mask, f"{var}_gr"
            ].mean()
        # Otherwise the link is set to 1
        else:
            group[f"{var}_link"] = 1.0
    return group


def apply_links(
    cf_df: pd.DataFrame,
    links_df: pd.DataFrame,
    imputed_vars_dict: dict[str, list[str]],
    imp_class_col: str,
) -> pd.DataFrame:
    """Apply the links to the carried forwards values.

    The links dataframe is merged with the carried forwards dataframe and the links are
    applied to the carried forwards values. If the link is null or 0, the carried
    forward value is not changed.

    Args:
        cf_df (pd.DataFrame): DataFrame of carried forwards values.
        links_df (pd.DataFrame): DataFrame containing calculated links.
        imp_class_col (str): Column name for the imputation class.
        imputed_vars_dict (dict[str, list[str]]): Dictionary of variables to be imputed.

    Returns
    -------
        pd.DataFrame: DataFrame with MoR imputed values.
    """
    target_vars, _ = get_imputation_lists(imputed_vars_dict)
    # Reduce the mor_df so we only have the variables we need and one row
    # per imputation class
    links_df = (
        links_df[[imp_class_col] + [f"{var}_link" for var in target_vars]]
        .groupby(imp_class_col)
        .first()
    )

    cf_df = cf_df.merge(links_df, on=imp_class_col, how="left", indicator=True)

    # Mask for values that are CF and also have a MoR link
    matched_mask = (cf_df["_merge"] == "both") & (cf_df["imp_marker"] == "CF")

    # Apply MoR for the target variables
    for var in target_vars:
        # Only apply MoR where the link is non null/0
        no_zero_mask = pd.notna(cf_df[f"{var}_link"]) & (cf_df[f"{var}_link"] != 0)
        mask = matched_mask & no_zero_mask
        # Apply the links to the previous values
        cf_df.loc[mask, f"{var}_imputed"] = (
            cf_df.loc[mask, f"{var}_imputed"] * cf_df.loc[mask, f"{var}_link"]
        )
        cf_df.loc[matched_mask, "imp_marker"] = "MoR"

    # Apply MoR for the other
    for target_var in imputed_vars_dict:
        for value in imputed_vars_dict[target_var]:
            # As above but using different elements to multiply
            no_zero_mask = pd.notna(cf_df[f"{target_var}_link"]) & (
                cf_df[f"{target_var}_link"] != 0
            )
            mask = matched_mask & no_zero_mask
            # Apply the links to the previous values
            cf_df.loc[mask, f"{value}_imputed"] = (
                cf_df.loc[mask, f"{value}_imputed"]
                * cf_df.loc[mask, f"{target_var}_link"]
            )
            cf_df.loc[matched_mask, "imp_marker"] = "MoR"

    # Drop _merge column
    cf_df = cf_df.drop("_merge", axis=1)
    return cf_df


def run_mor(
    df: pd.DataFrame,
    backdata: pd.DataFrame,
    ru_col: str,
    imp_class_col: str,
    imputed_vars_dict: dict[str, list[str]],
    link_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Implement Mean of Ratios method.

    This is implemented by first carrying forward data from last year
    for non-responders, and then calculating and applying growth rates
    for each imputation class.

    Args:
        df (pd.DataFrame): Processed full responses DataFrame
        backdata (pd.DataFrame): One period of backdata.
        impute_vars_dict (dict[str, list[str]]): Dictionary of variables to be imputed.
        link_threshold (int): Threshold for link calculation.

    Returns
    -------
        pd.DataFrame: df with MoR applied.
        pd.DataFrame: QA DataFrame showing how imputation links are calculated.
    """
    df = mor_preprocessing(df)

    # Carry forwards method
    target_vars, imputed_vars = get_imputation_lists(imputed_vars_dict)

    merged_df = join_current_backdata(df, backdata, ru_col, imp_class_col)

    carried_forwards_df = carry_forwards(merged_df, imputed_vars, ru_col, imp_class_col)

    # get list of columns to be imputed
    gr_df = calculate_growth_rates(carried_forwards_df, target_vars)
    links_df = calculate_links(gr_df, target_vars, imp_class_col, link_threshold)

    imputed_df = apply_links(
        carried_forwards_df, links_df, imputed_vars_dict, imp_class_col
    )

    return imputed_df, links_df
