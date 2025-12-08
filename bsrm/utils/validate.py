"""
The module provides schemas based validation for Dataframes.

Schames are defined in TOML files with expceted column names, data types and
nullability rules. The validator checks whether a dataframe matches the schema
and coerces columns to the required types.

Public Functions:
    * validate_with_schema
    * schema_name_from_filename
"""

# import packages
import logging
import pandas as pd

from pathlib import Path

from bsrm.utils.io_mods import load_toml

ValidationLogger = logging.getLogger(__name__)


def schema_name_from_filename(filename: str, config: dict) -> str:
    """
    Generate the schema file path based on the given filename.

    Parameters
    ----------
        filename (str): The base name of the file (without extension).
        config (dict): Configuration dictionary containing schema root path.

    Returns
    -------
        str: The full path to the TOML schema file.
    """
    file_stem = Path(filename).stem
    schema_filename = f"{file_stem}_schema.toml"
    toml_path = Path(config["input_schema_root"], schema_filename).as_posix()

    return toml_path


def validate_with_schema(
    df: pd.DataFrame, config: dict, filepath_name: str
) -> pd.DataFrame:
    """
    Validate and coerce a DataFrame to match a schema definition.

    The function:
    1. Loads the schema from a TOML file.
    2. Ensures required columns exist.
    3. Check for nullability violations.
    4. Coerces columns to the required data types.

    Parameters
    ----------
        df (pd.DataFrame): The input Dataframe to validate
        config (dict): The configuration dictionary containing paths.
        filepath_name (str): The path to the file being validated.

    Returns
    -------
        pd.DataFrame: A Dataframe thatb has been validated and coerced
            to the schema-defined data types.
    """
    ValidationLogger.info("Starting Data Schema Validation.")

    schema_name = schema_name_from_filename(Path(filepath_name).name, config)
    schema = load_toml(schema_name)

    for col, rules in schema.items():
        # Warn if expected column is missing
        if col not in df.columns:
            ValidationLogger.warning(f"Column {col} missing from DataFrame.")
            continue

        dtype = rules.get("dtype")
        nullable = rules.get("nullable", True)

        # Check nullability rule
        if not nullable and df[col].isna().any():
            ValidationLogger.warning(f"Column {col} contains NULLs but nullable=False.")

        # Try to coerce column into the required dtype
        try:
            if dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "int":
                if nullable:
                    ValidationLogger.warning(
                        f"Column {col} is defined as int but nullable=True. "
                        "Casting to float to accommodate NULLs."
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif df[col].isna().any():
                    ValidationLogger.warning(
                        f"Column {col} contains NULLs but nullable=False. "
                        "Cannot cast to int."
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        except Exception as e:
            ValidationLogger.warning(f"Failed to cast column {col} to {dtype}: {e}")
            df[col] = df[col].astype("string")

    ValidationLogger.info("Finished Data Schema Validation")

    return df
