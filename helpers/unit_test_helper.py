"""CSV to Unit Test - Code Generator.

This script processes CSV files and generates unit test code for functions that
operate on pandas DataFrames. It automates the conversion of CSV data into a format
suitable for unit tests by inferring column types and applying any specified overrides.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def validate_configuration(
    csv_directory: str,
    input_files: list[str],
    expected_file: str,
    function_name: str,
    column_type_override: dict[str, list[str]] | None = None,
) -> None:
    """Validate configuration parameters to ensure they meet expected criteria.

    Parameters
    ----------
    csv_directory (str): Directory path where CSV files are located.
    input_files (list[str]): List of input CSV filenames to process.
    expected_file (str): Expected output CSV filename.
    function_name (str): Name of the function to be tested.
    column_type_override (dict[str, list[str]] | None): Dictionary mapping column types
        to lists of columns. Defaults to None.

    Raises
    ------
    TypeError: If any attribute is not of the expected type.
    ValueError: If csv_directory is not a valid directory or contains non-CSV files.
    """
    if not isinstance(csv_directory, str):
        error_msg = "csv_directory must be a string"
        raise TypeError(error_msg)
    if not Path(csv_directory).is_dir():
        error_msg = f"Invalid CSV path: {csv_directory}"
        raise ValueError(error_msg)

    if not isinstance(input_files, list):
        error_msg = (
            "input_files must be a list of strings, "
            "even if there is only one file present it as in a list, i.e ['file.csv']"
        )
        raise TypeError(error_msg)
    all_files = input_files + [expected_file]
    if any(
        not isinstance(file, str) or not file.endswith(".csv") for file in all_files
    ):  # noqa: E501
        error_msg = "All files must be CSV files and must be strings"
        raise ValueError(error_msg)

    if not isinstance(function_name, str):
        error_msg = "function_name must be a string"
        raise TypeError(error_msg)
    if not function_name.isidentifier():
        error_msg = "function_name must be formatted as a function, i.e 'create_schema'"
        raise ValueError(error_msg)

    if column_type_override is None:
        column_type_override = {}

    example_err = (
        "i.e column_type_override={'string': ['names', 'cars'], 'float': ['wts']}"
    )
    if not isinstance(column_type_override, dict):
        error_msg = f"column_type_override must be a dictionary {example_err}"
        raise TypeError(error_msg)
    for key, value in column_type_override.items():
        if not isinstance(key, str):
            error_msg = f"Keys in column_type_override must be strings {example_err}"
            raise TypeError(error_msg)
        if not isinstance(value, list):
            error_msg = f"Values in column_type_override must be lists {example_err}"
            raise TypeError(error_msg)
        if any(not isinstance(col, str) for col in value):
            error_msg = (
                "All column names in column_type_override must be strings "
                f"{example_err}"
            )
            raise TypeError(error_msg)


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Infer the types of columns based on their values.

    This function analyzes each column in the DataFrame and determines its
    predominant type based on the values present. It categorizes columns
    as 'string', 'float', 'integer','boolean', 'date', or 'mixed'.

    Parameters
    ----------
    df (pd.DataFrame): The input DataFrame to analyze.

    Returns
    -------
    dict[str, list[str]]: Column type mappings as a dictionary with column
        names grouped by type. Mixed type columns are classified under 'mixed'.
    """
    type_dict: dict[str, list[str]] = {
        "string": [],
        "float": [],
        "integer": [],
        "boolean": [],
        "date": [],
        "mixed": [],
    }

    for col in df.columns:
        if ptypes.is_string_dtype(df[col]):
            type_dict["string"].append(col)
        elif ptypes.is_bool_dtype(df[col]):
            type_dict["boolean"].append(col)
        elif ptypes.is_integer_dtype(df[col]):
            type_dict["integer"].append(col)
        elif ptypes.is_float_dtype(df[col]):
            type_dict["float"].append(col)
        elif ptypes.is_datetime64_any_dtype(df[col]):
            type_dict["date"].append(col)
        else:
            # Default to 'string' if undetermined
            type_dict["string"].append(col)

    return type_dict


def dataframe_to_string(
    df: pd.DataFrame,
    file_name: str,
    column_type_override: dict[str, list[str]] | None = None,
) -> str:
    """Convert a DataFrame to a formatted string representation suitable for unit tests.

    This function infers column types and formats the DataFrame accordingly.
    Columns are converted to string representations with specific formatting based
    on their inferred types.

    Parameters
    ----------
    df (pd.DataFrame): The input DataFrame to convert.
    file_name (str): The name of the file the DataFrame was read from, used for logging.
    column_type_override (dict[str, list[str]] | None): Dictionary mapping column types
        to lists of columns. Defaults to None.

    Returns
    -------
    str: A string representation of the DataFrame formatted for use in unit tests.
    """
    if column_type_override is None:
        column_type_override = {}

    logging.info(f"Processing DataFrame from file: {file_name}")

    type_dict = infer_column_types(df)

    logging.debug(f"Inferred column types: {type_dict}")

    if column_type_override:
        non_existent_columns = []
        for col_type, cols in column_type_override.items():
            for col in cols:
                if col in type_dict.get(col_type, []):
                    type_dict[col_type].remove(col)
                if col in df.columns:
                    type_dict.setdefault(col_type, []).append(col)
                else:
                    non_existent_columns.append(col)
        if non_existent_columns:
            msg = (
                "The following columns to override do not exist in the DataFrame "
                f"'{file_name}': {', '.join(non_existent_columns)}"
            )
            logging.warning(msg)

    df = df.astype(str)

    # Format string columns with quotes
    for col in type_dict["string"]:
        mask = df[col] != "nan"
        df.loc[mask, col] = df.loc[mask, col].map(quote_string)

    # Format float columns with decimal points
    for col in type_dict["float"]:
        mask = df[col] != "nan"
        df.loc[mask, col] = df.loc[mask, col].map(ensure_decimal)

    df = df.replace("nan", "np.nan")

    tab = " " * 4
    col_string = "".join(f'{tab}"{col}",\n' for col in df.columns)
    rows_string = "\n".join(f"{tab}[{', '.join(row)}]," for row in df.to_numpy())
    data_string = f"columns = [\n{col_string}]\ndata = [\n{rows_string}\n]\n"

    logging.info(f"Data string generated for file: {file_name}")

    return data_string


def quote_string(value: str) -> str:
    """Wrap a value in double quotes."""
    return f'"{value}"'


def ensure_decimal(value: str) -> str:
    """Add a decimal suffix if a value looks like an integer."""
    return f"{value}.0" if "." not in value else value


def build_fixture_definition(file_name: str, data_string: str) -> tuple[str, str]:
    """Build one top-level pytest fixture function."""
    fixture_name = (
        file_name.replace(".csv", "").replace("-", "_").replace(" ", "_").lower()
    )
    fixture_defs = (
        f'\n@pytest.fixture(scope="function")\n'
        f"def {fixture_name}():\n"
        f'    """Data from {file_name}."""\n'
        f"{textwrap.indent(data_string, '    ')}"
        f"    return pd.DataFrame(columns=columns, data=data)\n"
    )
    return fixture_name, fixture_defs


def build_test_definition(function_name: str, fixture_names: list[str]) -> str:
    """Build one top-level test function."""
    fixture_args = ", ".join(fixture_names)
    input_fixtures = fixture_names[:-1]
    expected_fixture = fixture_names[-1]

    text = (
        f"\ndef test_{function_name}({fixture_args}):\n"
        f'    """General tests for {function_name}."""\n'
        f"    result = {function_name}({', '.join(input_fixtures)})\n"
        f"    assert result.equals({expected_fixture})"
    )
    return text


def generate_test_code(
    function_name: str, module_name: str, data_strings: dict[str, str]
) -> str:
    """Generate a unit test code string based on configuration and data strings.

    The function creates imports, fixture functions, and test functions necessary
    for unit testing a given function.

    Parameters
    ----------
    function_name (str): Name of the function to be tested.
    module_name (str): Module path for the function.
    data_strings (dict[str, str]): Dictionary mapping filenames to their corresponding
        data strings.

    Returns
    -------
    str: The generated unit test code as a string.
    """
    imports = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import pytest\n"
        f"from {module_name} import {function_name}\n"
    )

    fixture_defs = ""
    fixture_names = []

    for file_name, data_string in data_strings.items():
        fixture_name, fixture_def = build_fixture_definition(file_name, data_string)
        fixture_names.append(fixture_name)
        fixture_defs += fixture_def

    test_def = build_test_definition(
        function_name,
        fixture_names,
    )

    text = f"{imports}{fixture_defs}{test_def}"
    return text


def process_dataframe(
    csv_directory: str,
    input_files: list[str],
    expected_file: str,
    function_name: str,
    module_name: str,
    column_type_override: dict[str, list[str]] | None = None,
) -> None:
    """Process CSV files, generate unit test code, and save it to a Python (.py) file.

    This function reads CSV files, converts each DataFrame to a string representation
    suitable for unit tests, and generates test code. It handles file reading errors
    and logs relevant information.

    Parameters
    ----------
    csv_directory (str): Directory path where CSV files are located.
    input_files (list[str]): List of input CSV filenames to process.
    expected_file (str): Expected output CSV filename.
    function_name (str): Name of the function to be tested.
    module_name (str): Module path for the function.
    column_type_override (dict[str, list[str]] | None): Dictionary mapping column types
        to lists of columns. Defaults to None.

    Raises
    ------
    IOError: If there is an error writing the test code to the output file.
    """
    if column_type_override is None:
        column_type_override = {}

    files_to_check = input_files + [expected_file]
    missing_files = [
        file for file in files_to_check if not (Path(csv_directory) / file).is_file()
    ]

    if missing_files:
        logging.error(f"File(s) not found: {', '.join(missing_files)}")
        return

    data_strings: dict[str, str] = {}

    for file_name in files_to_check:
        try:
            df = pd.read_csv(Path(csv_directory) / file_name)
            data_strings[file_name] = dataframe_to_string(
                df, file_name, column_type_override
            )
            logging.info(f"Successfully read and processed file: {file_name}")
        except pd.errors.EmptyDataError:
            logging.error(f"File is empty: {file_name}")
        except pd.errors.ParserError:
            logging.error(f"File could not be parsed: {file_name}")
        except Exception as e:
            logging.error(f"Error reading or processing file {file_name}: {e}")
            return

    test_code = generate_test_code(function_name, module_name, data_strings)
    output_path = Path(csv_directory) / f"test_{function_name}.py"

    try:
        with Path(output_path).open("w") as text_file:
            text_file.write(test_code)
        logging.info(f"Successfully wrote output file: {output_path}")
    except IOError as e:
        logging.error(f"Error writing output file: {e}")


@dataclass
class TestConfig:
    """Configuration for unit test code generation.

    Attributes
    ----------
    csv_directory (str): Directory path where CSV files are located.
    input_files (list[str]): List of input CSV filenames to process.
    expected_file (str): Expected output CSV filename.
    function_name (str): Name of the function to be tested.
    module_name (str): Module path for the function.
    column_type_override (dict[str, list[str]]): Dictionary mapping column types
        to lists of columns for type override.
    """

    csv_directory: str
    input_files: list[str]
    expected_file: str
    function_name: str
    module_name: str
    column_type_override: dict[str, list[str]]


def main(config: TestConfig) -> None:
    """Call functions to generate unit test code based on provided configuration.

    Parameters
    ----------
    config (TestConfig): Configuration object containing all necessary parameters
        for test code generation.

    Raises
    ------
    ValueError: If required configuration variables are missing or invalid.
    """
    validate_configuration(
        config.csv_directory,
        config.input_files,
        config.expected_file,
        config.function_name,
        config.column_type_override,
    )
    process_dataframe(**asdict(config))


if __name__ == "__main__":
    config = TestConfig(
        csv_directory="Q:/IABS project/Test data/estimation_tests/",
        input_files=["estimation_component_test_input.csv"],
        expected_file="estimation_component_test_expected_output.csv",
        function_name="estimation_component",
        module_name="path.to.module",
        column_type_override={
            "string": ["reference"],
            "float": [],
        },
    )

    main(config)
