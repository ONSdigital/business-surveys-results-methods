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
    output_directory: str,
    input_files: list[str],
    expected_files: list[str],
    function_name: str,
    column_type_override: dict[str, list[str]] | None = None,
) -> None:
    """Validate configuration parameters to ensure they meet expected criteria.

    Parameters
    ----------
    csv_directory : str
        Directory path where CSV files are located.
    input_files : list[str]
        List of input CSV filenames to process.
    expected_files : list[str]
        List of expected output CSV filenames.
    function_name : str
        Name of the function to be tested.
    column_type_override : dict[str, list[str]] | None
        Dictionary mapping column types to lists of columns. Defaults to None.

    Raises
    ------
    TypeError
        If any attribute is not of the expected type.
    ValueError
        If csv_directory is not a valid directory or contains non-CSV files.
    """
    for dir_path in [csv_directory, output_directory]:
        if not isinstance(dir_path, str):
            error_msg = f"{dir_path} must be a string"
            raise TypeError(error_msg)
        if not Path(dir_path).is_dir():
            error_msg = f"Invalid directory path: {dir_path}"
            raise ValueError(error_msg)

    for file_list in [input_files, expected_files]:
        if not isinstance(file_list, list):
            error_msg = (
                f"{file_list} must be a list of strings, "
                "even if there is only one file present it as in a list, i.e ['file.csv']"  # noqa: E501
            )
            raise TypeError(error_msg)

    all_files = input_files + expected_files
    missing_files = [file for file in all_files if not (Path(csv_directory) / file).is_file()]

    if missing_files:
        error_msg = f"File(s) not found: {', '.join(missing_files)}"
        raise FileNotFoundError(error_msg)
    if any(not isinstance(file, str) or not file.endswith(".csv") for file in all_files):  # noqa: E501
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

    example_err = "i.e column_type_override={'string': ['names', 'cars'], 'float': ['wts']}"
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
            error_msg = f"All column names in column_type_override must be strings {example_err}"
            raise TypeError(error_msg)


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Infer the types of columns based on their values.

    This function analyzes each column in the DataFrame and determines its
    predominant type based on the values present. It categorizes columns
    as 'string', 'float', 'integer','boolean', 'date', or 'mixed'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    dict[str, list[str]]
        Column type mappings as a dictionary with column names grouped by type. Mixed type columns
        are classified under 'mixed'.
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
    rounding_precision: int = 4,
    padding: int = 20,
) -> str:
    """Convert a DataFrame to a formatted string representation suitable for unit tests.

    This function infers column types and formats the DataFrame accordingly.
    Columns are converted to string representations with specific formatting based
    on their inferred types.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to convert.
    file_name : str
        The name of the file the DataFrame was read from, used for logging.
    column_type_overrid : dict[str, list[str]] | None
        Dictionary mapping column types to lists of columns. Defaults to None.
    rounding_precision : int
        The number of decimal places to round float values.
    padding : int
        The number of spaces to pad each column in the output string for alignment.
        NOTE: padding is used for the commented out alternative method for right alignment

    Returns
    -------
    str
        A string representation of the DataFrame formatted for use in unit tests.
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
        df[col] = df[col].str.strip()
        df.loc[mask, col] = df.loc[mask, col].map(quote_string)

    # Format float columns with decimal points
    for col in type_dict["float"]:
        # round the float values to the specified precision
        df[col] = df[col].astype(float).round(rounding_precision).astype(str)
        mask = df[col] != "nan"
    df = df.replace("nan", "np.nan")

    first_row = [f'"{col}",' for col in df.columns.str.strip()]
    other_rows = [(row + ",") for row in df.to_numpy()]
    all_rows = [first_row, *other_rows]

    # this method aligns each column on the left
    for row in all_rows:
        row[0] = "(" + row[0]

    col_widths = [max(len(str(row[c])) for row in all_rows) for c in range(len(df.columns))]
    row_format = "".join(f"{{:<{w + 1}}}" for w in col_widths)

    data_string = "\n"
    for row in all_rows:
        data_string += "    " + row_format.format(*row).rstrip().removesuffix(",") + "),\n"
    data_string = f"df = create_dataframe([{data_string}])\nreturn df\n\n"

    # alternative method for columns that line up on the right, uaing a value for
    # padding, which will be set in the config
    # row_format = f"{{:>{padding}}}" * (1 + len(df.columns))

    # data_string = "\n"
    # for row in all_rows:
    #     row[0] = "(" + row[0]
    #     data_string += row_format.format("", *row).removesuffix(",") + ")," + "\n"
    # data_string = f"return create_dataframe([{data_string}])\n"

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
    fixture_name = file_name.replace(".csv", "").replace("-", "_").replace(" ", "_").lower()
    fixture_defs = (
        f'\n@pytest.fixture(scope="function")\n'
        f"def {fixture_name}():\n"
        f'    """Data from {file_name}."""\n'
        f"{textwrap.indent(data_string, '    ')}"
    )
    return fixture_name, fixture_defs


def build_test_definition(
    function_name: str, input_fixtures: list[str], output_fixtures: list[str]
) -> str:
    """Build one top-level test function."""
    fixture_args = ", ".join(input_fixtures + output_fixtures)

    text = (
        f"\ndef test_{function_name}({fixture_args}):\n"
        f'    """General tests for {function_name}."""\n'
        f"    result = {function_name}({', '.join(input_fixtures)})\n"
    )
    for output_fixture in output_fixtures:
        text += f"    assert result.equals({output_fixture})\n"
    return text


def generate_test_code(
    function_name: str,
    module_name: str,
    input_strings: dict[str, str],
    output_strings: dict[str, str],
) -> str:
    """Generate a unit test code string based on configuration and data strings.

    The function creates imports, fixture functions, and test functions necessary
    for unit testing a given function.

    Parameters
    ----------
    function_name : str
        Name of the function to be tested.
    module_name : str
        Module path for the function.
    input_strings : dict[str, str]
        Dictionary mapping input filenames to their corresponding data strings.
    output_strings : dict[str, str]
        Dictionary mapping expected output filenames to their corresponding data strings.

    Returns
    -------
    str
        The generated unit test code as a string.
    """
    imports = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import pytest\n"
        "from bsrm.utils.helpers import create_dataframe\n"
        f"from {module_name} import {function_name}\n"
    )

    fixture_defs = ""
    input_fixture_names = []
    output_fixture_names = []

    for file_name, data_string in input_strings.items():
        fixture_name, fixture_def = build_fixture_definition(file_name, data_string)
        input_fixture_names.append(fixture_name)
        fixture_defs += fixture_def

    for file_name, data_string in output_strings.items():
        fixture_name, fixture_def = build_fixture_definition(file_name, data_string)
        output_fixture_names.append(fixture_name)
        fixture_defs += fixture_def

    test_def = build_test_definition(
        function_name,
        input_fixture_names,
        output_fixture_names,
    )

    text = f"{imports}{fixture_defs}{test_def}"
    return text


def process_dataframe(
    csv_directory: str,
    output_directory: str,
    input_files: list[str],
    exp_output_files: list[str],
    function_name: str,
    module_name: str,
    column_type_override: dict[str, list[str]] | None = None,
    padding: int = 20,
) -> None:
    """Process CSV files, generate unit test code, and save it to a Python (.py) file.

    This function reads CSV files, converts each DataFrame to a string representation
    suitable for unit tests, and generates test code. It handles file reading errors
    and logs relevant information.

    Parameters
    ----------
    csv_directory : str
        Directory path where CSV files are located.
    output_directory : str
        Directory path where the generated test code will be saved.
    input_files : list[str]
        List of input CSV filenames to process.
    exp_output_files : list[str]
        List of expected output CSV filenames.
    function_name : str
        Name of the function to be tested.
    module_name : str
        Module path for the function.
    column_type_override : dict[str, list[str]] | None
        Dictionary mapping column types to lists of columns. Defaults to None.
    padding : int
        Padding for formatting the output string representation of the DataFrame.

    Raises
    ------
    IOError
        If there is an error writing the test code to the output file.
    FileNotFoundError
        If any of the input or expected output files are not found.
    """
    if column_type_override is None:
        column_type_override = {}

    input_strings: dict[str, str] = {}
    output_strings: dict[str, str] = {}

    for file_name in input_files:
        df = pd.read_csv(Path(csv_directory) / file_name)
        input_strings[file_name] = dataframe_to_string(
            df, file_name, column_type_override, padding=padding
        )
        logging.info(f"Successfully read and processed file: {file_name}")
    for file_name in exp_output_files:
        df = pd.read_csv(Path(csv_directory) / file_name)
        output_strings[file_name] = dataframe_to_string(
            df, file_name, column_type_override, padding=padding
        )
        logging.info(f"Successfully read and processed file: {file_name}")

    test_code = generate_test_code(function_name, module_name, input_strings, output_strings)
    output_path = Path(output_directory) / f"test_{function_name}.py"

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
    csv_directory : str)
        Directory path where CSV files are located.
    input_files : list[str]
        List of input CSV filenames to process.
    expected_file : str
        Expected output CSV filename.
    function_name : str
        Name of the function to be tested.
    module_name : str
        Module path for the function.
    column_type_override : dict[str, list[str]]
        Dictionary mapping column types to lists of columns for type override.
    padding : int
        Padding for formatting the output string representation of the DataFrame.
    """

    csv_directory: str
    output_directory: str
    input_files: list[str]
    exp_output_files: list[str]
    function_name: str
    module_name: str
    column_type_override: dict[str, list[str]]
    padding: int = 20


def main(config: TestConfig) -> None:
    """Call functions to generate unit test code based on provided configuration.

    Parameters
    ----------
    config : TestConfig
        Configuration object containing all necessary parameters for test code generation.

    Raises
    ------
    ValueError
        If required configuration variables are missing or invalid.
    """
    validate_configuration(
        config.csv_directory,
        config.output_directory,
        config.input_files,
        config.exp_output_files,
        config.function_name,
        config.column_type_override,
    )
    process_dataframe(**asdict(config))


if __name__ == "__main__":
    config = TestConfig(
        csv_directory="",
        output_directory="",
        input_files=["estimation_component_test_input.csv"],
        exp_output_files=[
            "estimation_component_test_expected_final_output.csv",
            "estimation_component_test_expected_qa_output.csv",
        ],
        function_name="newdataframeformat",
        module_name="path.to.module",
        column_type_override={
            "string": ["ruref", "cell_no", "k", "rusic_2007"],
            "float": [],
        },
        padding=10,  # Needs to be at least the length of the longest column or value, plus 4
    )

    main(config)
