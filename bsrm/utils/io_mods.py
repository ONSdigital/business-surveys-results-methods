"""IO modules for loading various file types."""

import logging
import yaml
import tomli

from pathlib import Path

logger = logging.getLogger(__name__)
IOModsLogger = logging.getLogger(__name__)


def safeload_yaml(file_path: str) -> dict:
    """Load a .yaml file from a path.

    Parameters
    ----------
        file_path (str): The path to load the .yaml file from.

    Raises
    ------
    FileNotFoundError: Raised if there is no file at the given path.
    TypeError: Raised if the file does not have the .yaml extension.
    yaml.YAMLError: Raised if there is an error decoding the .yaml file.

    Returns
    -------
    dict: The loaded yaml file as as dictionary.
    """
    if not Path(file_path).exists():
        e = "Attempted to load yaml at: {file_path}. File does not exist."
        raise FileNotFoundError(e)

    if Path(file_path).suffix != ".yaml":
        e = "Expected a .yaml file. Got {ext}"
        raise TypeError(e)

    try:
        with Path(file_path).open("rb") as f:
            yaml_dict = yaml.safe_load(f)
            return yaml_dict

    except yaml.YAMLError as e:
        IOModsLogger.error(f"Failed to decode YAML file: {e}")
        raise e


def load_toml(file_path: str) -> dict:
    """
    Load toml schema file into a dictionary.

    The schema defines expceted columns, their data types and whether
    they can contain null values.

    Parameters
    ----------
        file_path: str
            The path to the .toml file from.

    Returns
    -------
        dict : A dictionary representation of the schema.
    """
    # Create bool variable for checking if file exists
    if not Path(file_path).exists():
        e = f"Attempted to load toml at: {file_path}. File does not exist."
        raise FileNotFoundError(e)

    if Path(file_path).suffix != ".toml":
        e = f"Expected a .toml file. Got {Path(file_path).suffix}"
        raise TypeError(e)

    try:
        # Open the file and load toml data schema into dictionary
        with Path(file_path).open("rb") as file:
            toml_dict = tomli.load(file)
        return toml_dict
    except tomli.TOMLDecodeError as e:
        IOModsLogger.error(f"Failed to decode TOML file: {e}")
        raise e
