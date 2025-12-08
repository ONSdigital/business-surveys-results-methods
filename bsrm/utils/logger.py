"""Custom logger setup with colored console output."""

import logging
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """Define logging formatter with colors for different log levels."""

    dark_grey = "\x1b[90m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    base_format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: dark_grey + base_format + reset,
        logging.INFO: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Set color formatting for logger."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d - %H:%M:%S")
        formatted = formatter.format(record)
        return formatted


def logger_creator(config: dict) -> logging.Logger:
    """Create and configure a logger based on the provided config.

    Args:
        config (dict): Configuration dictionary with logging settings.

    Returns
    -------
        logging.Logger: Configured logger instance.
    """
    logs_folder_name = config["logs_foldername"]
    Path(logs_folder_name).mkdir(parents=True, exist_ok=True)

    """Set up logging with a custom formatter for console output."""
    logger = logging.getLogger()
    log_level = config["dev_global"]["logging_level"]
    logger.setLevel(log_level)

    # Remove any existing handlers
    logger.handlers.clear()

    # File handler (with default formatter)
    file_handler = logging.FileHandler("logs/main.log", mode="w")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s:%(message)s",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (with custom color formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    console_handler.setLevel(log_level)
    console_handler.terminator = "\n"

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
