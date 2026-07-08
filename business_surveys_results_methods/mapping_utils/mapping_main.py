"""The main file for the mapping module.

Public Functions:
    * run_mapping

Private Functions:
    * None
"""

import logging

from business_surveys_results_methods.utils.io_mods import safeload_yaml
from business_surveys_results_methods.utils.logger import logger_creator

MappingMainLogger = logging.getLogger(__name__)


def run_mapping(user_config_path: str) -> None:
    """
    Perform mapping to the responses dataframes and output QA to csv.

    Parameters
    ----------
        df (pd.DataFrame): bsrm dataframe.
        config (dict): The configuration settings.

    Returns
    -------
        pd.DataFrame: Dataframe created after mapping
    """
    # load user config
    config = safeload_yaml(user_config_path)

    # set up logger
    logger = logger_creator(config)  # noqa: F841

    MappingMainLogger.info("Finished Mapping.")

    return None
