"""Module to run the pipeline."""

from importlib import reload
from bsrm import pipeline as src

reload(src)

user_config_path = "configs/network_user_config.yaml"

src.run_pipeline(user_config_path)
