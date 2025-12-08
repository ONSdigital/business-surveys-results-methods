"""Define global settings and configuration for tests.

Note that you can use this config in any test file simply by referring to `test_config`.
"""
import pytest


@pytest.fixture
def test_config() -> dict:
    """A dummy test config for running tests."""
    config = {
        "mapping_module": {
            "gb_itl_col": "LAU121CD",
            "geo_cols": ["ITL121CD", "ITL121NM"]
        }
    }

    return config
