"""Define helper functions that wrap regularly-used functions."""

import pandas as pd


def create_test_dataframe(data: list[tuple], **kwargs) -> pd.DataFrame:  # noqa ANN003
    """Create a Pandas dataframe from a human-readable tuple.

    With a header for unit tests.
    """
    df = pd.DataFrame.from_records(data[1:], columns=data[0], **kwargs)
    return df
