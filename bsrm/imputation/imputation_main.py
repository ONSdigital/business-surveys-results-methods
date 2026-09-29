"""Script to set up and run the imputation process."""

import logging
import pandas as pd

from bsrm.imputation.mean_of_ratios import run_mor

ImpMainLogger = logging.getLogger(__name__)


# example usage
if __name__ == "__main__":
    input_path = "path/to/input.csv"
    backdata_path = "path/to/backdata.csv"

    df = pd.read_csv(input_path)
    backdata = pd.read_csv(backdata_path)

    imputed_vars = {"Q10": ["Q11", "Q12"], "Q20": ["Q21", "Q22"]}

    ru_col = "reference"
    imp_class_col = "imp_class"

    link_threshold = 5

    df, links_df = run_mor(
        df, backdata, ru_col, imp_class_col, imputed_vars, link_threshold
    )
