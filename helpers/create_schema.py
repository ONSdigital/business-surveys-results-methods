"""Create schema for the Data."""

import pandas as pd
import toml

from pathlib import Path

# Imputation file root and location
root = "D:/bsrm/"
input_file = "staging/anonymized_frozen_berd_data_complete.csv"

# Output folder for all schemas
out_dir = "./bsrm/schemas/input_schemas"

# Read the top 10 rows, inferrring the schema from csv
mypath = Path(root, input_file)

# check the file exists
if not Path.exists(mypath):
    e = f"File not found: {mypath}"
    raise FileNotFoundError(e)

# Read the csv file
df = pd.read_csv(mypath)


def is_nullable(series: pd.Series) -> bool:
    """Check if a pandas series is nullable."""
    return bool(series.isna().any())


schema = {}
for col in df.columns:
    schema[col] = {}
    schema[col]["data_type"] = df[col].dtype.name
    schema[col]["nullable"] = is_nullable(df[col])
    if ("int" in schema[col]["data_type"]) and schema[col]["nullable"]:
        schema[col]["data_type"] = "float"

# Output the schema toml file
csv_name = Path(mypath).stem
schema_path = Path(out_dir) / f"{csv_name}_schema.toml"

# Ensure directory exists
Path(out_dir).mkdir(parents=True, exist_ok=True)

# Write schema to file
with Path(schema_path).open("w") as f:
    toml.dump(schema, f)
