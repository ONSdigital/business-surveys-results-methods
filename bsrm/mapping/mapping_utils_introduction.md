# Introduction to the scripts in the mapping_utils module

The application of business survey methods often involves addtional mappings to the survey data.

One example is the "universe" information needed for estimation which relates the sample to the full population it was drawn from.

Mapping files are also used for application of manual outliers, or to define hierarchies in imputation classes.

The `mapping_helpers.py` script contains utility functions to validate mapping:

- `mapper_null_checks()`
	Checks for null values in specified columns of a mapper DataFrame. If any nulls are found, it raises an error or logs a warning, depending on the configuration.

- `join_with_null_check()`
	Performs a left join between a main DataFrame and a mapper DataFrame on a specified column, then checks for missing values in the join. Raises an error or logs a warning if any values in the main DataFrame do not have a match in the mapper.

- `check_mapping_unique()`
	Checks that all values in a specified column of a mapper DataFrame are unique. Raises an error if duplicates are found, ensuring that the column can be used as a unique key for mapping.
