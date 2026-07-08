import pytest
import re
import pandas as pd

from business_surveys_results_methods.mapping_utils.mapping_helpers import (
    mapper_null_checks,
    check_mapping_unique,
    join_with_null_check,
)


class TestMapperNullChecks:
    """Test to check that the mapper_null_checks function works as expected."""

    def create_mapper_df(self):
        """Create a sample mapper DataFrame."""
        columns = ["uni_count", "uni_employment"]
        data = [
            [23, 10],
            [14, 20],
            [13147, 30],
            [87577, 40],
            [9, None],
        ]
        return pd.DataFrame(data=data, columns=columns)

    def test_mapper_null_checks_warn_logs_warning(self, caplog):
        mapper_df = self.create_mapper_df()
        mapper_name = "test_mapper"
        warning_msg = "Mapper test_mapper contains nulls values in ['uni_employment']"

        with caplog.at_level("WARNING"):
            mapper_null_checks(mapper_df, mapper_name, validate_cols=None, warn=True)
            assert warning_msg in caplog.text


class TestJoinWithNullCheck(object):
    """Tests for join_with_null_check function."""

    def main_input_df(self):
        """Main input data for join_with_null_check tests."""
        columns = ["reference", "instance", "formtype", "cellnumber", "selectiontype"]
        data = [
            [49900001031, 0, "0006", 674, "C"],
            [49900001530, 0, "0006", 805, "P"],
            [49900001601, 0, "0001", 117, "C"],
            [49900001601, 1, "0001", 117, "C"],
            [49900003099, 0, "0006", 41, "L"],
        ]
        df = pd.DataFrame(columns=columns, data=data)
        return df

    def mapper_df(self):
        """Sample mapper for testing."""
        columns = ["cellnumber", "uni_count"]
        data = [
            [674, 23],
            [805, 14],
            [117, 13147],
            [41, 87577],
            [817, 9],
        ]
        df = pd.DataFrame(data=data, columns=columns)
        return df

    def expected_output(self):
        """Expected output for join_with_null_check tests."""
        columns = [
            "reference",
            "instance",
            "formtype",
            "cellnumber",
            "selectiontype",
            "uni_count",
        ]
        data = [
            [49900001031, 0, "0006", 674, "C", 23],
            [49900001530, 0, "0006", 805, "P", 14],
            [49900001601, 0, "0001", 117, "C", 13147],
            [49900001601, 1, "0001", 117, "C", 13147],
            [49900003099, 0, "0006", 41, "L", 87577],
        ]
        df = pd.DataFrame(columns=columns, data=data)
        return df

    def test_join_with_null_check_success(self):
        """General tests for join_with_null_check."""
        main_input_df = self.main_input_df()
        mapper_df = self.mapper_df()
        expected_output = self.expected_output()
        output = join_with_null_check(
            main_input_df, mapper_df, "test_mapper", "cellnumber"
        )
        assert output.equals(
            expected_output
        ), "join_with_null_check not behaving as expected."

    def test_join_with_null_check_failure(self):
        """Test the raises in join_with_null_check."""
        main_input_df = self.main_input_df()
        mapper_df = self.mapper_df()
        mapper_df = mapper_df.drop(0)
        error_msg = (
            "Nulls found in the join on cellnumber of test_mapper mapper."
            "Missing values: [674]"
        )

        with pytest.raises(ValueError, match=re.escape(error_msg)):
            join_with_null_check(main_input_df, mapper_df, "test_mapper", "cellnumber")


@pytest.fixture(scope="module")
def test_mapper_df():
    """Sample mapper for testing."""
    columns = ["ruref", "ultfoc"]
    data = [
        ["abc", "AB"],
        ["def", "EF"],
        ["ghi", "IJ"],
        ["jkl", "MN"],
        ["mno", "QR"],
        ["pqr", None],
    ]
    return pd.DataFrame(data=data, columns=columns)


class TestCheckMappingUnique(object):
    """Tests for check_mapping_unique."""

    @pytest.fixture(scope="function")
    def test_mapper_nonunique_df(self):
        """Sample mapper for testing."""
        columns = ["ruref", "ultfoc"]
        data = [
            ["abc", "AB"],
            ["def", "EF"],
            ["ghi", "AB"],
            ["jkl", "MN"],
            ["mno", "AB"],
            ["pqr", None],
        ]
        return pd.DataFrame(data=data, columns=columns)

    def test_check_mapping_unique_unique(self, test_mapper_nonunique_df):
        """Test check_mapping_unique for a column with unique values."""
        check_mapping_unique(test_mapper_nonunique_df, "ruref", "test_mapper")

    def test_check_mapping_unique_not_unique(self, test_mapper_nonunique_df):
        """Test check_mapping_unique for a column without unique values."""
        with pytest.raises(ValueError):
            check_mapping_unique(test_mapper_nonunique_df, "ultfoc", "test_mapper")
