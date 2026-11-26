import pytest

from bsrm.utils.validate_user_input import validate_estimation_config_dict


@pytest.fixture
def estimation_config_dict() -> dict:
    return {
        "ru_col": "ruref",
        "univ_count_col": "N",
        "round_val": 2,
        "incl_g_wts": True,
        "a_wgt_band": "a_wt_band",
        "g_wgt_band": "g_wt_band",
        "question_col": "question",
        "value_col": "value",
        "a_weight_columns": {
            "apply_cols": ["Q1", "Q2"],
        },
        "g_weight_columns": {
            "aux_var1": {
                "aux_col": "turnover",
                "univ_aux_col": "univ_turnover_sum",
                "apply_cols": ["Q1"],
            },
        },
    }


def test_validate_estimation_config_dict_for_redesigned_yaml(
    estimation_config_dict: dict,
) -> None:
    config = validate_estimation_config_dict(estimation_config_dict)

    assert config == {
        "a_wgt_band_col": "a_wt_band",
        "g_wgt_band_col": "g_wt_band",
        "a_weight_columns": ["Q1", "Q2"],
        "g_weight_columns": ["Q1"],
        "incl_g_wts": True,
        "round_val": 2,
        "ru_col": "ruref",
        "univ_count_col": "N",
        "aux_col": None,
        "univ_aux_col": None,
        "outliering": {},
        "g_wgt_type_col": None,
        "no_g_wgt_indicator": None,
        "question_col": "question",
        "value_col": "value",
    }

@pytest.fixture
def estimation_example_config_dict() -> dict:
    return {
        "ru_col": "ruref",
        "univ_count_col": "uni_count",
        "round_val": 2,
        "incl_g_wts": True,
        "a_wgt_band": "a_weight_band",
        "g_wgt_band": "a_weight_band",
        "question_col": "question",
        "value_col": "value",
        "aux_col": "aux_var",
        "univ_aux_col": "univ_aux_var",
        "g_wgt_type_col": "g_wgt_type",
        "no_g_wgt_indicator": "no_g_wgt",

        "a_weight_columns": {
            "apply_cols": ["Q101", "Q102", "Q103"],
        },
        "g_weight_columns": {
            "aux_var1": {
                "aux_col": "turnover",
                "univ_aux_col": "univ_turnover",
                "apply_cols": ["Q101", "Q102"],
            },
            "aux_var2": {
                "aux_col": "employment",
                "univ_aux_col": "univ_employment",
                "apply_cols": ["Q103"],
            },
        },
    }


def test_validate_estimation_config_dict_for_example_config(
    estimation_example_config_dict: dict,
) -> None:
    config = validate_estimation_config_dict(estimation_example_config_dict)

    assert config == {
        "a_wgt_band_col": "a_weight_band",
        "g_wgt_band_col": "a_weight_band",
        "a_weight_columns": ["Q101", "Q102", "Q103"],
        "g_weight_columns": ["Q101", "Q102", "Q103"],
        "incl_g_wts": True,
        "round_val": 2,
        "ru_col": "ruref",
        "univ_count_col": "uni_count",
        "aux_col": "aux_var",
        "univ_aux_col": "univ_aux_var",
        "g_wgt_type_col": "g_wgt_type",
        "no_g_wgt_indicator": "no_g_wgt",
        "outliering": {},
        "question_col": "question",
        "value_col": "value",
    }
