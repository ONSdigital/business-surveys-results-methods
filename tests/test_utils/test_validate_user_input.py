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
        "a_weight_columns": {
            "apply_cols": ["question"],
        },
        "g_weight_columns": {
            "aux_var1": {
                "aux_col": "turnover",
                "univ_aux_col": "univ_turnover_sum",
                "apply_cols": ["question"],
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
        "a_weight_columns": ["question"],
        "g_weight_columns": {"turnover": ["question"]},
        "incl_g_wts": True,
        "round_val": 2,
        "ru_col": "ruref",
        "univ_count_col": "N",
        "aux_cols": ["turnover"],
        "univ_aux_cols": ["univ_turnover_sum"],
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
        "g_weight_columns": {
            "turnover": ["Q101", "Q102"],
            "employment": ["Q103"],
        },
        "incl_g_wts": True,
        "round_val": 2,
        "ru_col": "ruref",
        "univ_count_col": "uni_count",
        "aux_cols": ["turnover", "employment"],
        "univ_aux_cols": ["univ_turnover", "univ_employment"],
    }
