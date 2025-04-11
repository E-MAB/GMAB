from contextlib import nullcontext

import pytest
from gmab.params import CategoricalParam

test_cat_param_data = [
    pytest.param(["a", "b"], [(0, 1)], nullcontext(), id="choice_str"),
    pytest.param([True, False], [(0, 1)], nullcontext(), id="choice_bool"),
    pytest.param([1, 2], [(0, 1)], nullcontext(), id="choice_int"),
    pytest.param([1.0, 2.0], [(0, 1)], nullcontext(), id="choice_float"),
    pytest.param(["a", False], [(0, 1)], nullcontext(), id="choice_mixed"),
    pytest.param(["a", None], [(0, 1)], nullcontext(), id="choice_mixed"),
    pytest.param({"a", "b"}, None, pytest.raises(ValueError), id="fail_choices_type"),
    pytest.param([["a", "b"], "c"], None, pytest.raises(ValueError), id="fail_choices_content"),
]


@pytest.mark.parametrize("choices, exp_bounds, expectation", test_cat_param_data)
def test_cat_param(choices, exp_bounds, expectation):
    with expectation:
        param = CategoricalParam(choices)

        bounds = param.bounds
        assert bounds == exp_bounds

        # Check if the exact choices are recreated using the mapping
        for idx in range(bounds[0][0], bounds[0][1] + 1):
            value = param.map_to_value([idx])
            exp_value = choices[idx]
            assert value == exp_value
            assert isinstance(value, type(exp_value))
