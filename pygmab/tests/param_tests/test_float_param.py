from contextlib import nullcontext

import pytest
from gmab.params import FloatParam

test_data = [
    pytest.param(0, 1, {}, [(0, 10)], [5], 0.5, nullcontext(), id="base"),
    pytest.param(
        0, 1, {"size": 2}, [(0, 10), (0, 10)], [0, 5], [0.0, 0.5], nullcontext(), id="vector"
    ),
    pytest.param(0, 1, {"step": 0.2}, [(0, 5)], [5], 1.0, nullcontext(), id="small_step"),
    pytest.param(
        0, 0.9, {"step": 0.2}, [(0, 5)], [5], 0.9, nullcontext(), id="small_step_edge_case"
    ),
    pytest.param(0, 3, {"step": 1.5}, [(0, 2)], [2], 3.0, nullcontext(), id="large_step"),
    pytest.param(
        0, 2.9, {"step": 1.5}, [(0, 2)], [5], 2.9, nullcontext(), id="large_step_edge_case"
    ),
    pytest.param(1, 0, {}, None, None, None, pytest.raises(ValueError), id="high_value"),
    pytest.param(0, 1, {"size": 0}, None, None, None, pytest.raises(ValueError), id="size_value"),
    pytest.param(0, 1, {"step": 0}, None, None, None, pytest.raises(ValueError), id="step_value"),
]


@pytest.mark.parametrize("low, high, kwargs, exp_bounds, action, value, expectation", test_data)
def test_float_param(low, high, kwargs, exp_bounds, action, value, expectation):
    with expectation:
        param = FloatParam(low, high, **kwargs)

        bounds = param.bounds
        for tuple in bounds:
            assert all(isinstance(x, int) for x in tuple)
        assert bounds == exp_bounds

        assert param.map_to_value(action) == value
