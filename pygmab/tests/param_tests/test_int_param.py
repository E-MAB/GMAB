from contextlib import nullcontext

import pytest
from gmab.params import IntParam

test_data = [
    pytest.param(0, 1, {}, [(0, 1)], [1], 1, nullcontext(), id="base"),
    pytest.param(0, 1, {"size": 2}, [(0, 1), (0, 1)], [1, 1], [1, 1], nullcontext(), id="vector"),
    pytest.param(3, 6, {"step": 3}, [(3, 4)], [1], 6, nullcontext(), id="step"),
    pytest.param(0, 4, {"step": 3}, [(0, 2)], [2], 4, nullcontext(), id="step_edge_case"),
    pytest.param(0, 0, {}, None, None, None, pytest.raises(ValueError), id="high_value"),
    pytest.param(0, 1, {"size": 0}, None, None, None, pytest.raises(ValueError), id="size_value"),
    pytest.param(0, 4, {"step": 0}, None, None, None, pytest.raises(ValueError), id="step_value"),
]


@pytest.mark.parametrize("low, high, kwargs, exp_bounds, action, value, expectation", test_data)
def test_int_param(low, high, kwargs, exp_bounds, action, value, expectation):
    with expectation:
        param = IntParam(low, high, **kwargs)
        assert param.bounds == exp_bounds
        assert param.map_to_value(action) == value
