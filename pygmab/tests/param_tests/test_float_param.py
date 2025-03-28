from contextlib import nullcontext

import pytest
from gmab.params import FloatParam

test_float_param_init_data = [
    pytest.param(0, 1, {}, [(0, 100)], id="base"),
    pytest.param(0, 1, {"size": 2}, [(0, 100), (0, 100)], id="vector"),
    pytest.param(0, 1, {"nsteps": 10}, [(0, 10)], id="nsteps"),
    pytest.param(1, 2, {"log": True}, [(0, 100)], id="log"),
    pytest.param(-1, 0, {"log": True}, [(0, 100)], id="log_and_offset"),
    pytest.param(0, 0, {"exp": pytest.raises(ValueError)}, None, id="high_value"),
    pytest.param(0, 1, {"size": 0, "exp": pytest.raises(ValueError)}, None, id="size_value"),
    pytest.param(0, 1, {"nsteps": 0, "exp": pytest.raises(ValueError)}, None, id="nsteps_value"),
]


@pytest.mark.parametrize("low, high, kwargs, exp_bounds", test_float_param_init_data)
def test_float_param_init(low, high, kwargs, exp_bounds):
    expectation = kwargs.pop("exp", nullcontext())
    with expectation:
        param = FloatParam(low, high, **kwargs)

        bounds = param.bounds
        assert bounds == exp_bounds  # evaluate bounds

        smallest_value = param.map_to_value([bounds[0][0]])
        assert isinstance(smallest_value, float)
        assert smallest_value == low

        largest_value = param.map_to_value([bounds[0][1]])
        assert isinstance(largest_value, float)
        assert largest_value == high


#         # Check if the expected values can be generated from the bounds
#         values = []
#         for x in range(bounds[0][0], bounds[0][1] + 1):
#             values.append(param.map_to_value([x]))
#         assert values == exp_values
