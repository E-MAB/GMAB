from contextlib import nullcontext

import pytest
from gmab import Gmab

from tests._functions import rosenbrock as rb

SEED = 42


@pytest.mark.parametrize(
    "bounds, budget, kwargs",
    [
        [[(0, 100), (0, 100)] * 5, 100, {}],
        [[(0, 100), (0, 100)] * 5, 100, {"seed": SEED}],
        # [[(0, 100), (0, 100)] * 5, 100, {"population_size": 10}],
        [[(0, 100), (0, 100)] * 5, 100, {"seed": float(SEED), "exp": pytest.raises(TypeError)}],
        # [[(0, 10), (0, 10)], 100, {"population_size": str(10), "exp": pytest.raises(TypeError)}],
        # [[(0, 100), (0, 100)] * 5, 100, {"population_size": 0, "exp": pytest.raises(ValueError)}],
        [[(0, 1), (0, 1)], None, {"exp": pytest.raises(RuntimeError)}],
    ],
    ids=[
        "success",
        "success_with_seed",
        # "success_with_popsize",
        "fail_seed_type",
        # "fail_population_size_type",
        # "fail_population_size_value",
        "fail_population_size_solution_size",
    ],
)
def test_gmab(bounds, budget, kwargs):
    expectation = kwargs.pop("exp", nullcontext())
    with expectation:
        gmab = Gmab(rb.function, bounds, **kwargs)
        _ = gmab.optimize(budget)
