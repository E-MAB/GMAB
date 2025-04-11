from contextlib import nullcontext

import pytest
from gmab import Gmab

SEED = 42


def rb_function(number: list):
    return sum(
        [
            100 * (number[i + 1] - number[i] ** 2) ** 2 + (1 - number[i]) ** 2
            for i in range(len(number) - 1)
        ]
    )


@pytest.mark.parametrize(
    "bounds, budget, kwargs",
    [
        [[(0, 100), (0, 100)] * 5, 100, {}],
        [[(0, 100), (0, 100)] * 5, 100, {"seed": SEED}],
        [[(0, 100), (0, 100)] * 5, 100, {"seed": float(SEED), "exp": pytest.raises(TypeError)}],
        [[(0, 1), (0, 1)], None, {"exp": pytest.raises(RuntimeError)}],
    ],
    ids=[
        "success",
        "success_with_seed",
        "fail_seed_value",
        "fail_population_size",
    ],
)
def test_gmab(bounds, budget, kwargs):
    expectation = kwargs.pop("exp", nullcontext())
    with expectation:
        gmab = Gmab(rb_function, bounds, **kwargs)
        _ = gmab.optimize(budget)
