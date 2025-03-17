import gmab
import numpy as np
import pytest

from tests._functions import rosenbrock as rb


def get_rng(number: list) -> float:
    """This function is designed showcase how an objective function without
    seeded rng will break the reproduction of gmab's results, even when the
    algorithm itself is seeded and reproducable"""
    rng = np.random.default_rng(number[0])
    return rng.uniform(low=number[1], high=number[1] + 100_000)


# ToDo: Improve the test cases as soon as Rust-gmab is seeded.
@pytest.mark.parametrize(
    "seed, objective, bounds, expect_reproduction",
    [
        pytest.param(42, rb.objective, rb.BOUNDS_2D, True, id="deterministic"),
        pytest.param(42, get_rng, [(42, 43), (0, 100_000)], True, id="fixed_rng"),
        pytest.param(None, get_rng, [(0, 43), (0, 100_000)], False, id="unfixed_rng"),
    ],
)
def test_reproduction(seed, objective, bounds, expect_reproduction):
    def execute_study(seed, objective, bounds):
        study = gmab.create_study(seed)
        study.optimize(objective, bounds, n_simulations=100)
        return study.best_trial

    first_result = execute_study(seed, objective, bounds)
    second_result = execute_study(seed, objective, bounds)

    if expect_reproduction:
        assert first_result == second_result
    else:
        assert first_result != second_result
