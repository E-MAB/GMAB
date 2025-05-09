# Copyright 2025 EvoBandits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import pytest
from evobandits import Arm, EvoBandits

from tests._functions import rosenbrock as rb


def test_arm():
    mock_av = [1, 1, 1]
    exp_dict = {"action_vector": mock_av, "mean_reward": 0.0, "num_pulls": 0}

    arm = Arm(mock_av)
    assert arm.action_vector == mock_av
    assert arm.num_pulls == 0
    assert arm.mean_reward == 0.0
    assert arm.to_dict == exp_dict


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"population_size": 10},
        {"mutation_rate": 0.1},
        {"crossover_rate": 0.9},
        {"mutation_span": 1.0},
    ],
    ids=[
        "default",
        "with_population_size",
        "with_mutation_rate",
        "with_crossover_rate",
        "with_mutation_span",
    ],
)
def test_evobandits_init(kwargs):
    expectation = kwargs.pop("exp", nullcontext())
    with expectation:
        evobandits = EvoBandits(**kwargs)
        assert isinstance(evobandits, EvoBandits)


@pytest.mark.parametrize(
    "bounds, budget, kwargs",
    [
        [[(0, 100), (0, 100)] * 5, 100, {}],
        [[(0, 100), (0, 100)] * 5, 100, {"seed": 42}],
        [[(0, 100), (0, 100)] * 5, 100, {"n_best": 2}],
        [[(0, 100), (0, 100)] * 5, 1, {"population_size": 2, "exp": pytest.raises(RuntimeError)}],
        [[(0, 100), (0, 100)] * 5, 1, {"n_best": 0, "exp": pytest.raises(RuntimeError)}],
        [[(0, 10), (0, 10)], 100, {"population_size": 0, "exp": pytest.raises(RuntimeError)}],
        [[(0, 10), (0, 10)], 100, {"mutation_rate": -0.1, "exp": pytest.raises(RuntimeError)}],
        [[(0, 10), (0, 10)], 100, {"crossover_rate": 1.1, "exp": pytest.raises(RuntimeError)}],
        [[(0, 10), (0, 10)], 100, {"mutation_span": -0.1, "exp": pytest.raises(RuntimeError)}],
        [[(0, 1), (0, 1)], 100, {"exp": pytest.raises(RuntimeError)}],
    ],
    ids=[
        "success",
        "success_with_seed",
        "success_with_n_best",
        "fail_budget_value",
        "fail_n_best_value",
        "fail_population_size_value",  # ToDo Issue #57: Err should be raised in the constructor
        "fail_mutation_rate_value",  # ToDo Issue #57: Err should be raised in the constructor
        "fail_crossover_rate_value",  # ToDo Issue #57: Err should be raised in the constructor
        "fail_mutation_span_value",  # ToDo Issue #57: Err should be raised in the constructor
        "fail_population_size_solution_size",
    ],
)
def test_evobandits(bounds, budget, kwargs):
    expectation = kwargs.pop("exp", nullcontext())
    seed = kwargs.pop("seed", None)
    n_best = kwargs.pop("n_best", 1)
    with expectation:
        evobandits = EvoBandits(**kwargs)
        result = evobandits.optimize(rb.function, bounds, budget, n_best, seed)

        assert all(isinstance(r, Arm) for r in result)
        assert len(result) == n_best


@pytest.mark.parametrize(
    "this, other, expected_eq",
    [
        [EvoBandits(), EvoBandits(), True],
        [EvoBandits(population_size=1), EvoBandits(population_size=1), True],
        [EvoBandits(), EvoBandits(population_size=1), False],
    ],
    ids=["default_eq", "modified_eq", "not_eq"],
)
def test_evobandits_eq(this, other, expected_eq):
    assert (this == other) == expected_eq
