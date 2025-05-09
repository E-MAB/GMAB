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

from collections.abc import Callable, Mapping
from typing import TypeAlias

import pandas as pd

from evobandits import logging
from evobandits.evobandits import (
    EvoBandits,
)
from evobandits.params import BaseParam

_logger = logging.get_logger(__name__)


ParamsType: TypeAlias = Mapping[str, BaseParam]


ALGORITHM_DEFAULT = EvoBandits()


class Study:
    """
    A Study represents an optimization task consisting of a set of trials.

    This class provides interfaces to optimize an objective function within specified bounds
    and to manage user-defined attributes related to the study.
    """

    def __init__(self, seed: int | None = None, algorithm=ALGORITHM_DEFAULT) -> None:
        """
        Initialize a Study instance.

        Args:
            seed: The seed for the Study. Defaults to None (use system entropy).
            algorithm: The optimization algorithm to use. Defaults to EvoBandits.
        """
        if seed is None:
            _logger.warning("No seed provided. Results will not be reproducible.")
        elif not isinstance(seed, int):
            raise TypeError(f"Seed must be integer: {seed}")

        self.seed: int | None = seed
        self.algorithm = algorithm  # ToDo Issue #23: type and input validation
        self.objective: Callable | None = None  # ToDo Issue #23: type and input validation
        self.params: ParamsType | None = None  # ToDo Issue #23: Input validation

        # 1 for minimization, -1 for maximization to avoid repeated branching during optimization.
        self._direction: int = 1
        self.results: list[dict] = []

    def _collect_bounds(self) -> list[tuple[int, int]]:
        """
        Collects the bounds of all parameters in the study.

        Returns:
            list[tuple[int, int]]: A list of tuples representing the bounds for each parameter.
        """
        bounds = []
        for param in self.params.values():
            bounds.extend(param.bounds)
        return bounds

    def _decode(self, action_vector: list) -> dict:
        """
        Decodes an action vector to a dictionary that contains the solution for each parameter.

        Args:
            action_vector (list): A list of actions to map.

        Returns:
            dict: The distinct solution for the action vector, formatted as dictionary.
        """
        result = {}
        idx = 0
        for key, param in self.params.items():
            result[key] = param.decode(action_vector[idx : idx + param.size])
            idx += param.size
        return result

    def _evaluate(self, action_vector: list) -> float:
        """
        Execute a trial with the given action vector.

        Args:
            action_vector (list): A list of actions to execute.

        Returns:
            float: The result of the objective function.
        """
        solution = self._decode(action_vector)
        evaluation = self._direction * self.objective(**solution)
        return evaluation

    def optimize(
        self,
        objective: Callable,
        params: ParamsType,
        trials: int,
        maximize: bool = False,
        n_best: int = 1,
    ) -> None:
        """
        Optimize the objective function, saving results to `study.results`.

        The optimization process involves selecting suitable hyperparameter values within
        specified bounds and running the objective function for a given number of trials.

        Args:
            objective (Callable): The objective function to optimize.
            params (dict): A dictionary of parameters with their bounds.
            trials (int): The number of trials to run.
            maximize (bool): Indicates if objective is maximized. Default is False.
            n_best (int): The number of results to return per run. Default is 1.

        """
        if not isinstance(maximize, bool):
            raise TypeError(f"maximize must be a bool, got {type(maximize)}.")
        self._direction = -1 if maximize else 1

        self.objective = objective
        self.params = params

        bounds = self._collect_bounds()
        best_arms = self.algorithm.optimize(self._evaluate, bounds, trials, n_best, self.seed)

        for i, arm in enumerate(best_arms, start=1):
            result = arm.to_dict
            action_vector = result.pop("action_vector")
            result["params"] = self._decode(action_vector)
            result["n_best"] = i
            self.results.append(result)

    def results_df(self) -> pd.DataFrame:
        if not self.results:
            raise AttributeError("Cannot access results. Run study.optimize() to generate.")

        processed_results = []

        for result in self.results:
            row = result.copy()
            params = row.pop("params")

            for key, value in params.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        row[f"params_{key}_{i}"] = v
                else:
                    row[f"params_{key}"] = value

            processed_results.append(row)

        return pd.DataFrame(processed_results)
