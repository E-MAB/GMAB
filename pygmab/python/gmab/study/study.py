from collections.abc import Callable

from gmab import gmab as rsgmab
from gmab import logging
from gmab.params import BaseParam

_logger = logging.get_logger(__name__)


class Study:
    """
    A Study represents an optimization task consisting of a set of trials.

    This class provides interfaces to optimize an objective function within specified bounds
    and to manage user-defined attributes related to the study.
    """

    def __init__(self, seed: int | None = None, algorithm=rsgmab.Gmab) -> None:
        """
        Initialize a Study instance.

        Args:
            seed: The seed for the Study. Defaults to None (use system entropy).
            algorithm: The optimization algorithm to use. Defaults to Gmab.
        """
        if seed is None:
            _logger.warning("No seed provided. Results will not be reproducible.")
        elif not isinstance(seed, int):
            raise TypeError(f"Seed must be integer: {seed}")

        self.seed: int | None = seed
        self.func: Callable | None = None
        self.params: dict[str, BaseParam] | None = None

        self._algorithm = algorithm
        self._best_trial: dict | None = None

    @property
    def best_trial(self) -> dict:
        """
        Retrieve the parameters of the best trial in the study.

        Returns:
            dict: A dictionary containing the parameters of the best trial.

        Raises:
            RuntimeError: If the best trial is not available yet.
        """
        if not self._best_trial:
            raise RuntimeError("best_trial is not available yet. Run study.optimize().")
        return self._best_trial

    def _map_to_solution(self, action_vector: list) -> dict:
        """
        Map an action vector to a dictionary that contains the solution value for each parameter.

        Args:
            action_vector (list): A list of actions to map.

        Returns:
            dict: The distinct solution for the action vector, formatted as dictionary.
        """
        result = {}
        idx = 0
        for key, param in self.params.items():
            result[key] = param.map_to_value(action_vector[idx : idx + param.size])
            idx += param.size
        return result

    def _run_trial(self, action_vector: list) -> float:
        """
        Execute a trial with the given action vector.

        Args:
            action_vector (list): A list of actions to execute.

        Returns:
            float: The result of the objective function.
        """
        solution = self._map_to_solution(action_vector)
        return self.func(**solution)

    def optimize(
        self,
        func: Callable,
        params: dict,
        trials: int,
        population_size: int = rsgmab.POPULATION_SIZE_DEFAULT,
        mutation_rate: float = rsgmab.MUTATION_RATE_DEFAULT,
        crossover_rate: float = rsgmab.CROSSOVER_RATE_DEFAULT,
        mutation_span: float = rsgmab.MUTATION_SPAN_DEFAULT,
    ) -> None:
        """
        Optimize the objective function.

        The optimization process involves selecting suitable hyperparameter values within
        specified bounds and running the objective function for a given number of trials.

        Args:
            func (Callable): The objective function to optimize.
            params (dict): A dictionary of parameters with their bounds.
            trials (int): The number of trials to run.
            population_size (int): The population size for the GMAB algorithm. Default is 20.
            mutation_rate (float): The mutation rate for the GMAB algorithm. Default is 0.25.
            crossover_rate (float): The crossover rate for the GMAB algorithm. Default is 1.0.
            mutation_span (float): The mutation span for the GMAB algorithm. Default is 0.1
        """
        self.func = func  # ToDo: Add input validation
        self.params = params  # ToDo: Add input validation

        # Retrieve the bounds for the parameters
        bounds = []
        for param in self.params.values():
            bounds.extend(param.bounds)

        gmab = self._algorithm(
            self._run_trial,
            bounds,
            seed=self.seed,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            mutation_span=mutation_span,
        )
        best_action_vector = gmab.optimize(trials)

        self._best_trial = self._map_to_solution(best_action_vector)
        _logger.info("completed")
