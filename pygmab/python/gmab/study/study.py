from collections.abc import Callable

from gmab import logging
from gmab.gmab import Gmab
from gmab.params import IntParam

_logger = logging.get_logger(__name__)


class Study:
    """A Study corresponds to an optimization task, i.e. a set of trials.

    This objecht provides interfaces to optimize an objective within its bounds,
    and to set/get attributes of the study itself that are user-defined.

    Note that the direct use of this constructor is not recommended.
    To create a study, use :func:`~gmab.create_study`.

    """

    def __init__(self, algorithm=Gmab) -> None:
        self.algorithm = algorithm
        self._best_trial: dict | None = None

    @property
    def best_trial(self) -> dict:
        """Return the parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.

        """
        if not self._best_trial:
            raise RuntimeError("best_trial is not available yet. Run study.optimize().")
        return self._best_trial

    def _collect_bounds(self, params: dict) -> list[tuple]:
        all_bounds = []
        for key, value in params.items():
            assert isinstance(value, IntParam), f"{key} is not valid. Try gmab.suggest_int."
            all_bounds += value.bounds
        return all_bounds

    def optimize(
        self,
        func: Callable,
        params: dict,
        trials: int,
    ) -> None:
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparmeter values within
        given ``bounds``

        The optimization trial will be stopped after ``n_simulations`` of the
        :func:`func`.

        Args:
            func:
                A callable that implements the objective function.
            bounds:
                A list of of tuples that define the bounds for each decision variable.
            trials:
                The number of simulations. An optimization will continue until the
                number of elapsed simulations reaches `trials`.
        """
        bounds = self._collect_bounds(params)
        gmab = self.algorithm(func, bounds)
        self._best_trial = gmab.optimize(trials)
        _logger.info("completed")


def create_study() -> Study:
    """Create a new :class:`~gmab.study.Study`."""
    return Study()
