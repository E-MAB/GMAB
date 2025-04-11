from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from gmab.study import Study

from tests._functions import rosenbrock as rb

# ToDo: Add tests for output formats and properties


@pytest.mark.parametrize(
    "seed, kwargs",
    [
        [42, {}],
        [None, {"log": ("WARNING", "No seed provided")}],
        [42.0, {"exp": pytest.raises(TypeError)}],
    ],
    ids=[
        "base",
        "log_warning_no_seed",
        "fail_seed_type",
        # ToDo: Add input validation / Typecheck for Algorithm
    ],
)
def test_study_init(seed, kwargs, caplog):
    expectation = kwargs.pop("exp", nullcontext())
    log = kwargs.pop("log", None)
    with expectation:
        study = Study(seed, **kwargs)
        assert study.seed == seed

        if log:
            level, msg = log
            matched = any(
                record.levelname == level and msg in record.message for record in caplog.records
            )
            assert matched, f"Expected {level} log containing '{msg}'"


@pytest.mark.parametrize(
    "func, params, trials, exp_bounds, exp_result, kwargs",
    [
        [rb.function, rb.PARAMS_2D, 100, rb.BOUNDS_2D, rb.RESULTS_2D, {}],
    ],
    ids=[
        "base",
        # ToDo: Input validation: Fail if func is not callable
        # ToDo: Input validation: Fail if params is not valid
        # ToDo: Input validation: Fail if trials is not positive integer
    ],
)
def test_study_optimize(func, params, trials, exp_bounds, exp_result, kwargs):
    # Mock GMAB Algorithm
    mock = MagicMock()
    mock.optimize.return_value = exp_result

    expectation = kwargs.pop("exp", nullcontext())
    with expectation:
        study = Study(algorithm=mock, **kwargs)
        study.optimize(func, params, trials)

        mock.assert_called_once_with(study._run_trial, exp_bounds, None)  # Use of Gmab(...)
        mock.return_value.optimize.assert_called_once_with(trials)  # GMAB.Optimize() called once
