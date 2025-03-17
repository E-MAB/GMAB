import gmab
import pytest
from pytest import LogCaptureFixture

from tests._functions import rosenbrock


@pytest.mark.parametrize(
    "seed, expected_warning",
    [
        pytest.param(42, "", id="base"),
        pytest.param(None, "Seed not provided", id="unseeded"),
    ],
)
def test_create_study(seed, expected_warning, caplog: LogCaptureFixture):
    study = gmab.create_study(seed)
    assert isinstance(study, gmab.Study)
    assert expected_warning in caplog.text


@pytest.mark.parametrize(
    "seed, skip_optimize, expected_info",
    [
        pytest.param(42, False, "completed", id="base"),
        pytest.param(
            42,
            True,
            "",
            marks=pytest.mark.xfail(raises=RuntimeError),
            id="skip_optimization",
        ),
    ],
)
def test_best_trial(seed, skip_optimize, expected_info, caplog: LogCaptureFixture):
    study = gmab.create_study(seed)

    if not skip_optimize:
        n_simulations = 10_000
        study.optimize(rosenbrock.objective, rosenbrock.BOUNDS_2D, n_simulations)
        assert expected_info in caplog.text

    result = study.best_trial
    assert result == rosenbrock.RESULT_2D
