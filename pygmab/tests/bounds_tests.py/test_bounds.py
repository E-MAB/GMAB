import gmab
import pytest


@pytest.mark.parametrize(
    "low, high, kwargs, exp_internal",
    [
        pytest.param(0, 1, {}, [(0, 1)]),
        pytest.param(0, 5, {"size": 1}, [(0, 5)]),
        pytest.param(0, 5, {"size": 2}, [(0, 5), (0, 5)]),
        pytest.param(0, 100, {"step": 10}, [(0, 10)]),
        pytest.param(0.0, 1, {}, [(0, 1)], marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 1.0, {}, [(0, 1)], marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 1, {"size": 0.0}, [(0, 1)], marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 1, {"step": 0.0}, [(0, 1)], marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(0, 0, {}, [(0, 0)], marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(0, 1, {"size": 0}, [], marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(0, 1, {"step": 0}, [], marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(0, 1, {"step": 2}, [], marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_suggest_int(low, high, kwargs, exp_internal):
    bounds = gmab.Bounds()
    bounds.suggest_int(low=low, high=high, **kwargs)
    assert bounds.internal == exp_internal


@pytest.mark.parametrize(
    "low, high, kwargs, internal, external",
    [
        pytest.param(100, 200, {}, [100], [200]),
        pytest.param(0, 5, {"step": 5}, [1], [5]),
        pytest.param(-10, 10, {"size": 3}, [0, 1, 20], [-10, -9, 10]),
    ],
)
def test_map_to_external_repr(low, high, kwargs, internal, external):
    bounds = gmab.Bounds()
    bounds.suggest_int(low=low, high=high, **kwargs)
    assert bounds.map_to_external_repr(internal) == external
