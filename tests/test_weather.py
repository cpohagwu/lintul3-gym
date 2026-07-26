"""Tests for seeded NASA sampling and persistent provider caching."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")

from lintul3_gym.envs.environment import Lintul3Env
from lintul3_gym.envs.types import WeatherConfig
from lintul3_gym.envs.weather import WeatherFactory


def test_nasa_random_sampling_is_seeded() -> None:
    """Identical seeds choose the same year/location candidate pair."""
    environment = object.__new__(Lintul3Env)
    environment.weather_config = WeatherConfig(
        source="nasa",
        locations=((1.0, 2.0), (3.0, 4.0)),
        years=(2001, 2002),
        random_weather_per_episode=True,
    )
    import numpy as np

    environment.np_random = np.random.default_rng(9)
    first = environment._sample_weather_context()
    environment.np_random = np.random.default_rng(9)
    second = environment._sample_weather_context()
    assert first == second


def test_deterministic_weather_cycles_through_combinations() -> None:
    """Non-random sampling round-robins through every (location, year) pair."""
    environment = object.__new__(Lintul3Env)
    environment.weather_config = WeatherConfig(
        source="nasa",
        locations=((1.0, 2.0), (3.0, 4.0)),
        years=(2001, 2002),
        random_weather_per_episode=False,
    )
    environment._weather_combinations = tuple(
        itertools.product(environment.weather_config.locations, environment.weather_config.years)
    )

    seen = []
    for episode in range(5):
        environment._episode_count = episode
        seen.append(environment._sample_weather_context())

    assert seen == [
        ((1.0, 2.0), 2001),
        ((1.0, 2.0), 2002),
        ((3.0, 4.0), 2001),
        ((3.0, 4.0), 2002),
        ((1.0, 2.0), 2001),
    ]


class _FakeNASAProvider:
    """Picklable stand-in for ``pcse.input.NASAPowerWeatherDataProvider`` that records calls.

    Must be module-level (not nested in the test function) so the cache's
    ``pickle.dump``/``pickle.load`` round-trip can actually locate the class.
    """

    calls: list[tuple[float, float]] = []

    def __init__(self, latitude: float, longitude: float) -> None:
        type(self).calls.append((latitude, longitude))


def test_nasa_provider_is_persistently_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached provider avoids a second NASA provider construction."""
    _FakeNASAProvider.calls = []
    import sys

    monkeypatch.setitem(sys.modules, "pcse.input", SimpleNamespace(NASAPowerWeatherDataProvider=_FakeNASAProvider))
    config = WeatherConfig(source="nasa", cache_dir=tmp_path)
    factory = WeatherFactory(config, tmp_path / "unused.xlsx")
    factory.create((1.0, 2.0))
    factory.create((1.0, 2.0))
    assert _FakeNASAProvider.calls == [(1.0, 2.0)]
