"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Weather provider construction and persistent NASA POWER caching.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from lintul3_gym.envs.types import WeatherConfig


def default_cache_dir() -> Path:
    """Return the platform-independent default cache directory.

    Returns:
        Path: ``~/.cache/lintul3_gym/nasa_power``, used to persist NASA POWER
        responses unless ``WeatherConfig.cache_dir`` overrides it.
    """
    return Path.home() / ".cache" / "lintul3_gym" / "nasa_power"


class WeatherFactory:
    """Creates PCSE weather providers without coupling environment logic to I/O."""

    def __init__(self, config: WeatherConfig, weather_file: Path) -> None:
        self._config = config
        self._weather_file = weather_file

    def create(self, location: tuple[float, float]) -> Any:
        """Build the configured provider for ``location``.

        Args:
            location: ``(latitude, longitude)`` to build a NASA POWER provider for;
                ignored for ``source="excel"``, which always reads the same fixed
                weather file regardless of location.

        Returns:
            Any: A PCSE weather data provider (``ExcelWeatherDataProvider`` for
            ``source="excel"``, otherwise a possibly-cached
            ``NASAPowerWeatherDataProvider``; see :meth:`_nasa_provider`).
        """
        if self._config.source == "excel":
            from pcse.input import ExcelWeatherDataProvider

            return ExcelWeatherDataProvider(str(self._weather_file))
        return self._nasa_provider(location)

    def _nasa_provider(self, location: tuple[float, float]) -> Any:
        """Build (or load from cache) a NASA POWER weather provider for ``location``.

        Args:
            location: ``(latitude, longitude)`` to fetch NASA POWER weather for.

        Returns:
            Any: A ``pcse.input.NASAPowerWeatherDataProvider``, either freshly
            constructed and cached to disk, or loaded from a previous run's cache
            (unless ``self._config.refresh_cache`` is set).
        """
        cache_dir = self._config.cache_dir or default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{location[0]:.4f}_{location[1]:.4f}.pickle"
        if cache_file.exists() and not self._config.refresh_cache:
            with cache_file.open("rb") as stream:
                return pickle.load(stream)

        from pcse.input import NASAPowerWeatherDataProvider

        provider = NASAPowerWeatherDataProvider(*location)
        with cache_file.open("wb") as stream:
            pickle.dump(provider, stream)
        return provider
