"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

The transparent Gymnasium environment backed by PCSE LINTUL3.
"""

from __future__ import annotations

import itertools
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lintul3_gym.envs.rewards import nitrogen_reward, storage_organ_growth
from lintul3_gym.envs.types import (
    EnvironmentPaths,
    EpisodeRecord,
    RewardConfig,
    SeasonConfig,
    WeatherConfig,
)
from lintul3_gym.envs.weather import WeatherFactory


CROP_FEATURES = (
    "DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO",
)
WEATHER_FEATURES = ("IRRAD", "TMIN", "TMAX", "RAIN")


def bundled_data_dir() -> Path:
    """Return the installed spring-wheat data directory.

    Returns:
        Path: The ``lintul3_gym/envs/data/springwheat`` directory shipped inside the
        installed package -- the default crop/site/soil/weather inputs used when
        ``Lintul3Env`` is constructed with no ``data_dir``.
    """
    return Path(str(files("lintul3_gym.envs").joinpath("data/springwheat")))


def _first_match(directory: Path, suffix: str) -> Path:
    """Return the lone ``*.<suffix>`` file in ``directory``, whatever its crop-specific stem.

    Args:
        directory: Directory to search (non-recursively) for a matching file.
        suffix: File extension to match, without the leading dot (e.g. ``"crop"``).

    Returns:
        Path: The first (alphabetically) matching file found. If none exists, a
        placeholder path ``directory / f"lintul3.{suffix}"`` is returned instead, so
        the caller can still report it by name in a "missing file" error message.
    """
    matches = sorted(directory.glob(f"*.{suffix}"))
    return matches[0] if matches else directory / f"lintul3.{suffix}"


def resolve_paths(data_dir: str | Path | None, weather_config: WeatherConfig | None) -> EnvironmentPaths:
    """Resolve required data inputs, raising a clear error for missing files.

    Crop, site, and soil parameter files are discovered by extension (see
    :func:`_first_match`) rather than a fixed filename, so any directory shaped like
    ``examples/envs/springwheat`` or ``examples/envs/winterwheat`` works. The Excel
    weather file (``nl1.xlsx``) is only required when ``weather_config`` calls for it
    (``source == "excel"``, the default when no ``weather_config`` is supplied at all);
    NASA POWER weather (``source == "nasa"``) never reads it.

    Args:
        data_dir: Directory containing the crop/site/soil (and, for Excel weather,
            ``nl1.xlsx``) input files. ``None`` uses :func:`bundled_data_dir`.
        weather_config: The environment's ``WeatherConfig``, or ``None`` if the caller
            hasn't supplied one yet (in which case Excel weather -- and therefore
            ``nl1.xlsx`` -- is assumed, matching ``Lintul3Env``'s own default).

    Returns:
        EnvironmentPaths: The resolved ``crop``/``site``/``soil``/``weather`` paths.

    Raises:
        FileNotFoundError: If any input required for the selected weather source is
            missing from ``data_dir``.
    """
    directory = Path(data_dir) if data_dir is not None else bundled_data_dir()
    paths = {
        "crop": _first_match(directory, "crop"),
        "site": _first_match(directory, "site"),
        "soil": _first_match(directory, "soil"),
        "weather": directory / "nl1.xlsx",
    }
    needs_weather_file = weather_config is None or weather_config.source == "excel"
    required_keys = ("crop", "site", "soil", "weather") if needs_weather_file else ("crop", "site", "soil")
    missing = [str(paths[key]) for key in required_keys if not paths[key].is_file()]
    valid_keys = [key for key in paths if paths[key].is_file()]

    if missing:
        raise FileNotFoundError("Missing LINTUL3 input files: " + ", ".join(missing))
    print(f"Using LINTUL3 input files from {directory}: {', '.join(valid_keys)}")
    return EnvironmentPaths(**paths)


def build_agromanagement(config: SeasonConfig) -> list[dict[date, dict[str, Any]]]:
    """Build a no-event PCSE crop calendar; the agent owns all N decisions.

    Args:
        config: The crop calendar (campaign/crop start and end dates and types,
            ``max_duration``) to translate into a PCSE agromanagement structure.

    Returns:
        list[dict[date, dict[str, Any]]]: A single-campaign PCSE agromanagement list,
        with ``TimedEvents``/``StateEvents`` both ``None`` -- fertilizer application is
        reserved entirely for the RL agent's own :meth:`Lintul3Env.step` actions.
    """
    return [{
        config.campaign_start: {
            "CropCalendar": {
                "crop_name": config.crop_name,
                "crop_start_date": config.crop_start,
                "crop_start_type": config.crop_start_type,
                "crop_end_date": config.crop_end,
                "crop_end_type": config.crop_end_type,
                "max_duration": config.max_duration,
            },
            "TimedEvents": None,
            "StateEvents": None,
        }
    }]


class Lintul3Env(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Optimize LINTUL3 nitrogen applications for a configurable crop and season.

    Actions are one-element float arrays containing g N/m². Observations keep
    crop, weather, and management fields separate for auditability. Use
    :class:`lintul3_gym.sb3.FlattenObservation` for an MLP-oriented SB3 view.
    Any crop with a LINTUL3 parameter set works via ``data_dir``/``season`` -- see
    ``examples/envs/springwheat`` (the default) and ``examples/envs/winterwheat`` for
    two worked examples.

    Args:
        data_dir: Directory holding the crop/site/soil (and optional ``nl1.xlsx``)
            input files; ``None`` uses the bundled spring-wheat default. See
            :func:`resolve_paths`.
        season: Crop calendar. ``None`` uses the default ``SeasonConfig()``
            (spring wheat, 2006).
        weather: Weather source/locations/years. ``None`` uses the default
            ``WeatherConfig()`` (bundled Excel weather, 2006).
        reward: Reward-function parameters. ``None`` uses the default
            ``RewardConfig()``.
        decision_interval: Days simulated per :meth:`step` call, i.e. how often the
            agent decides on a nitrogen dose. Must be at least 1.
        max_nitrogen: Upper bound of the continuous action space, in g N/m². Must be
            positive.
        render_mode: ``"human"`` prints each step's summary immediately;
            ``"ansi"``/``None`` leave rendering to an explicit :meth:`render` call.

    Raises:
        ValueError: If ``decision_interval`` is less than 1, or ``max_nitrogen`` is
            not positive.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        season: SeasonConfig | None = None,
        weather: WeatherConfig | None = None,
        reward: RewardConfig | None = None,
        decision_interval: int = 7,
        max_nitrogen: float = 20.0,
        render_mode: str | None = None,
    ) -> None:
        """Validate constructor arguments and build the Gymnasium spaces.

        See the class docstring for parameter details. Building the actual PCSE model
        is deferred to :meth:`reset` -- constructing the environment does not run any
        simulation.
        """
        if decision_interval < 1:
            raise ValueError("decision_interval must be at least one day.")
        if max_nitrogen <= 0:
            raise ValueError("max_nitrogen must be positive.")
        self.paths = resolve_paths(data_dir, weather)
        self.season = season or SeasonConfig()
        self.weather_config = weather or WeatherConfig()
        self.reward_config = reward or RewardConfig()
        self.decision_interval = decision_interval
        self.max_nitrogen = max_nitrogen
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(0.0, max_nitrogen, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "crop": gym.spaces.Box(-np.inf, np.inf, shape=(len(CROP_FEATURES),), dtype=np.float32),
            "weather": gym.spaces.Box(-np.inf, np.inf, shape=(len(WEATHER_FEATURES),), dtype=np.float32),
            "management": gym.spaces.Box(0.0, np.inf, shape=(2,), dtype=np.float32),
        })
        self._weather_factory = WeatherFactory(self.weather_config, self.paths.weather)
        self._weather_combinations: tuple[tuple[tuple[float, float], int], ...] = tuple(
            itertools.product(self.weather_config.locations, self.weather_config.years)
        )
        self._episode_count: int = 0
        self._model: Any | None = None
        self._provider: Any | None = None
        self._baseline_model: Any | None = None
        self._baseline_previous_wso = 0.0
        self._episode_season = self.season
        self._selected_location: tuple[float, float] | None = None
        self._selected_year: int | None = None
        self._previous_wso = 0.0
        self._cumulative_nitrogen = 0.0
        self._last_action = 0.0
        self.history: list[EpisodeRecord] = []

    def _sample_weather_context(self) -> tuple[tuple[float, float], int]:
        """Pick the ``(location, year)`` pair for the next episode.

        For NASA weather with ``random_weather_per_episode=True``, draws a seeded
        random location and year independently. Otherwise (including all Excel-weather
        episodes), advances round-robin through every ``(location, year)`` combination
        in ``weather_config`` in a fixed order, one combination per episode -- so
        repeated evaluation runs see every combination exactly once, in the same order.

        Returns:
            tuple[tuple[float, float], int]: The selected ``(latitude, longitude)`` and
            year.

        Raises:
            ValueError: If ``weather_config`` has no locations or no years configured.
        """
        locations = self.weather_config.locations
        years = self.weather_config.years
        if not locations or not years:
            raise ValueError("WeatherConfig requires at least one location and one year.")
        if self.weather_config.source == "nasa" and self.weather_config.random_weather_per_episode:
            return locations[int(self.np_random.integers(len(locations)))], years[int(self.np_random.integers(len(years)))]
        return self._weather_combinations[self._episode_count % len(self._weather_combinations)]

    def _create_model(self) -> None:
        self._model, self._provider = self._new_model()

    def _new_model(self) -> tuple[Any, Any]:
        """Build a fresh PCSE LINTUL3 model instance for the current episode.

        Reads the resolved crop/site/soil parameter files, builds the weather
        provider for ``self._selected_location``, and assembles the agromanagement
        calendar from ``self._episode_season``. Called once for the agent's own model
        in :meth:`reset`, and again for the zero-nitrogen baseline model when
        ``reward_config.relative_to_zero_nitrogen`` is enabled.

        Returns:
            tuple[Any, Any]: The constructed ``pcse.models.LINTUL3`` engine and the
            weather data provider it was built with.
        """
        from pcse.base import ParameterProvider
        from pcse.input import PCSEFileReader
        from pcse.models import LINTUL3

        crop = PCSEFileReader(str(self.paths.crop))
        site = PCSEFileReader(str(self.paths.site))
        soil = PCSEFileReader(str(self.paths.soil))
        parameters = ParameterProvider(cropdata=crop, sitedata=site, soildata=soil)
        provider = self._weather_factory.create(self._selected_location)
        return LINTUL3(parameterprovider=parameters, weatherdataprovider=provider, agromanagement=build_agromanagement(self._episode_season)), provider

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a new crop season and return the initial transparent state.

        Samples the episode's weather (location, year), builds a fresh LINTUL3 model
        (and, if configured, a synchronized zero-nitrogen baseline model for relative
        reward scoring), and clears the episode's ``history``.

        Args:
            seed: Seed for the episode's random number generator (weather sampling).
            options: Unused; accepted for Gymnasium API compatibility.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, Any]]: The initial observation
            (``crop``/``weather``/``management`` arrays) and its accompanying info
            dict (see :meth:`_episode_info`).
        """
        super().reset(seed=seed)
        self._selected_location, self._selected_year = self._sample_weather_context()
        self._episode_count += 1
        self._episode_season = self.season.for_year(self._selected_year) if self.weather_config.source == "nasa" else self.season
        self._cumulative_nitrogen = 0.0
        self._last_action = 0.0
        self.history = []
        self._create_model()
        output = self._latest_output()
        self._previous_wso = self._number(output.get("WSO"))
        self._baseline_model = None
        if self.reward_config.relative_to_zero_nitrogen:
            self._baseline_model, _ = self._new_model()
            baseline_output = self._latest_model_output(self._baseline_model)
            self._baseline_previous_wso = self._number(baseline_output.get("WSO"))
        observation = self._observation(output)
        return observation, self._episode_info(observation)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply nitrogen, advance LINTUL3, and expose all reward inputs.

        Args:
            action: A one-element array containing the nitrogen dose to apply, in
                g N/m², within ``[0, max_nitrogen]``.

        Returns:
            tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]: The
            standard Gymnasium 5-tuple -- observation, reward, ``terminated`` (the crop
            reached maturity/harvest), ``truncated`` (the crop calendar's
            ``max_duration``/``crop_end`` was reached first), and an info dict that
            additionally carries ``growth``, ``nitrogen_cost``, and the full
            ``EpisodeRecord`` for this step under ``"record"``.

        Raises:
            ValueError: If ``action`` isn't a valid one-element value within the
                environment's action space.
        """
        nitrogen = self._validate_action(action)
        assert self._model is not None
        self._apply_nitrogen(nitrogen)
        self._model.run(days=self.decision_interval)
        output = self._latest_output()
        current_wso = self._number(output.get("WSO"))
        growth = storage_organ_growth(self._previous_wso, current_wso)
        if self._baseline_model is not None:
            self._baseline_model.run(days=self.decision_interval)
            baseline_output = self._latest_model_output(self._baseline_model)
            baseline_wso = self._number(baseline_output.get("WSO"))
            growth -= storage_organ_growth(self._baseline_previous_wso, baseline_wso)
            self._baseline_previous_wso = baseline_wso
        reward, cost = nitrogen_reward(growth, nitrogen, self.reward_config)
        self._previous_wso = current_wso
        self._last_action = nitrogen
        self._cumulative_nitrogen += nitrogen
        observation = self._observation(output)

        # Check available model attributes at https://pcse.readthedocs.io/en/stable/_modules/pcse/engine.html#
        terminated = bool(getattr(self._model, "flag_terminate", False))
        # Learn more about the flag_crop_finish attribute at https://pcse.readthedocs.io/en/stable/_modules/pcse/engine.html#:~:text=2.%20After%20a-,CROP_FINISH,-signal%2C%20the%20engine
        truncated = bool(getattr(self._model, "flag_crop_finish", False))
        record = EpisodeRecord(
            step=len(self.history), date=output["day"], action_nitrogen=nitrogen,
            reward=reward, growth=growth, nitrogen_cost=cost,
            cumulative_nitrogen=self._cumulative_nitrogen,
            crop={name: self._number(output.get(name)) for name in CROP_FEATURES},
            weather=self._weather_values(output["day"]),
        )
        self.history.append(record)
        info = self._episode_info(observation)
        info.update({"growth": growth, "nitrogen_cost": cost, "record": record})
        if terminated:
            info["terminal_output"] = self._model.get_output()
        if self.render_mode == "human":
            print(self.render())
        return observation, float(reward), terminated, truncated, info

    def _validate_action(self, action: np.ndarray) -> float:
        """Check ``action`` against the action space and return its scalar value.

        Args:
            action: The raw action passed to :meth:`step`.

        Returns:
            float: The validated nitrogen dose, in g N/m².

        Raises:
            ValueError: If ``action`` isn't a one-element array within the action space.
        """
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (1,) or not self.action_space.contains(value):
            raise ValueError(f"Action must be within {self.action_space}: {action!r}")
        return float(value[0])

    def _apply_nitrogen(self, nitrogen: float) -> None:
        """Send the PCSE ``apply_n`` signal for the current step's nitrogen dose.

        Args:
            nitrogen: Nitrogen amount to apply, in g N/m². A fixed 0.7 recovery
                fraction is used, matching the bundled reference policies and
                PCSE-Gym's convention.
        """
        import pcse

        assert self._model is not None
        self._model._send_signal(
            signal=pcse.signals.apply_n,
            amount=nitrogen,
            recovery=0.7,
            N_amount=nitrogen * 10.0,
            N_recovery=0.7,
        )

    def _latest_output(self) -> dict[str, Any]:
        assert self._model is not None
        return self._latest_model_output(self._model)

    @staticmethod
    def _latest_model_output(model: Any) -> dict[str, Any]:
        """Return ``model``'s most recent daily output row, running one day if needed.

        A freshly-constructed PCSE engine has no output yet until it's been run at
        least once; this runs a single day on demand so callers always get a row back.

        Args:
            model: A PCSE engine instance (the agent's model or the zero-nitrogen
                baseline model).

        Returns:
            dict[str, Any]: The last entry of ``model.get_output()``.
        """
        output = model.get_output()
        if not output:
            model.run(days=1)
            output = model.get_output()
        return output[-1]

    @staticmethod
    def _number(value: Any) -> float:
        return 0.0 if value is None else float(value)

    def _weather_values(self, day: date) -> dict[str, float]:
        """Look up this episode's weather variables for a given day.

        Args:
            day: The calendar date to look up (typically the current model day).

        Returns:
            dict[str, float]: One value per name in ``WEATHER_FEATURES``, defaulting
            to 0.0 for any attribute the weather provider doesn't expose for that day.
        """
        assert self._provider is not None
        weather = self._provider(day)
        return {name: self._number(getattr(weather, name, 0.0)) for name in WEATHER_FEATURES}

    def _observation(self, output: dict[str, Any]) -> dict[str, np.ndarray]:
        """Build the ``crop``/``weather``/``management`` observation dict.

        Args:
            output: The latest PCSE model output row (see :meth:`_latest_output`).

        Returns:
            dict[str, np.ndarray]: Arrays for ``CROP_FEATURES``, ``WEATHER_FEATURES``,
            and ``(cumulative_nitrogen, last_action)``, matching ``observation_space``.
        """
        weather = self._weather_values(output["day"])
        return {
            "crop": np.asarray([self._number(output.get(name)) for name in CROP_FEATURES], dtype=np.float32),
            "weather": np.asarray([weather[name] for name in WEATHER_FEATURES], dtype=np.float32),
            "management": np.asarray([self._cumulative_nitrogen, self._last_action], dtype=np.float32),
        }

    def _episode_info(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        """Build the ``info`` dict accompanying an observation.

        Args:
            observation: The observation just built by :meth:`_observation`.

        Returns:
            dict[str, Any]: Weather source/location/year, the current episode index,
            the named field order for each observation group, and the observation
            itself (for convenient access alongside the other metadata).
        """
        return {
            "weather_source": self.weather_config.source,
            "weather_location": self._selected_location,
            "weather_year": self._selected_year,
            "episode_index": self._episode_count - 1,
            "observation_names": {"crop": CROP_FEATURES, "weather": WEATHER_FEATURES, "management": ("cumulative_nitrogen", "last_action")},
            "observation": observation,
        }

    def render(self) -> str | None:
        """Return or print a concise description of the latest decision.

        Returns:
            str | None: A one-line summary of the most recent step (date, nitrogen
            applied, growth, reward), or a placeholder message if no step has been
            taken yet. Returns ``None`` instead when ``render_mode == "human"``, since
            the summary is printed directly in that mode.
        """
        if not self.history:
            text = "LINTUL3-Gym has not received an action."
        else:
            record = self.history[-1]
            text = (f"{record.date}: N={record.action_nitrogen:.2f} g/m², "
                    f"growth={record.growth:.2f} g/m², reward={record.reward:.2f}")
        if self.render_mode == "human":
            print(text)
            return None
        return text
