"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Public configuration types for :mod:`lintul3_gym`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SeasonConfig:
    """Crop-calendar settings used to generate PCSE agromanagement.

    Mirrors a PCSE ``.agro`` file's ``CropCalendar`` section, but built in Python so
    the calendar can vary without editing YAML -- see
    ``lintul3_gym.envs.environment.build_agromanagement``.

    Attributes:
        campaign_start: Start date of the PCSE campaign (before crop emergence/sowing).
        crop_start: Date the crop starts (sowing or emergence, per ``crop_start_type``).
        crop_end: Date the crop ends (per ``crop_end_type``).
        crop_name: Descriptive crop name recorded in the agromanagement structure;
            purely a label -- crop physiology is entirely determined by the
            ``.crop``/``.site``/``.soil`` parameter files, not this string.
        crop_start_type: Whether ``crop_start`` marks ``"sowing"`` or ``"emergence"``.
        crop_end_type: Whether ``crop_end`` marks ``"maturity"``, ``"harvest"``, or
            the ``"earliest"`` of the two.
        max_duration: Maximum days the crop calendar may run, as a safety cap.
    """

    campaign_start: date = date(2006, 1, 1)
    crop_start: date = date(2006, 3, 31)
    crop_end: date = date(2006, 10, 20)
    crop_name: str = "spring-wheat"
    crop_start_type: Literal["sowing", "emergence"] = "emergence"
    crop_end_type: Literal["maturity", "harvest", "earliest"] = "earliest"
    max_duration: int = 300

    def for_year(self, year: int) -> "SeasonConfig":
        """Return this calendar with every date replaced by ``year``.

        Used when cycling through multiple NASA POWER weather years: each year gets
        its own copy of the crop calendar, shifted to that year.

        Args:
            year: The calendar year to move ``campaign_start``, ``crop_start``, and
                ``crop_end`` to (day and month unchanged).

        Returns:
            SeasonConfig: A new instance with the same crop/type/duration settings,
            but all three dates in ``year``.
        """
        return SeasonConfig(
            campaign_start=self.campaign_start.replace(year=year),
            crop_start=self.crop_start.replace(year=year),
            crop_end=self.crop_end.replace(year=year),
            crop_name=self.crop_name,
            crop_start_type=self.crop_start_type,
            crop_end_type=self.crop_end_type,
            max_duration=self.max_duration,
        )


@dataclass(frozen=True)
class RewardConfig:
    """Parameters for the PCSE-Gym-compatible nitrogen reward.

    Reward is ``(WSO_t - WSO_{t-1}) - β·N_t`` by default, where growth is
    additionally measured relative to a same-episode zero-nitrogen baseline
    run (``relative_to_zero_nitrogen=True``), matching the paper's r_t
    formula: r_t = (WSO_t^π - WSO_{t-1}^π) - (WSO_t^0 - WSO_{t-1}^0) - βN_t.

    Attributes:
        nitrogen_cost: β, the cost per unit of nitrogen applied (g N/m²), subtracted
            from growth to form the reward. See
            ``lintul3_gym.envs.rewards.nitrogen_reward``.
        relative_to_zero_nitrogen: If ``True``, reward is growth relative to a
            synchronized zero-nitrogen baseline run of the same season; if ``False``,
            reward is simply this episode's own growth minus the nitrogen cost.
    """

    nitrogen_cost: float = 10.0
    relative_to_zero_nitrogen: bool = True


@dataclass(frozen=True)
class WeatherConfig:
    """Select and configure the weather provider.

    Excel mode reads ``nl1.xlsx`` from the selected data directory. NASA mode
    uses PCSE's NASA POWER provider. When ``random_weather_per_episode`` is
    enabled (NASA only), each reset independently draws a random location and
    a random year. Otherwise, each reset deterministically advances through
    every ``(location, year)`` combination in round-robin order, one
    combination per episode.

    Attributes:
        source: ``"excel"`` reads the bundled ``nl1.xlsx``; ``"nasa"`` queries PCSE's
            NASA POWER API (see ``lintul3_gym.envs.weather.WeatherFactory``).
        locations: Candidate ``(latitude, longitude)`` pairs to sample or cycle
            through. Only the first is used for ``source="excel"``.
        years: Candidate years to sample or cycle through. Only the first is used for
            ``source="excel"``.
        random_weather_per_episode: NASA-only. If ``True``, each ``reset()`` draws an
            independent seeded random ``(location, year)`` pair; if ``False``, resets
            round-robin through every combination in a fixed order.
        cache_dir: Directory to cache NASA POWER responses in. ``None`` uses
            ``lintul3_gym.envs.weather.default_cache_dir()``
            (``~/.cache/lintul3_gym/nasa_power``).
        refresh_cache: If ``True``, ignore any cached NASA POWER response and
            re-fetch it.
    """

    source: Literal["excel", "nasa"] = "excel"
    locations: tuple[tuple[float, float], ...] = ((51.97, 5.67),)
    years: tuple[int, ...] = (2006,)
    random_weather_per_episode: bool = False
    cache_dir: Path | None = None
    refresh_cache: bool = False


@dataclass(frozen=True)
class EnvironmentPaths:
    """Resolved input paths for a LINTUL3 environment.

    Attributes:
        crop: Path to the ``.crop`` parameter file.
        site: Path to the ``.site`` parameter file.
        soil: Path to the ``.soil`` parameter file.
        weather: Path to the Excel weather file (``nl1.xlsx``), whether or not it's
            actually required for the configured weather source -- see
            ``lintul3_gym.envs.environment.resolve_paths``.
    """

    crop: Path
    site: Path
    soil: Path
    weather: Path


@dataclass
class EpisodeRecord:
    """A serializable record of one agent-environment transition.

    Attributes:
        step: Zero-based index of this transition within its episode.
        date: The model's calendar date after this transition.
        action_nitrogen: Nitrogen dose applied this step, in g N/m².
        reward: The reward returned for this transition.
        growth: Storage-organ growth this step, in g/m² (see
            ``lintul3_gym.envs.rewards.storage_organ_growth``) -- relative to the
            zero-nitrogen baseline run when ``RewardConfig.relative_to_zero_nitrogen``
            is enabled.
        nitrogen_cost: The reward's nitrogen-cost component this step (``β · N_t``).
        cumulative_nitrogen: Total nitrogen applied so far this episode, in g N/m².
        crop: One value per name in ``lintul3_gym.envs.environment.CROP_FEATURES``.
        weather: One value per name in
            ``lintul3_gym.envs.environment.WEATHER_FEATURES``.
    """

    step: int
    date: date
    action_nitrogen: float
    reward: float
    growth: float
    nitrogen_cost: float
    cumulative_nitrogen: float
    crop: dict[str, float]
    weather: dict[str, float]
