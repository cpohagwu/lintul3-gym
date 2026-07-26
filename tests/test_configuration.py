"""Fast tests for configuration and local asset behavior."""

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from lintul3_gym.envs.environment import build_agromanagement, resolve_paths
from lintul3_gym.envs.rewards import nitrogen_reward, storage_organ_growth
from lintul3_gym.envs.types import RewardConfig, SeasonConfig


def test_generated_agromanagement_has_no_agent_competing_events() -> None:
    """The generated calendar must reserve fertilization for the agent."""
    management = build_agromanagement(SeasonConfig())
    campaign = management[0][date(2006, 1, 1)]
    assert campaign["TimedEvents"] is None
    assert campaign["StateEvents"] is None
    assert campaign["CropCalendar"]["crop_start_date"] == date(2006, 3, 31)


def test_season_year_replacement_changes_every_calendar_date() -> None:
    """NASA years map to an internally regenerated crop calendar."""
    season = SeasonConfig().for_year(2005)
    assert season.campaign_start == date(2005, 1, 1)
    assert season.crop_start == date(2005, 3, 31)
    assert season.crop_end == date(2005, 10, 20)


def test_bundled_assets_resolve() -> None:
    """All four environment inputs are distributed with the package."""
    paths = resolve_paths(None, None)
    assert all(path.is_file() for path in (paths.crop, paths.site, paths.soil, paths.weather))


def test_external_data_directory_requires_all_runtime_inputs(tmp_path: Path) -> None:
    """External directories fail clearly instead of silently using defaults."""
    with pytest.raises(FileNotFoundError, match="Missing LINTUL3 input files"):
        resolve_paths(tmp_path, None)


def test_missing_weather_file_is_tolerated_for_nasa_source(tmp_path: Path) -> None:
    """A NASA WeatherConfig doesn't need nl1.xlsx, but still requires crop/site/soil."""
    from lintul3_gym.envs.types import WeatherConfig

    for name in ("a.crop", "a.site", "a.soil"):
        (tmp_path / name).write_text("")
    paths = resolve_paths(tmp_path, WeatherConfig(source="nasa"))
    assert paths.crop.is_file() and paths.site.is_file() and paths.soil.is_file()

    (tmp_path / "a.crop").unlink()
    with pytest.raises(FileNotFoundError, match="Missing LINTUL3 input files"):
        resolve_paths(tmp_path, WeatherConfig(source="nasa"))


def test_pcse_gym_nitrogen_cost_formula() -> None:
    """Reward remains growth minus nitrogen cost in LINTUL units."""
    assert storage_organ_growth(2.0, 7.5) == 5.5
    reward, cost = nitrogen_reward(5.5, 0.2, RewardConfig(nitrogen_cost=10.0))
    assert cost == 2.0
    assert reward == 3.5
