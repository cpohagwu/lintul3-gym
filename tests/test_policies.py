"""Fast, network-free tests for reference policy scheduling logic."""

from dataclasses import dataclass
from datetime import date

import pytest

from lintul3_gym.policies import ExpertPolicy, StandardPracticePolicy


@dataclass
class _FakeModel:
    """Minimal stand-in carrying only what ``StandardPracticePolicy`` reads."""

    day: date


@dataclass
class _FakeEnv:
    """Minimal stand-in for a running Lintul3Env, avoiding a real PCSE model."""

    _model: _FakeModel
    decision_interval: int = 7


def test_standard_practice_splits_dose_across_three_dates() -> None:
    """Each of the three fixed fertilization dates receives an equal share."""
    policy = StandardPracticePolicy(total_nitrogen=9.0)
    for month, day in ((2, 24), (3, 26), (4, 29)):
        env = _FakeEnv(_FakeModel(date(2007, month, day)))
        assert policy.action(env)[0] == pytest.approx(3.0)


def test_standard_practice_applies_nothing_outside_the_fertilization_weeks() -> None:
    """Weeks that contain none of the three fixed dates get a zero dose."""
    policy = StandardPracticePolicy(total_nitrogen=9.0)
    env = _FakeEnv(_FakeModel(date(2007, 6, 1)))
    assert policy.action(env)[0] == pytest.approx(0.0)


def test_standard_practice_defaults_to_the_papers_reported_total() -> None:
    """The default total matches Table 2/3's SP nitrogen figure (170.7 kg N/ha)."""
    policy = StandardPracticePolicy()
    assert policy.total_nitrogen == 17.07


def test_expert_policy_applies_each_scheduled_dose() -> None:
    """Each of the bundled .agro schedule's two dates yields its own dose."""
    policy = ExpertPolicy()
    for month, day, dose in ((4, 10, 10.0), (5, 5, 5.0)):
        env = _FakeEnv(_FakeModel(date(2006, month, day)))
        assert policy.action(env)[0] == pytest.approx(dose)


def test_expert_policy_applies_nothing_outside_the_fertilization_weeks() -> None:
    """Weeks that contain neither scheduled date get a zero dose."""
    policy = ExpertPolicy()
    env = _FakeEnv(_FakeModel(date(2006, 6, 1)))
    assert policy.action(env)[0] == pytest.approx(0.0)


def test_both_policies_raise_without_a_reset_environment() -> None:
    """The shared calendar-dose helper guards against reading a model that doesn't exist yet."""
    env = _FakeEnv(_model=None)
    with pytest.raises(RuntimeError, match="Reset the environment"):
        ExpertPolicy().action(env)
    with pytest.raises(RuntimeError, match="Reset the environment"):
        StandardPracticePolicy().action(env)
