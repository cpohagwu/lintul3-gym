"""Tests for the VecNormalize-aware SB3 training/eval wrapper stack."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pcse")
pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from lintul3_gym import Lintul3Env, WeatherConfig
from lintul3_gym.policies import ZeroNitrogenPolicy, evaluate_policy_over_weather, evaluate_sb3_policy
from lintul3_gym.sb3 import make_eval_env, make_training_env

from stable_baselines3.common.vec_env import VecNormalize


class _ConstantModel:
    """Minimal stand-in for an SB3 model: always predicts the same discrete action."""

    def __init__(self, action: int = 1) -> None:
        self._action = action

    def predict(self, observation, deterministic: bool = True):
        del observation, deterministic
        return np.array([self._action]), None


def test_make_training_env_returns_batched_vecnormalize() -> None:
    """Training-env observations carry the DummyVecEnv batch dimension VecNormalize expects."""
    train_env = make_training_env(Lintul3Env(decision_interval=7))
    assert isinstance(train_env, VecNormalize)
    observation = train_env.reset()
    assert observation.shape == (1, 15)


def test_make_eval_env_loads_frozen_stats(tmp_path: Path) -> None:
    """Evaluation envs load saved normalization statistics and freeze further updates."""
    train_env = make_training_env(Lintul3Env(decision_interval=7))
    train_env.reset()
    stats_path = tmp_path / "vec_normalize_stats.pkl"
    train_env.save(str(stats_path))

    eval_env = make_eval_env(Lintul3Env(decision_interval=7), stats_path)
    assert eval_env.training is False
    assert eval_env.norm_reward is False
    observation = eval_env.reset()
    assert observation.shape == (1, 15)


def test_evaluate_sb3_policy_runs_requested_episode_count(tmp_path: Path) -> None:
    """The eval helper produces one auditable EvaluationResult per episode."""
    train_env = make_training_env(Lintul3Env(decision_interval=7))
    train_env.reset()
    stats_path = tmp_path / "vec_normalize_stats.pkl"
    train_env.save(str(stats_path))

    eval_env = make_eval_env(Lintul3Env(decision_interval=7), stats_path)
    results = evaluate_sb3_policy(eval_env, _ConstantModel(1), n_episodes=2)

    assert len(results) == 2
    for result in results:
        assert result.history
        assert result.final_wso == result.history[-1].crop["WSO"]


def test_evaluate_policy_over_weather_covers_every_combination() -> None:
    """Round-robin evaluation visits every (location, year) pair exactly once."""
    environment = Lintul3Env(
        decision_interval=7,
        weather=WeatherConfig(locations=((1.0, 2.0), (3.0, 4.0)), years=(2001, 2002)),
    )
    results = evaluate_policy_over_weather(environment, ZeroNitrogenPolicy())
    assert len(results) == 4
    assert all(result.history for result in results)
