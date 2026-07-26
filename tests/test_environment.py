"""Integration tests for Gymnasium and bundled Excel weather."""

import numpy as np
import pytest

pytest.importorskip("pcse")
gym = pytest.importorskip("gymnasium")

from lintul3_gym import Lintul3Env
from lintul3_gym.policies import ExpertPolicy, ZeroNitrogenPolicy, evaluate_policy
from lintul3_gym.sb3 import DiscretizeNitrogen, make_sb3_env


def test_environment_reset_step_and_transparency() -> None:
    """The bundled scenario follows the Gymnasium API and returns audit data."""
    environment = Lintul3Env(decision_interval=7)
    observation, info = environment.reset(seed=2)
    assert environment.observation_space.contains(observation)
    assert info["weather_source"] == "excel"
    observation, reward, _, _, info = environment.step(np.asarray([0.0], dtype=np.float32))
    assert environment.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert info["nitrogen_cost"] == 0.0
    assert info["record"].crop.keys() == set(environment._episode_info(observation)["observation_names"]["crop"])


def test_sb3_wrapper_has_flat_finite_feature_order() -> None:
    """The optional SB3 adapter keeps a stable MLP observation shape."""
    environment = make_sb3_env(Lintul3Env(decision_interval=7))
    observation, _ = environment.reset(seed=2)
    assert observation.shape == environment.observation_space.shape


def test_discretize_nitrogen_maps_indices_to_kg_per_ha_doses() -> None:
    """The discrete SB3 adapter offers a {0, 20, 40} kg N/ha menu in g N/m² internally."""
    environment = DiscretizeNitrogen(Lintul3Env(decision_interval=7))
    assert environment.action_space == gym.spaces.Discrete(3)
    environment.reset(seed=2)
    for index, expected_kg_per_ha in enumerate((0.0, 20.0, 40.0)):
        _, _, _, _, info = environment.step(index)
        assert info["record"].action_nitrogen == pytest.approx(expected_kg_per_ha / 10.0)


def test_reference_policies_produce_auditable_episodes() -> None:
    """Both requested comparison baselines can run the bundled season."""
    environment = Lintul3Env(decision_interval=7)
    expert = evaluate_policy(environment, ExpertPolicy(), seed=2)
    zero = evaluate_policy(environment, ZeroNitrogenPolicy(), seed=2)
    assert expert.history
    assert zero.history
    assert expert.total_nitrogen == pytest.approx(15.0)
