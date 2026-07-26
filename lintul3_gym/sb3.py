"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Stable-Baselines3-friendly adapters without a mandatory SB3 dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from lintul3_gym.envs.environment import Lintul3Env

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class FlattenObservation(gym.ObservationWrapper):
    """
    Flatten observations from a LINTUL3-Gym environment in crop, weather, management order.
    Suitable for MLP policies in Stable-Baselines3 and other RL libraries.
    """

    feature_groups = ("crop", "weather", "management")

    def __init__(self, environment: Lintul3Env) -> None:
        """Wrap ``environment`` and compute the flattened observation space's size.

        Args:
            environment: The ``Lintul3Env`` (or another wrapper around one) whose
                ``Dict`` observation space should be flattened.
        """
        super().__init__(environment)
        size = sum(int(np.prod(environment.observation_space[group].shape)) for group in self.feature_groups)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return the flattened float32 representation used by MLP policies.

        Args:
            observation: The wrapped environment's ``crop``/``weather``/``management``
                observation dict.

        Returns:
            np.ndarray: The three groups concatenated in that fixed order.
        """
        return np.concatenate([observation[group] for group in self.feature_groups]).astype(np.float32)


def make_sb3_env(environment: Lintul3Env) -> FlattenObservation:
    """Create a flattened environment suitable for SB3 ``MlpPolicy`` models from a LINTUL3-Gym environment.

    Args:
        environment: The environment to wrap. Keeps its continuous ``Box`` action
            space -- use :func:`make_discrete_sb3_env` instead for a discrete-dose menu.

    Returns:
        FlattenObservation: The wrapped environment.
    """
    return FlattenObservation(environment)


class DiscretizeNitrogen(gym.ActionWrapper):
    """Expose a fixed {0, 20, 40} kg N/ha menu over Lintul3Env's continuous g N/m² action.
    {0, 20, 40} kg N/ha corresponds to {0, 2, 4} g N/m²."""

    doses_kg_per_ha = (0.0, 20.0, 40.0)

    def __init__(self, environment: Lintul3Env) -> None:
        """Wrap ``environment`` and replace its action space with ``Discrete(3)``.

        Args:
            environment: The ``Lintul3Env`` (or another wrapper around one) whose
                continuous nitrogen action should be discretized.
        """
        super().__init__(environment)
        self.action_space = gym.spaces.Discrete(len(self.doses_kg_per_ha))

    def action(self, action: int) -> np.ndarray:
        """Map a discrete dose index to the g N/m² amount Lintul3Env expects.

        Args:
            action: Index into ``doses_kg_per_ha`` (0, 1, or 2).

        Returns:
            np.ndarray: A one-element array with the corresponding dose in g N/m².
        """
        kg_per_ha = self.doses_kg_per_ha[int(action)]
        return np.asarray([kg_per_ha / 10.0], dtype=np.float32)


def make_discrete_sb3_env(environment: Lintul3Env) -> FlattenObservation:
    """Create a flattened, discrete-dose ({0, 20, 40} kg N/ha) environment for SB3 models.

    Args:
        environment: The environment to wrap.

    Returns:
        FlattenObservation: ``environment`` wrapped in :class:`DiscretizeNitrogen` then
        :class:`FlattenObservation`.
    """
    return FlattenObservation(DiscretizeNitrogen(environment))


def make_vec_env(environment: Lintul3Env, *, log_dir: str | None = None) -> "DummyVecEnv":
    """Wrap a discretized Lintul3Env the way SB3 vectorized training/eval expects.

    Args:
        environment: The environment to wrap.
        log_dir: Directory for Stable-Baselines3's ``Monitor`` episode logs, or
            ``None`` to disable file logging.

    Returns:
        DummyVecEnv: A single-environment vectorized env: ``Monitor`` wrapping
        :func:`make_discrete_sb3_env`, inside a ``DummyVecEnv``.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([lambda: Monitor(make_discrete_sb3_env(environment), log_dir)])


def make_training_env(
    environment: Lintul3Env, *, log_dir: str | None = None, norm_reward: bool = True, clip_obs: float = 10.0
) -> "VecNormalize":
    """Build the ``VecNormalize``-wrapped environment a model should train against.

    Save its running statistics after training with ``vec_env.save(path)`` --
    :func:`make_eval_env` needs that file to reproduce the same observation scale.

    Args:
        environment: The environment to wrap (see :func:`make_vec_env`).
        log_dir: Directory for Stable-Baselines3's ``Monitor`` episode logs.
        norm_reward: Whether to normalize rewards during training (SB3's usual
            recommendation for PPO-style algorithms).
        clip_obs: Absolute value to clip normalized observations to.

    Returns:
        VecNormalize: The training-ready wrapped environment, with fresh (not yet
        fitted) running statistics.
    """
    from stable_baselines3.common.vec_env import VecNormalize

    return VecNormalize(make_vec_env(environment, log_dir=log_dir), norm_reward=norm_reward, clip_obs=clip_obs)


def make_eval_env(environment: Lintul3Env, vecnormalize_path: str | Path, *, log_dir: str | None = None) -> "VecNormalize":
    """Rebuild the exact training wrapper stack for evaluation, with frozen statistics.

    A model trained through :func:`make_training_env` expects a batched input
    (from ``DummyVecEnv``) scaled by a moving average (from ``VecNormalize``).
    Evaluating it against a raw ``Lintul3Env`` silently feeds out-of-distribution
    observations instead of raising an error. This loads the saved statistics
    and disables further stat updates and reward normalization, so evaluation
    reflects the trained policy rather than a moving eval-time target.

    Args:
        environment: The environment to evaluate on -- may differ from the training
            environment (e.g. a held-out set of years/locations), but must share the
            same observation/action shape.
        vecnormalize_path: Path to the ``VecNormalize`` statistics file saved during
            training (via ``train_env.save(path)``).
        log_dir: Directory for Stable-Baselines3's ``Monitor`` episode logs.

    Returns:
        VecNormalize: The wrapped environment, with training's saved statistics
        loaded and frozen (``training=False``, ``norm_reward=False``).
    """
    from stable_baselines3.common.vec_env import VecNormalize

    vec_env = VecNormalize.load(str(vecnormalize_path), make_vec_env(environment, log_dir=log_dir))
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env
