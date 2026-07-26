"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

LINTUL3-Gym: a transparent, reproducible Gymnasium environment for nitrogen-management
reinforcement learning.
"""

from lintul3_gym.envs.environment import Lintul3Env
from lintul3_gym.envs.types import RewardConfig, SeasonConfig, WeatherConfig
from lintul3_gym.logging_utils import silence_pcse_log_rotation

try:
    from gymnasium.envs.registration import registry, register

    if "Lintul3Gym-v0" not in registry:
        register(id="Lintul3Gym-v0", entry_point="lintul3_gym.envs:Lintul3Env")
except ImportError:
    pass

__all__ = [
    "Lintul3Env",
    "RewardConfig",
    "SeasonConfig",
    "WeatherConfig",
    "silence_pcse_log_rotation",
]
