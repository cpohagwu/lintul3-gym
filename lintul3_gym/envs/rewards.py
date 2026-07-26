"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Reward calculations used by the LINTUL3 environment.
"""

from __future__ import annotations

from lintul3_gym.envs.types import RewardConfig


def storage_organ_growth(previous_wso: float, current_wso: float) -> float:
    """Return LINTUL3 storage-organ growth in g/m².

    Args:
        previous_wso: Weight of storage organs (g/m²) at the previous step.
        current_wso: Weight of storage organs (g/m²) at the current step.

    Returns:
        float: ``current_wso - previous_wso``.
    """
    return current_wso - previous_wso


def nitrogen_reward(growth: float, nitrogen: float, config: RewardConfig) -> tuple[float, float]:
    """Return reward and its nitrogen-cost component.

    Args:
        growth: Storage-organ growth this step, in g/m² (see
            :func:`storage_organ_growth`) -- already computed relative to a
            zero-nitrogen baseline run if ``config.relative_to_zero_nitrogen`` is set.
        nitrogen: Nitrogen dose applied this step, in g N/m².
        config: Reward parameters; ``config.nitrogen_cost`` is β in the reward
            formula ``growth - β · nitrogen``.

    Returns:
        tuple[float, float]: ``(reward, cost)``, where ``cost = config.nitrogen_cost
        * nitrogen`` and ``reward = growth - cost``.
    """
    cost = config.nitrogen_cost * nitrogen
    return growth - cost, cost
