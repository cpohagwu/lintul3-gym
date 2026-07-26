"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Transparent reference policies and evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np

from lintul3_gym.envs.environment import Lintul3Env
from lintul3_gym.envs.types import EpisodeRecord

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize


class Policy(Protocol):
    """Protocol implemented by simple, inspectable environment policies."""

    def action(self, environment: Lintul3Env) -> np.ndarray:
        """Return the next nitrogen action.

        Args:
            environment: The (already reset) environment to act in.

        Returns:
            np.ndarray: A one-element array containing the nitrogen dose to apply,
            in g N/m², suitable for passing straight to :meth:`Lintul3Env.step`.
        """


class ZeroNitrogenPolicy():
    """Reference policy that never applies nitrogen."""

    def action(self, environment: Lintul3Env) -> np.ndarray:
        """Return a zero nitrogen dose regardless of environment state.

        Args:
            environment: Unused; accepted to satisfy the :class:`Policy` protocol.

        Returns:
            np.ndarray: ``[0.0]``.
        """
        del environment
        return np.asarray([0.0], dtype=np.float32)


def _calendar_dose(
    environment: Lintul3Env, events: tuple[tuple[int, int, float], ...]
) -> np.ndarray:
    """Sum the ``(month, day, dose)`` events in ``events`` due within the current decision window.

    Shared by every "fixed calendar schedule" reference policy (:class:`ExpertPolicy`,
    :class:`StandardPracticePolicy`): each such policy is just a different set of
    ``events`` applied through this same date-window lookup.

    Args:
        environment: The (already reset) environment being acted in; its current
            model day and ``decision_interval`` define the window to check.
        events: ``(month, day, dose)`` triples -- ``dose`` in g N/m², applied once per
            calendar year on ``month``/``day`` if that date falls within
            ``[environment's current day, current day + decision_interval)``.

    Returns:
        np.ndarray: A one-element array with the summed dose (0.0 if no event in
        ``events`` falls within the current window).

    Raises:
        RuntimeError: If ``environment`` hasn't been reset yet (no model exists).
    """
    model = environment._model
    if model is None:
        raise RuntimeError("Reset the environment before requesting an action.")
    start = model.day
    end = start + timedelta(days=environment.decision_interval)
    amount = sum(
        dose
        for month, day, dose in events
        if start <= date(start.year, month, day) < end
    )
    return np.asarray([amount], dtype=np.float32)


class ExpertPolicy():
    """The nitrogen schedule from the bundled spring-wheat `.agro` file.

    A fixed calendar schedule (see :func:`_calendar_dose`): two doses, 10 and
    5 g N/m² on Apr 10 and May 5.
    """

    _events: tuple[tuple[int, int, float], ...] = ((4, 10, 10.0), (5, 5, 5.0))

    def action(self, environment: Lintul3Env) -> np.ndarray:
        """Apply every scheduled dose falling within the current interval.

        Args:
            environment: The (already reset) environment to act in.

        Returns:
            np.ndarray: A one-element array with the total dose due this step, in
            g N/m² (see :func:`_calendar_dose`).
        """
        return _calendar_dose(environment, self._events)


class StandardPracticePolicy():
    """The "Standard Practice" (SP) baseline from Kallenberg et al. (2023).

    Another fixed calendar schedule (see :func:`_calendar_dose`, and compare
    :class:`ExpertPolicy`): splits ``total_nitrogen`` evenly across three fixed
    calendar dates -- Feb 24, Mar 26, and Apr 29 -- the real fertilization
    dates used by the paper's reference implementation for its
    "standard-practice" policy (WUR-AI/PCSE-Gym
    `pcse_gym/utils/eval.py::evaluate_policy`). ``total_nitrogen`` defaults to
    17.07 g N/m² (=170.7 kg N/ha), the SP amount the paper reports (Table
    2/3) -- though the paper's own code applies that particular number as a
    single early "start-dump" dose rather than these three dates (see
    `examples/envs/winterwheat/README.md` for why that makes this policy's
    yield an approximation of, not identical to, the paper's published
    numbers).
    """

    _fertilization_days: tuple[tuple[int, int], ...] = ((2, 24), (3, 26), (4, 29))

    def __init__(self, total_nitrogen: float = 17.07) -> None:
        """Split ``total_nitrogen`` evenly across the three fertilization dates.

        Args:
            total_nitrogen: Total nitrogen to apply over the season, in g N/m².
                Defaults to the paper's reported Standard Practice amount
                (17.07 g N/m² = 170.7 kg N/ha).
        """
        self.total_nitrogen = total_nitrogen
        dose = total_nitrogen / len(self._fertilization_days)
        self._events: tuple[tuple[int, int, float], ...] = tuple(
            (month, day, dose) for month, day in self._fertilization_days
        )

    def action(self, environment: Lintul3Env) -> np.ndarray:
        """Apply a third of ``total_nitrogen`` for every scheduled date in the current interval.

        Args:
            environment: The (already reset) environment to act in.

        Returns:
            np.ndarray: A one-element array with the total dose due this step, in
            g N/m² (see :func:`_calendar_dose`).
        """
        return _calendar_dose(environment, self._events)


@dataclass(frozen=True)
class EvaluationResult:
    """Completed episode metrics and auditable transition history.

    Attributes:
        total_reward: Sum of every step's reward over the episode.
        final_wso: Weight of storage organs (grain yield, g/m²) at the episode's
            last recorded step.
        total_nitrogen: Cumulative nitrogen applied over the episode, in g N/m².
        history: The complete, ordered sequence of per-step ``EpisodeRecord``s.
    """

    total_reward: float
    final_wso: float
    total_nitrogen: float
    history: tuple[EpisodeRecord, ...]


def is_sb3_model(model: object) -> bool:
    """Check if the given model is a Stable Baselines3 model.

    Args:
        model: The object to check -- typically a :class:`Policy` implementation or a
            trained Stable-Baselines3 algorithm instance.

    Returns:
        bool: ``True`` if ``model`` is a Stable-Baselines3 ``BaseAlgorithm``.
    """
    import stable_baselines3 as sb3
    return isinstance(model, sb3.common.base_class.BaseAlgorithm)


def evaluate_policy(environment: Lintul3Env, policy: Policy, *, seed: int | None = None) -> EvaluationResult:
    """Run one policy episode and return its complete transparent record.

    Works with either a simple :class:`Policy` (e.g. :class:`ExpertPolicy`,
    :class:`ZeroNitrogenPolicy`, :class:`StandardPracticePolicy`) or a trained
    Stable-Baselines3 model, acting directly on the raw (unwrapped) ``environment`` --
    for an already-wrapped SB3 ``VecNormalize`` environment, use
    :func:`evaluate_sb3_policy` instead.

    Args:
        environment: The environment to run the episode in.
        policy: The policy (or SB3 model) to evaluate.
        seed: Seed passed to ``environment.reset()``.

    Returns:
        EvaluationResult: The episode's total reward, final yield, total nitrogen
        applied, and complete step-by-step history.
    """
    observation, _ = environment.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    while not terminated:
        if not is_sb3_model(policy):
            action = policy.action(environment)
        else:
            action, _ = policy.predict(observation, deterministic=True)

        _, reward, terminated, truncated, _ = environment.step(action)
        total_reward += reward
        if truncated:
            break

    # Get the final WSO and total nitrogen from the environment's history
    if not is_sb3_model(policy):
        metrics_environment = environment
    else:
        metrics_environment = environment.unwrapped
    final_wso = metrics_environment.history[-1].crop["WSO"] if metrics_environment.history else 0.0
    return EvaluationResult(
        total_reward=total_reward,
        final_wso=final_wso,
        total_nitrogen=metrics_environment.history[-1].cumulative_nitrogen if metrics_environment.history else 0.0,
        history=tuple(metrics_environment.history),
    )


def evaluate_policy_over_weather(
    environment: Lintul3Env, policy: Policy, *, seed: int | None = None
) -> tuple[EvaluationResult, ...]:
    """Evaluate a policy once for every (location, year) combination, in round-robin order.

    Args:
        environment: The environment to run each episode in. Its ``weather_config``
            determines how many combinations there are; each ``evaluate_policy`` call
            advances ``environment`` to the next combination via its internal
            round-robin cycling (see ``Lintul3Env._sample_weather_context``).
        policy: The policy to evaluate.
        seed: Seed passed to every ``evaluate_policy`` call.

    Returns:
        tuple[EvaluationResult, ...]: One result per ``(location, year)`` combination,
        in the same fixed order ``environment`` itself cycles through -- so calling
        this again with a different policy on the same ``environment`` configuration
        lines up one-to-one with a previous call's results.
    """
    n_combinations = len(environment.weather_config.locations) * len(environment.weather_config.years)
    return tuple(evaluate_policy(environment, policy, seed=seed) for _ in range(n_combinations))


def evaluate_sb3_policy(
    vec_env: "VecNormalize", model: "BaseAlgorithm", *, n_episodes: int = 1
) -> tuple[EvaluationResult, ...]:
    """Evaluate an SB3 model through its (already wrapped, frozen) vec env, ``n_episodes`` times.

    ``vec_env`` should come from :func:`lintul3_gym.sb3.make_eval_env` so the
    model receives observations on the same scale it was trained on. Each
    step's ``info["record"]`` is read directly rather than the wrapped
    Lintul3Env's ``.history``: a VecEnv auto-resets the underlying env the
    instant an episode ends, inside the same ``step()`` call, which clears
    ``.history`` before the caller can read it back.

    Args:
        vec_env: A frozen ``VecNormalize`` environment (see
            :func:`lintul3_gym.sb3.make_eval_env`) wrapping a single ``Lintul3Env``.
        model: The trained Stable-Baselines3 model to evaluate.
        n_episodes: Number of episodes to run in sequence.

    Returns:
        tuple[EvaluationResult, ...]: One result per episode, in the order run.
    """
    results = []
    observation = vec_env.reset()
    for _ in range(n_episodes):
        records: list[EpisodeRecord] = []
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, infos = vec_env.step(action)
            records.append(infos[0]["record"])
            done = bool(dones[0])
        results.append(EvaluationResult(
            total_reward=sum(record.reward for record in records),
            final_wso=records[-1].crop["WSO"] if records else 0.0,
            total_nitrogen=records[-1].cumulative_nitrogen if records else 0.0,
            history=tuple(records),
        ))
    return tuple(results)
