"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Small plotting helpers for transparent policy analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from lintul3_gym.envs.types import EpisodeRecord

_MONTH_STARTS = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _pyplot():
    """Import and return ``matplotlib.pyplot``, with a friendlier error if it's missing.

    Returns:
        module: The imported ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If ``matplotlib`` isn't installed, with a message pointing at
            this package's optional ``[viz]`` extra.
    """
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError("Install visualization support with `pip install lintul3-gym[viz]`.") from error
    return pyplot


def _aggregate_by_day_of_year(
    histories: Iterable[Iterable[EpisodeRecord]],
    extractor: Callable[[EpisodeRecord], float],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Bucket ``extractor(record)`` values by calendar day-of-year across histories.

    Returns sorted day-of-year x-values alongside the median, 2.5th, and
    97.5th percentile of the values observed at each x-value.

    Args:
        histories: One or more episode histories (e.g. one per simulated
            year/replicate) to pool together by day-of-year.
        extractor: Function selecting the value to aggregate from each
            ``EpisodeRecord`` (e.g. ``lambda record: record.crop["WSO"]``).

    Returns:
        tuple[list[int], list[float], list[float], list[float]]: Sorted
        day-of-year values, and the median, 2.5th-percentile, and 97.5th-percentile
        of ``extractor``'s output observed on each of those days across all
        ``histories``.
    """
    values_by_day: dict[int, list[float]] = defaultdict(list)
    for history in histories:
        for record in history:
            values_by_day[record.date.timetuple().tm_yday].append(extractor(record))
    days = sorted(values_by_day)
    medians = [float(np.median(values_by_day[day])) for day in days]
    lows = [float(np.percentile(values_by_day[day], 2.5)) for day in days]
    highs = [float(np.percentile(values_by_day[day], 97.5)) for day in days]
    return days, medians, lows, highs


def _set_month_xticks(axis, days: Iterable[int]) -> None:
    """Relabel a day-of-year x-axis with the month names it spans.

    Args:
        axis: The Matplotlib axis to relabel.
        days: The day-of-year values actually plotted on ``axis``; only month
            boundaries within their range get a tick. No-op if empty.
    """
    days = list(days)
    if not days:
        return
    lo, hi = min(days), max(days)
    ticks = [start for start in _MONTH_STARTS if lo - 31 <= start <= hi]
    labels = [_MONTH_NAMES[_MONTH_STARTS.index(start)] for start in ticks]
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)


def _median_and_iqr(values_per_x: Iterable[Iterable[float]]) -> tuple[list[float], list[float], list[float]]:
    """Return the median, 25th, and 75th percentile of each inner sequence in ``values_per_x``.

    Args:
        values_per_x: One inner sequence of values per x-position (e.g. per test
            year), such as one nitrogen total per training seed.

    Returns:
        tuple[list[float], list[float], list[float]]: Parallel ``(medians, lows,
        highs)`` lists, one entry per inner sequence in ``values_per_x``.
    """
    medians, lows, highs = [], [], []
    for values in values_per_x:
        values = list(values)
        medians.append(float(np.median(values)))
        lows.append(float(np.percentile(values, 25)))
        highs.append(float(np.percentile(values, 75)))
    return medians, lows, highs


def plot_nitrogen_vs_rainfall(
    baseline_points: dict[str, tuple[Iterable[float], Iterable[float]]],
    ensemble_points: dict[str, tuple[Iterable[float], Iterable[Iterable[float]]]],
    *,
    xlabel: str = "Average daily rainfall (mm/day)",
    ylabel: str = "Total nitrogen applied (g N/m²)",
    title: str | None = None,
):
    """Scatter nitrogen applied against seasonal rainfall, one point per test year.

    Reproduces the non-Ceres half of Kallenberg et al. (2023)'s Figure 4: a
    fixed-dose baseline (e.g. Standard Practice) as one dot per test year, and
    an ensemble policy (e.g. several independently-seeded RL runs) as a
    median-plus-IQR error bar per test year. Each series gets a linear
    regression line and a hollow "median" marker at its overall median. Ceres
    is not included since this package does not implement it.

    Args:
        baseline_points: label -> (rainfall, nitrogen), one value per test year.
        ensemble_points: label -> (rainfall, nitrogen_per_seed), where
            ``nitrogen_per_seed[i]`` holds one value per seed for test year i.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional axis title; omitted if ``None``.
    """
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6, 6))
    median_labeled = False

    def _median_marker(x: float, y: float, color) -> None:
        nonlocal median_labeled
        axis.scatter(
            x, y, s=150, edgecolors=color, facecolors="none", linewidths=2, zorder=3,
            label=None if median_labeled else "median",
        )
        median_labeled = True

    def _regression_line(x: np.ndarray, y: np.ndarray, color) -> None:
        if len(x) < 2:
            return
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        axis.plot(xs, slope * xs + intercept, color=color, alpha=0.3)

    for label, (rainfall, nitrogen) in baseline_points.items():
        rainfall_arr = np.asarray(list(rainfall), dtype=float)
        nitrogen_arr = np.asarray(list(nitrogen), dtype=float)
        scatter = axis.scatter(rainfall_arr, nitrogen_arr, label=label, marker="o")
        color = scatter.get_facecolor()[0]
        _regression_line(rainfall_arr, nitrogen_arr, color)
        _median_marker(float(np.median(rainfall_arr)), float(np.median(nitrogen_arr)), color)

    for label, (rainfall, nitrogen_per_seed) in ensemble_points.items():
        rainfall_arr = np.asarray(list(rainfall), dtype=float)
        medians, lows, highs = _median_and_iqr(nitrogen_per_seed)
        medians_arr, lows_arr, highs_arr = np.asarray(medians), np.asarray(lows), np.asarray(highs)
        errorbar = axis.errorbar(
            rainfall_arr, medians_arr, yerr=[medians_arr - lows_arr, highs_arr - medians_arr],
            fmt="o", label=f"{label} (median+IQR)", alpha=0.7, capsize=3,
        )
        color = errorbar.lines[0].get_color()
        _regression_line(rainfall_arr, medians_arr, color)
        _median_marker(float(np.median(rainfall_arr)), float(np.median(medians_arr)), color)

    axis.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    pyplot.show()


def plot_episode(history: Iterable[EpisodeRecord], variables: tuple[str, ...] = ("WSO", "LAI", "TNSOIL")):
    """Plot actions, reward, and selected crop variables for an episode.

    Args:
        history: The episode's ordered ``EpisodeRecord`` sequence (e.g.
            ``EvaluationResult.history``).
        variables: Names of ``record.crop`` entries to plot, one panel each, below
            the nitrogen-action and reward panels.
    """
    records = list(history)
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(2 + len(variables), 1, sharex=True, figsize=(9, 2.4 * (2 + len(variables))))
    dates = [record.date for record in records]
    axes[0].step(dates, [record.action_nitrogen for record in records], where="post")
    axes[0].set_ylabel("N (g/m²)")
    axes[1].plot(dates, [record.reward for record in records])
    axes[1].set_ylabel("Reward")
    for axis, variable in zip(axes[2:], variables):
        axis.plot(dates, [record.crop[variable] for record in records])
        axis.set_ylabel(variable)
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    pyplot.show()


def plot_comparison(results: dict[str, Iterable[EpisodeRecord]], variable: str = "WSO"):
    """Plot one crop variable from named agent/reference episode histories.

    Args:
        results: Mapping from a policy/agent label to its single episode history
            (e.g. ``{"expert": expert.history, "RL model": rl_result.history}``).
        variable: Name of the ``record.crop`` entry to plot against calendar date.
    """
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(9, 4))
    for label, history in results.items():
        records = list(history)
        axis.plot([record.date for record in records], [record.crop[variable] for record in records], label=label)
    axis.set(xlabel="Date", ylabel=variable)
    axis.legend()
    figure.tight_layout()
    pyplot.show()


def plot_complete_comparison_monthly(results: dict[str, list[Iterable[EpisodeRecord]]]) -> None:
    """Plot median and 95% interval bands across multiple years, labeled by month.

    Unlike :func:`plot_comparison`, this expects *multiple* episode histories
    per policy label (one per simulated year/replicate) and aggregates them
    onto a shared day-of-year x-axis labeled with month names, since a single
    calendar-date axis does not make sense across several years of data.

    Args:
        results: Mapping from policy label to a list of episode histories,
            one history per simulated year/replicate -- e.g. the results of
            calling :func:`lintul3_gym.policies.evaluate_policy` once per
            year in a multi-year ``WeatherConfig``.
    """
    pyplot = _pyplot()

    crop_variables = ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN",
                    "TNSOIL", "TRAIN", "TRANRF", "WSO"]
    weather_variables = ["TMIN", "TMAX", "RAIN", "IRRAD"]
    metric_variables = [
        ("action_nitrogen", "Action nitrogen"),
        ("reward", "Reward"),
        ("growth", "Growth"),
        ("nitrogen_cost", "Nitrogen cost"),
        ("cumulative_nitrogen", "Cumulative nitrogen"),
    ]

    fig_crop, axes_crop = pyplot.subplots(nrows=3, ncols=3, figsize=(16, 12), sharex=True)
    fig_weather, axes_weather = pyplot.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)
    fig_metrics, axes_metrics = pyplot.subplots(nrows=2, ncols=3, figsize=(16, 10), sharex=True)

    def _plot_panel(axis, extractor: Callable[[EpisodeRecord], float]) -> None:
        all_days: list[int] = []
        for label, histories in results.items():
            days, medians, lows, highs = _aggregate_by_day_of_year(histories, extractor)
            all_days.extend(days)
            line, = axis.plot(days, medians, label=label, linewidth=1.8)
            axis.fill_between(days, lows, highs, alpha=0.2, color=line.get_color())
        _set_month_xticks(axis, all_days)
        axis.grid(alpha=0.25)

    for ax, variable in zip(axes_crop.flat, crop_variables):
        _plot_panel(ax, lambda record, variable=variable: record.crop[variable])
        ax.set_title(variable)
        ax.set_ylabel(variable)

    for ax, variable in zip(axes_weather.flat, weather_variables):
        _plot_panel(ax, lambda record, variable=variable: record.weather[variable])
        ax.set_title(variable)
        ax.set_ylabel(variable)

    for ax, (metric_name, title) in zip(axes_metrics.flat, metric_variables):
        _plot_panel(ax, lambda record, metric_name=metric_name: getattr(record, metric_name))
        ax.set_title(title)
        ax.set_ylabel(title)

    for ax in axes_metrics.flat[len(metric_variables):]:
        ax.axis("off")

    axes_crop[-1, -1].legend(loc="best")
    axes_weather[-1, -1].legend(loc="best")
    axes_metrics[0, 0].legend(loc="best")

    fig_crop.tight_layout()
    fig_weather.tight_layout()
    fig_metrics.tight_layout()

    fig_crop.suptitle("Crop Variables Comparison (median, 95% interval by month)", fontsize=16, y=1.02)
    fig_weather.suptitle("Weather Variables Comparison (median, 95% interval by month)", fontsize=16, y=1.02)
    fig_metrics.suptitle("Episode Metrics Comparison (median, 95% interval by month)", fontsize=16, y=1.02)

    pyplot.show()


def plot_complete_comparison_yearly(results: dict[str, list]) -> None:
    """Plot a comparison of crop, weather, and episode metrics for multiple policies.

    Args:
        results: A dictionary mapping policy names to their corresponding history records.
    """
    pyplot = _pyplot()

    # Build a clean comparison plot with separate panels for crop, weather, and episode metrics.
    # Each panel shows all models side by side for the same variable.

    crop_variables = ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN",
                    "TNSOIL", "TRAIN", "TRANRF", "WSO"]
    weather_variables = ["TMIN", "TMAX", "RAIN", "IRRAD"]
    metric_variables = [
        ("action_nitrogen", "Action nitrogen"),
        ("reward", "Reward"),
        ("growth", "Growth"),
        ("nitrogen_cost", "Nitrogen cost"),
        ("cumulative_nitrogen", "Cumulative nitrogen"),
    ]

    # Create separate figures for crop, weather, and episode metrics.
    fig_crop, axes_crop = pyplot.subplots(nrows=3, ncols=3, figsize=(16, 12), sharex=True)
    fig_weather, axes_weather = pyplot.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)
    fig_metrics, axes_metrics = pyplot.subplots(nrows=2, ncols=3, figsize=(16, 10), sharex=True)

    for ax, variable in zip(axes_crop.flat, crop_variables):
        for label, history in results.items():
            records = list(history)
            ax.plot(
                [record.date for record in records],
                [record.crop[variable] for record in records],
                label=label,
                linewidth=1.8,
            )
        ax.set_title(variable)
        ax.set_ylabel(variable)
        ax.grid(alpha=0.25)

    for ax, variable in zip(axes_weather.flat, weather_variables):
        for label, history in results.items():
            records = list(history)
            ax.plot(
                [record.date for record in records],
                [record.weather[variable] for record in records],
                label=label,
                linewidth=1.8,
            )
        ax.set_title(variable)
        ax.set_ylabel(variable)
        ax.grid(alpha=0.25)

    for ax, (metric_name, title) in zip(axes_metrics.flat, metric_variables):
        for label, history in results.items():
            records = list(history)
            ax.plot(
                [record.date for record in records],
                [getattr(record, metric_name) for record in records],
                label=label,
                linewidth=1.8,
            )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)

    for ax in axes_metrics.flat[len(metric_variables):]:
        ax.axis("off")

    axes_crop[-1, -1].legend(loc="best")
    axes_weather[-1, -1].legend(loc="best")
    axes_metrics[0, 0].legend(loc="best")

    fig_crop.tight_layout()
    fig_crop.autofmt_xdate()
    fig_weather.tight_layout()
    fig_weather.autofmt_xdate()
    fig_metrics.tight_layout()
    fig_metrics.autofmt_xdate()

    # Add a title to the entire figure for each of the three figures.
    fig_crop.suptitle("Crop Variables Comparison", fontsize=16, y=1.02)
    fig_weather.suptitle("Weather Variables Comparison", fontsize=16, y=1.02)
    fig_metrics.suptitle("Episode Metrics Comparison", fontsize=16, y=1.02)

    pyplot.show()
