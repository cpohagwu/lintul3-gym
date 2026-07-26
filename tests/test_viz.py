"""Tests for the month-aggregated, multi-year comparison plot helpers."""

from dataclasses import dataclass
from datetime import date

import pytest

pytest.importorskip("matplotlib")

from lintul3_gym.viz.plots import _aggregate_by_day_of_year, _median_and_iqr, _set_month_xticks


@dataclass
class _FakeRecord:
    """Minimal stand-in for EpisodeRecord carrying only what these helpers read."""

    date: date
    value: float


def test_aggregate_by_day_of_year_computes_median_and_band_across_years() -> None:
    """Values recorded on the same day-of-year across different years are grouped together."""
    year_one = [_FakeRecord(date(2001, 4, 1), 10.0), _FakeRecord(date(2001, 4, 8), 20.0)]
    year_two = [_FakeRecord(date(2002, 4, 1), 30.0), _FakeRecord(date(2002, 4, 8), 20.0)]

    days, medians, lows, highs = _aggregate_by_day_of_year([year_one, year_two], lambda record: record.value)

    assert days == [date(2001, 4, 1).timetuple().tm_yday, date(2001, 4, 8).timetuple().tm_yday]
    assert medians == [20.0, 20.0]
    assert lows[0] < 20.0 < highs[0]
    assert lows[1] == highs[1] == 20.0


def test_aggregate_by_day_of_year_handles_no_histories() -> None:
    """An empty set of histories yields empty series rather than an error."""
    days, medians, lows, highs = _aggregate_by_day_of_year([], lambda record: record.value)
    assert days == medians == lows == highs == []


def test_set_month_xticks_labels_the_spanned_months() -> None:
    """Day-of-year ticks are relabeled with the month names present in the data."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots()
    try:
        # Spans late March through late April (day-of-year 90-110).
        _set_month_xticks(axis, [90, 95, 100, 105, 110])
        labels = [label.get_text() for label in axis.get_xticklabels()]
        assert "Mar" in labels
        assert "Apr" in labels
        assert "Jan" not in labels
    finally:
        pyplot.close(figure)


def test_set_month_xticks_handles_empty_days() -> None:
    """No days present means no tick changes and no error."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots()
    try:
        _set_month_xticks(axis, [])
    finally:
        pyplot.close(figure)


def test_median_and_iqr_summarizes_each_inner_sequence() -> None:
    """Each per-year sequence of per-seed values collapses to (median, q25, q75)."""
    medians, lows, highs = _median_and_iqr([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0]])
    assert medians == [2.5, 10.0]
    assert lows[0] < 2.5 < highs[0]
    assert lows[1] == highs[1] == 10.0


def test_plot_nitrogen_vs_rainfall_renders_without_error() -> None:
    """The Figure-4-style scatter accepts baseline and ensemble series and renders cleanly."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    from lintul3_gym.viz.plots import plot_nitrogen_vs_rainfall

    try:
        plot_nitrogen_vs_rainfall(
            baseline_points={"Standard Practice": ([1.0, 2.0, 3.0], [17.0, 17.0, 17.0])},
            ensemble_points={"RL": ([1.0, 2.0, 3.0], [[10.0, 12.0], [14.0, 15.0], [20.0, 22.0]])},
        )
    finally:
        pyplot.close("all")
