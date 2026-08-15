import datetime


import numpy as np
import pytest


from bench_runner import interactive_plot
from bench_runner import plot


# ---------------------------------------------------------------------------
# _clamped_range
# ---------------------------------------------------------------------------


def test_clamped_range_inside_bounds():
    # Everything within (low, high), so the axis is pinned rather than
    # autoscaled, matching the matplotlib plots.
    assert interactive_plot._clamped_range([0.95, 1.0, 1.05], 0.75, 1.25) == [
        0.75,
        1.25,
    ]


@pytest.mark.parametrize("values", [[0.5, 1.0], [1.0, 2.0], [0.5, 2.0]])
def test_clamped_range_outside_bounds_autoscales(values):
    # A value outside the window means plotly should pick the range itself.
    assert interactive_plot._clamped_range(values, 0.75, 1.25) is None


def test_clamped_range_empty():
    assert interactive_plot._clamped_range([]) is None


def test_clamped_range_boundaries_are_exclusive():
    # Exactly on the boundary does not count as inside.
    assert interactive_plot._clamped_range([0.75, 1.0], 0.75, 1.25) is None
    assert interactive_plot._clamped_range([1.0, 1.25], 0.75, 1.25) is None


def test_clamped_range_accepts_ndarray():
    # The top-level violin chart passes the concatenated ndarray straight in.
    values = np.array([0.95, 1.0, 1.05], dtype=np.float64)
    assert interactive_plot._clamped_range(values, 0.75, 1.25) == [0.75, 1.25]
    assert interactive_plot._clamped_range(np.array([]), 0.75, 1.25) is None


# ---------------------------------------------------------------------------
# _vertical_spacing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [0, 1])
def test_vertical_spacing_single_row(rows):
    assert interactive_plot._vertical_spacing(rows) == 0.0


def test_vertical_spacing_respects_plotly_limit():
    # plotly rejects spacing greater than 1 / (rows - 1).
    for rows in range(2, 40):
        spacing = interactive_plot._vertical_spacing(rows)
        assert spacing <= 1.0 / (rows - 1)


def test_vertical_spacing_caps_at_default():
    # With few rows the limit is generous, so the default wins.
    assert interactive_plot._vertical_spacing(2, default=0.06) == 0.06


# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------


def test_subsample_shorter_than_count_is_unchanged():
    values = np.arange(10.0)
    result = interactive_plot._subsample(values, 100)
    np.testing.assert_array_equal(result, values)


def test_subsample_keeps_endpoints_and_count():
    values = np.arange(1000.0)
    result = interactive_plot._subsample(values, 200)
    assert len(result) == 200
    # The endpoints matter: they are the min and max of the distribution.
    assert result[0] == values[0]
    assert result[-1] == values[-1]


def test_subsample_is_monotonic_for_sorted_input():
    values = np.sort(np.random.default_rng(0).normal(size=5000))
    result = interactive_plot._subsample(values, 200)
    assert np.all(np.diff(result) >= 0)


# ---------------------------------------------------------------------------
# rolling_stats
# ---------------------------------------------------------------------------


def test_rolling_stats_length_matches_input():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    means, stds = interactive_plot.rolling_stats(values, 2)
    assert len(means) == len(values)
    assert len(stds) == len(values)


def test_rolling_stats_uses_partial_leading_window():
    # The window is trailing but partial: a mean is produced as soon as there
    # is any data at all, rather than waiting for a full window.
    means, _ = interactive_plot.rolling_stats([1.0, 2.0, 3.0, 4.0], 3)
    assert means[0] == pytest.approx(1.0)
    assert means[1] == pytest.approx(1.5)
    assert means[2] == pytest.approx(2.0)


def test_rolling_stats_nan_only_where_window_is_empty():
    # NaN appears only where the whole window was missing.
    means, _ = interactive_plot.rolling_stats([None, None, 3.0], 2)
    assert np.isnan(means[0])
    assert np.isnan(means[1])
    assert means[2] == pytest.approx(3.0)


def test_rolling_stats_computes_trailing_mean():
    means, stds = interactive_plot.rolling_stats([1.0, 2.0, 3.0, 4.0], 2)
    # Window of 2: the point at index 1 averages the first two values.
    assert means[1] == pytest.approx(1.5)
    assert means[3] == pytest.approx(3.5)
    assert stds[1] == pytest.approx(0.5)


def test_rolling_stats_skips_none_values():
    # Missing comparisons show up as None and must not poison the window.
    means, _ = interactive_plot.rolling_stats([1.0, None, 3.0, 5.0], 2)
    assert len(means) == 4
    assert not np.isnan(means[3])


def test_rolling_stats_all_none_is_all_nan():
    means, stds = interactive_plot.rolling_stats([None, None, None], 2)
    assert np.all(np.isnan(means))
    assert np.all(np.isnan(stds))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_direction_title_shows_both_directions():
    title = interactive_plot._direction_title(("slower", "faster"))
    assert "slower" in title
    assert "faster" in title


def test_ratio_axis_formats_as_ratio():
    axis = interactive_plot._ratio_axis()
    # Ratios render as "1.05×", matching plot.formatter in the .svg charts.
    assert axis["ticksuffix"] == "×"
    assert axis["tickformat"] == ".2f"
    assert axis["zeroline"] is False


def test_plotlyjs_mode_default_is_cdn():
    # Inlining plotly.js would add ~3MB to every generated chart, and a results
    # repository keeps one chart per result per base.
    assert interactive_plot.PLOTLYJS_MODE == "cdn"


# ---------------------------------------------------------------------------
# plot cache round-trip
# ---------------------------------------------------------------------------


def test_plot_cache_round_trip(tmp_path):
    output = tmp_path / "chart.html"
    data = {"a": [1, 2, 3], "b": "value"}
    interactive_plot._save_plot_cache(output, data)
    assert interactive_plot._load_plot_cache(output) == data


def test_load_plot_cache_missing_returns_empty(tmp_path):
    assert interactive_plot._load_plot_cache(tmp_path / "nope.html") == {}


# ---------------------------------------------------------------------------
# add_rolling_average
# ---------------------------------------------------------------------------


def test_add_rolling_average_returns_trace_indices():
    go, _ = interactive_plot._plotly()
    fig = go.Figure()
    dates = [datetime.datetime(2024, 1, d) for d in range(1, 11)]
    values = [1.0 + 0.01 * i for i in range(10)]
    style = plot.TraceStyle("runner", "#1f77b4", "solid", "circle")

    indices = interactive_plot.add_rolling_average(fig, dates, values, style, window=3)

    assert len(indices) > 0
    assert len(fig.data) == len(indices)


def test_add_rolling_average_all_nan_adds_nothing():
    go, _ = interactive_plot._plotly()
    fig = go.Figure()
    dates = [datetime.datetime(2024, 1, 1)]
    style = plot.TraceStyle("runner", "#1f77b4", "solid", "circle")

    # A single point with a window of 5 never fills the window.
    added = interactive_plot.add_rolling_average(fig, dates, [None], style, window=5)
    assert added == []
    assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# save_html
# ---------------------------------------------------------------------------


def test_save_html_defaults_to_cdn_delivery(tmp_path):
    go, _ = interactive_plot._plotly()
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    output = tmp_path / "chart.html"

    interactive_plot.save_html(fig, output)

    content = output.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    # A CDN link rather than ~3MB of inlined library.
    assert output.stat().st_size < 1_000_000


def test_save_html_config_override_is_merged(tmp_path):
    go, _ = interactive_plot._plotly()
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    output = tmp_path / "chart.html"

    interactive_plot.save_html(
        fig, output, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    content = output.read_text(encoding="utf-8")
    assert "displayModeBar" in content
