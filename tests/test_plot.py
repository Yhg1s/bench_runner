import numpy as np
import pytest


from bench_runner import plot
from bench_runner import runners


# ---------------------------------------------------------------------------
# plot.py: the matplotlib -> plotly style translation
# ---------------------------------------------------------------------------


def test_to_css_color_named_and_cycle_colors():
    assert plot.to_css_color("red") == "#ff0000"
    # "C0" is a matplotlib cycle color, which plotly has never heard of.
    assert plot.to_css_color("C0").startswith("#")


def test_to_css_color_with_alpha_returns_rgba():
    assert plot.to_css_color("red", 0.5) == "rgba(255, 0, 0, 0.5)"


def test_to_css_color_passes_through_hex():
    assert plot.to_css_color("#ddd") == "#dddddd"


@pytest.mark.parametrize(
    "mpl,plotly",
    [("-", "solid"), ("--", "dash"), ("-.", "dashdot"), (":", "dot")],
)
def test_dash_styles_cover_matplotlib_shorthand(mpl, plotly):
    assert plot.DASH_STYLES[mpl] == plotly


@pytest.mark.parametrize(
    "mpl,plotly",
    [("o", "circle"), ("s", "square"), ("^", "triangle-up"), ("D", "diamond")],
)
def test_marker_symbols_cover_common_markers(mpl, plotly):
    assert plot.MARKER_SYMBOLS[mpl] == plotly


def _runner(**plot_config):
    # Runner.__post_init__ builds the PlotConfig from a plain dict.
    return runners.Runner(
        nickname="nick",
        os="linux",
        arch="x86_64",
        hostname="host",
        plot=plot_config or None,
    )


def test_get_trace_style_uses_runner_plot_config():
    style = plot.get_trace_style(
        _runner(name="Pretty", color="red", style="--", marker="^")
    )
    assert style.name == "Pretty"
    assert style.color == "#ff0000"
    assert style.dash == "dash"
    assert style.symbol == "triangle-up"


def test_get_trace_style_defaults_to_nickname_when_runner_has_no_plot_section():
    # A runner with no [plot] table still gets a PlotConfig, named after the
    # nickname, from Runner.__post_init__.
    style = plot.get_trace_style(_runner())
    assert style.name == "nick"
    assert style.dash == "solid"
    assert style.symbol == "square"


def test_get_trace_style_unknown_style_and_marker_fall_back():
    style = plot.get_trace_style(
        _runner(name="N", color="C0", style="squiggly", marker="unheard-of")
    )
    assert style.dash == "solid"
    assert style.symbol == "circle"


def test_get_trace_style_none_plot_config_fallback():
    # Defensive branch in get_trace_style. Runner.__post_init__ always
    # populates .plot, so this is not reachable through a real Runner; it is
    # covered here so the fallback stays honest if that ever changes.
    class _NoPlotRunner:
        nickname = "nick"
        plot = None

    style = plot.get_trace_style(_NoPlotRunner())
    assert style.name == "nick"
    assert style.dash == "solid"
    assert style.symbol == "circle"


# ---------------------------------------------------------------------------
# plot.plot_diff_pair: the violin data collection
# ---------------------------------------------------------------------------


def test_plot_diff_pair_empty_data():
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()
    try:
        assert list(plot.plot_diff_pair(ax, [])) == []
    finally:
        plt.close("all")


def test_plot_diff_pair_concatenates_all_samples():
    from matplotlib import pyplot as plt

    data = [
        ("a", np.array([0.9, 1.0, 1.1]), 1.0),
        ("b", np.array([1.2, 1.3]), 1.25),
    ]
    _, ax = plt.subplots()
    try:
        all_data = plot.plot_diff_pair(ax, data)
    finally:
        plt.close("all")

    # One combined array holding every sample, not a boxed Python list.
    assert isinstance(all_data, np.ndarray)
    assert len(all_data) == 5
    assert sorted(all_data) == pytest.approx([0.9, 1.0, 1.1, 1.2, 1.3])


def test_plot_diff_pair_handles_insignificant_benchmarks():
    from matplotlib import pyplot as plt

    # A None distribution means "insignificant"; it contributes a single 1.0
    # so the benchmark still gets a row.
    data = [
        ("a", np.array([0.9, 1.1]), 1.0),
        ("insignificant", None, 1.0),
    ]
    _, ax = plt.subplots()
    try:
        all_data = plot.plot_diff_pair(ax, data)
    finally:
        plt.close("all")

    assert isinstance(all_data, np.ndarray)
    assert len(all_data) == 3
    assert 1.0 in all_data
