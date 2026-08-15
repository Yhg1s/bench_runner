from collections import defaultdict
from pathlib import Path


import pytest


from bench_runner.scripts import profiling_plot


def _categories():
    return [(0.40, "interpreter"), (0.25, "library"), (0.10, "gc")]


def _results():
    results = defaultdict(lambda: defaultdict(float))
    for benchmark in ("nbody", "chaos", "go"):
        for category, value in (("interpreter", 0.4), ("library", 0.25), ("gc", 0.1)):
            results[benchmark][category] = value
    return results


def _html(path):
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# get_css_color_and_pattern
# ---------------------------------------------------------------------------


def test_css_color_and_pattern_returns_css_color():
    color, pattern = profiling_plot.get_css_color_and_pattern("interpreter")
    assert color.startswith("#")
    assert pattern in profiling_plot.PATTERN_SHAPES.values()


def test_css_color_and_pattern_unknown_category_is_grey():
    # get_color_and_hatch returns the "other" grey for anything unrecognised.
    color, pattern = profiling_plot.get_css_color_and_pattern("not-a-category")
    assert color == "#dddddd"
    assert pattern == ""


def test_pattern_shapes_translate_matplotlib_hatches():
    assert profiling_plot.PATTERN_SHAPES["//"] == "/"
    assert profiling_plot.PATTERN_SHAPES["\\\\"] == "\\"
    assert profiling_plot.PATTERN_SHAPES[""] == ""


# ---------------------------------------------------------------------------
# plot_bargraph_interactive
# ---------------------------------------------------------------------------


def test_bargraph_interactive_writes_html(tmp_path):
    output = tmp_path / "bar.html"
    profiling_plot.plot_bargraph_interactive(
        _results(), _categories(), output, include_plotlyjs="cdn"
    )
    assert output.is_file()
    assert "plotly" in _html(output).lower()


def test_bargraph_interactive_one_trace_per_category_plus_other(tmp_path):
    fig = profiling_plot.plot_bargraph_interactive(
        _results(), _categories(), tmp_path / "bar.html", include_plotlyjs="cdn"
    )
    names = [trace.name for trace in fig.data]
    # One trace per category so it can be isolated from the legend, plus the
    # unaccounted-for remainder.
    assert len(names) == len(_categories()) + 1
    assert "(other functions)" in names
    assert any(name.startswith("interpreter") for name in names)


def test_bargraph_interactive_skips_unknown_category(tmp_path):
    categories = _categories() + [(0.05, "unknown")]
    fig = profiling_plot.plot_bargraph_interactive(
        _results(), categories, tmp_path / "bar.html", include_plotlyjs="cdn"
    )
    assert not any(str(trace.name).startswith("unknown") for trace in fig.data)


def test_bargraph_interactive_is_stacked_with_toggle(tmp_path):
    fig = profiling_plot.plot_bargraph_interactive(
        _results(), _categories(), tmp_path / "bar.html", include_plotlyjs="cdn"
    )
    assert fig.layout.barmode == "stack"
    labels = [
        button.label for menu in fig.layout.updatemenus for button in menu.buttons
    ]
    assert "Stacked" in labels
    assert "Grouped" in labels


def test_bargraph_interactive_rows_are_benchmarks(tmp_path):
    fig = profiling_plot.plot_bargraph_interactive(
        _results(), _categories(), tmp_path / "bar.html", include_plotlyjs="cdn"
    )
    for trace in fig.data:
        assert set(trace.y) == {"nbody", "chaos", "go"}


# ---------------------------------------------------------------------------
# plot_pie_interactive
# ---------------------------------------------------------------------------


def test_pie_interactive_writes_html(tmp_path):
    output = tmp_path / "pie.html"
    profiling_plot.plot_pie_interactive(_categories(), output, include_plotlyjs="cdn")
    assert output.is_file()
    assert "plotly" in _html(output).lower()


def test_pie_interactive_adds_remainder_slice_when_under_one(tmp_path):
    # The categories sum to 0.75, so the missing quarter becomes its own slice.
    fig = profiling_plot.plot_pie_interactive(
        _categories(), tmp_path / "pie.html", include_plotlyjs="cdn"
    )
    labels = list(fig.data[0].labels)
    assert "(other functions)" in labels
    assert sum(fig.data[0].values) == pytest.approx(1.0)


def test_pie_interactive_no_remainder_when_categories_sum_to_one(tmp_path):
    categories = [(0.5, "interpreter"), (0.5, "library")]
    fig = profiling_plot.plot_pie_interactive(
        categories, tmp_path / "pie.html", include_plotlyjs="cdn"
    )
    assert "(other functions)" not in list(fig.data[0].labels)


def test_pie_interactive_labels_only_the_largest_slices(tmp_path):
    categories = [(1.0 / 30.0, f"cat{i}") for i in range(30)]
    fig = profiling_plot.plot_pie_interactive(
        categories, tmp_path / "pie.html", include_plotlyjs="cdn"
    )
    labelled = [text for text in fig.data[0].text if text]
    # Only the first PIE_LABELS slices get a drawn label, as in plot_pie.
    assert len(labelled) == profiling_plot.PIE_LABELS


def test_pie_interactive_keeps_category_order(tmp_path):
    fig = profiling_plot.plot_pie_interactive(
        _categories(), tmp_path / "pie.html", include_plotlyjs="cdn"
    )
    assert fig.data[0].sort is False
    assert list(fig.data[0].labels)[:3] == ["interpreter", "library", "gc"]


# ---------------------------------------------------------------------------
# plotly.js delivery
# ---------------------------------------------------------------------------


def test_interactive_plots_default_to_the_shared_plotlyjs_mode(tmp_path, monkeypatch):
    from bench_runner import interactive_plot

    monkeypatch.setattr(interactive_plot, "PLOTLYJS_MODE", "cdn")
    output = tmp_path / "pie.html"
    profiling_plot.plot_pie_interactive(_categories(), output)
    # "cdn" pulls plotly.js from a URL rather than inlining megabytes of it.
    assert "https://" in _html(output)
    assert output.stat().st_size < 1_000_000


def test_inline_plotlyjs_is_self_contained(tmp_path):
    output = tmp_path / "pie.html"
    profiling_plot.plot_pie_interactive(_categories(), output, include_plotlyjs=True)
    # Inlining is what the --plotly-js=inline option buys: a large, offline file.
    assert output.stat().st_size > 1_000_000


def test_bargraph_defaults_to_the_shared_plotlyjs_mode(tmp_path, monkeypatch):
    from bench_runner import interactive_plot

    monkeypatch.setattr(interactive_plot, "PLOTLYJS_MODE", "cdn")
    output = tmp_path / "bar.html"
    profiling_plot.plot_bargraph_interactive(_results(), _categories(), output)
    assert "https://" in _html(output)
    assert output.stat().st_size < 1_000_000
