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


# ---------------------------------------------------------------------------
# Interaction and layout of the violin comparison chart
# ---------------------------------------------------------------------------


def _diff_data(count: int = 12, insignificant_every: int = 5):
    """A CombinedData shaped like a real comparison, sorted by mean."""
    rng = np.random.default_rng(0)
    data = []
    for i in range(count):
        name = f"bm_{i:02d}"
        mean = 1.0 + (i - count // 2) * 0.01
        if insignificant_every and i % insignificant_every == 2:
            data.append((name, None, mean))
        else:
            data.append((name, rng.normal(mean, 0.02, 40), mean))
    data.sort(key=lambda entry: entry[2])
    return data


@pytest.fixture
def diff_figure(tmp_path):
    def build(**kwargs):
        return interactive_plot.plot_diff_interactive(
            _diff_data(**kwargs),
            tmp_path / "diff.html",
            "Timings",
            ("slower", "faster"),
            include_plotlyjs=False,
        )

    return build


def test_the_wheel_is_not_hijacked_for_zoom():
    # These charts are thousands of pixels tall, so the page always scrolls,
    # and the chart covers nearly the full width. A chart that swallows the
    # wheel leaves the reader nowhere to scroll from.
    assert interactive_plot.PLOTLY_CONFIG["scrollZoom"] is False


def test_the_config_reaches_the_generated_html(tmp_path, diff_figure):
    fig = diff_figure()
    interactive_plot.save_html(fig, tmp_path / "out.html", include_plotlyjs=False)

    assert '"scrollZoom": false' in (tmp_path / "out.html").read_text().replace(
        '"scrollZoom":false', '"scrollZoom": false'
    )


def test_every_row_gets_its_own_label(diff_figure):
    # plotly's automatic tick placement thins labels once a categorical axis
    # is dense, and a thinned label no longer names the row beside it.
    fig = diff_figure(count=40)
    yaxis = fig.layout.yaxis

    assert yaxis.tickmode == "array"
    assert list(yaxis.tickvals) == list(yaxis.categoryarray)
    assert list(yaxis.ticktext) == list(yaxis.categoryarray)


def test_labels_cover_the_categories_exactly(diff_figure):
    fig = diff_figure(count=25)
    categories = list(fig.layout.yaxis.categoryarray)

    # Every benchmark, plus the ALL summary, labelled once and only once.
    assert len(categories) == 26
    assert len(set(categories)) == len(categories)
    assert list(fig.layout.yaxis.ticktext) == categories


def test_rows_have_separators(diff_figure):
    fig = diff_figure(count=12)
    rows = len(fig.layout.yaxis.categoryarray)

    separators = [s for s in fig.layout.shapes if s.type == "line" and s.yref == "y"]

    # One boundary between each adjacent pair of rows.
    assert len(separators) == rows - 1
    # Categories sit at integer positions, so a boundary is at i + 0.5.
    assert sorted(s.y0 for s in separators) == [i + 0.5 for i in range(rows - 1)]
    assert all(s.y0 == s.y1 for s in separators)


def test_separators_span_the_plot_and_sit_beneath_the_data(diff_figure):
    fig = diff_figure(count=8)
    separators = [s for s in fig.layout.shapes if s.type == "line" and s.yref == "y"]

    for s in separators:
        assert s.xref == "paper" and (s.x0, s.x1) == (0, 1)
        # Beneath, or a violin filling its row would be sliced by its own
        # boundary.
        assert s.layer == "below"
        assert s.line.width == interactive_plot.ROW_SEPARATOR_WIDTH


def test_every_other_row_separator():
    go, _ = interactive_plot._plotly()
    fig = go.Figure()

    interactive_plot._add_row_separators(fig, 10, every=2)

    assert [s.y0 for s in fig.layout.shapes] == [1.5, 3.5, 5.5, 7.5]


def test_one_row_needs_no_separators():
    go, _ = interactive_plot._plotly()
    fig = go.Figure()
    interactive_plot._add_row_separators(fig, 1)
    assert fig.layout.shapes == ()


def test_points_sit_over_their_own_violin(diff_figure):
    """
    plotly offsets sample points to one side of the violin by default, which
    for a horizontal violin puts them in a strip below the row -- so they read
    as belonging to the row beneath, and the violin they describe sits alone.
    """
    fig = diff_figure()
    violins = [t for t in fig.data if t.type == "violin"]

    assert violins
    for violin in violins:
        assert violin.pointpos == 0
        # And the jitter band stays inside the row: half of 0.3 is well
        # within the 0.9 width.
        assert violin.jitter is not None
        assert violin.jitter / 2 < violin.width


def test_points_are_styled_to_read_over_the_fill(diff_figure):
    fig = diff_figure()
    violin = next(t for t in fig.data if t.type == "violin")

    assert violin.marker.size == 3
    assert 0 < violin.marker.opacity < 1


def test_the_points_button_still_toggles_them(diff_figure):
    # The layout fix must not disturb the control that turns them on.
    fig = diff_figure()
    labels = [
        button.label for menu in fig.layout.updatemenus for button in menu.buttons
    ]

    assert "Points: all" in labels and "Points: off" in labels


# ---------------------------------------------------------------------------
# Re-ordering the comparison chart
# ---------------------------------------------------------------------------


def _ordering_data():
    """
    Benchmarks whose four orderings are all different, so a test that passes
    under one ordering cannot pass by accident under another.
    """
    rng = np.random.default_rng(3)
    spec = [
        # name, mean ratio, spread of the distribution
        ("zlib", 1.20, 0.005),
        ("apple", 0.85, 0.060),
        ("mango", 1.02, 0.002),
        ("banana", 0.98, 0.050),
        ("cherry", 1.10, 0.030),
        ("date", None, 0.0),  # not significant
        ("elder", 0.95, 0.010),
        ("fig", 1.30, 0.080),
    ]
    data = [
        (
            name,
            None if mean is None else np.sort(rng.normal(mean, sd, 200)),
            mean or 0.0,
        )
        for name, mean, sd in spec
    ]
    data.sort(key=lambda entry: entry[2])
    return data


def test_spread_measures_the_width_of_the_distribution():
    narrow = np.sort(np.linspace(0.99, 1.01, 200))
    wide = np.sort(np.linspace(0.80, 1.20, 200))

    assert interactive_plot._spread(narrow) < interactive_plot._spread(wide)
    # The middle 90%, so the extremes do not set the answer on their own.
    assert interactive_plot._spread(wide) < 0.4


def test_spread_of_an_insignificant_result():
    assert interactive_plot._spread(None) == 0.0
    assert interactive_plot._spread(np.array([])) == 0.0


def test_delta_ordering_is_the_input_order():
    # The default view has to be exactly what it was before there was a
    # choice of orderings.
    data = _ordering_data()
    orders = interactive_plot._sort_orders(data)

    assert orders["delta"] == [name for name, _, _ in data]


def test_name_ordering_reads_alphabetically_from_the_top():
    orders = interactive_plot._sort_orders(_ordering_data())

    # Rows run bottom to top, so reading the chart downwards is the reverse.
    assert list(reversed(orders["name"])) == [
        "apple",
        "banana",
        "cherry",
        "date",
        "elder",
        "fig",
        "mango",
        "zlib",
    ]


def test_change_ordering_puts_the_biggest_movers_at_the_top():
    orders = interactive_plot._sort_orders(_ordering_data())
    significant = [n for n in orders["change"] if n != "date"]

    # fig moved 0.30, apple 0.15, zlib 0.20 ... biggest last, i.e. at the top.
    assert significant[-1] == "fig"
    assert significant[-2] == "zlib"
    # And it is a magnitude, so a slowdown ranks with a speedup of the same
    # size: apple (0.85) outranks cherry (1.10).
    assert significant.index("apple") > significant.index("cherry")


def test_spread_ordering_puts_the_noisiest_at_the_top():
    orders = interactive_plot._sort_orders(_ordering_data())
    significant = [n for n in orders["spread"] if n != "date"]

    assert significant[-1] == "fig"  # widest distribution
    assert significant[0] == "mango"  # narrowest


def test_metric_orderings_park_insignificant_results_at_the_bottom():
    # They have no distribution and no meaningful delta, so interleaving them
    # on a value that does not mean anything would be misleading.
    orders = interactive_plot._sort_orders(_ordering_data())

    for key in ("delta", "change", "spread"):
        assert orders[key][0] == "date", key


def test_name_ordering_keeps_insignificant_results_in_the_alphabet():
    # Sorting by name is for finding a benchmark, and it should be where the
    # alphabet says whether or not it moved.
    orders = interactive_plot._sort_orders(_ordering_data())

    assert list(reversed(orders["name"])).index("date") == 3


def test_every_ordering_covers_every_benchmark_exactly_once():
    data = _ordering_data()
    names = {name for name, _, _ in data}

    for key, order in interactive_plot._sort_orders(data).items():
        assert sorted(order) == sorted(names), key
        assert len(set(order)) == len(order), key


@pytest.fixture
def ordered_figure(tmp_path):
    def build(**kwargs):
        return interactive_plot.plot_diff_interactive(
            _ordering_data(),
            tmp_path / "diff.html",
            "Timings",
            ("slower", "faster"),
            include_plotlyjs=False,
            **kwargs,
        )

    return build


def _sort_menu(fig):
    return next(m for m in fig.layout.updatemenus if m.type == "dropdown")


def test_the_chart_offers_every_ordering(ordered_figure):
    menu = _sort_menu(ordered_figure())

    assert [b.label for b in menu.buttons] == [
        label for _, label in interactive_plot.SORT_CHOICES
    ]


def test_re_sorting_happens_in_the_browser_so_it_needs_no_regeneration(ordered_figure):
    # The whole point of doing this client side: the reader re-sorts, nobody
    # re-runs the pipeline.
    menu = _sort_menu(ordered_figure())

    for button in menu.buttons:
        # "update" rather than "relayout": a sort also restores every row, so
        # that sorting a collapsed chart cannot leave blank bands behind.
        assert button.method == "update"
        args = button.args[1]
        # categoryorder is repeated because relayout replaces only what it
        # names, and "array" is what makes categoryarray mean anything.
        assert args["yaxis.categoryorder"] == "array"
        assert args["yaxis.categoryarray"]


def test_each_button_carries_a_different_order(ordered_figure):
    menu = _sort_menu(ordered_figure())
    arrays = [tuple(b.args[1]["yaxis.categoryarray"]) for b in menu.buttons]

    assert len(set(arrays)) == len(arrays)


def test_every_ordering_holds_the_same_rows(ordered_figure):
    fig = ordered_figure()
    rows = set(fig.layout.yaxis.categoryarray)

    for button in _sort_menu(fig).buttons:
        assert set(button.args[1]["yaxis.categoryarray"]) == rows


def test_the_summary_row_stays_pinned_to_the_top(ordered_figure):
    # ALL summarises the others, so it is not sorted among them.
    fig = ordered_figure()

    assert fig.layout.yaxis.categoryarray[-1] == "ALL"
    for button in _sort_menu(fig).buttons:
        assert button.args[1]["yaxis.categoryarray"][-1] == "ALL"


def test_the_chart_opens_on_the_default_ordering(ordered_figure):
    fig = ordered_figure()
    menu = _sort_menu(fig)

    assert menu.active == 0
    assert list(fig.layout.yaxis.categoryarray) == list(
        menu.buttons[0].args[1]["yaxis.categoryarray"]
    )


def test_the_initial_ordering_is_selectable(ordered_figure):
    fig = ordered_figure(order_by="name")
    menu = _sort_menu(fig)

    # The rows are laid out that way, under the pinned ALL summary...
    assert list(fig.layout.yaxis.categoryarray)[-3:] == ["banana", "apple", "ALL"]
    # ...and the closed dropdown says so, rather than claiming the default.
    assert menu.buttons[menu.active].label.endswith("(A–Z)")


def test_an_unknown_ordering_is_refused(ordered_figure):
    with pytest.raises(ValueError, match="Unknown ordering"):
        ordered_figure(order_by="by-vibes")


def test_re_sorting_does_not_disturb_the_row_separators(ordered_figure):
    # The separators are at fixed positions between rows, not attached to
    # particular benchmarks, so they stay right whatever order the rows are in.
    fig = ordered_figure()
    rows = len(fig.layout.yaxis.categoryarray)
    separators = [s for s in fig.layout.shapes if s.type == "line" and s.yref == "y"]

    assert len(separators) == rows - 1
    assert sorted(s.y0 for s in separators) == [i + 0.5 for i in range(rows - 1)]


def test_labels_follow_the_rows_under_any_ordering(ordered_figure):
    # tickvals are category names, so plotly maps each to wherever that
    # category currently sits: the labels move with the rows.
    fig = ordered_figure()

    assert set(fig.layout.yaxis.tickvals) == set(fig.layout.yaxis.categoryarray)


# ---------------------------------------------------------------------------
# Grouping benchmarks into families, and collapsing them
# ---------------------------------------------------------------------------


def test_families_come_from_the_name_prefix():
    groups = interactive_plot._benchmark_groups(
        ["scimark_fft", "scimark_lu", "regex_dna", "regex_v8", "nbody"]
    )

    assert groups["scimark"] == ["scimark_fft", "scimark_lu"]
    assert groups["regex"] == ["regex_dna", "regex_v8"]


def test_a_prefix_of_one_is_not_a_family():
    groups = interactive_plot._benchmark_groups(["nbody", "chaos", "go"])

    assert list(groups) == [interactive_plot.UNGROUPED]
    assert groups[interactive_plot.UNGROUPED] == ["nbody", "chaos", "go"]


def test_every_benchmark_lands_in_exactly_one_family():
    names = ["scimark_fft", "scimark_lu", "regex_dna", "nbody", "json_dumps", "go"]

    groups = interactive_plot._benchmark_groups(names)
    members = [name for group in groups.values() for name in group]

    assert sorted(members) == sorted(names)
    assert len(members) == len(set(members))


def test_ungrouped_sorts_last():
    # The control that lists the families should end with the leftovers.
    groups = interactive_plot._benchmark_groups(
        ["zlib_a", "zlib_b", "alone", "abc_x", "abc_y"]
    )

    assert list(groups) == ["abc", "zlib", interactive_plot.UNGROUPED]


def test_an_override_wins_over_the_prefix():
    # So a repository can curate its own grouping without changing this code.
    groups = interactive_plot._benchmark_groups(
        ["scimark_fft", "scimark_lu", "nbody", "float"],
        {"nbody": "math", "float": "math", "scimark_fft": "math"},
    )

    assert groups["math"] == ["scimark_fft", "nbody", "float"]
    # scimark_lu is on its own now, so it is no longer a family.
    assert groups[interactive_plot.UNGROUPED] == ["scimark_lu"]


def _grouped_data():
    rng = np.random.default_rng(5)
    names = [
        "regex_compile",
        "regex_dna",  # insignificant, inside a family
        "regex_v8",
        "scimark_fft",
        "scimark_lu",
        "json_dumps",
        "json_loads",
        "nbody",
        "chaos",  # insignificant, outside any family
    ]
    data = []
    for i, name in enumerate(names):
        if name in ("regex_dna", "chaos"):
            data.append((name, None, 0.0))
        else:
            mean = 1.0 + i * 0.02
            data.append((name, np.sort(rng.normal(mean, 0.02, 80)), mean))
    data.sort(key=lambda entry: entry[2])
    return data


@pytest.fixture
def grouped_figure(tmp_path):
    def build(**kwargs):
        return interactive_plot.plot_diff_interactive(
            _grouped_data(),
            tmp_path / "diff.html",
            "Timings",
            ("slower", "faster"),
            include_plotlyjs=False,
            **kwargs,
        )

    return build


def _menus(fig):
    dropdowns = [m for m in fig.layout.updatemenus if m.type == "dropdown"]
    return dropdowns[0], dropdowns[1]  # sort, collapse


def test_the_legend_lists_families_not_benchmarks(grouped_figure):
    """
    One legend entry per benchmark made the legend a second list of every row,
    differently spaced, running down the side of the chart -- which is what
    made the names look out of step with the rows.
    """
    fig = grouped_figure()
    entries = [t.name for t in fig.data if t.showlegend]

    assert entries == ["json", "regex", "scimark", interactive_plot.UNGROUPED]
    # Every violin is off the legend, and in a family.
    for trace in fig.data:
        if trace.type == "violin":
            assert trace.showlegend is False
            assert trace.legendgroup


def test_clicking_a_family_in_the_legend_toggles_the_whole_family(grouped_figure):
    fig = grouped_figure()
    assert fig.layout.legend.groupclick == "togglegroup"


def test_the_legend_proxies_draw_nothing(grouped_figure):
    fig = grouped_figure()
    proxies = [t for t in fig.data if t.showlegend]

    for proxy in proxies:
        assert list(proxy.x) == [None] and list(proxy.y) == [None]
        assert proxy.hoverinfo == "skip"


def test_collapsing_to_a_family_keeps_only_its_rows(grouped_figure):
    fig = grouped_figure()
    _, collapse = _menus(fig)

    only_regex = next(b for b in collapse.buttons if b.label.startswith("Only: regex"))
    rows = list(only_regex.args[1]["yaxis.categoryarray"])

    assert set(rows) == {"regex_compile", "regex_dna", "regex_v8", "ALL"}


def test_collapsing_keeps_an_insignificant_member_of_the_family(grouped_figure):
    # regex_dna has no distribution, but it is still a regex benchmark.
    fig = grouped_figure()
    _, collapse = _menus(fig)

    only_regex = next(b for b in collapse.buttons if b.label.startswith("Only: regex"))

    assert "regex_dna" in only_regex.args[1]["yaxis.categoryarray"]


def test_collapsing_keeps_the_summary_row(grouped_figure):
    fig = grouped_figure()
    _, collapse = _menus(fig)

    for button in collapse.buttons:
        assert "ALL" in button.args[1]["yaxis.categoryarray"]


def test_expanding_restores_every_row(grouped_figure):
    fig = grouped_figure()
    _, collapse = _menus(fig)

    show_all = collapse.buttons[0]
    assert show_all.label == "Show: every family"
    assert list(show_all.args[1]["yaxis.categoryarray"]) == list(
        fig.layout.yaxis.categoryarray
    )
    assert all(show_all.args[0]["visible"])


def test_the_chart_opens_expanded(grouped_figure):
    _, collapse = _menus(grouped_figure())
    assert collapse.active == 0


def test_a_family_of_one_gets_no_collapse_entry(grouped_figure):
    # Collapsing to a single row is not worth a menu entry.
    fig = grouped_figure()
    _, collapse = _menus(fig)
    families = interactive_plot._benchmark_groups(
        [name for name, _, _ in _grouped_data()]
    )

    offered = {
        b.label.split(": ", 1)[1].rsplit(" (", 1)[0] for b in collapse.buttons[1:]
    }
    assert offered == {name for name, m in families.items() if len(m) > 1}


def _rows_drawn(fig, button):
    """Every category some visible trace actually puts a mark in."""
    drawn = set()
    for index, visible in enumerate(button.args[0]["visible"]):
        if not visible:
            continue
        for y in fig.data[index].y or ():
            if y:
                drawn.add(y)
    return drawn


def test_no_state_leaves_a_blank_row_or_a_homeless_mark(grouped_figure):
    """
    The invariant that makes collapse safe: in every state the axis offers, the
    set of rows on the axis is exactly the set of rows something is drawn in.
    A row on the axis with nothing in it is a blank band; a mark whose row is
    not on the axis is a point plotly has to put somewhere it does not belong.
    """
    fig = grouped_figure()

    for menu in _menus(fig):
        for button in menu.buttons:
            rows = set(button.args[1]["yaxis.categoryarray"])
            assert rows == _rows_drawn(fig, button), button.label


def test_sorting_restores_a_collapsed_chart(grouped_figure):
    # Each control sets a whole view. Sorting means "all of it, in this order",
    # so it cannot leave a collapsed family's categories on the axis with
    # nothing drawn in them.
    fig = grouped_figure()
    sort, _ = _menus(fig)

    for button in sort.buttons:
        assert all(button.args[0]["visible"])
        assert button.method == "update"


def test_collapsing_preserves_the_charts_ordering(grouped_figure):
    # Within a family the rows keep the order the chart was built with, rather
    # than reverting to something of the collapse control's own.
    fig = grouped_figure(order_by="name")
    _, collapse = _menus(fig)

    only_regex = next(b for b in collapse.buttons if b.label.startswith("Only: regex"))
    rows = [r for r in only_regex.args[1]["yaxis.categoryarray"] if r != "ALL"]
    full = [r for r in fig.layout.yaxis.categoryarray if r.startswith("regex")]

    assert rows == full


def test_grouping_does_not_disturb_the_row_separators(grouped_figure):
    fig = grouped_figure()
    rows = len(fig.layout.yaxis.categoryarray)
    separators = [s for s in fig.layout.shapes if s.type == "line" and s.yref == "y"]

    assert len(separators) == rows - 1
