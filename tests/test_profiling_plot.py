"""
Tests for `bench_runner.scripts.profiling_plot`, which turns Linux `perf`
output into summary tables and charts.
"""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path


import matplotlib
import pytest


matplotlib.use("agg")


from bench_runner.scripts import profiling_plot  # noqa: E402


# The columns `perf script` is post-processed into, in order.
CSV_HEADER = ["self_time", "pid", "command", "obj", "sym"]


def _write_csv(path, rows, header=CSV_HEADER):
    with Path(path).open("w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(header)
        writer.writerows(rows)
    return Path(path)


def _collectors():
    results = defaultdict(lambda: defaultdict(float))
    categories = defaultdict(lambda: defaultdict(float))
    return results, categories


# ---------------------------------------------------------------------------
# get_color_and_hatch
# ---------------------------------------------------------------------------


def test_color_and_hatch_unknown_category_is_grey():
    assert profiling_plot.get_color_and_hatch("no-such-category") == ("#ddd", "")


def test_color_and_hatch_uses_the_matplotlib_cycle():
    color, hatch = profiling_plot.get_color_and_hatch(profiling_plot.COLOR_ORDER[0])
    assert color == "C0"
    assert hatch == ""


def test_color_and_hatch_wraps_colors_and_changes_hatch_after_ten():
    # Only ten cycle colors exist, so past the tenth category the hatch is what
    # keeps them distinguishable.
    order = profiling_plot.COLOR_ORDER
    if len(order) <= 10:
        pytest.skip("fewer than 11 categories defined")
    color, hatch = profiling_plot.get_color_and_hatch(order[10])
    assert color == "C0"
    assert hatch == "//"


def test_color_and_hatch_is_unique_per_category():
    # Every distinct category must be visually distinguishable in the chart.
    names = set(profiling_plot.COLOR_ORDER)
    pairs = [profiling_plot.get_color_and_hatch(c) for c in names]
    assert len(set(pairs)) == len(pairs)


def test_color_order_lists_library_twice():
    # COLOR_ORDER is ["jit", "kernel", "libc", "library"] + CATEGORIES keys, and
    # CATEGORIES also defines "library", so the name appears at index 3 and 9.
    # Harmless -- .index() always resolves to the first -- but it means the list
    # holds 24 slots for 23 categories, so one colour/hatch pair goes unused.
    # Pinned here so that changing it is a deliberate act.
    order = profiling_plot.COLOR_ORDER
    assert order.count("library") == 2
    assert profiling_plot.get_color_and_hatch("library") == ("C3", "")


# ---------------------------------------------------------------------------
# category_for_obj_sym
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj,sym,expected",
    [
        ("[kernel.kallsyms]", "anything", "kernel"),
        ("libc.so.6", "malloc", "libc"),
        ("libc-2.31.so", "free", "libc"),
        ("[JIT]", "whatever", "jit"),
        ("libssl.so", "SSL_read", "library"),
        ("libfoo.so.1.2", "bar", "library"),
        ("some-random-binary", "sym", "unknown"),
    ],
)
def test_category_for_obj_sym_by_object(obj, sym, expected):
    assert profiling_plot.category_for_obj_sym(obj, sym) == expected


def test_category_for_obj_sym_kernel_wins_over_everything():
    # The kernel object is checked first, before the .so pattern.
    assert profiling_plot.category_for_obj_sym("[kernel.kallsyms]", "x") == "kernel"


def test_category_for_obj_sym_matches_interpreter_symbols():
    assert profiling_plot.category_for_obj_sym("python", "_PyEval_EvalFrame") == (
        "interpreter"
    )
    assert profiling_plot.category_for_obj_sym("python", "PyEval_EvalCode") == (
        "interpreter"
    )


def test_category_for_obj_sym_matches_lookup_symbols():
    assert profiling_plot.category_for_obj_sym("python", "_PyType_Lookup") == "lookup"


def test_category_for_obj_sym_unknown_python_symbol():
    assert profiling_plot.category_for_obj_sym("python", "totally_made_up") == "unknown"


def test_category_for_obj_sym_strips_symbol_suffixes():
    # Symbols arrive with trailing detail ("sym.cold", "sym +0x10"); only the
    # leading name is matched against the category patterns.
    assert profiling_plot.category_for_obj_sym("python", "_PyEval_Foo.cold") == (
        "interpreter"
    )
    assert profiling_plot.category_for_obj_sym("python", "_PyEval_Foo +0x10") == (
        "interpreter"
    )


def test_category_for_obj_sym_anchors_the_pattern():
    # Patterns are anchored, so a symbol that merely contains a category name
    # must not match.
    assert profiling_plot.category_for_obj_sym("python", "not_PyEval_Foo") == "unknown"


def test_category_for_obj_sym_only_categorises_the_python_binary():
    # The same symbol in a non-python object is not an interpreter symbol.
    assert profiling_plot.category_for_obj_sym("otherbin", "_PyEval_EvalFrame") == (
        "unknown"
    )


# ---------------------------------------------------------------------------
# handle_benchmark
# ---------------------------------------------------------------------------


def test_handle_benchmark_totals_self_time(tmp_path):
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["50", "1", "python", "python", "_PyType_Lookup"],
        ],
    )
    results, categories = _collectors()
    total = profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert total == pytest.approx(150.0)


def test_handle_benchmark_groups_by_category(tmp_path):
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["50", "1", "python", "python", "_PyType_Lookup"],
        ],
    )
    results, categories = _collectors()
    profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert results["nbody"]["interpreter"] == pytest.approx(100.0)
    assert results["nbody"]["lookup"] == pytest.approx(50.0)


def test_handle_benchmark_names_the_benchmark_from_the_filename(tmp_path):
    path = _write_csv(
        tmp_path / "chaos.perf.csv",
        [["10", "1", "python", "python", "_PyEval_EvalFrame"]],
    )
    results, categories = _collectors()
    md = io.StringIO()
    profiling_plot.handle_benchmark(path, md, results, categories)
    # The stem is truncated at the first dot.
    assert "chaos" in results
    assert "## chaos\n" in md.getvalue()


def test_handle_benchmark_skips_non_python_commands(tmp_path):
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["999", "2", "gcc", "cc1", "compile"],
        ],
    )
    results, categories = _collectors()
    total = profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert total == pytest.approx(100.0)


def test_handle_benchmark_skips_orchestrating_python_by_pid(tmp_path):
    # A pid whose samples reference a pythonX.XX shared object belongs to the
    # orchestrating interpreter, not the one under benchmark, so every sample
    # from that pid is dropped.
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["7", "2", "python", "python3.12", "_PyEval_EvalFrame"],
            ["500", "2", "python", "python", "_PyEval_EvalFrame"],
        ],
    )
    results, categories = _collectors()
    total = profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    # Only pid 1 survives; both pid 2 rows go, including the untainted one.
    assert total == pytest.approx(100.0)


def test_handle_benchmark_taints_on_libpython_shared_object(tmp_path):
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["7", "2", "python", "libpython3.12.so", "_PyEval_EvalFrame"],
            ["500", "2", "python", "python", "_PyEval_EvalFrame"],
        ],
    )
    results, categories = _collectors()
    total = profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert total == pytest.approx(100.0)


def test_handle_benchmark_merges_all_jit_samples(tmp_path):
    # JIT frames have meaningless symbol names, so they collapse into one row.
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["30", "1", "python", "[JIT]", "anonymous_block_1"],
            ["20", "1", "python", "[JIT]", "anonymous_block_2"],
        ],
    )
    results, categories = _collectors()
    total = profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert total == pytest.approx(50.0)
    assert categories["jit"][("[JIT]", "jit")] == pytest.approx(50.0)


def test_handle_benchmark_writes_a_markdown_table(tmp_path):
    path = _write_csv(
        tmp_path / "nbody.csv",
        [["100", "1", "python", "python", "_PyEval_EvalFrame"]],
    )
    results, categories = _collectors()
    md = io.StringIO()
    profiling_plot.handle_benchmark(path, md, results, categories)
    content = md.getvalue()
    assert "| percentage | object | symbol | category |" in content
    assert "100.00%" in content
    assert "interpreter" in content


def test_handle_benchmark_omits_rows_below_the_threshold(tmp_path):
    # Rows under 0.25% are aggregated but not listed, to keep the table short.
    rows = [["10000", "1", "python", "python", "_PyEval_EvalFrame"]]
    rows.append(["1", "1", "python", "python", "_PyType_Lookup"])
    path = _write_csv(tmp_path / "nbody.csv", rows)
    results, categories = _collectors()
    md = io.StringIO()
    profiling_plot.handle_benchmark(path, md, results, categories)
    content = md.getvalue()
    assert "_PyEval_EvalFrame" in content
    assert "_PyType_Lookup" not in content
    # Still counted in the totals, just not printed.
    assert results["nbody"]["lookup"] == pytest.approx(1.0)


def test_handle_benchmark_accumulates_across_calls(tmp_path):
    results, categories = _collectors()
    for name in ("a", "b"):
        path = _write_csv(
            tmp_path / f"{name}.csv",
            [["10", "1", "python", "python", "_PyEval_EvalFrame"]],
        )
        profiling_plot.handle_benchmark(path, io.StringIO(), results, categories)
    assert set(results) == {"a", "b"}
    assert categories["interpreter"][("python", "_PyEval_EvalFrame")] == (
        pytest.approx(20.0)
    )


# ---------------------------------------------------------------------------
# plot_bargraph / plot_pie
# ---------------------------------------------------------------------------


def _plot_inputs():
    results = defaultdict(lambda: defaultdict(float))
    for benchmark in ("nbody", "chaos"):
        results[benchmark]["interpreter"] = 0.6
        results[benchmark]["lookup"] = 0.4
    categories = [(0.6, "interpreter"), (0.4, "lookup")]
    return results, categories


def test_plot_bargraph_writes_an_svg(tmp_path):
    results, categories = _plot_inputs()
    output = tmp_path / "out.svg"
    profiling_plot.plot_bargraph(results, categories, output)
    assert output.is_file()
    assert output.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_plot_bargraph_ignores_the_unknown_category(tmp_path):
    results, categories = _plot_inputs()
    output = tmp_path / "out.svg"
    profiling_plot.plot_bargraph(results, categories + [(0.1, "unknown")], output)
    # "unknown" is deliberately not drawn as its own band.
    assert "unknown" not in output.read_text(encoding="utf-8")


def test_plot_pie_writes_a_png_sized_file(tmp_path):
    _, categories = _plot_inputs()
    output = tmp_path / "out.pie.svg"
    profiling_plot.plot_pie(categories, output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_pie_handles_categories_summing_over_one(tmp_path):
    # Absolute times rather than fractions must not produce a negative slice.
    output = tmp_path / "out.pie.svg"
    profiling_plot.plot_pie([(100.0, "interpreter"), (50.0, "lookup")], output)
    assert output.is_file()


# ---------------------------------------------------------------------------
# handle_tail_call_stats
# ---------------------------------------------------------------------------


def test_tail_call_stats_no_tail_call_symbols_writes_nothing(tmp_path):
    categories = defaultdict(lambda: defaultdict(float))
    categories["interpreter"][("python", "_PyEval_EvalFrame")] = 10.0
    profiling_plot.handle_tail_call_stats(tmp_path, categories, tmp_path / "out")
    assert not (tmp_path / "out.tail_calls.csv").exists()


def test_tail_call_stats_without_pystats_is_skipped(tmp_path, capsys):
    categories = defaultdict(lambda: defaultdict(float))
    categories["interpreter"][("python", "_TAIL_CALL_LOAD_FAST")] = 10.0
    profiling_plot.handle_tail_call_stats(tmp_path, categories, tmp_path / "out")
    assert not (tmp_path / "out.tail_calls.csv").exists()
    assert "No pystats.json" in capsys.readouterr().out


def test_tail_call_stats_writes_csv(tmp_path):
    categories = defaultdict(lambda: defaultdict(float))
    categories["interpreter"][("python", "_TAIL_CALL_LOAD_FAST")] = 300.0
    categories["interpreter"][("python", "_TAIL_CALL_STORE_FAST")] = 100.0
    (tmp_path / "pystats.json").write_text(
        json.dumps(
            {
                "opcode[LOAD_FAST].execution_count": 1000,
                "opcode[STORE_FAST].execution_count": 500,
                "unrelated": 1,
            }
        )
    )

    profiling_plot.handle_tail_call_stats(tmp_path, categories, tmp_path / "out")

    with (tmp_path / "out.tail_calls.csv").open(newline="") as fd:
        rows = list(csv.reader(fd))

    assert rows[0] == [
        "Bytecode",
        "% time",
        "count",
        "% count",
        "time per count (μs)",
    ]
    # Sorted by time spent, descending.
    assert rows[1][0] == "LOAD_FAST"
    assert rows[2][0] == "STORE_FAST"
    assert rows[1][1] == "75.00%"
    assert rows[1][2] == "1000"


def test_tail_call_stats_skips_bytecodes_with_no_recorded_count(tmp_path):
    categories = defaultdict(lambda: defaultdict(float))
    categories["interpreter"][("python", "_TAIL_CALL_LOAD_FAST")] = 300.0
    categories["interpreter"][("python", "_TAIL_CALL_NEVER_RUN")] = 100.0
    (tmp_path / "pystats.json").write_text(
        json.dumps({"opcode[LOAD_FAST].execution_count": 1000})
    )

    profiling_plot.handle_tail_call_stats(tmp_path, categories, tmp_path / "out")

    with (tmp_path / "out.tail_calls.csv").open(newline="") as fd:
        rows = list(csv.reader(fd))
    assert [r[0] for r in rows[1:]] == ["LOAD_FAST"]


# ---------------------------------------------------------------------------
# _main
# ---------------------------------------------------------------------------


def test_main_with_missing_directory_is_a_no_op(tmp_path, capsys):
    profiling_plot._main(tmp_path / "nope", tmp_path / "out")
    assert "No profiling data" in capsys.readouterr().out
    assert not (tmp_path / "out.md").exists()


def test_main_with_no_csv_files_is_a_no_op(tmp_path, capsys):
    (tmp_path / "input").mkdir()
    profiling_plot._main(tmp_path / "input", tmp_path / "out")
    assert "No profiling data" in capsys.readouterr().out


def test_main_writes_report_and_charts(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_csv(
        input_dir / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["50", "1", "python", "python", "_PyType_Lookup"],
        ],
    )
    output = tmp_path / "out"

    profiling_plot._main(input_dir, output)

    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".svg").is_file()
    assert output.with_suffix(".pie.svg").is_file()
    content = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "## nbody" in content
    assert "## Categories" in content
    assert "### interpreter" in content


def test_main_ignores_previously_generated_tail_call_csv(tmp_path):
    # Its own .tail_calls.csv output must not be read back in as input.
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_csv(
        input_dir / "nbody.csv",
        [["100", "1", "python", "python", "_PyEval_EvalFrame"]],
    )
    (input_dir / "out.tail_calls.csv").write_text("Bytecode,% time\nX,1%\n")

    profiling_plot._main(input_dir, tmp_path / "out")

    content = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert "## out" not in content


def test_handle_benchmark_stops_at_non_positive_self_time(tmp_path):
    # Rows are walked in descending order, so the first non-positive sample
    # ends the table.
    path = _write_csv(
        tmp_path / "nbody.csv",
        [
            ["100", "1", "python", "python", "_PyEval_EvalFrame"],
            ["0", "1", "python", "python", "_PyType_Lookup"],
        ],
    )
    results, categories = _collectors()
    md = io.StringIO()
    total = profiling_plot.handle_benchmark(path, md, results, categories)
    assert total == pytest.approx(100.0)
    assert "_PyType_Lookup" not in md.getvalue()
    # The zero-time row is never categorised at all.
    assert "lookup" not in results["nbody"]


def test_plot_pie_adds_a_remainder_slice_when_under_one(tmp_path):
    # Fractions that do not account for all the time leave a grey remainder.
    output = tmp_path / "out.pie.svg"
    profiling_plot.plot_pie([(0.3, "interpreter"), (0.2, "lookup")], output)
    assert output.is_file()


def test_main_truncates_the_category_table_at_tiny_fractions(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_csv(
        input_dir / "nbody.csv",
        [
            ["1000000", "1", "python", "python", "_PyEval_EvalFrame"],
            ["1", "1", "python", "python", "_Py_dict_lookup"],
        ],
    )
    profiling_plot._main(input_dir, tmp_path / "out")

    content = (tmp_path / "out.md").read_text(encoding="utf-8")
    # Below 0.0025% of total, so it is counted but not listed.
    assert "_PyEval_EvalFrame" in content
    assert "_Py_dict_lookup" not in content


def test_main_entry_point_parses_arguments(tmp_path, monkeypatch):
    import sys

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_csv(
        input_dir / "nbody.csv",
        [["100", "1", "python", "python", "_PyEval_EvalFrame"]],
    )
    output = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", ["profiling_plot", str(input_dir), str(output)])
    profiling_plot.main()

    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".svg").is_file()


# ---------------------------------------------------------------------------
# Interactive (plotly) charts
# ---------------------------------------------------------------------------


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
