"""
Interactive (plotly) renderings of the benchmark plots.

This is the counterpart to `bench_runner.plot`: the same inputs and the same
`bench_runner.toml` configuration, rendered to an interactive .html file rather
than an .svg. The two modules deliberately share the same `.json` value caches,
so generating both formats from the same data doesn't recompute any comparison.

Styling (`to_css_color`, `get_trace_style`, and the matplotlib-to-plotly
translation tables) lives in `bench_runner.plot` and is used from there, so that
a runner's `PlotConfig` has a single interpretation across both renderers.
"""

from __future__ import annotations


from collections import defaultdict
import datetime
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence


import numpy as np


from . import config as mconfig

# `plot` is bound as a module rather than `from .plot import ...` on purpose:
# `plot` imports `result`, which imports this module, so at that moment `plot`
# is only partially initialised and its names do not exist yet. Binding the
# module and resolving attributes at call time sidesteps that, and is the same
# pattern `plot` itself uses for `result`.
from . import plot as mplot
from . import result
from . import runners as mrunners
from .util import PathLike


# The number of samples in the trailing window used by the rolling average
# overlays.
ROLLING_WINDOW = 5

# The plotly template used by all of the interactive plots.
PLOTLY_TEMPLATE = "plotly_white"

# How plotly.js is delivered for the .html files written by the generation
# pipeline.
#
# The results repository keeps one comparison plot per result per base, forever
# (or at least until `purge` runs), so inlining the ~3.6MB library into every
# one of them would add gigabytes to the repository. The pipeline therefore
# links the library from a CDN, which costs about 12KB per file but needs
# network access to view. Set this to True for self-contained files that work
# offline, or to "directory" to share a single plotly.min.js per directory.
PLOTLYJS_MODE: bool | str = "cdn"

# Options passed to plotly.js itself, as opposed to the figure.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


def _plotly():
    """
    Import plotly.

    This is deliberately lazy. `bench_runner.plot` is imported (by way of
    `result` and `config`) by every entry point, including the ones that run on
    the benchmarking machines, and none of those need plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


def save_html(
    fig,
    output_filename: PathLike,
    *,
    include_plotlyjs: bool | str | None = None,
    config: dict | None = None,
) -> None:
    """
    Write a plotly figure to an .html file. This is the interactive
    counterpart to `savefig`.

    How plotly.js is delivered defaults to `PLOTLYJS_MODE`, i.e. a CDN link.
    Pass `include_plotlyjs` explicitly to override it for one file:

    - `"cdn"`: link the library from a CDN. ~35KB per file, needs network
      access to view.
    - `True`: inline the whole library. Self-contained and works offline and
      from a `file://` URL, at roughly 3MB per file.
    - `"directory"`: reference a single `plotly.min.js` shared by every .html
      file in the same directory. That file has to be written out separately.
    - `False`: emit no script tag at all.

    `None` means "use `PLOTLYJS_MODE`", and is resolved here rather than in the
    signature so that changing the module-level setting at runtime takes
    effect.
    """
    output_filename = Path(output_filename)

    if include_plotlyjs is None:
        include_plotlyjs = PLOTLYJS_MODE

    plotly_config = dict(PLOTLY_CONFIG)
    plotly_config["toImageButtonOptions"] = {
        "format": "svg",
        "filename": output_filename.stem,
    }
    if config is not None:
        plotly_config.update(config)

    fig.write_html(
        output_filename,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config=plotly_config,
        auto_open=False,
    )


def _load_plot_cache(output_filename: Path) -> dict:
    cache_filename = output_filename.with_suffix(".json")
    if cache_filename.is_file():
        with cache_filename.open() as fd:
            return json.load(fd)
    return {}


def _save_plot_cache(output_filename: Path, data: dict) -> None:
    with output_filename.with_suffix(".json").open("w") as fd:
        json.dump(data, fd, indent=2)


def _ratio_axis() -> dict:
    """
    Axis options for an axis holding a "1.05×"-style ratio.
    """
    return {"tickformat": ".2f", "ticksuffix": "×", "zeroline": False}


def _direction_title(differences: tuple[str, str]) -> str:
    return f"⟵ {differences[0]}          {differences[1]} ⟶"


def _clamped_range(
    values: Sequence[float] | np.ndarray, low: float = 0.9, high: float = 1.3
) -> list[float] | None:
    """
    Match the matplotlib plots' behaviour of not zooming in any further than a
    fixed range. Returns None when plotly should autoscale instead.
    """
    if not len(values):
        return None
    if min(values) > low and max(values) < high:
        return [low, high]
    return None


def _vertical_spacing(rows: int, default: float = 0.06) -> float:
    """
    plotly requires the spacing between subplots to be no more than
    1 / (rows - 1).
    """
    if rows <= 1:
        return 0.0
    return min(default, (1.0 / (rows - 1)) * 0.5)


def _subsample(values: np.ndarray, count: int) -> np.ndarray:
    """
    Take at most `count` evenly spaced samples from `values`, which keeps the
    generated HTML to a sane size. The values are already sorted by
    `Comparison._get_combined_data`, so this preserves the distribution.
    """
    if len(values) <= count:
        return values
    idx = np.round(np.linspace(0, len(values) - 1, count)).astype(int)
    return values[idx]


def rolling_stats(
    values: Sequence[float | None], window: int = ROLLING_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trailing rolling mean and standard deviation, ignoring missing values.

    Both arrays are the same length as `values`, and hold NaN wherever the
    window contained no data at all (plotly draws NaN as a gap).
    """
    data = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    means = np.full(len(data), np.nan)
    stds = np.full(len(data), np.nan)
    for i in range(len(data)):
        chunk = data[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            means[i] = chunk.mean()
            stds[i] = chunk.std()
    return means, stds


def add_rolling_average(
    fig,
    dates: Sequence[datetime.datetime],
    values: Sequence[float | None],
    style: mplot.TraceStyle,
    *,
    window: int = ROLLING_WINDOW,
    row: int | None = None,
    col: int | None = None,
) -> list[int]:
    """
    Add a trailing rolling average line, and a ±1 standard deviation band
    around it, for an already-plotted series.

    The band is drawn as a single closed polygon rather than a pair of filled
    traces, so that it doesn't depend on the order the traces were added in.
    Both traces share the series' legend group, so clicking the series in the
    legend toggles them with it.

    Returns the indices of the traces that were added.
    """
    go, _ = _plotly()

    means, stds = rolling_stats(values, window)
    if np.all(np.isnan(means)):
        return []

    where = {} if row is None else {"row": row, "col": col}
    indices = []

    valid = ~np.isnan(means)
    if valid.any():
        upper = (means + stds)[valid]
        lower = (means - stds)[valid]
        xs = [dates[int(i)] for i in np.nonzero(valid)[0]]
        fig.add_trace(
            go.Scatter(
                x=[*xs, *xs[::-1]],
                y=[*upper, *lower[::-1]],
                fill="toself",
                fillcolor=mplot.to_css_color(style.color, 0.13),
                line={"width": 0},
                mode="lines",
                hoverinfo="skip",
                legendgroup=style.name,
                showlegend=False,
                name=f"{style.name} ±1σ",
            ),
            **where,
        )
        indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=means,
            mode="lines",
            line={"color": style.color, "width": 2, "dash": "dot"},
            legendgroup=style.name,
            showlegend=False,
            name=f"{style.name} ({window}-point average)",
            hovertemplate=(
                f"<b>{style.name}</b><br>{window}-point average: "
                "%{y:.3f}×<extra></extra>"
            ),
        ),
        **where,
    )
    indices.append(len(fig.data) - 1)

    return indices


def add_interactive_controls(
    fig,
    *,
    line_traces: Sequence[int] = (),
    rolling_traces: Sequence[int] = (),
) -> None:
    """
    Add the button groups that control line smoothing and the visibility of
    the rolling average overlays.

    Zoom, pan and toggling individual series from the legend are all built in
    to plotly and don't need any setup here.
    """
    menus = []

    if len(line_traces):
        menus.append(
            {
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"b": 6},
                "buttons": [
                    {
                        "label": "Straight lines",
                        "method": "restyle",
                        "args": [{"line.shape": "linear"}, list(line_traces)],
                    },
                    {
                        "label": "Smoothed",
                        "method": "restyle",
                        "args": [{"line.shape": "spline"}, list(line_traces)],
                    },
                ],
            }
        )

    if len(rolling_traces):
        menus.append(
            {
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.25,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"b": 6},
                "buttons": [
                    {
                        "label": "Rolling average: on",
                        "method": "restyle",
                        "args": [{"visible": True}, list(rolling_traces)],
                    },
                    {
                        "label": "Rolling average: off",
                        "method": "restyle",
                        "args": [{"visible": False}, list(rolling_traces)],
                    },
                ],
            }
        )

    if menus:
        fig.update_layout(updatemenus=menus)


def plot_diff_interactive(
    combined_data: result.CombinedData,
    output_filename: PathLike,
    title: str,
    differences: tuple[str, str],
    *,
    max_points: int = 200,
    include_plotlyjs: bool | str | None = None,
):
    """
    The interactive version of `plot_diff`: one violin per benchmark showing
    the distribution of the ratio between the two runs, plus an "ALL" summary.

    Each benchmark is a separate trace, so it can be isolated or hidden from
    the legend, and buttons are provided to show and hide the inner box plot
    and the underlying samples.
    """
    go, _ = _plotly()

    fig = go.Figure()
    # Kept as arrays and concatenated at the end. `.tolist()` on each one would
    # box every element into a float object, which for a whole comparison is
    # tens of megabytes that are then immediately subsampled away.
    parts: list[np.ndarray] = []
    all_values = np.empty(0, dtype=np.float64)
    insignificant: list[str] = []
    violins: list[int] = []

    def add_violin(name: str, values: np.ndarray, color: str, mean: float) -> None:
        fig.add_trace(
            go.Violin(
                x=_subsample(values, max_points),
                y=[name] * min(len(values), max_points),
                name=name,
                orientation="h",
                box_visible=True,
                meanline_visible=True,
                points=False,
                width=0.9,
                spanmode="hard",
                line={"width": 1, "color": mplot.to_css_color(color)},
                fillcolor=mplot.to_css_color(color, 0.5),
                legendgroup=name,
                hovertemplate=(
                    f"<b>{name}</b><br>mean {mean:.4f}×<br>%{{x:.4f}}×<extra></extra>"
                ),
            )
        )
        violins.append(len(fig.data) - 1)

    for name, values, mean in combined_data:
        if values is None:
            insignificant.append(name)
            continue
        parts.append(values)
        color = "red" if name in mplot.INTERPRETER_HEAVY else "C0"
        add_violin(name, values, color, mean)

    if len(parts):
        all_values = np.concatenate(parts)
        all_values.sort()
        add_violin("ALL", all_values, "C2", float(all_values.mean()))

    if len(insignificant):
        fig.add_trace(
            go.Scatter(
                x=[1.0] * len(insignificant),
                y=insignificant,
                mode="markers",
                name="insignificant",
                marker={"symbol": "line-ns-open", "size": 10, "color": "#888"},
                hovertemplate="<b>%{y}</b><br>insignificant<extra></extra>",
            )
        )

    # The violins are added in `combined_data` order, but the insignificant
    # results are collected into a single trace at the end, so the category
    # order has to be pinned explicitly to keep everything sorted by mean.
    order = [name for name, _, _ in combined_data]
    if len(all_values):
        order.append("ALL")

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=240 + 26 * (len(order) + 1),
        margin={"l": 220, "r": 40, "t": 100, "b": 60},
        violinmode="overlay",
        hovermode="closest",
        legend={"traceorder": "reversed"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"b": 6},
                "buttons": [
                    {
                        "label": "Box: on",
                        "method": "restyle",
                        "args": [{"box.visible": True}, violins],
                    },
                    {
                        "label": "Box: off",
                        "method": "restyle",
                        "args": [{"box.visible": False}, violins],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.18,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"b": 6},
                "buttons": [
                    {
                        "label": "Points: off",
                        "method": "restyle",
                        "args": [{"points": False}, violins],
                    },
                    {
                        "label": "Points: all",
                        "method": "restyle",
                        "args": [{"points": "all"}, violins],
                    },
                ],
            },
        ],
    )
    fig.update_xaxes(
        title_text=_direction_title(differences),
        range=_clamped_range(all_values, 0.75, 1.25),
        **_ratio_axis(),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order, automargin=True)
    fig.add_vline(x=1.0, line_width=1, line_color="#444")

    save_html(fig, output_filename, include_plotlyjs=include_plotlyjs)

    return fig


def _annotate_versions(
    fig,
    results: Sequence[result.Result],
    dates: Sequence[datetime.datetime],
    changes: Sequence[float | None],
    row: int,
) -> None:
    """
    Label the first appearance of each released micro version, the same way
    `longitudinal_plot` does.
    """
    annotations = set()
    for r, date, change in zip(results, dates, changes):
        if change is None:
            continue
        micro = mplot.get_micro_version(r.version)
        if micro not in annotations and not r.version.endswith("+"):
            annotations.add(micro)
            fig.add_annotation(
                x=date,
                y=change,
                text=micro,
                textangle=-90,
                font={"size": 9, "color": "#888"},
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor="#888",
                ax=0,
                ay=-26,
                row=row,
                col=1,
            )


def longitudinal_plot_interactive(
    results: Iterable[result.Result],
    output_filename: PathLike,
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.geometric_mean_float,
    differences: tuple[str, str] = ("slower", "faster"),
    title="Performance improvement by configuration",
    *,
    rolling_window: int = ROLLING_WINDOW,
    show_rolling: bool = True,
    include_plotlyjs: bool | str | None = None,
):
    """
    The interactive version of `longitudinal_plot`.
    """
    go, make_subplots = _plotly()

    cfg = mconfig.get_config()
    if cfg.longitudinal_plot is None or not cfg.longitudinal_plot.subplots:
        print("No longitudinal plot config found. Skipping.")
        return None
    all_cfg = cfg.longitudinal_plot.subplots

    output_filename = Path(output_filename)
    data = _load_plot_cache(output_filename)

    def get_comparison_value(ref, r, base):
        key = ",".join((str(ref.filename)[8:], str(r.filename)[8:], base))
        if key in data:
            return data[key]
        value = getter(result.BenchmarkComparison(ref, r, base))
        data[key] = value
        return value

    results = [r for r in results if r.fork == "python"]
    runners = cfg.runners.values()

    subtitles = []
    for subcfg in all_cfg:
        titleflags = f" ({','.join(subcfg.flags)})" if len(subcfg.flags) else ""
        subtitles.append(f"Python {subcfg.version}.x{titleflags} vs. {subcfg.base}")

    fig = make_subplots(
        rows=len(all_cfg),
        cols=1,
        subplot_titles=subtitles,
        vertical_spacing=_vertical_spacing(len(all_cfg)),
    )

    legend_seen: set[str] = set()
    line_traces: list[int] = []
    rolling_traces: list[int] = []

    for row, subcfg in enumerate(all_cfg, start=1):
        version = [int(x) for x in subcfg.version.split(".")]
        ver_results = [
            r for r in results if list(r.parsed_version.release[0:2]) == version
        ]
        if subcfg.runners:
            cfg_runners = [r for r in runners if r.nickname in subcfg.runners]
        else:
            cfg_runners = runners

        row_values: list[float] = []
        first_runner = True

        for runner in cfg_runners:
            runner_results = [
                r
                for r in ver_results
                if r.nickname == runner.nickname and r.flags == subcfg.flags
            ]

            for r in results:
                if (
                    r.nickname == runner.nickname
                    and r.version == subcfg.base
                    and r.flags == []
                ):
                    ref = r
                    break
            else:
                continue

            runner_results.sort(
                key=lambda x: datetime.datetime.fromisoformat(x.commit_datetime)
            )
            dates = [
                datetime.datetime.fromisoformat(x.commit_datetime)
                for x in runner_results
            ]
            changes = [
                get_comparison_value(ref, r, subcfg.base) for r in runner_results
            ]

            if not any(x is not None for x in changes):
                continue

            style = mplot.get_trace_style(runner)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=changes,
                    mode="lines+markers",
                    name=style.name,
                    legendgroup=style.name,
                    showlegend=style.name not in legend_seen,
                    opacity=0.9,
                    line={"color": style.color, "dash": style.dash, "width": 2},
                    marker={"symbol": style.symbol, "size": 6},
                    customdata=[[r.cpython_hash] for r in runner_results],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>"
                        "%{customdata[0]}<br>%{y:.3f}×<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            line_traces.append(len(fig.data) - 1)
            legend_seen.add(style.name)
            row_values.extend([x for x in changes if x is not None])

            if show_rolling:
                rolling_traces.extend(
                    add_rolling_average(
                        fig,
                        dates,
                        changes,
                        style,
                        window=rolling_window,
                        row=row,
                        col=1,
                    )
                )

            if first_runner:
                _annotate_versions(fig, runner_results, dates, changes, row)
                first_runner = False

        fig.add_hline(y=1.0, line_width=1, line_color="#888", row=row, col=1)
        fig.update_yaxes(
            title_text=_direction_title(differences),
            title_font={"size": 10},
            range=_clamped_range(row_values),
            row=row,
            col=1,
            **_ratio_axis(),
        )

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(420, 380 * len(all_cfg)),
        margin={"t": 120},
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
    )
    add_interactive_controls(
        fig, line_traces=line_traces, rolling_traces=rolling_traces
    )

    save_html(fig, output_filename, include_plotlyjs=include_plotlyjs)
    _save_plot_cache(output_filename, data)

    return fig


def flag_effect_plot_interactive(
    results: Iterable[result.Result],
    output_filename: PathLike,
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.geometric_mean_float,
    differences: tuple[str, str] = ("slower", "faster"),
    title="Performance improvement by configuration",
    *,
    rolling_window: int = ROLLING_WINDOW,
    show_rolling: bool = True,
    include_plotlyjs: bool | str | None = None,
):
    """
    The interactive version of `flag_effect_plot`.

    The subplots share an x axis, which both matches `_standardize_xlims` in
    the matplotlib version and means zooming one subplot zooms them all.
    """
    go, make_subplots = _plotly()

    cfg = mconfig.get_config()
    if cfg.flag_effect_plot is None or not cfg.flag_effect_plot.subplots:
        print("No flag effect plot config found. Skipping.")
        return None
    subplots = cfg.flag_effect_plot.subplots

    output_filename = Path(output_filename)
    data = _load_plot_cache(output_filename)

    def get_comparison_value(ref, r, force_valid):
        key = ",".join((str(ref.filename)[8:], str(r.filename)[8:]))
        if key in data:
            return data[key]
        value = getter(
            result.BenchmarkComparison(ref, r, "default", force_valid=force_valid)
        )
        data[key] = value
        return value

    results = [r for r in results if r.fork == "python"]

    commits: dict[str, dict[tuple[str, ...], dict[str, result.Result]]] = {}
    for r in results:
        commits.setdefault(r.nickname, {}).setdefault(tuple(r.flags), {})[
            r.cpython_hash
        ] = r

    fig = make_subplots(
        rows=len(subplots),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Effect of {subplot.name}" for subplot in subplots],
        vertical_spacing=_vertical_spacing(len(subplots)),
    )

    legend_seen: set[str] = set()
    line_traces: list[int] = []
    rolling_traces: list[int] = []

    for row, subplot in enumerate(subplots, start=1):
        version = tuple(int(x) for x in subplot.version.split("."))
        assert len(version) == 2, (
            f"Version config in {subplot.name}" " should only be major.minor"
        )

        row_values: list[float] = []

        for runner in cfg.runners.values():
            if subplot.runners and runner.nickname not in subplot.runners:
                continue
            runner_is_mapped = runner.nickname in subplot.runner_map
            if subplot.runner_map and not runner_is_mapped:
                continue

            head_results = commits.get(runner.nickname, {}).get(
                tuple(subplot.head_flags), {}
            )
            base_results = commits.get(
                subplot.runner_map.get(runner.nickname, runner.nickname), {}
            ).get(tuple(subplot.base_flags), {})

            line = []
            for cpython_hash, r in head_results.items():
                if cpython_hash in base_results:
                    if r.parsed_version.release[0:2] != version:
                        continue
                    line.append(
                        (
                            r.commit_datetime,
                            get_comparison_value(
                                base_results[cpython_hash], r, runner_is_mapped
                            ),
                            cpython_hash,
                        )
                    )
            line.sort(key=lambda x: datetime.datetime.fromisoformat(x[0]))

            dates = [datetime.datetime.fromisoformat(x[0]) for x in line]
            changes = [x[1] for x in line]

            if not any(x is not None for x in changes):
                continue

            style = mplot.get_trace_style(runner)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=changes,
                    mode="lines+markers",
                    name=style.name,
                    legendgroup=style.name,
                    showlegend=style.name not in legend_seen,
                    opacity=0.9,
                    line={"color": style.color, "dash": style.dash, "width": 2},
                    marker={"symbol": style.symbol, "size": 6},
                    customdata=[[x[2]] for x in line],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>"
                        "%{customdata[0]}<br>%{y:.3f}×<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            line_traces.append(len(fig.data) - 1)
            legend_seen.add(style.name)
            row_values.extend([x for x in changes if x is not None])

            if show_rolling:
                rolling_traces.extend(
                    add_rolling_average(
                        fig,
                        dates,
                        changes,
                        style,
                        window=rolling_window,
                        row=row,
                        col=1,
                    )
                )

        fig.add_hline(y=1.0, line_width=1, line_color="#888", row=row, col=1)
        fig.update_yaxes(
            title_text=_direction_title(differences),
            title_font={"size": 10},
            range=_clamped_range(row_values),
            row=row,
            col=1,
            **_ratio_axis(),
        )

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(420, 380 * len(subplots)),
        margin={"t": 120},
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
    )
    add_interactive_controls(
        fig, line_traces=line_traces, rolling_traces=rolling_traces
    )

    save_html(fig, output_filename, include_plotlyjs=include_plotlyjs)
    _save_plot_cache(output_filename, data)

    return fig


def benchmark_longitudinal_plot_interactive(
    results: Iterable[result.Result],
    output_filename: PathLike,
    *,
    rolling_window: int = ROLLING_WINDOW,
    show_rolling: bool = False,
    include_plotlyjs: bool | str | None = None,
):
    """
    The interactive version of `benchmark_longitudinal_plot`: one row per
    benchmark, sharing a single x axis so that zooming or panning any of them
    moves all of them together.

    The rolling average overlays are off by default here, because this plot
    already has a trace per benchmark per runner.
    """
    go, make_subplots = _plotly()

    cfg = mconfig.get_config()
    if cfg.benchmark_longitudinal_plot is None:
        print("No benchmark longitudinal plot config found. Skipping.")
        return None
    cfg = cfg.benchmark_longitudinal_plot

    output_filename = Path(output_filename)
    cache = _load_plot_cache(output_filename)

    results = [r for r in results if r.fork == "python" and r.nickname in cfg.runners]

    base = None
    for r in results:
        if r.version == cfg.base and r.flags == cfg.base_flags:
            base = r
            break
    else:
        raise ValueError(f"Base version {cfg.base} not found")

    results = [
        r
        for r in results
        if r.version.startswith(cfg.version) and r.flags == cfg.head_flags
    ]

    by_benchmark = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.filename.name not in cache:
            comparison = result.BenchmarkComparison(base, r, "")
            # Set up the entry unconditionally: a result where nothing was
            # significant has no timings to record, but it has still been
            # looked at.
            entry = cache.setdefault(r.filename.name, {})
            for name, _diff, mean in comparison.get_timing_diff():
                # Don't include insignificant results
                if mean > 0.0:
                    entry[name] = [r.commit_date, mean, r.cpython_hash]

        for name, value in cache[r.filename.name].items():
            by_benchmark[name][r.nickname].append(value)

    _save_plot_cache(output_filename, cache)

    # Exclude any benchmarks where we don't have enough data to make a
    # meaningful plot
    by_benchmark = {
        k: v for k, v in by_benchmark.items() if any(len(x) > 2 for x in v.values())
    }

    if not len(by_benchmark):
        print("Not enough data for a benchmark longitudinal plot. Skipping.")
        return None

    benchmarks = sorted(by_benchmark.items())

    fig = make_subplots(
        rows=len(benchmarks),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=_vertical_spacing(len(benchmarks), 0.004),
    )

    legend_seen: set[str] = set()
    line_traces: list[int] = []
    rolling_traces: list[int] = []

    for row, (benchmark, runner_timings) in enumerate(benchmarks, start=1):
        for runner_name, timings in runner_timings.items():
            runner = mrunners.get_runner_by_nickname(runner_name)
            style = mplot.get_trace_style(runner)

            timings.sort(key=lambda x: datetime.datetime.fromisoformat(x[0]))
            dates = [datetime.datetime.fromisoformat(x[0]) for x in timings]
            changes = [x[1] for x in timings]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=changes,
                    mode="lines+markers",
                    name=style.name,
                    legendgroup=style.name,
                    showlegend=style.name not in legend_seen,
                    line={"color": style.color, "dash": style.dash, "width": 1.5},
                    marker={"symbol": style.symbol, "size": 3},
                    customdata=[[benchmark, x[2]] for x in timings],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> — %{fullData.name}<br>"
                        "%{x|%Y-%m-%d}<br>%{customdata[1]}<br>"
                        "%{y:.3f}×<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            line_traces.append(len(fig.data) - 1)
            legend_seen.add(style.name)

            if show_rolling:
                rolling_traces.extend(
                    add_rolling_average(
                        fig,
                        dates,
                        changes,
                        style,
                        window=rolling_window,
                        row=row,
                        col=1,
                    )
                )

        fig.add_hline(y=1.0, line_width=1, line_color="#666", row=row, col=1)
        fig.update_yaxes(showticklabels=False, row=row, col=1, **_ratio_axis())

    # The benchmark names go in the left margin as horizontal labels; a
    # rotated y axis title per row would be unreadable at this height.
    for row, (benchmark, _) in enumerate(benchmarks, start=1):
        axis = fig.layout["yaxis" if row == 1 else f"yaxis{row}"]
        fig.add_annotation(
            text=benchmark,
            xref="paper",
            yref="paper",
            x=0.0,
            y=(axis.domain[0] + axis.domain[1]) / 2,
            xanchor="right",
            yanchor="middle",
            xshift=-8,
            showarrow=False,
            font={"size": 9},
        )

    fig.update_layout(
        title=f"Performance change by benchmark on {cfg.version} vs. {cfg.base}",
        template=PLOTLY_TEMPLATE,
        height=max(420, 90 * len(benchmarks)),
        margin={"l": 200, "r": 40, "t": 120, "b": 60},
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
    )
    add_interactive_controls(
        fig, line_traces=line_traces, rolling_traces=rolling_traces
    )

    save_html(fig, output_filename, include_plotlyjs=include_plotlyjs)

    return fig
