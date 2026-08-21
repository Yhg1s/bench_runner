from __future__ import annotations


import argparse
import csv
import json
import os
from operator import itemgetter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable


import rich_argparse


from bench_runner import benchmark_definitions
from bench_runner import config
from bench_runner import flags
from bench_runner import git
from bench_runner.result import Result
from bench_runner.table import md_link
from bench_runner.util import PathLike


REPO_ROOT = Path()
BENCHMARK_JSON = REPO_ROOT / "benchmark.json"
PROFILING_RESULTS = REPO_ROOT / "profiling" / "results"
GITHUB_URL = "https://github.com/" + os.environ.get(
    "GITHUB_REPOSITORY", "faster-cpython/bench_runner"
)
# Environment variables that control the execution of CPython
ENV_VARS = ["PYTHON_JIT", "PYPERF_PERF_RECORD_EXTRA_OPTS"]
LOOPS_FILE_ENV_VAR = "PYPERFORMANCE_LOOPS_FILE"
NO_CALIBRATE_ENV_VAR = "PYPERFORMANCE_NO_CALIBRATE"
# Where a loops table lives when nothing says otherwise. This is the name the
# README tells people to use, and what the variable defaulted to before it
# existed.
DEFAULT_LOOPS_FILE = "loops.json"


class NoBenchmarkError(Exception):
    pass


def _runner_setting(name: str):
    """
    What the runner this process is running as says about `name`.

    None when it says nothing, which is also what a repo with no
    `bench_runner.toml` gets: run_benchmarks is usable without one, so a
    missing config means "not configured" rather than an error.
    """
    from bench_runner import runners

    try:
        runner = runners.get_current_runner()
    except FileNotFoundError:
        return None
    return getattr(runner, name, None)


def get_loops_file() -> str | None:
    """
    The loops table to use, or None to let pyperf calibrate.

    Per-runner config wins over the environment variable. Loop counts describe
    the machine that measured them, so which table applies is a property of the
    runner, whereas the variable is one value for a whole repo.
    """
    configured = _runner_setting("loops_table_file")
    if configured is not None:
        return configured
    return os.environ.get(LOOPS_FILE_ENV_VAR)


def get_no_calibrate() -> bool:
    """
    Whether to refuse to calibrate.

    Per-runner config wins over the environment variable, including when it
    says False: a runner whose table is incomplete can opt out while the rest
    of the repo keeps the variable set.
    """
    configured = _runner_setting("no_calibrate")
    if configured is not None:
        return bool(configured)
    return bool(os.environ.get(NO_CALIBRATE_ENV_VAR))


def check_loops_table(loops_path: Path) -> None:
    """
    Fail fast, and locally, on a loops file pyperf would reject.

    Both failure modes below otherwise surface from pyperf, inside a worker
    process, on the runner, after the interpreter has already been built -- and
    with an error that mentions neither bench_runner nor what to do about it.

    The format changed: before pyperf grew --loops-table, the file this
    variable pointed at was a pyperformance *results* file, and that is still
    what the older documented `ln -s results/.../bm-....json loops.json`
    produces. Such a file has no `table_version`, so pyperf refuses it.
    """
    if not loops_path.is_file():
        raise FileNotFoundError(
            f"{LOOPS_FILE_ENV_VAR} points at {loops_path}, which does not "
            "exist. Generate it with `python -m bench_runner "
            "synthesize_loops_file`, or unset the variable to let pyperf "
            "calibrate loop counts on every run."
        )
    try:
        with loops_path.open() as fd:
            data = json.load(fd)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{loops_path} is not readable as JSON: {exc}") from exc
    if not isinstance(data, dict) or "table_version" not in data:
        raise ValueError(
            f"{loops_path} is not a pyperf loops table. Loop counts used to be "
            "taken from a benchmark results file, and a symlink to one is "
            "probably what this is; the format is not the same. Regenerate it "
            "with `python -m bench_runner synthesize_loops_file -o "
            f"{loops_path.name} <results>`."
        )


def get_benchmark_names(benchmarks: str) -> list[str]:
    if benchmarks.strip() == "":
        benchmarks = "all"

    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pyperformance",
            "list",
            "--manifest",
            "benchmarks.manifest",
            "--benchmarks",
            benchmarks,
        ],
        encoding="utf-8",
    )

    return [line[2:].strip() for line in output.splitlines() if line.startswith("- ")]


def generate_loops_table(python: PathLike, benchmarks: str, /) -> Path:
    """
    Calibrate the selected benchmarks and write a loops table.

    Returns the path written, and points LOOPS_FILE_ENV_VAR at it so the run
    that follows uses it.

    This is the other way to get a table than `synthesize_loops_file`, which
    reuses counts recorded in an earlier results file. Calibrating here costs a
    pass over the benchmarks, but the counts come from this machine and the
    interpreter that was just built, so there is no question of whether they
    still apply.
    """
    if benchmarks.strip() == "":
        benchmarks = "all"

    # Generated where this runner will later look for it, so a runner with its
    # own loops_table_file calibrates into that file rather than into a shared
    # default that the next runner would overwrite.
    #
    # Resolved for the reason run_benchmarks() resolves it: pyperformance runs
    # each benchmark from its own directory, so a relative path would not mean
    # the same thing by the time it is used.
    loops_path = Path(get_loops_file() or DEFAULT_LOOPS_FILE).resolve()

    extra_args = []
    if affinity := os.environ.get("CPU_AFFINITY"):
        # How many loops fit in --min-time depends on which CPU runs them, so
        # calibrate under the affinity the benchmarks will be run with.
        extra_args.append(f"--affinity={affinity}")

    args = [
        sys.executable,
        "-m",
        "pyperformance",
        "loops_table",
        "-o",
        loops_path,
        "--manifest",
        "benchmarks.manifest",
        "--benchmarks",
        benchmarks,
        "--python",
        python,
        "--inherit-environ",
        ",".join(ENV_VARS),
        *extra_args,
    ]

    print(f"RUNNING: {' '.join(str(x) for x in args)}")

    # Not check_call: pyperformance exits nonzero when any single benchmark
    # fails to calibrate, and still writes the counts that worked. That is the
    # same tolerance run_benchmarks() extends to the benchmark run itself, and
    # for the same reason -- one benchmark that will not build its requirements
    # should not stop the other hundred. An unusable table is caught below.
    subprocess.call(args)

    check_loops_table(loops_path)

    with loops_path.open() as fd:
        count = len(json.load(fd).get("loops", {}))
    print(f"Calibrated {count} benchmark functions into {loops_path}")

    # A benchmark missing from the table is calibrated as usual by the run that
    # follows, unless NO_CALIBRATE_ENV_VAR is set, in which case that run stops
    # and names it. Either way the table is worth using, so point at it.
    os.environ[LOOPS_FILE_ENV_VAR] = str(loops_path)

    return loops_path


def run_benchmarks(
    python: PathLike,
    benchmarks: str,
    /,
    test_mode: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    if benchmarks.strip() == "":
        benchmarks = "all"

    if BENCHMARK_JSON.is_file():
        BENCHMARK_JSON.unlink()

    if test_mode:
        fast_arg = ["--fast"]
    else:
        fast_arg = []

    if extra_args is None:
        extra_args = []

    loops_file = get_loops_file()
    if loops_file:
        # Resolved, not just made absolute: pyperformance runs each benchmark
        # from its own directory, so a relative path would not resolve by the
        # time pyperf reads it, and the documented way to nominate a table is a
        # symlink, which .absolute() would leave unresolved.
        loops_path = Path(loops_file).resolve()
        check_loops_table(loops_path)
        extra_args.append(f"--loops-table={loops_path}")

    if get_no_calibrate():
        if not loops_file:
            # pyperf would say this too, but only after the interpreter has
            # been built and a worker has started, and without naming any of
            # the settings involved. Both routes are named because either can
            # be the one that turned calibration off.
            raise ValueError(
                "Calibration is turned off, but no loops table is named, so "
                "there is no loop count to use instead. Set the runner's "
                f"loops_table_file or {LOOPS_FILE_ENV_VAR} to a table from "
                "`python -m bench_runner synthesize_loops_file`, or turn "
                "calibration back on (the runner's no_calibrate, or "
                f"{NO_CALIBRATE_ENV_VAR})."
            )
        extra_args.append("--no-calibrate")

    if affinity := os.environ.get("CPU_AFFINITY"):
        extra_args.append(f"--affinity={affinity}")

    args = [
        sys.executable,
        "-m",
        "pyperformance",
        "run",
        *fast_arg,
        "-o",
        BENCHMARK_JSON,
        "--manifest",
        "benchmarks.manifest",
        "--benchmarks",
        benchmarks,
        "--python",
        python,
        "--inherit-environ",
        ",".join(ENV_VARS),
        *extra_args,
    ]

    print(f"RUNNING: {' '.join(str(x) for x in args)}")

    subprocess.call(args)

    # pyperformance frequently returns an error if any of the benchmarks failed.
    # We only want to fail if things are worse than that.

    if not BENCHMARK_JSON.is_file():
        raise NoBenchmarkError(
            f"No benchmark file created at {BENCHMARK_JSON.resolve()}."
        )
    with BENCHMARK_JSON.open() as fd:
        contents = json.load(fd)
    if len(contents.get("benchmarks", [])) == 0:
        raise NoBenchmarkError("No benchmarks were run.")


def collect_pystats(
    python: PathLike,
    benchmarks: str,
    fork: str | None = None,
    ref: str | None = None,
    individual: bool = True,
    flags: Iterable[str] | None = None,
) -> None:
    pystats_dir = Path("/tmp/py_stats")

    all_benchmarks = get_benchmark_names(benchmarks)

    # Default to loops.json if not explicitly set, like before the
    # environment variable was added.
    if LOOPS_FILE_ENV_VAR not in os.environ:
        os.environ[LOOPS_FILE_ENV_VAR] = DEFAULT_LOOPS_FILE

    extra_args = ["--hook", "pystats", "--warmups", "0"]

    if flags is None:
        flags = []

    # Clear all files in /tmp/py_stats. The _pystats.yml workflow already
    # does this, but this helps when running and testing things locally.
    for filename in pystats_dir.glob("*"):
        filename.unlink()

    # We could technically run each benchmark in parallel (since we don't care
    # about performance timings), however, since the stats are written to the
    # same directory, they would get intertwined. At some point, specifying an
    # output directory for stats might make sense for this.

    with tempfile.TemporaryDirectory() as tempdir:
        for benchmark in all_benchmarks:
            try:
                run_benchmarks(python, benchmark, extra_args=extra_args)
            except NoBenchmarkError:
                pass
            else:
                if individual and fork is not None and ref is not None:
                    run_summarize_stats(python, fork, ref, benchmark, flags=flags)

            for filename in pystats_dir.iterdir():
                os.rename(filename, Path(tempdir) / filename.name)

        for filename in Path(tempdir).iterdir():
            os.rename(filename, pystats_dir / filename.name)

        if individual:
            benchmark_links = all_benchmarks
        else:
            benchmark_links = []

        if fork is not None and ref is not None:
            run_summarize_stats(python, fork, ref, "all", benchmark_links, flags=flags)


def get_perf_lines(files: Iterable[PathLike]) -> Iterable[str]:
    for filename in files:
        p = subprocess.Popen(
            [
                "perf",
                "report",
                "--stdio",
                "-g",
                "none",
                "--show-total-period",
                "-s",
                "pid,symbol,dso",
                "-i",
                str(filename),
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            bufsize=1,
        )
        assert p.stdout is not None  # for pyright
        yield from iter(p.stdout.readline, "")
        p.kill()


def perf_to_csv(lines: Iterable[str], output: PathLike) -> bool:
    """
    Convert the output of `perf report` to a csv file.

    Returns whether there was anything to convert. If `lines` contains no
    profiling data -- because it is empty, or holds nothing but headers and
    zero-period entries -- nothing is written and this returns False, rather
    than leaving behind a csv file with only a header row for the plotting
    step to choke on.
    """
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            pass
        else:
            _, period, command, _, symbol, shared, _ = line.split(maxsplit=6)
            pid, command = command.split(":")
            period = float(period)
            if period > 0.0:
                rows.append([period, pid, command, shared, symbol])

    if not rows:
        return False

    rows.sort(key=itemgetter(0), reverse=True)

    with Path(output).open("w") as fd:
        csvwriter = csv.writer(fd)
        csvwriter.writerow(["self", "pid", "command", "shared_obj", "symbol"])
        for row in rows:
            csvwriter.writerow(row)

    return True


def collect_perf(python: PathLike, benchmarks: str):
    all_benchmarks = get_benchmark_names(benchmarks)

    if PROFILING_RESULTS.is_dir():
        shutil.rmtree(PROFILING_RESULTS)
    PROFILING_RESULTS.mkdir()

    perf_data_glob = "perf.data.*"
    for benchmark in all_benchmarks:
        for filename in Path(".").glob(perf_data_glob):
            filename.unlink()

        run_benchmarks(
            python,
            benchmark,
            extra_args=["--hook", "perf_record"],
        )

        if not perf_to_csv(
            get_perf_lines(Path(".").glob(perf_data_glob)),
            PROFILING_RESULTS / f"{benchmark}.perf.csv",
        ):
            print(f"No profiling data collected for {benchmark}", file=sys.stderr)

    for filename in Path(".").glob(perf_data_glob):
        filename.unlink()


def update_metadata(
    filename: PathLike,
    fork: str,
    ref: str,
    cpython: PathLike = Path("cpython"),
    run_id: str | None = None,
) -> None:
    with Path(filename).open() as fd:
        content = json.load(fd)

    metadata = content.setdefault("metadata", {})

    metadata["commit_id"] = git.get_git_hash(cpython)
    metadata["commit_fork"] = fork
    metadata["commit_branch"] = ref
    metadata["commit_date"] = git.get_git_commit_date(cpython)
    merge_base = git.get_git_merge_base(cpython)
    if merge_base is not None:
        metadata["commit_merge_base"] = merge_base
    metadata["benchmark_hash"] = benchmark_definitions.get_benchmark_hash()
    if run_id is not None:
        metadata["github_action_url"] = f"{GITHUB_URL}/actions/runs/{run_id}"
    actor = os.environ.get("GITHUB_ACTOR")
    if actor is not None:
        metadata["github_actor"] = actor

    with Path(filename).open("w") as fd:
        json.dump(content, fd, indent=2)


def copy_to_directory(
    filename: PathLike, python: PathLike, fork: str, ref: str, flags: Iterable[str]
) -> None:
    result = Result.from_scratch(python, fork, ref, flags=flags)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filename, result.filename)


def run_summarize_stats(
    python: PathLike,
    fork: str,
    ref: str,
    benchmark: str,
    benchmarks: Iterable[str] | None = None,
    flags: Iterable[str] | None = None,
) -> None:
    if benchmarks is None:
        benchmarks = []

    if flags is None:
        flags = []

    summarize_stats_path = (
        Path(python).parent / "Tools" / "scripts" / "summarize_stats.py"
    )

    parts = ["pystats"]
    if benchmark != "all":
        parts.append(benchmark)
    result = Result.from_scratch(python, fork, ref, parts, flags=flags)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    pystats_json = result.filename.with_suffix(".json")

    args = [str(python), summarize_stats_path]
    args.extend(["--json-output", pystats_json])

    table = subprocess.check_output(args, encoding="utf-8")

    header = textwrap.dedent(
        f"""
    # Pystats results

    - benchmark: {benchmark}
    - fork: {fork}
    - ref: {ref}
    - commit hash: {git.get_git_hash(Path('cpython'))[:7]}
    - commit date: {git.get_git_commit_date(Path('cpython'))}

    """
    )

    with result.filename.with_suffix(".md").open("w") as fd:
        fd.write(header)
        if benchmarks:
            fd.write("- ")
            for name in benchmarks:
                fd.write(
                    md_link(
                        name,
                        str(Path(result.filename.name).with_suffix("")) + f"-{name}.md",
                    )
                )
                fd.write(", ")
            fd.write("\n")
        fd.write(table)

    if pystats_json.is_file():
        update_metadata(pystats_json, fork, ref)
    else:
        print(
            "WARNING: No pystats.json file generated. "
            "This is expected with CPython < 3.12"
        )


def select_benchmarks(benchmarks: str):
    cfg = config.get_config()
    if benchmarks == "all":
        return ",".join(
            ["all", *[f"-{x}" for x in cfg.benchmarks.excluded_benchmarks if x]]
        )
    elif benchmarks == "all_and_excluded":
        return "all"
    return benchmarks


def _main(
    mode: str,
    python: PathLike,
    fork: str,
    ref: str,
    benchmarks: str,
    test_mode: bool,
    run_id: str | None,
    individual: bool,
    flags: Iterable[str],
    generate_loops: bool = False,
) -> None:
    benchmarks = select_benchmarks(benchmarks)

    # Before the mode dispatch, and after select_benchmarks(), so the table is
    # calibrated for exactly the benchmarks that are about to run: a table is
    # looked up per benchmark function, so covering the wrong set is the one
    # way to get a table that silently does nothing.
    if generate_loops:
        generate_loops_table(python, benchmarks)

    if mode == "benchmark":
        run_benchmarks(python, benchmarks, test_mode=test_mode)
        update_metadata(BENCHMARK_JSON, fork, ref, run_id=run_id)
        copy_to_directory(BENCHMARK_JSON, python, fork, ref, flags)
    elif mode == "perf":
        collect_perf(python, benchmarks)
    elif mode == "pystats":
        collect_pystats(python, benchmarks, fork, ref, individual, flags)


def main():
    print("Environment variables:")
    for var in ENV_VARS:
        print(f"{var}={os.environ.get(var, '<unset>')}")

    parser = argparse.ArgumentParser(
        description="""
        Run benchmarks in `pyperformance` with the given python executable. Add
        additional metadata to a benchmark results file and then copy it to the
        correct location in the results tree.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "mode", choices=["benchmark", "perf", "pystats"], help="The mode of execution"
    )
    parser.add_argument("python", help="The path to the Python executable")
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument("benchmarks", help="The benchmarks to run")
    parser.add_argument("flags", help="Configuration flags")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in a special mode for unit testing",
    )
    parser.add_argument("--run_id", default=None, type=str, help="The github run id")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="For pystats mode, collect stats for each individual benchmark",
    )
    parser.add_argument(
        "--generate-loops",
        action="store_true",
        help="Calibrate the selected benchmarks first and write a loops table, "
        f"then use it for this run (see {LOOPS_FILE_ENV_VAR})",
    )
    args = parser.parse_args()

    if args.test_mode:
        import socket

        def gethostname():
            return "pyperf"

        socket.gethostname = gethostname

        def dummy(*args, **kwargs):
            return None

        git.get_git_merge_base = dummy

    _main(
        args.mode,
        Path(args.python),
        args.fork,
        args.ref,
        args.benchmarks,
        args.test_mode,
        args.run_id,
        args.individual,
        flags.parse_flags(args.flags),
        args.generate_loops,
    )


if __name__ == "__main__":
    # This lets pytest-cov collect coverage data in a subprocess
    if "COV_CORE_SOURCE" in os.environ:
        try:
            from pytest_cov.embed import init

            init()
        except Exception:
            sys.stderr.write("pytest-cov: Failed to setup subprocess coverage.")

    main()
