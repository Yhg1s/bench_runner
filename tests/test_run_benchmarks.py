import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


import pytest


from bench_runner import benchmark_definitions
from bench_runner import cache
from bench_runner import git
from bench_runner.scripts import run_benchmarks


DATA_PATH = Path(__file__).parent / "data"


# A sample of what `perf report --stdio -g none --show-total-period
# -s pid,symbol,dso` writes: some `#` comment lines, a blank line, then one row
# per symbol. The last two columns are IPC and IPC coverage, which bench_runner
# ignores.
PERF_REPORT_LINES = [
    "# Overhead    Period  Pid:Command  Symbol  Shared Object  IPC  [IPC Coverage]",
    "# ........  ........  ...........  ......  .............  ...  ..............",
    "#",
    "",
    "    12.82%  33745507  111:python   [.] PyLong_FromLong      python     -  -",
    "     8.82%  23207644  111:python   [.] _PyObject_Malloc     python     -  -",
    "     1.74%   4576089  222:python   [.] __memcmp_evex_movbe  libc.so.6  -  -",
]


def dont_get_git_merge_base(monkeypatch):
    def dummy(*args, **kwargs):
        return None

    monkeypatch.setattr(git, "get_git_merge_base", dummy)


def hardcode_benchmark_hash(monkeypatch):
    def dummy(*args, **kwargs):
        return "215d35"

    monkeypatch.setattr(benchmark_definitions, "get_benchmark_hash", dummy)


def test_update_metadata(benchmarks_checkout, monkeypatch):
    dont_get_git_merge_base(monkeypatch)
    hardcode_benchmark_hash(monkeypatch)

    shutil.copy(
        DATA_PATH
        / "results"
        / "bm-20211208-3.11.0a3-2e91dba"
        / "bm-20211208-linux-x86_64-python-main-3.11.0a3-2e91dba.json",
        benchmarks_checkout / "benchmarks.json",
    )
    run_benchmarks.update_metadata(
        benchmarks_checkout / "benchmarks.json",
        "myfork",
        "myref",
        benchmarks_checkout / "cpython",
        "12345",
    )

    with open(benchmarks_checkout / "benchmarks.json") as fd:
        content = json.load(fd)

    metadata = content["metadata"]

    assert metadata["commit_id"] == "9d38120"
    assert metadata["commit_fork"] == "myfork"
    assert metadata["commit_branch"] == "myref"
    assert metadata["commit_date"].startswith("2022-03-23T20:12:04")
    assert "commit_merge_base" not in metadata
    assert metadata["benchmark_hash"] == "215d35"
    assert (
        metadata["github_action_url"]
        == "https://github.com/faster-cpython/bench_runner/actions/runs/12345"
    )


def test_run_benchmarks(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)

    shutil.copyfile(
        DATA_PATH / "bench_runner.toml", benchmarks_checkout / "bench_runner.toml"
    )

    venv_dir = benchmarks_checkout / "venv"
    venv_python = venv_dir / "bin" / "python"

    shutil.copy(
        DATA_PATH / "benchmarks.manifest",
        benchmarks_checkout / "benchmarks.manifest",
    )

    # Now actually run the run_benchmarks.py script
    subprocess.check_call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "deepcopy",
            ",,",
            "--test_mode",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )

    with open(
        benchmarks_checkout
        / "results"
        / f"bm-20220323-{platform.python_version()}-9d38120"
        / f"bm-20220323-{platform.system().lower()}-{platform.machine()}-"
        f"python-main-{platform.python_version()}-9d38120.json"
    ) as fd:
        content = json.load(fd)

    metadata = content["metadata"]
    benchmarks = content["benchmarks"]

    assert metadata["commit_id"] == "9d38120"
    assert metadata["commit_fork"] == "python"
    assert metadata["commit_branch"] == "main"
    assert metadata["commit_date"].startswith("2022-03-23T20:12:04")
    assert "commit_merge_base" not in metadata
    assert metadata["benchmark_hash"] == "215d35"
    assert (
        metadata["github_action_url"]
        == "https://github.com/faster-cpython/bench_runner/actions/runs/12345"
    )

    assert len(benchmarks) == 3
    assert all(len(benchmark["runs"]) > 1 for benchmark in benchmarks)
    assert set(bm["metadata"]["name"] for bm in benchmarks) == {
        "deepcopy",
        "deepcopy_memo",
        "deepcopy_reduce",
    }

    # Run an unknown benchmark, expect an error
    returncode = subprocess.call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "foo",
            ",,",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )
    assert returncode == 1


def test_run_benchmarks_flags(benchmarks_checkout):
    shutil.copyfile(
        DATA_PATH / "bench_runner.toml", benchmarks_checkout / "bench_runner.toml"
    )

    venv_dir = benchmarks_checkout / "venv"
    venv_python = venv_dir / "bin" / "python"

    shutil.copy(
        DATA_PATH / "benchmarks.manifest",
        benchmarks_checkout / "benchmarks.manifest",
    )

    # Now actually run the run_benchmarks.py script
    subprocess.check_call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "nbody",
            "tier2,,",
            "--test_mode",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )

    with open(
        benchmarks_checkout
        / "results"
        / f"bm-20220323-{platform.python_version()}-9d38120-PYTHON_UOPS"
        / f"bm-20220323-{platform.system().lower()}-{platform.machine()}-"
        f"python-main-{platform.python_version()}-9d38120.json"
    ) as fd:
        json.load(fd)


def test_perf_to_csv_writes_a_row_per_symbol(tmp_path):
    output = tmp_path / "deepcopy.perf.csv"

    assert run_benchmarks.perf_to_csv(PERF_REPORT_LINES, output) is True

    with output.open(newline="") as fd:
        rows = list(csv.reader(fd))

    assert rows[0] == ["self", "pid", "command", "shared_obj", "symbol"]
    # Sorted by self time, descending.
    assert rows[1:] == [
        ["33745507.0", "111", "python", "python", "PyLong_FromLong"],
        ["23207644.0", "111", "python", "python", "_PyObject_Malloc"],
        ["4576089.0", "222", "python", "libc.so.6", "__memcmp_evex_movbe"],
    ]


def test_perf_to_csv_reads_a_one_shot_iterator_from_the_first_line(tmp_path):
    # `get_perf_lines` is a generator, so perf_to_csv gets one pass over the
    # lines and must not need the caller to have peeked at them first. Every
    # line has to land in the csv, including the first.
    output = tmp_path / "deepcopy.perf.csv"

    assert run_benchmarks.perf_to_csv(iter(PERF_REPORT_LINES), output) is True

    with output.open(newline="") as fd:
        symbols = [row[-1] for row in list(csv.reader(fd))[1:]]

    assert symbols == ["PyLong_FromLong", "_PyObject_Malloc", "__memcmp_evex_movbe"]


def test_perf_to_csv_returns_false_for_an_empty_iterable(tmp_path):
    output = tmp_path / "deepcopy.perf.csv"

    assert run_benchmarks.perf_to_csv([], output) is False
    assert not output.exists()


def test_perf_to_csv_returns_false_for_an_empty_generator(tmp_path):
    output = tmp_path / "deepcopy.perf.csv"

    assert run_benchmarks.perf_to_csv((line for line in []), output) is False
    assert not output.exists()


def test_perf_to_csv_returns_false_when_there_are_only_headers(tmp_path):
    # perf still prints its banner when it recorded nothing worth reporting.
    output = tmp_path / "deepcopy.perf.csv"

    assert run_benchmarks.perf_to_csv(PERF_REPORT_LINES[:4], output) is False
    assert not output.exists()


def test_perf_to_csv_returns_false_when_every_period_is_zero(tmp_path):
    # Zero-period rows are dropped, which can empty out an otherwise
    # non-empty report.
    output = tmp_path / "deepcopy.perf.csv"
    lines = [
        *PERF_REPORT_LINES[:4],
        "     0.00%         0  111:python   [.] PyLong_FromLong  python  -  -",
    ]

    assert run_benchmarks.perf_to_csv(lines, output) is False
    assert not output.exists()


def test_perf_to_csv_does_not_overwrite_an_earlier_csv_with_nothing(tmp_path):
    output = tmp_path / "deepcopy.perf.csv"
    output.write_text("previous contents\n")

    assert run_benchmarks.perf_to_csv([], output) is False
    assert output.read_text() == "previous contents\n"


def _fake_collect_perf(monkeypatch, tmp_path, perf_data_files, lines):
    """
    Set `collect_perf` up to run in `tmp_path` against a fake benchmark, whose
    run drops `perf_data_files` for `get_perf_lines` to (pretend to) read.

    Returns the list the fake `get_perf_lines` records the files it saw in.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profiling").mkdir()

    monkeypatch.setattr(run_benchmarks, "get_benchmark_names", lambda _: ["deepcopy"])

    def fake_run_benchmarks(*args, **kwargs):
        for name in perf_data_files:
            (tmp_path / name).touch()

    monkeypatch.setattr(run_benchmarks, "run_benchmarks", fake_run_benchmarks)

    seen = []

    def fake_get_perf_lines(files):
        names = sorted(Path(f).name for f in files)
        seen.extend(names)
        if names:
            yield from lines

    monkeypatch.setattr(run_benchmarks, "get_perf_lines", fake_get_perf_lines)

    return seen


def test_collect_perf_uses_every_perf_data_file(tmp_path, monkeypatch):
    perf_data_files = ["perf.data.1", "perf.data.2", "perf.data.3"]
    seen = _fake_collect_perf(monkeypatch, tmp_path, perf_data_files, PERF_REPORT_LINES)

    run_benchmarks.collect_perf(sys.executable, "deepcopy")

    # None of them may be swallowed by a check for whether there are any.
    assert seen == perf_data_files
    assert (tmp_path / "profiling" / "results" / "deepcopy.perf.csv").is_file()
    # And they are cleaned up afterwards.
    assert list(tmp_path.glob("perf.data.*")) == []


def test_collect_perf_reports_a_benchmark_with_no_profiling_data(
    tmp_path, monkeypatch, capsys
):
    # perf produced files, but nothing usable came out of them.
    seen = _fake_collect_perf(monkeypatch, tmp_path, ["perf.data.1"], [])

    run_benchmarks.collect_perf(sys.executable, "deepcopy")

    assert seen == ["perf.data.1"]
    assert not (tmp_path / "profiling" / "results" / "deepcopy.perf.csv").exists()
    assert "No profiling data collected for deepcopy" in capsys.readouterr().err


def test_collect_perf_reports_a_benchmark_that_produced_no_perf_data(
    tmp_path, monkeypatch, capsys
):
    seen = _fake_collect_perf(monkeypatch, tmp_path, [], PERF_REPORT_LINES)

    run_benchmarks.collect_perf(sys.executable, "deepcopy")

    assert seen == []
    assert not (tmp_path / "profiling" / "results" / "deepcopy.perf.csv").exists()
    assert "No profiling data collected for deepcopy" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --loops-table passthrough
# ---------------------------------------------------------------------------


def _write_loops_table(path, loops=None):
    path.write_text(
        json.dumps(
            {
                "table_version": 1,
                "min_time": 0.1,
                "machine": {
                    "hostname": "h",
                    "platform": "p",
                    "python": "3.14.0",
                    "date": "2026-01-01T00:00:00",
                },
                "loops": loops or {"nbody": 64},
            }
        )
    )
    return path


def _stub_pyperformance(tmp_path, monkeypatch):
    """
    Capture the pyperformance command line without running it.

    run_benchmarks unlinks the output file before the run and insists it exists
    afterwards, so the stub has to produce it.
    """
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(run_benchmarks, "BENCHMARK_JSON", output)
    captured = []

    def fake_call(args, **kwargs):
        captured.append(args)
        output.write_text(json.dumps({"benchmarks": [{"metadata": {"name": "nbody"}}]}))
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    return captured


def test_check_loops_table_accepts_a_table(tmp_path):
    run_benchmarks.check_loops_table(_write_loops_table(tmp_path / "loops.json"))


def test_check_loops_table_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_benchmarks.check_loops_table(tmp_path / "nope.json")


def test_check_loops_table_rejects_a_results_file(tmp_path):
    # The pre---loops-table setup symlinked loops.json to a results file. That
    # must fail here, with an actionable message, rather than inside pyperf on
    # the runner after the interpreter has already been built.
    results = next(DATA_PATH.glob("results/**/*.json"))
    target = tmp_path / "loops.json"
    target.write_text(results.read_text())
    with pytest.raises(ValueError, match="not a pyperf loops table"):
        run_benchmarks.check_loops_table(target)


def test_check_loops_table_rejects_junk(tmp_path):
    target = tmp_path / "loops.json"
    target.write_text("not json at all")
    with pytest.raises(ValueError, match="not readable as JSON"):
        run_benchmarks.check_loops_table(target)


def test_loops_table_is_passed_to_pyperformance(tmp_path, monkeypatch):
    """
    The one line this feature adds to the product path: the env var becomes a
    --loops-table argument on pyperformance's command line.
    """
    table = _write_loops_table(tmp_path / "loops.json")
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(table))

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert captured, "pyperformance was never invoked"
    assert f"--loops-table={table.resolve()}" in captured[0]


def test_loops_table_path_is_resolved_through_a_symlink(tmp_path, monkeypatch):
    # The documented setup nominates the table with a symlink, and each
    # benchmark runs from its own directory, so the path handed over has to be
    # both absolute and dereferenced.
    real = _write_loops_table(tmp_path / "real-table.json")
    link = tmp_path / "loops.json"
    link.symlink_to(real)
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(link))

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--loops-table={real.resolve()}" in captured[0]


def test_no_loops_table_argument_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert not [a for a in captured[0] if str(a).startswith("--loops-table")]


# ---------------------------------------------------------------------------
# --no-calibrate passthrough
# ---------------------------------------------------------------------------


def test_no_calibrate_is_passed_to_pyperformance(tmp_path, monkeypatch):
    table = _write_loops_table(tmp_path / "loops.json")
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(table))
    monkeypatch.setenv(run_benchmarks.NO_CALIBRATE_ENV_VAR, "1")

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert "--no-calibrate" in captured[0]
    assert f"--loops-table={table.resolve()}" in captured[0]


def test_no_calibrate_argument_absent_when_unset(tmp_path, monkeypatch):
    table = _write_loops_table(tmp_path / "loops.json")
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(table))
    monkeypatch.delenv(run_benchmarks.NO_CALIBRATE_ENV_VAR, raising=False)

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert "--no-calibrate" not in captured[0]


def test_no_calibrate_without_a_loops_file_is_refused(tmp_path, monkeypatch):
    # pyperf would refuse this too, but only from inside a worker process on
    # the runner, naming neither variable.
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    monkeypatch.setenv(run_benchmarks.NO_CALIBRATE_ENV_VAR, "1")

    captured = _stub_pyperformance(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=run_benchmarks.LOOPS_FILE_ENV_VAR):
        run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert not captured, "pyperformance should not have been invoked"


# ---------------------------------------------------------------------------
# loops table generation
# ---------------------------------------------------------------------------


def _stub_calibration(tmp_path, monkeypatch, loops=None, returncode=0):
    """
    Capture the pyperformance command line and write the table it would write.
    """
    captured = []

    def fake_call(args, **kwargs):
        captured.append(args)
        if loops is not None:
            _write_loops_table(Path(args[args.index("-o") + 1]), loops)
        return returncode

    monkeypatch.setattr(subprocess, "call", fake_call)
    return captured


def test_generate_loops_table_calls_the_pyperformance_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "loops.json"))
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    args = [str(a) for a in captured[0]]
    assert "loops_table" in args
    assert "--benchmarks" in args and args[args.index("--benchmarks") + 1] == "nbody"
    # The interpreter being benchmarked, not the one running bench_runner: a
    # count calibrated against a different build is the wrong count.
    assert args[args.index("--python") + 1] == sys.executable


def test_generate_loops_table_points_the_run_at_what_it_wrote(tmp_path, monkeypatch):
    # Without this the table would be calibrated and then ignored, which looks
    # exactly like the feature working.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    written = run_benchmarks.generate_loops_table(sys.executable, "nbody")

    assert written == (tmp_path / run_benchmarks.DEFAULT_LOOPS_FILE).resolve()
    assert os.environ[run_benchmarks.LOOPS_FILE_ENV_VAR] == str(written)


def test_generate_loops_table_writes_an_absolute_path(tmp_path, monkeypatch):
    # pyperformance runs each benchmark from its own directory, so a relative
    # -o would not land where the run later looks for it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, "loops.json")
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    args = [str(a) for a in captured[0]]
    assert Path(args[args.index("-o") + 1]).is_absolute()


def test_generate_loops_table_calibrates_under_the_run_affinity(tmp_path, monkeypatch):
    # How many loops fit in --min-time depends on which CPU runs them.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    monkeypatch.setenv("CPU_AFFINITY", "2-5")
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    assert "--affinity=2-5" in [str(a) for a in captured[0]]


def test_generate_loops_table_keeps_a_partial_table(tmp_path, monkeypatch):
    # pyperformance exits nonzero when any one benchmark fails to calibrate and
    # still writes the rest. One benchmark that will not install its
    # requirements should not cost the other hundred their counts.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64}, returncode=1)

    written = run_benchmarks.generate_loops_table(sys.executable, "nbody,broken")

    assert json.loads(written.read_text())["loops"] == {"nbody": 64}


def test_generate_loops_table_refuses_an_unusable_table(tmp_path, monkeypatch):
    # Nothing written at all is not a partial result, it is a broken run, and
    # saying so here beats failing inside pyperf on the runner later.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    _stub_calibration(tmp_path, monkeypatch, loops=None, returncode=1)

    with pytest.raises(FileNotFoundError):
        run_benchmarks.generate_loops_table(sys.executable, "nbody")


def test_generated_table_is_used_by_the_run_that_follows(tmp_path, monkeypatch):
    # The whole point: generation and consumption have to meet on the same path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})
    written = run_benchmarks.generate_loops_table(sys.executable, "nbody")

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--loops-table={written}" in captured[0]


# ---------------------------------------------------------------------------
# per-runner loops table configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_config(tmp_path, monkeypatch):
    """
    Make the current runner one configured by the test.

    get_config() caches on the relative path "bench_runner.toml", so a config
    another test loaded from its own directory would otherwise be handed back
    here instead of this one. Cleared on the way in and on the way out, so this
    test's config does not escape either.
    """
    from bench_runner import config as mconfig

    def configure(**runner_fields):
        lines = [
            "[bases]",
            'versions = ["3.12.0"]',
            "",
            "[runners.testrunner]",
            'os = "linux"',
            'arch = "x86_64"',
            'hostname = "testhost"',
        ]
        lines += [
            f"{key} = {json.dumps(value)}" for key, value in runner_fields.items()
        ]
        (tmp_path / "bench_runner.toml").write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BENCHMARK_MACHINE_NICKNAME", "testrunner")
        mconfig._load_config.cache_clear()

    mconfig._load_config.cache_clear()
    yield configure
    mconfig._load_config.cache_clear()


def test_runner_loops_table_file_beats_the_env_var(
    tmp_path, monkeypatch, runner_config
):
    # The point of the feature: one repo, several runners, and a loops table
    # describes the machine that measured it.
    table = _write_loops_table(tmp_path / "runner-table.json")
    _write_loops_table(tmp_path / "global-table.json")
    monkeypatch.setenv(
        run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "global-table.json")
    )
    runner_config(loops_table_file=str(table))

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--loops-table={table.resolve()}" in captured[0]


def test_runner_loops_table_file_used_with_no_env_var(
    tmp_path, monkeypatch, runner_config
):
    table = _write_loops_table(tmp_path / "runner-table.json")
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    runner_config(loops_table_file=str(table))

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--loops-table={table.resolve()}" in captured[0]


def test_env_var_still_used_when_the_runner_says_nothing(
    tmp_path, monkeypatch, runner_config
):
    table = _write_loops_table(tmp_path / "global-table.json")
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(table))
    runner_config()

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--loops-table={table.resolve()}" in captured[0]


def test_runner_no_calibrate_is_passed(tmp_path, monkeypatch, runner_config):
    table = _write_loops_table(tmp_path / "runner-table.json")
    monkeypatch.delenv(run_benchmarks.NO_CALIBRATE_ENV_VAR, raising=False)
    runner_config(loops_table_file=str(table), no_calibrate=True)

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert "--no-calibrate" in captured[0]


def test_runner_no_calibrate_false_overrides_the_env_var(
    tmp_path, monkeypatch, runner_config
):
    # The reason no_calibrate is tri-state. One runner whose table is
    # incomplete opts out while the variable stays set for the rest.
    table = _write_loops_table(tmp_path / "runner-table.json")
    monkeypatch.setenv(run_benchmarks.NO_CALIBRATE_ENV_VAR, "1")
    runner_config(loops_table_file=str(table), no_calibrate=False)

    captured = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert "--no-calibrate" not in captured[0]


def test_settings_fall_back_to_the_env_without_a_config_file(tmp_path, monkeypatch):
    # run_benchmarks is usable in a checkout with no bench_runner.toml, and
    # consulting the runner must not turn that into an error.
    from bench_runner import config as mconfig

    mconfig._load_config.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "bench_runner.toml").exists()
    table = _write_loops_table(tmp_path / "loops.json")
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(table))

    try:
        assert run_benchmarks.get_loops_file() == str(table)
        assert run_benchmarks.get_no_calibrate() is False
    finally:
        mconfig._load_config.cache_clear()


def test_generation_targets_the_runners_own_table(tmp_path, monkeypatch, runner_config):
    # Otherwise every runner would calibrate into the same default file and
    # the last one to run would win.
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    runner_config(loops_table_file="runner-table.json")
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    written = run_benchmarks.generate_loops_table(sys.executable, "nbody")

    assert written == (tmp_path / "runner-table.json").resolve()
    assert str(written) in [str(a) for a in captured[0]]


# ------------------------------------------- the cross-run package cache


def _inherited_environ(args) -> list[str]:
    """The variables one pyperformance command line passes through to pip."""
    args = [str(a) for a in args]
    return args[args.index("--inherit-environ") + 1].split(",")


def test_pip_cache_vars_are_allowlisted():
    # pyperformance builds the benchmark venvs itself and hands pip an
    # allowlisted environment (venv.py:_get_envvars), so a variable not named
    # in --inherit-environ never reaches the pip that installs into them.
    from bench_runner import cache

    for var in cache.PIP_ENV_VARS:
        assert var in run_benchmarks.ENV_VARS

    # Still carrying what it carried before.
    assert "PYTHON_JIT" in run_benchmarks.ENV_VARS
    assert "PYPERF_PERF_RECORD_EXTRA_OPTS" in run_benchmarks.ENV_VARS


def test_run_benchmarks_inherits_the_pip_cache_vars(tmp_path, monkeypatch):
    from bench_runner import cache

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    inherited = _inherited_environ(captured[0])
    for var in cache.PIP_ENV_VARS:
        assert var in inherited


def test_loops_table_generation_inherits_the_pip_cache_vars(tmp_path, monkeypatch):
    # Calibration builds the same benchmark venvs, so it needs the cache just
    # as much as the run that follows it.
    from bench_runner import cache

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "loops.json"))
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    inherited = _inherited_environ(captured[0])
    for var in cache.PIP_ENV_VARS:
        assert var in inherited


# --------------------------------------------- local dependency result marking


def _result_file(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps({"benchmarks": [{"metadata": {"name": "nbody"}}], "metadata": {}})
    )
    return path


def _cpython_checkout(tmp_path, monkeypatch):
    """Just enough of a checkout for update_metadata's git calls."""
    from bench_runner import git as mgit

    cpython = tmp_path / "cpython"
    cpython.mkdir()
    subprocess.check_call(["git", "init", "-q", "."], cwd=cpython)
    (cpython / "README").write_text("x")
    subprocess.check_call(["git", "add", "-A"], cwd=cpython)
    subprocess.check_call(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
        cwd=cpython,
    )
    # Finding a merge base needs the network, and is not what these test.
    monkeypatch.setattr(mgit, "get_git_merge_base", lambda dirname: None)
    return cpython


def test_metadata_records_local_deps(tmp_path, monkeypatch):
    from bench_runner import benchmark_definitions as bd
    from bench_runner import local_deps

    pyperf = tmp_path / "pyperf"
    pyperf.mkdir()
    subprocess.check_call(["git", "init", "-q", "."], cwd=pyperf)
    (pyperf / "runner.py").write_text("original")
    subprocess.check_call(["git", "add", "-A"], cwd=pyperf)
    subprocess.check_call(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
        cwd=pyperf,
    )
    (pyperf / "runner.py").write_text("edited but not committed")

    cpython = _cpython_checkout(tmp_path, monkeypatch)
    result = _result_file(tmp_path)
    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={pyperf}")
    bd.get_local_repo_state.cache_clear()

    run_benchmarks.update_metadata(result, "python", "main", cpython=cpython)

    metadata = json.loads(result.read_text())["metadata"]
    recorded = metadata["local_deps"]["pyperf"]
    assert recorded["path"] == str(pyperf)
    assert len(recorded["commit"]) == 40
    assert recorded["dirty"] is True
    assert recorded["diff_sha"]
    bd.get_local_repo_state.cache_clear()


def test_metadata_has_no_local_deps_key_normally(tmp_path, monkeypatch):
    from bench_runner import local_deps

    monkeypatch.delenv(local_deps.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    cpython = _cpython_checkout(tmp_path, monkeypatch)
    result = _result_file(tmp_path)

    run_benchmarks.update_metadata(result, "python", "main", cpython=cpython)

    assert "local_deps" not in json.loads(result.read_text())["metadata"]


def test_a_stale_local_deps_key_is_removed(tmp_path, monkeypatch):
    # update_metadata updates whatever is already in the file, so a key left
    # over from an earlier write would claim a local checkout this run did not
    # use -- exactly the confusion the key exists to prevent.
    from bench_runner import local_deps

    monkeypatch.delenv(local_deps.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    cpython = _cpython_checkout(tmp_path, monkeypatch)
    result = tmp_path / "benchmark.json"
    result.write_text(
        json.dumps(
            {
                "benchmarks": [{"metadata": {"name": "nbody"}}],
                "metadata": {"local_deps": {"pyperf": {"path": "/gone"}}},
            }
        )
    )

    run_benchmarks.update_metadata(result, "python", "main", cpython=cpython)

    assert "local_deps" not in json.loads(result.read_text())["metadata"]


# ---------------------------------------------- --local-dep passthrough


@pytest.fixture
def local_dep_checkout(tmp_path, monkeypatch):
    """A configured local pyperf, as BENCH_RUNNER_LOCAL_DEPS would give it."""
    from bench_runner import local_deps

    checkout = tmp_path / "pyperf"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text('[project]\nname = "pyperf"\n')
    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={checkout}")
    return checkout


@pytest.fixture(autouse=True)
def no_ambient_local_deps(monkeypatch):
    from bench_runner import local_deps

    monkeypatch.delenv(local_deps.ENV_VAR, raising=False)


def test_no_local_dep_args_when_nothing_is_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_benchmarks.get_local_dep_args() == []


def test_local_dep_args(tmp_path, monkeypatch, local_dep_checkout):
    monkeypatch.chdir(tmp_path)
    assert run_benchmarks.get_local_dep_args() == [
        f"--local-dep=pyperf={local_dep_checkout}:editable"
    ]


def test_local_dep_args_are_sorted(tmp_path, monkeypatch):
    from bench_runner import local_deps

    for name in ("zzz", "aaa"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        local_deps.ENV_VAR, f"zzz={tmp_path / 'zzz'},aaa={tmp_path / 'aaa'}"
    )

    args = run_benchmarks.get_local_dep_args()

    # Deterministic, so two runs build the same command line.
    assert [a.split("=")[1] for a in args] == ["aaa", "zzz"]


def test_run_benchmarks_passes_local_dep(tmp_path, monkeypatch, local_dep_checkout):
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    captured = _stub_pyperformance(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--local-dep=pyperf={local_dep_checkout}:editable" in [
        str(a) for a in captured[0]
    ]


def test_loops_table_passes_local_dep(tmp_path, monkeypatch, local_dep_checkout):
    # Calibration builds the same benchmark venvs: a loop count measured
    # against a different pyperf is not the count the run wants.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "loops.json"))
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    assert f"--local-dep=pyperf={local_dep_checkout}:editable" in [
        str(a) for a in captured[0]
    ]


def test_benchmark_names_does_not_pass_local_dep(
    tmp_path, monkeypatch, local_dep_checkout
):
    """
    `pyperformance list` does not accept --local-dep and does not need it: it
    builds no venv, and it runs under the outer venv's pyperformance, which is
    already the local checkout when one is configured.
    """
    captured = []

    def fake_check_output(args, **kwargs):
        captured.append([str(a) for a in args])
        return "- nbody\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    monkeypatch.chdir(tmp_path)

    assert run_benchmarks.get_benchmark_names("nbody") == ["nbody"]
    assert not any(a.startswith("--local-dep") for a in captured[0])


def test_the_spec_pyperformance_is_given_is_the_one_it_parses(tmp_path, monkeypatch):
    """
    A cross-repository contract with no single home: bench_runner emits the
    spec and pyperformance parses it, and the two chose opposite defaults for a
    bare NAME=PATH (editable here, non-editable there). They agree only because
    an editable dep is never emitted bare. Pin that, so neither side can drift.
    """
    from bench_runner import local_deps

    for editable in (True, False):
        dep = local_deps.LocalDep(name="pyperf", path="/x/pyperf", editable=editable)
        spec = dep.to_spec()

        # Reparsed the way pyperformance's _localdeps.LocalDep.parse does it.
        rest = spec.partition("=")[2]
        reparsed_editable = False
        for suffix, value in ((":editable", True), (":copy", False)):
            if rest.endswith(suffix):
                reparsed_editable = value
                rest = rest[: -len(suffix)]
                break

        assert reparsed_editable is editable, spec
        assert rest == "/x/pyperf"


# ------------------------------------------------- --venvs-dir passthrough


@pytest.fixture
def venvs_dir_config(tmp_path, monkeypatch):
    """A repository whose [cache] points somewhere disposable."""
    from bench_runner import cache
    from bench_runner import config as mconfig

    def configure(**settings):
        settings.setdefault("dir", str(tmp_path / "cache"))
        lines = [
            "[bases]",
            'versions = ["3.12.0"]',
            "",
            "[runners.testrunner]",
            'os = "linux"',
            'arch = "x86_64"',
            'hostname = "testhost"',
            "",
            "[cache]",
        ]
        for key, value in settings.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            else:
                lines.append(f'{key} = "{value}"')
        (tmp_path / "bench_runner.toml").write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        mconfig._load_config.cache_clear()
        return tmp_path / "cache"

    for name in (
        cache.CACHE_DIR_ENV_VAR,
        cache.NO_CACHE_ENV_VAR,
        cache.OFFLINE_ENV_VAR,
        cache.ABI_SCOPE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    mconfig._load_config.cache_clear()
    yield configure
    mconfig._load_config.cache_clear()


def test_venvs_dir_args(venvs_dir_config):
    root = venvs_dir_config()
    assert run_benchmarks.get_venvs_dir_args() == [f"--venvs-dir={root / 'venvs'}"]


def test_venvs_dir_is_outside_the_results_repo(venvs_dir_config, tmp_path):
    """
    The whole point of D1: pyperformance's default nests the benchmark venvs at
    ./venv/<runid>, where the bootstrap's rebuild of the outer venv destroys
    them. With nothing configured they must land outside the results repository
    altogether -- a results repo's .gitignore knows about `venv/` and no more,
    so a directory inside it could be swept into a commit by a `git add`.
    """
    # No [cache].dir, so the platform cache directory: what a real repo gets.
    venvs_dir_config(dir="")

    target = Path(run_benchmarks.get_venvs_dir_args()[0].split("=", 1)[1])

    assert target.is_absolute()
    assert not target.is_relative_to(tmp_path)  # tmp_path is the results repo
    assert target == cache.get_cache_root() / "venvs"


def test_no_venvs_dir_when_caching_is_off(venvs_dir_config):
    # pyperformance's own default is left in place, so behaviour is exactly
    # what it was before any of this existed.
    venvs_dir_config(enabled=False)
    assert run_benchmarks.get_venvs_dir_args() == []


def test_venvs_dir_follows_the_cache_dir_env_var(
    venvs_dir_config, tmp_path, monkeypatch
):
    from bench_runner import cache

    venvs_dir_config()
    monkeypatch.setenv(cache.CACHE_DIR_ENV_VAR, str(tmp_path / "elsewhere"))

    assert run_benchmarks.get_venvs_dir_args() == [
        f"--venvs-dir={tmp_path / 'elsewhere' / 'venvs'}"
    ]


def test_run_benchmarks_passes_venvs_dir(tmp_path, monkeypatch, venvs_dir_config):
    root = venvs_dir_config()
    monkeypatch.delenv(run_benchmarks.LOOPS_FILE_ENV_VAR, raising=False)
    captured = _stub_pyperformance(tmp_path, monkeypatch)

    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    assert f"--venvs-dir={root / 'venvs'}" in [str(a) for a in captured[0]]


def test_loops_table_passes_venvs_dir(tmp_path, monkeypatch, venvs_dir_config):
    # Calibration must build its venvs where the run that follows will look.
    root = venvs_dir_config()
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "loops.json"))
    captured = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})

    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    assert f"--venvs-dir={root / 'venvs'}" in [str(a) for a in captured[0]]


def test_calibration_and_run_agree_on_the_venvs_dir(
    tmp_path, monkeypatch, venvs_dir_config
):
    # If they disagreed, calibration would build a venv the run then rebuilds.
    venvs_dir_config()
    monkeypatch.setenv(run_benchmarks.LOOPS_FILE_ENV_VAR, str(tmp_path / "loops.json"))
    calibration = _stub_calibration(tmp_path, monkeypatch, loops={"nbody": 64})
    run_benchmarks.generate_loops_table(sys.executable, "nbody")

    run = _stub_pyperformance(tmp_path, monkeypatch)
    run_benchmarks.run_benchmarks(sys.executable, "nbody")

    def venvs_dir(args):
        return [a for a in map(str, args) if a.startswith("--venvs-dir=")]

    assert venvs_dir(calibration[0]) == venvs_dir(run[0]) != []


def test_benchmark_names_does_not_pass_venvs_dir(
    tmp_path, monkeypatch, venvs_dir_config
):
    # `pyperformance list` builds no venv and does not accept the flag.
    venvs_dir_config()
    captured = []

    def fake_check_output(args, **kwargs):
        captured.append([str(a) for a in args])
        return "- nbody\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert run_benchmarks.get_benchmark_names("nbody") == ["nbody"]
    assert not any(a.startswith("--venvs-dir") for a in captured[0])
