import csv
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


from bench_runner import benchmark_definitions
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
