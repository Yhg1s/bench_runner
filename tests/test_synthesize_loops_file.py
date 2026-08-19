import json
import pathlib
import sys
import tempfile

import pytest

from bench_runner.scripts import synthesize_loops_file

DATA_PATH = pathlib.Path(__file__).parent / "data"


def run_synthesize(
    output: pathlib.Path,
    datadir: pathlib.Path,
    *,
    update: bool = False,
    overwrite: bool = False,
    merger: str = "max",
):
    files = datadir.glob("results/**/*.json")
    synthesize_loops_file._main(
        loops_file=output,
        update=update,
        overwrite=overwrite,
        merger=merger,
        results=files,
    )


def check_loops(output: pathlib.Path):
    """
    Check the file is a loops table pyperf would accept, and return the counts.
    """
    with output.open() as f:
        data = json.load(f)
    # The shape pyperf's LoopsTable requires; it refuses anything else.
    assert isinstance(data["table_version"], int)
    assert isinstance(data["min_time"], (int, float))
    for key in ("hostname", "platform", "python", "date"):
        assert data["machine"][key]
    for name, loops in data["loops"].items():
        assert isinstance(name, str)
        assert isinstance(loops, int)

    # Separate assertion, separate reason: the fallback writer must keep
    # claiming the version pyperf expects, or pyperf refuses the file outright.
    assert (
        data["table_version"] == synthesize_loops_file.FALLBACK_TABLE_VERSION
    ), "FALLBACK_TABLE_VERSION has drifted from the version being written"

    # Round-trip through the real reader where there is one: check_loops is a
    # re-implementation of pyperf's validation and can drift from it, which is
    # the whole failure mode this table format was adopted to avoid.
    try:
        from pyperf._loops_table import LoopsTable
    except ImportError:
        pass
    else:
        LoopsTable(data)

    return data["loops"]


def set_loops(output, value):
    with output.open() as f:
        data = json.load(f)
    data["loops"] = {name: value for name in data["loops"]}
    with output.open("w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def test_synthesize():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = pathlib.Path(tmpdir) / "loops.json"
        run_synthesize(output, DATA_PATH)
        expected_data = check_loops(output)

        with pytest.raises(FileExistsError):
            run_synthesize(output, DATA_PATH)

        run_synthesize(output, DATA_PATH, update=True)
        assert expected_data == check_loops(output)

        set_loops(output, 0)
        run_synthesize(output, DATA_PATH, update=True)
        assert expected_data == check_loops(output)

        set_loops(output, sys.maxsize)
        run_synthesize(output, DATA_PATH, overwrite=True)
        assert expected_data == check_loops(output)

        run_synthesize(output, DATA_PATH, overwrite=True, merger="min")
        expected_data = check_loops(output)
        set_loops(output, sys.maxsize)
        run_synthesize(output, DATA_PATH, update=True, merger="min")
        assert expected_data == check_loops(output)


def test_fallback_matches_pyperfs_own_writer():
    """
    The fallback exists because pyperf pins older than the loops table
    feature cannot provide a writer. It is only safe while it produces what
    pyperf itself would, so check that whenever pyperf can tell us.
    """
    pyperf_build = synthesize_loops_file._pyperf_build_table
    if pyperf_build is None:
        pytest.skip("installed pyperf has no loops table support")

    theirs = pyperf_build({"a": 8, "b": 64}, 0.1)

    # Actually exercise the fallback, rather than restating it as a literal: a
    # hand-written copy of the expected dict cannot catch a typo in the code
    # under test, which is the only thing this test exists to catch.
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(synthesize_loops_file, "_pyperf_build_table", None)
        ours = synthesize_loops_file.build_table({"a": 8, "b": 64}, 0.1)

    # Compare the machine block too. pyperf gaining a key there is the obvious
    # way these drift, and excluding it hides exactly that.
    assert ours.keys() == theirs.keys()
    assert ours["machine"].keys() == theirs["machine"].keys()
    assert ours["table_version"] == theirs["table_version"]
    assert ours["min_time"] == theirs["min_time"]
    assert ours["loops"] == theirs["loops"]
    # hostname/platform/python are the same on this machine; date may tick over
    # between the two calls, so compare everything else exactly.
    for key in ("hostname", "platform", "python"):
        assert ours["machine"][key] == theirs["machine"][key]


def test_load_table_rejects_a_results_file():
    """
    The pre---loops-table setup symlinked loops.json to a results file. Feeding
    one to -u must say so, rather than producing a table from nonsense.
    """
    results = next(DATA_PATH.glob("results/**/*.json"))
    with pytest.raises(SystemExit, match="not a loops table"):
        synthesize_loops_file.load_table(results)


def test_load_table_reads_a_table_back(tmp_path):
    output = tmp_path / "loops.json"
    run_synthesize(output, DATA_PATH)
    table = synthesize_loops_file.load_table(output)
    assert table["loops"]
    assert table["table_version"] == synthesize_loops_file.FALLBACK_TABLE_VERSION


def test_machine_comes_from_the_results_not_this_host(tmp_path):
    """
    A table describes the machine its counts were measured on. This script runs
    wherever results are collected, which is not that machine.
    """
    output = tmp_path / "loops.json"
    run_synthesize(output, DATA_PATH)
    with output.open() as f:
        machine = json.load(f)["machine"]

    hostnames = set()
    for results in DATA_PATH.glob("results/**/*.json"):
        with results.open() as f:
            hostname = json.load(f).get("metadata", {}).get("hostname")
        if hostname:
            hostnames.add(hostname)
    assert machine["hostname"] in hostnames


def test_results_from_several_machines_are_refused(tmp_path):
    results = sorted(DATA_PATH.glob("results/**/*.json"))[:1]
    with results[0].open() as f:
        data = json.load(f)

    other = tmp_path / "other-machine.json"
    data.setdefault("metadata", {})["hostname"] = "a-different-machine"

    with other.open("w") as f:
        json.dump(data, f)

    with pytest.raises(SystemExit, match="more than one machine"):
        synthesize_loops_file._main(
            loops_file=tmp_path / "loops.json",
            update=False,
            overwrite=False,
            merger="max",
            results=[*results, other],
        )


def test_min_time_is_recorded_when_given(tmp_path):
    output = tmp_path / "loops.json"
    files = list(DATA_PATH.glob("results/**/*.json"))
    synthesize_loops_file._main(
        loops_file=output,
        update=False,
        overwrite=False,
        merger="max",
        results=files,
        min_time=0.25,
    )
    with output.open() as f:
        assert json.load(f)["min_time"] == 0.25


def test_min_time_defaults_to_pyperfs_default(tmp_path):
    # pyperf does not record --min-time in results, so it cannot be recovered;
    # the table says so by falling back to pyperf's default.
    output = tmp_path / "loops.json"
    run_synthesize(output, DATA_PATH)
    with output.open() as f:
        assert json.load(f)["min_time"] == synthesize_loops_file.DEFAULT_MIN_TIME


def test_conflicting_min_time_is_refused_not_averaged(tmp_path):
    """
    Counts calibrated against different targets are not comparable, so the
    table cannot describe both. Refusing beats silently writing a third value.
    """
    output = tmp_path / "loops.json"
    files = list(DATA_PATH.glob("results/**/*.json"))
    synthesize_loops_file._main(
        loops_file=output,
        update=False,
        overwrite=False,
        merger="max",
        results=files,
        min_time=0.1,
    )
    with pytest.raises(SystemExit, match="disagrees with"):
        synthesize_loops_file._main(
            loops_file=output,
            update=True,
            overwrite=False,
            merger="max",
            results=files,
            min_time=0.5,
        )
