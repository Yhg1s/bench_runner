import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


import pytest


from bench_runner import benchmark_definitions
from bench_runner import cache
from bench_runner import git
from bench_runner.scripts import generate_results
from bench_runner.scripts import workflow


DATA_PATH = Path(__file__).parent / "data"


def _copy_repo(tmp_path):
    repo_path = tmp_path / "repo"
    shutil.copytree(DATA_PATH, repo_path)
    return repo_path


def hardcode_benchmark_hash(monkeypatch):
    def dummy(*args, **kwargs):
        return "215d35"

    monkeypatch.setattr(benchmark_definitions, "get_benchmark_hash", dummy)


def test_run_in_venv(tmpdir):
    venv_dir = tmpdir / "venv"

    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    workflow.run_in_venv(venv_dir, "pip", ["install", "rich"])
    workflow.run_in_venv(venv_dir, "rich", [])


def test_should_run_exists_noforce(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is False
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_diff_machine_noforce(benchmarks_checkout, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "darwin",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_all_noforce(benchmarks_checkout, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "all",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_noexists_noforce(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    result = workflow.should_run(
        False,
        "python",
        "main",
        "linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_exists_force(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)

    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    removed_paths = []

    def remove(repo, path):
        removed_paths.append(path)
        (repo / path).unlink()

    monkeypatch.setattr(git, "remove", remove)

    generate_results._main(repo, force=False, bases=["3.11.0b3"])
    result = workflow.should_run(
        True,
        "python",
        "main",
        "linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()
    assert set(x.name for x in removed_paths) == {
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.svg",
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.html",
        "README.md",
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.md",
    }


def test_should_run_noexists_force(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    result = workflow.should_run(
        True,
        "python",
        "main",
        "linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_checkout_failed(tmp_path, capsys, monkeypatch):
    repo = _copy_repo(tmp_path)
    monkeypatch.chdir(repo)
    cpython_path = tmp_path / "cpython"
    cpython_path.mkdir()
    subprocess.check_call(["git", "init"], cwd=cpython_path)

    with pytest.raises(SystemExit):
        workflow.should_run(
            True,
            "python",
            "main",
            "linux",
            False,
            [],
            cpython_path,
            repo / "results",
        )

    captured = capsys.readouterr()
    assert "The checkout of cpython failed" in captured.err
    assert "You specified fork 'python' and ref 'main'" in captured.err


@pytest.mark.long_running
def test_whole_workflow(tmpdir):
    """
    Tests the whole workflow from a clean benchmarking repo.
    """
    repo = tmpdir / "repo"
    venv_dir = repo / "outer_venv"
    bench_runner_checkout = DATA_PATH.parents[1]
    if sys.platform.startswith("win"):
        binary = venv_dir / "Scripts" / "python.exe"
    else:
        binary = venv_dir / "bin" / "python"

    repo.mkdir()

    with contextlib.chdir(repo):
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call(
            [
                str(binary),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )
        subprocess.check_call(
            [str(binary), "-m", "pip", "install", f"{bench_runner_checkout}[test]"]
        )
        subprocess.check_call([str(binary), "-m", "bench_runner", "install"])
        # install --check should never fail immediately after install
        subprocess.check_call(
            [
                str(binary),
                "-m",
                "bench_runner",
                "install",
                "--check",
            ]
        )
        # Now edit one of the generated files to make the check fail
        with open("workflow_bootstrap.py", "a") as fd:
            fd.write("# EXTRA CONTENT\n\n")

        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    str(binary),
                    "-m",
                    "bench_runner",
                    "install",
                    "--check",
                ]
            )

        with open("requirements.txt", "w") as fd:
            fd.write(f"{str(bench_runner_checkout)}\n")
        subprocess.check_call(
            [
                str(binary),
                "workflow_bootstrap.py",
                "python",
                "main",
                "linux-x86_64-linux",
                "deltablue",
                ",,,",
                "--_fast",
            ]
        )


@pytest.mark.long_running
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_pystats(tmpdir):
    """
    Tests the whole workflow from a clean benchmarking repo.
    """
    tmpdir = Path(tmpdir)
    repo = tmpdir / "repo"
    venv_dir = repo / "outer_venv"
    bench_runner_checkout = DATA_PATH.parents[1]
    if sys.platform.startswith("win"):
        binary = venv_dir / "Scripts" / "python.exe"
    else:
        binary = venv_dir / "bin" / "python"
    profiling_dir = Path(repo / "profiling" / "results")

    repo.mkdir()
    profiling_dir.mkdir(parents=True)

    shutil.copyfile(DATA_PATH / "loops.json", repo / "loops.json")

    with contextlib.chdir(repo):
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call(
            [
                str(binary),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )
        subprocess.check_call(
            [str(binary), "-m", "pip", "install", f"{bench_runner_checkout}[test]"]
        )
        subprocess.check_call([str(binary), "-m", "bench_runner", "install"])
        with open("requirements.txt", "w") as fd:
            fd.write(f"{str(bench_runner_checkout)}\n")
        subprocess.check_call(
            [
                str(binary),
                "workflow_bootstrap.py",
                "python",
                "main",
                "linux-x86_64-linux",
                "deltablue",
                ",,,",
                "--_fast",
                "--pystats",
            ]
        )

        deltablue_output = list((repo / "results").glob("**/*-pystats-deltablue.*"))
        assert len(deltablue_output) == 2
        all_output = list((repo / "results").glob("**/*-pystats.*"))
        assert len(all_output) == 2


# ---------------------------------------------------------------------------
# nickname_for_machine
# ---------------------------------------------------------------------------


def test_nickname_for_machine_simple():
    assert workflow.nickname_for_machine("linux-x86_64-pyperf") == "pyperf"


def test_nickname_for_machine_with_dashes():
    """
    The nickname is user-controlled and the README's own examples use dashes,
    so splitting on every dash crashed before any benchmarking started.
    """
    assert workflow.nickname_for_machine("linux-x86_64-linux2-gcc12") == "linux2-gcc12"
    assert workflow.nickname_for_machine("darwin-arm64-m1-mini-2") == "m1-mini-2"


@pytest.mark.parametrize("machine", ["all", "__really_all"])
def test_nickname_for_machine_passes_through_the_wildcards(machine):
    assert workflow.nickname_for_machine(machine) == machine


# ------------------------------------------------- the cross-run package cache


@pytest.fixture
def cache_config(tmp_path, monkeypatch):
    """
    A repository whose bench_runner.toml has the given [cache] settings, with
    the cache pointed somewhere disposable.
    """
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
        lines += [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
        (tmp_path / "bench_runner.toml").write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        mconfig._load_config.cache_clear()

    for name in (
        cache.CACHE_DIR_ENV_VAR,
        cache.ABI_SCOPE_ENV_VAR,
        cache.OFFLINE_ENV_VAR,
        cache.NO_CACHE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    # export_benchmark_pip_cache() writes to os.environ for real -- that is its
    # whole job -- so monkeypatch cannot undo it and the variables would
    # otherwise leak into every test that runs after this one.
    saved = {name: os.environ.get(name) for name in cache.PIP_ENV_VARS}
    mconfig._load_config.cache_clear()
    yield configure
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    mconfig._load_config.cache_clear()


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    """
    A `venv/` whose interpreter is really this one, so the ABI probe works
    without building anything.
    """
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "Scripts").mkdir(parents=True)
    for name in ("bin/python", "Scripts/python.exe"):
        (venv / name).symlink_to(sys.executable)
    cache._probe.cache_clear()
    yield venv
    cache._probe.cache_clear()


@pytest.fixture
def captured_venv_commands(monkeypatch):
    """Record what run_in_venv would have run, instead of running it."""
    calls = []

    def fake_run_in_venv(venv, module, cmd, sudo=False, env=None):
        calls.append((module, cmd, env))

    monkeypatch.setattr(workflow, "run_in_venv", fake_run_in_venv)
    return calls


def test_run_in_venv_passes_the_environment_through(tmp_path, monkeypatch):
    captured = {}

    def fake_check_call(args, env=None):
        captured["args"] = args
        captured["env"] = env

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setenv("SOME_EXISTING_VAR", "kept")

    workflow.run_in_venv(tmp_path, "pip", ["install", "x"], env={"PIP_NO_INDEX": "1"})

    assert captured["env"] is not None
    assert captured["env"]["PIP_NO_INDEX"] == "1"
    # Added to the environment, not substituted for it: pip needs the rest.
    assert captured["env"]["SOME_EXISTING_VAR"] == "kept"


def test_run_in_venv_inherits_the_environment_by_default(tmp_path, monkeypatch):
    captured = {}

    def fake_check_call(args, env=None):
        captured["env"] = env

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    workflow.run_in_venv(tmp_path, "pip", ["install", "x"])

    assert captured["env"] is None


def test_the_pip_cache_is_not_purged(
    cache_config, fake_venv, captured_venv_commands, tmp_path
):
    # D2: this used to run unconditionally, which is why nothing was ever
    # cached between runs.
    cache_config()

    env = workflow.setup_pip_cache(fake_venv, purge=False)

    assert ["cache", "purge"] not in [cmd for _, cmd, _ in captured_venv_commands]
    assert Path(env["PIP_CACHE_DIR"]).is_relative_to(tmp_path / "cache")


def test_setup_pip_cache_returns_the_host_partition(cache_config, fake_venv, tmp_path):
    cache_config(abi_scope="version")

    env = workflow.setup_pip_cache(fake_venv, purge=False)

    partition = Path(env["PIP_CACHE_DIR"])
    assert partition.parent == tmp_path / "cache" / "pip"
    # The outer venv, so the host partition, and the scope the config asked for.
    assert partition.name.startswith("host-")
    assert partition.name.endswith("-v")
    assert (partition / "wheels").is_dir()
    assert (partition / "http-v2").is_symlink()


def test_setup_pip_cache_honours_offline(cache_config, fake_venv):
    cache_config(offline=True)
    env = workflow.setup_pip_cache(fake_venv, purge=False)
    assert env["PIP_NO_INDEX"] == "1"


def test_purge_only_happens_when_asked(cache_config, fake_venv, captured_venv_commands):
    cache_config()

    env = workflow.setup_pip_cache(fake_venv, purge=True)

    assert ("pip", ["cache", "purge"], None) in captured_venv_commands
    # Still cached afterwards: the purge empties pip's own default cache, not
    # ours, so the run still gets a partition.
    assert "PIP_CACHE_DIR" in env


def test_disabling_the_cache_restores_the_old_behaviour(
    cache_config, fake_venv, captured_venv_commands
):
    cache_config(enabled=False)

    env = workflow.setup_pip_cache(fake_venv, purge=False)

    # No environment to merge, so pip uses its own default cache...
    assert env == {}
    # ...and that cache is purged, exactly as it was before any of this.
    assert ("pip", ["cache", "purge"], None) in captured_venv_commands


def test_the_purge_never_targets_our_own_cache(
    cache_config, fake_venv, captured_venv_commands
):
    # `pip cache purge` empties whatever PIP_CACHE_DIR points at, and our
    # partitions share one download directory by symlink, so purging a
    # partition would throw away every other ABI's downloads too.
    cache_config()

    workflow.setup_pip_cache(fake_venv, purge=True)

    for module, cmd, env in captured_venv_commands:
        if cmd == ["cache", "purge"]:
            assert env is None


def test_install_pyperformance_passes_the_cache_environment(captured_venv_commands):
    workflow.install_pyperformance("venv", {"PIP_CACHE_DIR": "/somewhere"})

    assert captured_venv_commands == [
        ("pip", ["install", "./pyperformance"], {"PIP_CACHE_DIR": "/somewhere"})
    ]


def test_venv_python():
    assert workflow.venv_python("venv").parent.name in ("bin", "Scripts")


def test_export_benchmark_pip_cache(cache_config, fake_venv, monkeypatch, tmp_path):
    # The benchmark venvs are built by pyperformance, several layers down, so
    # the only way to reach their pip is through the environment.
    cache_config()
    for var in cache.PIP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")

    partition = Path(env["PIP_CACHE_DIR"])
    assert partition.name.startswith("bm-")
    assert os.environ["PIP_CACHE_DIR"] == str(partition)
    assert os.environ["PIP_FIND_LINKS"] == str(
        tmp_path / "cache" / "wheelhouse" / "pure"
    )


def test_the_two_venvs_get_different_partitions(cache_config, fake_venv, monkeypatch):
    # The outer venv runs the system Python; the benchmark venvs run the
    # CPython that was just built. Usually different interpreters, and between
    # two commits of main, differently built ones.
    cache_config()

    host = workflow.setup_pip_cache(fake_venv, purge=False)
    bench = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")

    assert host["PIP_CACHE_DIR"] != bench["PIP_CACHE_DIR"]
    assert Path(host["PIP_CACHE_DIR"]).name.startswith("host-")
    assert Path(bench["PIP_CACHE_DIR"]).name.startswith("bm-")
    # ...but they share the downloads.
    assert (Path(host["PIP_CACHE_DIR"]) / "http-v2").resolve() == (
        Path(bench["PIP_CACHE_DIR"]) / "http-v2"
    ).resolve()


def test_export_benchmark_pip_cache_when_disabled(cache_config, fake_venv, monkeypatch):
    cache_config(enabled=False)
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)

    assert workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python") == {}
    # Nothing exported, so pyperformance behaves exactly as it did before.
    assert "PIP_CACHE_DIR" not in os.environ


def test_the_benchmark_cache_reaches_pyperformance(
    cache_config, fake_venv, monkeypatch
):
    """
    The two halves of B-3 together: exporting the variables is only useful
    because run_benchmarks names them in --inherit-environ.
    """
    from bench_runner.scripts import run_benchmarks

    cache_config()
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")

    for var in env:
        assert var in run_benchmarks.ENV_VARS, f"{var} would be stripped by pyperf"


def _make_wheel(partition: Path, name: str) -> Path:
    """A wheel where pip would have left one it built itself."""
    path = partition / "wheels" / "3a" / "4b" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return path


def test_collect_pure_wheels(cache_config, fake_venv, tmp_path):
    cache_config()
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")
    partition = Path(env["PIP_CACHE_DIR"])
    built = _make_wheel(partition, "chameleon-4.0-py3-none-any.whl")
    _make_wheel(partition, "greenlet-3.2-cp314-cp314-linux_x86_64.whl")

    added = workflow.collect_pure_wheels(env)

    assert [wheel.name for wheel in added] == ["chameleon-4.0-py3-none-any.whl"]
    wheelhouse = tmp_path / "cache" / "wheelhouse" / "pure"
    assert [w.name for w in wheelhouse.iterdir()] == ["chameleon-4.0-py3-none-any.whl"]
    # Hardlinked, so sharing costs no disk.
    assert (wheelhouse / built.name).stat().st_ino == built.stat().st_ino


def test_collected_wheels_are_on_find_links(cache_config, fake_venv, tmp_path):
    # The point of collecting them: the next run's pip is already looking here.
    cache_config()
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")
    _make_wheel(Path(env["PIP_CACHE_DIR"]), "chameleon-4.0-py3-none-any.whl")

    workflow.collect_pure_wheels(env)

    wheelhouse = tmp_path / "cache" / "wheelhouse" / "pure"
    assert env["PIP_FIND_LINKS"] == str(wheelhouse)
    host = workflow.setup_pip_cache(fake_venv, purge=False)
    assert host["PIP_FIND_LINKS"] == str(wheelhouse)
    assert (wheelhouse / "chameleon-4.0-py3-none-any.whl").is_file()


def test_only_this_run_s_partition_is_swept(cache_config, fake_venv, tmp_path):
    cache_config()
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")
    _make_wheel(Path(env["PIP_CACHE_DIR"]), "mine-1.0-py3-none-any.whl")
    # A partition left behind by some other interpreter.
    other = cache.partition_dir(tmp_path / "cache", "bm", "some-other-abi")
    _make_wheel(other, "theirs-1.0-py3-none-any.whl")

    added = workflow.collect_pure_wheels(env)

    assert [wheel.name for wheel in added] == ["mine-1.0-py3-none-any.whl"]


def test_a_local_dep_wheel_is_never_swept(cache_config, fake_venv, tmp_path):
    """
    The invariant that keeps a local checkout from being masked by a stale
    build. pip refuses to persist a wheel built from an editable install or a
    plain path into `wheels/`, so sweeping only from there is sufficient -- but
    it has to actually be only from there.
    """
    cache_config()
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")
    partition = Path(env["PIP_CACHE_DIR"])
    for stray in (
        partition / "http-v2" / "8a" / "pyperf-2.9-py3-none-any.whl",
        partition / "pyperf-2.9-py3-none-any.whl",
        tmp_path / "cache" / "pyperf-2.9-py3-none-any.whl",
    ):
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("built from a local checkout")

    assert workflow.collect_pure_wheels(env) == []
    assert not (tmp_path / "cache" / "wheelhouse" / "pure").exists() or not list(
        (tmp_path / "cache" / "wheelhouse" / "pure").iterdir()
    )


def test_collect_pure_wheels_when_sharing_is_off(cache_config, fake_venv):
    cache_config(share_pure_wheels=False)
    env = workflow.export_benchmark_pip_cache(fake_venv / "bin" / "python")
    _make_wheel(Path(env["PIP_CACHE_DIR"]), "chameleon-4.0-py3-none-any.whl")

    assert workflow.collect_pure_wheels(env) == []
    # ...and nothing points pip at a wheelhouse either.
    assert "PIP_FIND_LINKS" not in env


def test_collect_pure_wheels_when_the_cache_is_off(cache_config, fake_venv):
    cache_config(enabled=False)
    assert workflow.collect_pure_wheels({}) == []


# ------------------------------------------------- local dependency overrides


@pytest.fixture
def local_checkouts(tmp_path):
    """Plausible local checkouts to override the pinned repositories with."""
    root = tmp_path / "checkouts"
    for name in ("pyperf", "pyperformance", "pyston-benchmarks"):
        (root / name).mkdir(parents=True)
    (root / "pyperf" / "pyproject.toml").write_text('[project]\nname = "pyperf"\n')
    (root / "pyperformance" / "pyproject.toml").write_text(
        '[project]\nname = "pyperformance"\n'
    )
    return root


@pytest.fixture
def captured_clones(monkeypatch):
    """Record which repositories would have been fetched."""
    cloned = []

    def fake_clone(dirname, url, branch=None, depth=None):
        cloned.append(str(dirname))

    monkeypatch.setattr(git, "clone", fake_clone)
    return cloned


def _dep(path, name, editable=True):
    from bench_runner import local_deps

    return local_deps.LocalDep(name=name, path=str(path / name), editable=editable)


def test_checkout_benchmarks_fetches_everything_by_default(captured_clones):
    workflow.checkout_benchmarks()

    assert captured_clones == [
        repo.dirname for repo in benchmark_definitions.BENCHMARK_REPOS
    ]


def test_checkout_benchmarks_skips_an_overridden_repo(
    captured_clones, local_checkouts, capsys
):
    # git.clone() hard-resets an existing directory to the pin, so for a
    # repository someone is editing, "checking out" destroys the very changes
    # the run is meant to measure.
    workflow.checkout_benchmarks(
        {"pyperformance": _dep(local_checkouts, "pyperformance")}
    )

    assert "pyperformance" not in captured_clones
    assert "pyston-benchmarks" in captured_clones

    out = capsys.readouterr().out
    assert "USING LOCAL pyperformance" in out
    assert str(local_checkouts / "pyperformance") in out


def test_checkout_benchmarks_can_skip_every_repo(captured_clones, local_checkouts):
    workflow.checkout_benchmarks(
        {
            "pyperformance": _dep(local_checkouts, "pyperformance"),
            "pyston-benchmarks": _dep(local_checkouts, "pyston-benchmarks"),
        }
    )

    assert captured_clones == []


def test_a_local_dep_that_is_not_a_repo_does_not_skip_a_clone(
    captured_clones, local_checkouts
):
    # pyperf is installed, never cloned, so overriding it changes nothing here.
    workflow.checkout_benchmarks({"pyperf": _dep(local_checkouts, "pyperf")})

    assert captured_clones == [
        repo.dirname for repo in benchmark_definitions.BENCHMARK_REPOS
    ]


def test_install_pyperformance_from_the_pinned_checkout(captured_venv_commands):
    workflow.install_pyperformance("venv")

    assert captured_venv_commands == [("pip", ["install", "./pyperformance"], None)]


def test_install_pyperformance_from_a_local_checkout(
    captured_venv_commands, local_checkouts
):
    workflow.install_pyperformance("venv", None, _dep(local_checkouts, "pyperformance"))

    assert captured_venv_commands == [
        ("pip", ["install", "-e", str(local_checkouts / "pyperformance")], None)
    ]


def test_install_pyperformance_from_a_local_checkout_non_editable(
    captured_venv_commands, local_checkouts
):
    workflow.install_pyperformance(
        "venv", None, _dep(local_checkouts, "pyperformance", editable=False)
    )

    assert captured_venv_commands == [
        ("pip", ["install", str(local_checkouts / "pyperformance")], None)
    ]


def test_install_local_outer_deps(captured_venv_commands, local_checkouts):
    installed = workflow.install_local_outer_deps(
        "venv",
        {
            "pyperf": _dep(local_checkouts, "pyperf"),
            # Handled by install_pyperformance, so not installed twice...
            "pyperformance": _dep(local_checkouts, "pyperformance"),
            # ...and this one is read in place, never installed.
            "pyston-benchmarks": _dep(local_checkouts, "pyston-benchmarks"),
        },
    )

    assert installed == ["pyperf"]
    assert captured_venv_commands == [
        ("pip", ["install", "-e", str(local_checkouts / "pyperf")], None)
    ]


def test_install_local_outer_deps_passes_the_cache_environment(
    captured_venv_commands, local_checkouts
):
    workflow.install_local_outer_deps(
        "venv", {"pyperf": _dep(local_checkouts, "pyperf")}, {"PIP_NO_INDEX": "1"}
    )

    assert captured_venv_commands[0][2] == {"PIP_NO_INDEX": "1"}


def test_install_local_outer_deps_refuses_an_uninstallable_path(
    captured_venv_commands, local_checkouts
):
    # A directory with no packaging metadata, named as an ordinary dependency
    # rather than as a benchmark repository, is a configuration error. pip
    # would say so too, but only after the CPython build.
    from bench_runner import local_deps

    dep = local_deps.LocalDep(
        name="notapackage", path=str(local_checkouts / "pyston-benchmarks")
    )
    with pytest.raises(ValueError, match="pyproject.toml"):
        workflow.install_local_outer_deps("venv", {"notapackage": dep})

    assert captured_venv_commands == []


def test_nothing_changes_without_local_deps(captured_venv_commands):
    assert workflow.install_local_outer_deps("venv", {}) == []
    assert captured_venv_commands == []


# ------------------------------------------ the CI gate and the venv location


@pytest.fixture(autouse=True)
def not_in_ci(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv(workflow.ALLOW_LOCAL_DEPS_ENV_VAR, raising=False)
    monkeypatch.delenv(workflow.VENV_ENV_VAR, raising=False)


def test_in_github_actions(monkeypatch):
    assert workflow.in_github_actions() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert workflow.in_github_actions() is True
    # Any non-empty value counts: this gates a safety check, so the failure to
    # avoid is deciding we are not in CI when we are.
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    assert workflow.in_github_actions() is True
    monkeypatch.setenv("GITHUB_ACTIONS", "")
    assert workflow.in_github_actions() is False


def test_local_deps_are_fine_outside_ci(local_checkouts):
    workflow.check_local_deps_allowed({"pyperf": _dep(local_checkouts, "pyperf")})


def test_no_local_deps_is_fine_in_ci(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    workflow.check_local_deps_allowed({})


def test_local_deps_in_ci_are_refused(monkeypatch, local_checkouts):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    with pytest.raises(RuntimeError) as exc:
        workflow.check_local_deps_allowed({"pyperf": _dep(local_checkouts, "pyperf")})

    # The message has to say which dep, and how to proceed on purpose.
    assert "pyperf" in str(exc.value)
    assert str(local_checkouts / "pyperf") in str(exc.value)
    assert workflow.ALLOW_LOCAL_DEPS_ENV_VAR in str(exc.value)


def test_local_deps_in_ci_can_be_allowed(monkeypatch, local_checkouts, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv(workflow.ALLOW_LOCAL_DEPS_ENV_VAR, "1")

    workflow.check_local_deps_allowed({"pyperf": _dep(local_checkouts, "pyperf")})

    assert "benchmarking unpushed code" in capsys.readouterr().out


def test_the_allow_flag_must_say_something(monkeypatch, local_checkouts):
    # An empty or "0" value is not an opt-in.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for value in ("", "0"):
        monkeypatch.setenv(workflow.ALLOW_LOCAL_DEPS_ENV_VAR, value)
        with pytest.raises(RuntimeError):
            workflow.check_local_deps_allowed(
                {"pyperf": _dep(local_checkouts, "pyperf")}
            )


def test_local_workflow_is_exempt(monkeypatch, local_checkouts):
    # local_workflow is the deliberate local-development entry point and stays
    # usable even on a runner.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    workflow.check_local_deps_allowed(
        {"pyperf": _dep(local_checkouts, "pyperf")}, exempt=True
    )


def test_local_workflow_passes_the_exemption(monkeypatch):
    from bench_runner.scripts import local_workflow

    captured = {}
    monkeypatch.setattr(local_workflow, "set_environment_for", lambda machine: None)
    monkeypatch.setattr(
        workflow, "_main", lambda *args, **kwargs: captured.update(kwargs)
    )

    local_workflow._main("python", "main", "linux-x86_64-linux", "nbody")

    assert captured["allow_local_deps"] is True


def test_venv_dir_defaults_to_the_running_venv(monkeypatch):
    # No reason a local developer's outer venv has to sit at
    # <results-repo>/venv.
    monkeypatch.setattr(sys, "prefix", "/somewhere/myvenv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")

    assert workflow.get_venv_dir() == Path("/somewhere/myvenv")


def test_venv_dir_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv(workflow.VENV_ENV_VAR, str(tmp_path / "elsewhere"))
    assert workflow.get_venv_dir() == tmp_path / "elsewhere"

    # The override wins in CI too.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert workflow.get_venv_dir() == tmp_path / "elsewhere"


def test_venv_dir_in_ci_is_unchanged(monkeypatch):
    # The bootstrap creates ./venv and runs the workflow inside it, so this
    # path stays exactly what it always was.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(sys, "prefix", "/somewhere/myvenv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")

    assert workflow.get_venv_dir() == Path("venv")


def test_venv_dir_never_targets_a_system_python(monkeypatch):
    # Not running inside a venv: pip installing into sys.prefix would mean
    # system site-packages.
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")

    assert workflow.get_venv_dir() == Path("venv")
