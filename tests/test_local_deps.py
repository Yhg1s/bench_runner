import json
from pathlib import Path


import pytest


from bench_runner import local_deps


DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def checkouts(tmp_path):
    """A directory of plausible local checkouts to point at."""
    root = tmp_path / "checkouts"
    for name in ("pyperf", "pyperformance", "pyston-benchmarks"):
        (root / name).mkdir(parents=True)
    (root / "pyperf" / "pyproject.toml").write_text('[project]\nname = "pyperf"\n')
    (root / "pyperformance" / "setup.py").write_text("from setuptools import setup\n")
    return root


def toml_value(value) -> str:
    """
    Render a Python value as TOML. json.dumps is close enough for strings and
    booleans, but a table is `{key = value}`, not JSON's `{"key": value}`.
    """
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{json.dumps(key)} = {toml_value(item)}" for key, item in value.items()
        )
        return "{" + pairs + "}"
    return json.dumps(value)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """
    Write a bench_runner.toml and make this process the runner it names.

    The config is cached by path, so it has to be cleared on the way in and on
    the way out or one test's config leaks into the next.
    """
    from bench_runner import config as mconfig

    def configure(*, dev_local_deps=None, runner_fields=None):
        lines = [
            "[bases]",
            'versions = ["3.12.0"]',
            "",
            "[runners.testrunner]",
            'os = "linux"',
            'arch = "x86_64"',
            'hostname = "testhost"',
        ]
        for key, value in (runner_fields or {}).items():
            lines.append(f"{key} = {toml_value(value)}")
        if dev_local_deps is not None:
            lines += ["", "[dev.local_deps]"]
            for name, dep in dev_local_deps.items():
                lines.append(f"{json.dumps(name)} = {toml_value(dep)}")

        path = tmp_path / "bench_runner.toml"
        path.write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BENCHMARK_MACHINE_NICKNAME", "testrunner")
        mconfig._load_config.cache_clear()
        return path

    mconfig._load_config.cache_clear()
    yield configure
    mconfig._load_config.cache_clear()


@pytest.fixture(autouse=True)
def no_local_deps_in_the_environment(monkeypatch):
    monkeypatch.delenv(local_deps.ENV_VAR, raising=False)


# ------------------------------------------------------------- spec parsing


def test_from_spec():
    dep = local_deps.LocalDep.from_spec("pyperf=~/python/pyperf")
    assert dep.name == "pyperf"
    assert dep.path == "~/python/pyperf"
    assert dep.editable is True


def test_from_spec_editable_flags():
    assert local_deps.LocalDep.from_spec("a=/x:editable").editable is True
    assert local_deps.LocalDep.from_spec("a=/x:noneditable").editable is False
    assert local_deps.LocalDep.from_spec("a=/x:EDITABLE").editable is True
    assert local_deps.LocalDep.from_spec("a=/x:noneditable").path == "/x"


def test_from_spec_strips_whitespace():
    dep = local_deps.LocalDep.from_spec("  pyperf = ~/python/pyperf : editable ")
    assert dep.name == "pyperf"
    assert dep.path == "~/python/pyperf"
    assert dep.editable is True


def test_from_spec_keeps_a_windows_drive_letter():
    # The flag is only a flag when it is one of the flag words, otherwise
    # every Windows path would lose its drive.
    dep = local_deps.LocalDep.from_spec(r"pyperf=C:\src\pyperf")
    assert dep.path == r"C:\src\pyperf"
    assert dep.editable is True

    dep = local_deps.LocalDep.from_spec(r"pyperf=C:\src\pyperf:noneditable")
    assert dep.path == r"C:\src\pyperf"
    assert dep.editable is False


@pytest.mark.parametrize("spec", ["pyperf", "=/x", "pyperf=", "  ", "pyperf=:editable"])
def test_from_spec_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        local_deps.LocalDep.from_spec(spec)


def test_to_spec_round_trips():
    for spec in ("pyperf=/x/pyperf:editable", "pyperf=/x/pyperf"):
        dep = local_deps.LocalDep.from_spec(spec)
        assert local_deps.LocalDep.from_spec(dep.to_spec()) == dep

    assert local_deps.LocalDep("a", "/x", editable=False).to_spec() == "a=/x"
    assert local_deps.LocalDep("a", "/x").to_spec() == "a=/x:editable"


# ---------------------------------------------------------- the environment


def test_parse_env():
    deps = local_deps.parse_env(
        "pyperf=~/python/pyperf:editable,pyperformance=~/python/pyperformance"
    )
    assert deps is not None
    assert sorted(deps) == ["pyperf", "pyperformance"]
    assert deps["pyperf"].editable is True
    assert deps["pyperformance"].path == "~/python/pyperformance"


def test_parse_env_ignores_empty_entries():
    deps = local_deps.parse_env("pyperf=/x, ,")
    assert deps is not None
    assert sorted(deps) == ["pyperf"]


def test_parse_env_says_nothing_when_unset(monkeypatch):
    assert local_deps.parse_env() is None
    assert local_deps.parse_env("") is None
    assert local_deps.parse_env("   ") is None

    monkeypatch.setenv(local_deps.ENV_VAR, "pyperf=/x")
    parsed = local_deps.parse_env()
    assert parsed is not None
    assert sorted(parsed) == ["pyperf"]


# --------------------------------------------------------- config parsing


def test_config_parses_the_dev_section(config_file):
    from bench_runner import config as mconfig

    cfgpath = config_file(
        dev_local_deps={
            "pyperf": {"path": "~/python/pyperf", "editable": True},
            "pyston-benchmarks": {"path": "~/python/python-macrobenchmarks"},
        }
    )

    dev = mconfig.get_config(cfgpath).dev

    assert isinstance(dev, local_deps.Dev)
    assert sorted(dev.local_deps) == ["pyperf", "pyston-benchmarks"]
    for name, dep in dev.local_deps.items():
        assert isinstance(dep, local_deps.LocalDep)
        # The name comes from the key it was written under.
        assert dep.name == name
    assert dev.local_deps["pyperf"].editable is True
    # Editable defaults on: everything in this dependency chain is pure Python.
    assert dev.local_deps["pyston-benchmarks"].editable is True


def test_config_without_a_dev_section_gets_an_empty_one():
    from bench_runner import config as mconfig

    dev = mconfig.get_config(DATA_PATH / "bench_runner.toml").dev
    assert isinstance(dev, local_deps.Dev)
    assert dev.local_deps == {}


def test_runner_local_deps_are_coerced(config_file):
    from bench_runner import runners as mrunners

    cfgpath = config_file(
        runner_fields={"local_deps": {"pyperf": {"path": "/x", "editable": False}}}
    )

    runner = mrunners.get_runner_by_nickname("testrunner", cfgpath)

    assert runner.local_deps is not None
    dep = runner.local_deps["pyperf"]
    assert isinstance(dep, local_deps.LocalDep)
    assert dep.name == "pyperf"
    assert dep.editable is False


def test_a_runner_says_nothing_by_default(config_file):
    from bench_runner import runners as mrunners

    cfgpath = config_file()
    assert mrunners.get_runner_by_nickname("testrunner", cfgpath).local_deps is None


# ----------------------------------------------------------- path resolution


def test_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    dep = local_deps.LocalDep("pyperf", "~/python/pyperf")
    assert dep.resolved_path() == tmp_path / "python" / "pyperf"


def test_resolves_a_relative_path_against_the_repo_root(tmp_path):
    dep = local_deps.LocalDep("pyperf", "../pyperf")
    assert dep.resolved_path(tmp_path / "results") == tmp_path / "pyperf"


def test_leaves_an_absolute_path_alone(tmp_path):
    dep = local_deps.LocalDep("pyperf", str(tmp_path / "pyperf"))
    assert dep.resolved_path(tmp_path / "elsewhere") == tmp_path / "pyperf"


def test_is_benchmark_repo():
    assert local_deps.LocalDep("pyperformance", "/x").is_benchmark_repo
    assert local_deps.LocalDep("pyston-benchmarks", "/x").is_benchmark_repo
    # A distribution that is installed but never cloned.
    assert not local_deps.LocalDep("pyperf", "/x").is_benchmark_repo


def test_is_installable(checkouts):
    assert local_deps.LocalDep("pyperf", str(checkouts / "pyperf")).is_installable
    assert local_deps.LocalDep(
        "pyperformance", str(checkouts / "pyperformance")
    ).is_installable
    # A checkout that is only ever read in place.
    assert not local_deps.LocalDep(
        "pyston-benchmarks", str(checkouts / "pyston-benchmarks")
    ).is_installable


def test_pip_args():
    assert local_deps.LocalDep("pyperf", "/x").pip_args() == ["-e", "/x"]
    assert local_deps.LocalDep("pyperf", "/x", editable=False).pip_args() == ["/x"]


# ------------------------------------------------------ resolution and precedence


def test_nothing_configured(config_file):
    cfgpath = config_file()
    assert local_deps.get_local_deps(cfgpath) == {}


def test_no_config_file_at_all(tmp_path, monkeypatch, checkouts):
    # Local deps are most useful in a checkout with no bench_runner.toml, so
    # consulting the config must not turn that into an error.
    from bench_runner import config as mconfig

    mconfig._load_config.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "bench_runner.toml").exists()
    try:
        assert local_deps.get_local_deps() == {}

        monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={checkouts / 'pyperf'}")
        assert sorted(local_deps.get_local_deps()) == ["pyperf"]
    finally:
        mconfig._load_config.cache_clear()


def test_from_the_config_section(config_file, checkouts):
    cfgpath = config_file(
        dev_local_deps={"pyperf": {"path": str(checkouts / "pyperf")}}
    )

    deps = local_deps.get_local_deps(cfgpath)

    assert sorted(deps) == ["pyperf"]
    assert deps["pyperf"].path == str(checkouts / "pyperf")


def test_paths_come_back_resolved(config_file, checkouts, tmp_path):
    # Relative to the results repository, which is the working directory.
    relative = (checkouts / "pyperf").relative_to(tmp_path)
    cfgpath = config_file(dev_local_deps={"pyperf": {"path": str(relative)}})

    deps = local_deps.get_local_deps(cfgpath)

    assert Path(deps["pyperf"].path).is_absolute()
    assert Path(deps["pyperf"].path) == checkouts / "pyperf"


def test_the_env_var_beats_the_config_section(config_file, checkouts, monkeypatch):
    cfgpath = config_file(
        dev_local_deps={"pyperf": {"path": str(checkouts / "pyperf")}}
    )
    monkeypatch.setenv(
        local_deps.ENV_VAR, f"pyperformance={checkouts / 'pyperformance'}"
    )

    # Whole-set replacement, not a merge: the most specific source that says
    # anything provides all of it.
    assert sorted(local_deps.get_local_deps(cfgpath)) == ["pyperformance"]


def test_the_runner_beats_the_env_var(config_file, checkouts, monkeypatch):
    cfgpath = config_file(
        dev_local_deps={"pyperf": {"path": str(checkouts / "pyperf")}},
        runner_fields={
            "local_deps": {
                "pyston-benchmarks": {"path": str(checkouts / "pyston-benchmarks")}
            }
        },
    )
    monkeypatch.setenv(
        local_deps.ENV_VAR, f"pyperformance={checkouts / 'pyperformance'}"
    )

    assert sorted(local_deps.get_local_deps(cfgpath)) == ["pyston-benchmarks"]


def test_a_runner_can_turn_local_deps_off(config_file, checkouts, monkeypatch):
    # The reason the runner field is tri-state. One repository, one development
    # machine and several real runners: the real ones opt out.
    cfgpath = config_file(
        dev_local_deps={"pyperf": {"path": str(checkouts / "pyperf")}},
        runner_fields={"local_deps": {}},
    )
    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={checkouts / 'pyperf'}")

    assert local_deps.get_local_deps(cfgpath) == {}


def test_a_missing_checkout_is_refused(config_file, tmp_path):
    cfgpath = config_file(
        dev_local_deps={"pyperf": {"path": str(tmp_path / "not-here")}}
    )

    with pytest.raises(ValueError) as exc:
        local_deps.get_local_deps(cfgpath)

    assert "pyperf" in str(exc.value)
    assert "[dev.local_deps]" in str(exc.value)
    assert str(tmp_path / "not-here") in str(exc.value)


def test_a_checkout_that_is_a_file_is_refused(config_file, tmp_path, monkeypatch):
    not_a_checkout = tmp_path / "pyperf.tar.gz"
    not_a_checkout.write_text("")
    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={not_a_checkout}")
    cfgpath = config_file()

    with pytest.raises(ValueError) as exc:
        local_deps.get_local_deps(cfgpath)

    assert "not a directory" in str(exc.value)
    # The error says which source to go and fix.
    assert local_deps.ENV_VAR in str(exc.value)


def test_a_bad_env_var_is_not_parsed_when_the_runner_overrides_it(
    config_file, checkouts, monkeypatch
):
    cfgpath = config_file(
        runner_fields={"local_deps": {"pyperf": {"path": str(checkouts / "pyperf")}}}
    )
    monkeypatch.setenv(local_deps.ENV_VAR, "this is not a spec")

    assert sorted(local_deps.get_local_deps(cfgpath)) == ["pyperf"]
