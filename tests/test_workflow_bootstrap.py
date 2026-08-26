"""The venv-reuse behaviour of the workflow_bootstrap template.

The template is not part of the installed package -- `bench_runner install`
copies it into a results repository -- so it is loaded here straight from the
templates directory.
"""

import importlib.util
from pathlib import Path
import sys

import pytest


TEMPLATE = (
    Path(__file__).parents[1]
    / "bench_runner"
    / "templates"
    / "workflow_bootstrap.src.py"
)


@pytest.fixture(scope="module")
def bootstrap():
    spec = importlib.util.spec_from_file_location("workflow_bootstrap", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_venv(root: Path) -> Path:
    """A directory that looks enough like a venv for the reuse check."""
    exe = root / ("Scripts" if sys.platform.startswith("win") else "bin")
    exe.mkdir(parents=True)
    (exe / ("python.exe" if sys.platform.startswith("win") else "python")).touch()
    return root


def test_reuse_is_off_by_default(bootstrap, monkeypatch):
    monkeypatch.delenv(bootstrap.REUSE_VENV_ENV_VAR, raising=False)
    assert bootstrap.get_reuse_venv() is False


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("yes", True), ("", False), ("0", False),
     ("false", False)],
)
def test_reuse_env_var(bootstrap, monkeypatch, value, expected):
    monkeypatch.setenv(bootstrap.REUSE_VENV_ENV_VAR, value)
    assert bootstrap.get_reuse_venv() is expected


def test_rebuilds_by_default(bootstrap, monkeypatch, tmp_path):
    """Without the variable, an existing venv is removed and made again."""
    monkeypatch.delenv(bootstrap.REUSE_VENV_ENV_VAR, raising=False)
    venv = make_venv(tmp_path / "venv")
    marker = venv / "lib" / "keep-me"
    marker.parent.mkdir()
    marker.touch()

    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "check_call", lambda args, **kw: calls.append(args)
    )

    bootstrap.create_venv(venv)

    assert len(calls) == 1, "expected `python -m venv` to be run"
    assert calls[0][1:3] == ["-m", "venv"]
    assert not marker.exists(), "the old venv should have been removed"


def test_reuse_keeps_an_existing_venv(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setenv(bootstrap.REUSE_VENV_ENV_VAR, "1")
    venv = make_venv(tmp_path / "venv")
    marker = venv / "lib" / "keep-me"
    marker.parent.mkdir()
    marker.touch()

    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "check_call", lambda args, **kw: calls.append(args)
    )

    bootstrap.create_venv(venv)

    assert calls == [], "nothing should have been rebuilt"
    assert marker.exists()


def test_reuse_keeps_nested_benchmark_venvs(bootstrap, monkeypatch, tmp_path):
    """D1: pyperformance nests its benchmark venvs in ./venv/<runid>.

    Rebuilding the outer venv therefore throws away every benchmark venv, which
    costs a full reinstall of every benchmark's requirements. Reuse is one of
    the two ways out; pyperformance's --venvs-dir is the other.
    """
    monkeypatch.setenv(bootstrap.REUSE_VENV_ENV_VAR, "1")
    venv = make_venv(tmp_path / "venv")
    benchmark_venv = make_venv(venv / "cpython3.14-aaaa-compat-bbbb")
    monkeypatch.setattr(bootstrap.subprocess, "check_call", lambda args, **kw: None)

    bootstrap.create_venv(venv)

    assert benchmark_venv.exists()


def test_reuse_still_builds_when_there_is_nothing_to_reuse(
    bootstrap, monkeypatch, tmp_path
):
    monkeypatch.setenv(bootstrap.REUSE_VENV_ENV_VAR, "1")
    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "check_call", lambda args, **kw: calls.append(args)
    )

    bootstrap.create_venv(tmp_path / "venv")

    assert len(calls) == 1
    assert calls[0][1:3] == ["-m", "venv"]


def test_reuse_rebuilds_a_broken_venv(bootstrap, monkeypatch, tmp_path):
    """A directory with no interpreter in it is not something to reuse."""
    monkeypatch.setenv(bootstrap.REUSE_VENV_ENV_VAR, "1")
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "junk").touch()

    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "check_call", lambda args, **kw: calls.append(args)
    )

    bootstrap.create_venv(venv)

    assert len(calls) == 1
    assert not (venv / "junk").exists()


def test_run_in_venv_uses_the_venv_interpreter(bootstrap, monkeypatch, tmp_path):
    venv = make_venv(tmp_path / "venv")
    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "check_call", lambda args, **kw: calls.append(args)
    )

    bootstrap.run_in_venv(venv, "pip", ["install", "-r", "requirements.txt"])

    assert calls == [
        [str(bootstrap.venv_python(venv)), "-m", "pip", "install", "-r",
         "requirements.txt"]
    ]
