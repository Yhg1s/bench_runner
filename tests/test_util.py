"""
Tests for `bench_runner.util`, the shared helper module.
"""

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


import pytest


from bench_runner import util


# ---------------------------------------------------------------------------
# apply_suffix
# ---------------------------------------------------------------------------


def test_apply_suffix_replaces_the_extension():
    assert util.apply_suffix(Path("a/b/c.txt"), ".svg") == Path("a/b/c.svg")


def test_apply_suffix_allows_a_compound_suffix():
    # This is why it exists: Path.with_suffix rejects "-mem.svg".
    assert util.apply_suffix(Path("a/b/c.txt"), "-mem.svg") == Path("a/b/c-mem.svg")


def test_apply_suffix_accepts_a_string_path():
    assert util.apply_suffix("a/b/c.txt", ".md") == Path("a/b/c.md")


def test_apply_suffix_keeps_the_parent_directory():
    assert util.apply_suffix(Path("/x/y/z.json"), ".html").parent == Path("/x/y")


def test_apply_suffix_on_a_name_without_an_extension():
    assert util.apply_suffix(Path("plain"), ".svg") == Path("plain.svg")


def test_apply_suffix_only_strips_the_last_extension():
    assert util.apply_suffix(Path("a.pie.svg"), ".png") == Path("a.pie.png")


# ---------------------------------------------------------------------------
# has_any_element
# ---------------------------------------------------------------------------


def test_has_any_element_true_for_a_non_empty_list():
    assert util.has_any_element([1, 2, 3]) is True


def test_has_any_element_false_for_an_empty_list():
    assert util.has_any_element([]) is False


def test_has_any_element_false_for_an_exhausted_generator():
    assert util.has_any_element(x for x in []) is False


def test_has_any_element_consumes_one_element_of_the_original():
    # BUG, pinned rather than fixed: the docstring and the itertools.tee call
    # both say the original must not be consumed, but tee's children are local
    # and discarded on return, while the caller's own iterator has been
    # advanced by one. So the first element is lost.
    #
    # This bites the only call site, run_benchmarks.py:253:
    #
    #     fileiter = Path(".").glob(perf_data_glob)
    #     if util.has_any_element(fileiter):
    #         perf_to_csv(get_perf_lines(fileiter), ...)
    #
    # which silently drops the first perf.data file of every profiling run.
    #
    # Asserting current behaviour keeps the suite honest; a fix will flip this
    # test, which is the point.
    generator = (x for x in [1, 2, 3])
    assert util.has_any_element(generator) is True
    assert list(generator) == [2, 3]


def test_has_any_element_is_safe_for_re_iterable_sequences():
    # Lists are unaffected, since iter() gives a fresh iterator each time.
    items = [1, 2, 3]
    assert util.has_any_element(items) is True
    assert list(items) == [1, 2, 3]


# ---------------------------------------------------------------------------
# safe_which
# ---------------------------------------------------------------------------


def test_safe_which_returns_the_resolved_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    assert util.safe_which("git") == "/usr/bin/git"


def test_safe_which_raises_when_not_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not found in PATH"):
        util.safe_which("definitely-not-a-command")


# ---------------------------------------------------------------------------
# get_brew_prefix
# ---------------------------------------------------------------------------


def test_get_brew_prefix_strips_the_output(monkeypatch):
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **k: b"/opt/homebrew/opt/pyenv\n"
    )
    assert util.get_brew_prefix("pyenv") == "/opt/homebrew/opt/pyenv"


def test_get_brew_prefix_raises_when_brew_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "brew")

    monkeypatch.setattr(subprocess, "check_output", boom)
    with pytest.raises(RuntimeError, match="Unable to find brew"):
        util.get_brew_prefix("nope")


# ---------------------------------------------------------------------------
# get_simple_platform
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _platform(monkeypatch, name):
    # get_simple_platform is cached, so the cache has to be dropped on the way
    # in and on the way out to avoid leaking a fake platform into other tests.
    util.get_simple_platform.cache_clear()
    monkeypatch.setattr(sys, "platform", name)
    try:
        yield
    finally:
        monkeypatch.undo()
        util.get_simple_platform.cache_clear()


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("linux", "linux"),
        ("linux2", "linux"),
        ("darwin", "macos"),
        ("win32", "windows"),
        ("windows", "windows"),
    ],
)
def test_get_simple_platform(monkeypatch, platform, expected):
    with _platform(monkeypatch, platform):
        assert util.get_simple_platform() == expected


def test_get_simple_platform_rejects_the_unknown(monkeypatch):
    with _platform(monkeypatch, "sunos5"):
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            util.get_simple_platform()


def test_get_simple_platform_is_cached(monkeypatch):
    with _platform(monkeypatch, "linux"):
        first = util.get_simple_platform()
        monkeypatch.setattr(sys, "platform", "darwin")
        # Still the cached answer, not the new one.
        assert util.get_simple_platform() == first


# ---------------------------------------------------------------------------
# format_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "0.00 ns"),
        (1e-9, "1.00 ns"),
        (5e-9, "5.00 ns"),
        (1e-6, "1.00 us"),
        (1.5e-6, "1.50 us"),
        (0.001, "1.00 ms"),
        (0.0123, "12.3 ms"),
        (0.5, "500 ms"),
        (1.0, "1.00 sec"),
        (12.0, "12.0 sec"),
        (123.0, "123 sec"),
    ],
)
def test_format_seconds(value, expected):
    assert util.format_seconds(value) == expected


def test_format_seconds_keeps_three_significant_figures():
    # The precision shrinks as the mantissa grows, within each unit.
    assert util.format_seconds(1.0).endswith("sec")
    for value in (1.0, 12.0, 123.0):
        digits = util.format_seconds(value).split()[0].replace(".", "")
        assert len(digits) == 3


def test_format_seconds_switches_unit_at_each_thousand():
    assert util.format_seconds(1.0).endswith("sec")
    assert util.format_seconds(0.1).endswith("ms")
    assert util.format_seconds(0.0001).endswith("us")
    assert util.format_seconds(0.0000001).endswith("ns")


# ---------------------------------------------------------------------------
# valid_version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["3.14", "3.14.0", "3.14.0a1", "1!2.3", "2020.1"])
def test_valid_version_accepts_pep440(version):
    assert util.valid_version(version) is True


@pytest.mark.parametrize("version", ["not-a-version", "", "3.14.x", "main"])
def test_valid_version_rejects_junk(version):
    assert util.valid_version(version) is False


# ---------------------------------------------------------------------------
# TYPE_TO_ICON
# ---------------------------------------------------------------------------


def test_type_to_icon_covers_the_generated_artifact_types():
    for file_type in ("table", "time plot", "memory plot"):
        assert file_type in util.TYPE_TO_ICON
        assert util.TYPE_TO_ICON[file_type]


def test_type_to_icon_values_are_unique():
    # Two artifact types sharing an icon would be unreadable in the index.
    icons = list(util.TYPE_TO_ICON.values())
    assert len(set(icons)) == len(icons)


# ---------------------------------------------------------------------------
# smart_rmtree
# ---------------------------------------------------------------------------


def test_smart_rmtree_removes_a_tree(tmp_path):
    target = tmp_path / "tree"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "file.txt").write_text("x")

    util.smart_rmtree(target)

    assert not target.exists()


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="read-only handling is the Windows path"
)
def test_smart_rmtree_is_plain_rmtree_off_windows():
    assert util.smart_rmtree is shutil.rmtree


# ---------------------------------------------------------------------------
# log_group / track
# ---------------------------------------------------------------------------


def test_log_group_is_a_context_manager(capsys):
    with util.log_group("doing a thing"):
        pass
    captured = capsys.readouterr()
    assert "doing a thing" in captured.out + captured.err


def test_log_group_propagates_exceptions():
    with pytest.raises(ValueError):
        with util.log_group("failing"):
            raise ValueError("boom")


def test_track_yields_every_element():
    assert list(util.track([1, 2, 3], "counting")) == [1, 2, 3]


def test_track_of_an_empty_iterable():
    assert list(util.track([], "nothing")) == []


def test_track_preserves_order():
    items = list(range(20))
    assert list(util.track(items, "ordered")) == items


def test_log_group_uses_github_actions_markers(monkeypatch):
    # In CI the group markers are what make the log foldable. The branch is
    # chosen at import time, so this reimports the module under the env var.
    import importlib

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    reloaded = importlib.reload(util)
    try:
        import io

        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        with reloaded.log_group("phase"):
            pass
        output = stderr.getvalue()
        assert "::group::phase" in output
        assert "::endgroup::" in output
    finally:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        importlib.reload(util)


def test_track_under_github_actions_still_yields_everything(monkeypatch):
    import importlib

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    reloaded = importlib.reload(util)
    try:
        assert list(reloaded.track([1, 2, 3], "phase")) == [1, 2, 3]
    finally:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        importlib.reload(util)


def test_module_reload_restores_the_normal_helpers():
    # Guards the two tests above: if a reload leaked, TYPE_TO_ICON would still
    # be present but log_group would be the CI variant.
    assert os.getenv("GITHUB_ACTIONS") != "true"
    assert callable(util.log_group)
    assert callable(util.track)
