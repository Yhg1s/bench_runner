import dataclasses
import json
import os
from pathlib import Path
import sys
import time


import pytest


from bench_runner import cache


PYTHON_INSTALLS = Path("~/python-installs").expanduser()


# A stand-in for `sysconfig` that reports whatever include directory, SOABI and
# platform the test wants. Putting it on PYTHONPATH shadows the standard
# library module for the probe subprocess, which lets us exercise the real
# probe -- subprocess, JSON and all -- against synthetic interpreters, without
# needing to build one.
STUB_SYSCONFIG = """
import importlib.machinery
import importlib.util
import os

_stdlib = os.path.dirname(os.__file__)
_spec = importlib.machinery.PathFinder.find_spec("sysconfig", [_stdlib])
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)


def __getattr__(name):
    return getattr(_real, name)


def get_paths(*args, **kwargs):
    paths = dict(_real.get_paths(*args, **kwargs))
    paths["include"] = os.environ["STUB_INCLUDE"]
    return paths


def get_config_h_filename():
    return os.path.join(os.environ["STUB_INCLUDE"], "pyconfig.h")


def get_config_var(name):
    if name == "SOABI":
        return os.environ["STUB_SOABI"]
    return _real.get_config_var(name)


def get_platform():
    return os.environ["STUB_PLATFORM"]
"""


@dataclasses.dataclass
class StubInterpreter:
    """A synthetic interpreter whose ABI inputs the test controls."""

    stub_dir: Path
    include: Path
    soabi: str = "cpython-999-stub"
    platform: str = "stub-platform"

    def write_header(self, relpath: str, content: str) -> None:
        path = self.include / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def key(self, monkeypatch, scope=None) -> str:
        monkeypatch.setenv("PYTHONPATH", str(self.stub_dir))
        monkeypatch.setenv("STUB_INCLUDE", str(self.include))
        monkeypatch.setenv("STUB_SOABI", self.soabi)
        monkeypatch.setenv("STUB_PLATFORM", self.platform)
        # Same executable every time, so the memoized probe has to be dropped.
        cache._probe.cache_clear()
        return cache.abi_key(sys.executable, scope)


@pytest.fixture
def stub(tmp_path) -> StubInterpreter:
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "sysconfig.py").write_text(STUB_SYSCONFIG)
    include = tmp_path / "include"
    include.mkdir()
    interpreter = StubInterpreter(stub_dir=stub_dir, include=include)
    interpreter.write_header("Python.h", "#define Py_LIMITED_API 1\n")
    interpreter.write_header("cpython/pystate.h", "struct PyThreadState { int x; };\n")
    interpreter.write_header("pyconfig.h", "#define WITH_PYMALLOC 1\n")
    return interpreter


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (
        cache.CACHE_DIR_ENV_VAR,
        cache.ABI_SCOPE_ENV_VAR,
        cache.OFFLINE_ENV_VAR,
        cache.NO_CACHE_ENV_VAR,
        "PYTHONPATH",
    ):
        monkeypatch.delenv(name, raising=False)
    cache._probe.cache_clear()
    yield
    cache._probe.cache_clear()


def find_install(name: str) -> Path | None:
    for exe in ("python3", "python"):
        path = PYTHON_INSTALLS / name / "bin" / exe
        if path.exists():
            return path
    return None


def requires_install(name: str) -> Path:
    path = find_install(name)
    if path is None:
        pytest.skip(f"No {name} interpreter in {PYTHON_INSTALLS}")
    return path


# ---------------------------------------------------------------- cache root


def test_get_cache_root_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(cache.CACHE_DIR_ENV_VAR, str(tmp_path / "from-env"))
    assert cache.get_cache_root() == tmp_path / "from-env"
    # The environment wins over the configured value.
    assert cache.get_cache_root(tmp_path / "from-config") == tmp_path / "from-env"


def test_get_cache_root_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.get_cache_root(tmp_path / "from-config") == tmp_path / "from-config"
    # An empty `[cache].dir` means "unset", not "the current directory".
    assert cache.get_cache_root("") == tmp_path / "xdg" / "bench_runner"


def test_get_cache_root_expands_user(monkeypatch):
    monkeypatch.setenv(cache.CACHE_DIR_ENV_VAR, "~/some-cache")
    assert cache.get_cache_root() == Path.home() / "some-cache"


def test_get_cache_root_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.get_cache_root() == tmp_path / "xdg" / "bench_runner"


def test_get_cache_root_home_fallback(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache.get_cache_root() == Path.home() / ".cache" / "bench_runner"


def test_get_cache_root_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.get_cache_root() == tmp_path / "AppData" / "bench_runner"


# ----------------------------------------------------------------- ABI scope


def test_resolve_abi_scope():
    assert cache.resolve_abi_scope() == "headers"
    assert cache.resolve_abi_scope("build") == "build"
    assert cache.resolve_abi_scope(None) == "headers"


def test_resolve_abi_scope_env_var(monkeypatch):
    monkeypatch.setenv(cache.ABI_SCOPE_ENV_VAR, "version")
    assert cache.resolve_abi_scope() == "version"
    assert cache.resolve_abi_scope("build") == "version"


def test_resolve_abi_scope_invalid(monkeypatch):
    with pytest.raises(ValueError):
        cache.resolve_abi_scope("paranoid")
    monkeypatch.setenv(cache.ABI_SCOPE_ENV_VAR, "paranoid")
    with pytest.raises(ValueError):
        cache.resolve_abi_scope()


# ------------------------------------------------------------------- ABI key


def test_abi_key_is_stable(stub, monkeypatch):
    assert stub.key(monkeypatch) == stub.key(monkeypatch)


def test_abi_key_shape(stub, monkeypatch):
    key = stub.key(monkeypatch)
    major, minor = sys.version_info[:2]
    prefix = (
        f"{sys.implementation.name}{major}.{minor}-cpython-999-stub-stub-platform-h"
    )
    assert key.startswith(prefix)
    assert len(key) == len(prefix) + cache.DIGEST_LENGTH
    # Usable as a directory name.
    assert Path(key).name == key


def test_abi_key_changes_when_a_header_changes(stub, monkeypatch):
    before = stub.key(monkeypatch)
    stub.write_header("cpython/pystate.h", "struct PyThreadState { int x; int y; };\n")
    assert stub.key(monkeypatch) != before


def test_abi_key_changes_when_a_header_is_added(stub, monkeypatch):
    before = stub.key(monkeypatch)
    stub.write_header("internal/pycore_new.h", "#define NEW 1\n")
    assert stub.key(monkeypatch) != before


def test_abi_key_changes_when_a_header_is_removed(stub, monkeypatch):
    before = stub.key(monkeypatch)
    (stub.include / "cpython" / "pystate.h").unlink()
    assert stub.key(monkeypatch) != before


def test_abi_key_changes_when_pyconfig_h_changes(stub, monkeypatch):
    """A --enable-experimental-jit build is partitioned by pyconfig.h alone."""
    before = stub.key(monkeypatch)
    stub.write_header("pyconfig.h", "#define WITH_PYMALLOC 1\n#define _Py_JIT 1\n")
    assert stub.key(monkeypatch) != before


def test_abi_key_ignores_non_headers(stub, monkeypatch):
    before = stub.key(monkeypatch)
    (stub.include / "README.txt").write_text("not a header")
    (stub.include / "cpython" / "notes.md").write_text("still not a header")
    assert stub.key(monkeypatch) == before


def test_abi_key_soabi_and_platform(stub, monkeypatch):
    before = stub.key(monkeypatch)

    stub.soabi = "cpython-999t-stub"
    free_threaded = stub.key(monkeypatch)
    assert free_threaded != before
    assert "cpython-999t-stub" in free_threaded

    stub.soabi = "cpython-999-stub"
    stub.platform = "stub-other"
    assert stub.key(monkeypatch) != before


def test_abi_key_version_scope_ignores_headers(stub, monkeypatch):
    before = stub.key(monkeypatch, "version")
    assert before.endswith("-v")
    stub.write_header("cpython/pystate.h", "struct PyThreadState { double x; };\n")
    assert stub.key(monkeypatch, "version") == before


def test_abi_key_build_scope_ignores_headers(stub, monkeypatch):
    before = stub.key(monkeypatch, "build")
    assert "-b" in before
    stub.write_header("cpython/pystate.h", "struct PyThreadState { double x; };\n")
    assert stub.key(monkeypatch, "build") == before


def test_abi_key_scopes_never_collide(stub, monkeypatch):
    keys = {scope: stub.key(monkeypatch, scope) for scope in cache.ABI_SCOPES}
    assert len(set(keys.values())) == len(cache.ABI_SCOPES)


def test_abi_key_without_headers_falls_back_to_build(stub, monkeypatch):
    """
    An interpreter with no headers installed tells us nothing about its ABI, so
    it must not land in a partition shared with every other such interpreter.
    """
    for header in sorted(stub.include.glob("**/*.h")):
        header.unlink()
    key = stub.key(monkeypatch)
    _, _, digest = key.rpartition("-")
    assert digest.startswith("b")
    assert key == stub.key(monkeypatch, "build")


def test_abi_key_of_a_broken_interpreter(tmp_path):
    with pytest.raises(cache.CacheError):
        cache.abi_key(tmp_path / "does-not-exist" / "python3")


# ------------------------------------------- ABI key, against real CPythons


def test_abi_key_across_versions():
    """3.13, 3.14 and 3.14t must never share a partition."""
    exes = {name: find_install(name) for name in ("3.13", "3.14", "3.14t")}
    missing = [name for name, exe in exes.items() if exe is None]
    if missing:
        pytest.skip(f"No {', '.join(missing)} interpreter in {PYTHON_INSTALLS}")

    for scope in cache.ABI_SCOPES:
        keys = {name: cache.abi_key(exe, scope) for name, exe in exes.items()}
        assert len(set(keys.values())) == 3, f"{scope}: {keys}"


def test_abi_key_across_builds_of_one_version():
    """
    The hazard the ABI key exists for: two builds of the same X.Y with the same
    wheel compatibility tag, which pip's own tag filter cannot tell apart.
    """
    plain = requires_install("3.14")
    other = requires_install("3.14-opt")

    assert cache.abi_key(plain, "version") == cache.abi_key(other, "version")
    assert cache.abi_key(plain, "headers") != cache.abi_key(other, "headers")
    assert cache.abi_key(plain, "build") != cache.abi_key(other, "build")


def test_abi_key_does_not_depend_on_the_executable_path():
    """One install reached by two names is one ABI, not two."""
    exe = requires_install("3.14")
    alias = exe.parent / "python3.14"
    if not alias.exists():
        pytest.skip(f"No {alias}")
    assert cache.abi_key(exe) == cache.abi_key(alias)


# -------------------------------------------------------------------- layout


def test_pip_env_layout(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)  # Sets up the stub environment.
    env = cache.pip_env(sys.executable, "bm", root=root)

    partition = Path(env["PIP_CACHE_DIR"])
    assert partition.parent == root / "pip"
    assert partition.name.startswith("bm-")
    assert (partition / "wheels").is_dir()
    assert not (partition / "wheels").is_symlink()

    for name in cache.HTTP_CACHE_DIRS:
        link = partition / name
        assert link.is_symlink()
        assert link.resolve() == (root / "pip" / "_shared" / name).resolve()
        # Relative, so the whole cache can be moved.
        assert not os.path.isabs(os.readlink(link))

    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["PIP_FIND_LINKS"] == str(root / "wheelhouse" / "pure")
    assert (root / "wheelhouse" / "pure").is_dir()
    assert "PIP_NO_INDEX" not in env
    assert set(env) <= set(cache.PIP_ENV_VARS)


def test_pip_env_shares_downloads_between_partitions(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    host = Path(cache.pip_env(sys.executable, "host", root=root)["PIP_CACHE_DIR"])
    bench = Path(cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"])

    assert host != bench
    assert (host / "wheels").resolve() != (bench / "wheels").resolve()
    assert (host / "http-v2").resolve() == (bench / "http-v2").resolve()

    (host / "http-v2" / "downloaded").write_text("bytes")
    assert (bench / "http-v2" / "downloaded").read_text() == "bytes"


def test_pip_env_partitions_by_abi(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    first = cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"]

    stub.write_header("cpython/pystate.h", "struct PyThreadState { double x; };\n")
    stub.key(monkeypatch)
    second = cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"]

    assert first != second


def test_pip_env_is_idempotent(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    first = cache.pip_env(sys.executable, "bm", root=root)
    partition = Path(first["PIP_CACHE_DIR"])
    (partition / "wheels" / "keep-me").write_text("x")

    stub.key(monkeypatch)
    assert cache.pip_env(sys.executable, "bm", root=root) == first
    assert (partition / "wheels" / "keep-me").read_text() == "x"
    assert (partition / "http-v2").is_symlink()


def test_pip_env_offline(stub, monkeypatch, tmp_path):
    stub.key(monkeypatch)
    env = cache.pip_env(sys.executable, root=tmp_path, offline=True)
    assert env["PIP_NO_INDEX"] == "1"

    stub.key(monkeypatch)
    assert "PIP_NO_INDEX" not in cache.pip_env(sys.executable, root=tmp_path)

    monkeypatch.setenv(cache.OFFLINE_ENV_VAR, "1")
    stub.key(monkeypatch)
    env = cache.pip_env(sys.executable, root=tmp_path, offline=False)
    assert env["PIP_NO_INDEX"] == "1"


def test_pip_env_find_links(stub, monkeypatch, tmp_path):
    stub.key(monkeypatch)
    env = cache.pip_env(
        sys.executable,
        root=tmp_path,
        share_pure_wheels=False,
        extra_find_links=[tmp_path / "elsewhere"],
    )
    assert env["PIP_FIND_LINKS"] == str(tmp_path / "elsewhere")

    stub.key(monkeypatch)
    env = cache.pip_env(sys.executable, root=tmp_path, share_pure_wheels=False)
    assert "PIP_FIND_LINKS" not in env


def test_pip_env_invalid_kind(stub, monkeypatch, tmp_path):
    stub.key(monkeypatch)
    with pytest.raises(ValueError):
        cache.pip_env(sys.executable, "outer", root=tmp_path)  # pyright: ignore


def test_pip_env_without_symlinks(stub, monkeypatch, tmp_path):
    """Windows without developer mode: a per-ABI download cache, not a crash."""

    def no_symlinks(self, target, target_is_directory=False):
        raise OSError("symbolic links are not supported")

    monkeypatch.setattr(Path, "symlink_to", no_symlinks)

    root = tmp_path / "cache"
    stub.key(monkeypatch)
    host = Path(cache.pip_env(sys.executable, "host", root=root)["PIP_CACHE_DIR"])
    bench = Path(cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"])

    for partition in (host, bench):
        for name in cache.HTTP_CACHE_DIRS:
            assert (partition / name).is_dir()
            assert not (partition / name).is_symlink()
    assert (host / "http-v2").resolve() != (bench / "http-v2").resolve()

    # An existing real directory is left alone once symlinks work again.
    monkeypatch.undo()
    stub.key(monkeypatch)
    cache.pip_env(sys.executable, "host", root=root)
    assert not (host / "http-v2").is_symlink()


# ------------------------------------------------------------ wheel sweeping


def make_wheel(partition: Path, name: str, *, subdir: str = "3a/4b") -> Path:
    path = partition / "wheels" / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return path


def test_sweep_pure_wheels(tmp_path):
    root = tmp_path / "cache"
    bench = cache.partition_dir(root, "bm", "abi-one")
    pure = make_wheel(bench, "chameleon-4.0-py3-none-any.whl")
    make_wheel(bench, "greenlet-3.2-cp314-cp314-linux_x86_64.whl")

    added = cache.sweep_pure_wheels(root)

    assert [path.name for path in added] == ["chameleon-4.0-py3-none-any.whl"]
    wheelhouse = cache.pure_wheelhouse_dir(root)
    assert [path.name for path in wheelhouse.iterdir()] == [
        "chameleon-4.0-py3-none-any.whl"
    ]
    # Hardlinked, so the wheelhouse costs nothing.
    linked = wheelhouse / "chameleon-4.0-py3-none-any.whl"
    assert linked.stat().st_ino == pure.stat().st_ino

    # Already there the second time around.
    assert cache.sweep_pure_wheels(root) == []


def test_sweep_pure_wheels_across_partitions(tmp_path):
    root = tmp_path / "cache"
    make_wheel(cache.partition_dir(root, "bm", "abi-one"), "a-1-py3-none-any.whl")
    make_wheel(cache.partition_dir(root, "host", "abi-two"), "b-1-py3-none-any.whl")

    assert sorted(path.name for path in cache.sweep_pure_wheels(root)) == [
        "a-1-py3-none-any.whl",
        "b-1-py3-none-any.whl",
    ]


def test_sweep_pure_wheels_only_from_wheels_dirs(tmp_path):
    """
    The rule that keeps local dependencies out of the shared wheelhouse: only
    pip's `wheels/` directory is swept, and pip never persists a wheel built
    from an editable install or a local path there.
    """
    root = tmp_path / "cache"
    bench = cache.partition_dir(root, "bm", "abi-one")
    for stray in (
        bench / "http-v2" / "8a" / "pyperf-2.9-py3-none-any.whl",
        bench / "pyperf-2.9-py3-none-any.whl",
        root / "pyperf-2.9-py3-none-any.whl",
    ):
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("local")
    (bench / "wheels").mkdir(parents=True, exist_ok=True)

    assert cache.sweep_pure_wheels(root) == []


def test_sweep_pure_wheels_partition_filter(tmp_path):
    root = tmp_path / "cache"
    make_wheel(cache.partition_dir(root, "bm", "abi-one"), "a-1-py3-none-any.whl")
    make_wheel(cache.partition_dir(root, "host", "abi-two"), "b-1-py3-none-any.whl")

    added = cache.sweep_pure_wheels(root, partitions=["bm-abi-one"])
    assert [path.name for path in added] == ["a-1-py3-none-any.whl"]


def test_sweep_pure_wheels_no_cache(tmp_path):
    assert cache.sweep_pure_wheels(tmp_path / "nothing-here") == []


# ------------------------------------------------------------------- pruning


def age(path: Path, days: float) -> None:
    when = time.time() - days * 86400
    os.utime(path, (when, when))


def test_prune(tmp_path):
    root = tmp_path / "cache"
    bench = cache.partition_dir(root, "bm", "abi-one")
    old = make_wheel(bench, "old-1-py3-none-any.whl")
    new = make_wheel(bench, "new-1-py3-none-any.whl")
    age(old, 45)
    age(new, 3)

    result = cache.prune(30, root)

    assert not old.exists()
    assert new.exists()
    assert result.files_removed == 1
    assert result.bytes_removed == len("old-1-py3-none-any.whl")
    assert result.partitions_removed == []


def test_prune_keeps_recently_read_files(tmp_path):
    root = tmp_path / "cache"
    bench = cache.partition_dir(root, "bm", "abi-one")
    wheel = make_wheel(bench, "old-1-py3-none-any.whl")
    # Written long ago, but read on the last run.
    os.utime(wheel, (time.time(), time.time() - 45 * 86400))

    assert cache.prune(30, root).files_removed == 0
    assert wheel.exists()


def test_prune_counts_shared_downloads_once(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    cache.pip_env(sys.executable, "host", root=root)
    stub.key(monkeypatch)
    cache.pip_env(sys.executable, "bm", root=root)

    download = cache.shared_dir(root) / "http-v2" / "0a" / "body"
    download.parent.mkdir(parents=True, exist_ok=True)
    download.write_text("x" * 100)
    age(download, 45)

    result = cache.prune(30, root)

    assert result.files_removed == 1
    assert result.bytes_removed == 100
    assert not download.exists()
    # The symlinks survive, and still point somewhere real.
    for kind in ("host", "bm"):
        for partition in (root / "pip").glob(f"{kind}-*"):
            for name in cache.HTTP_CACHE_DIRS:
                assert (partition / name).is_symlink()
                assert (partition / name).is_dir()


def test_prune_removes_dead_partitions(tmp_path):
    root = tmp_path / "cache"
    dead = cache.partition_dir(root, "bm", "abi-dead")
    wheel = make_wheel(dead, "old-1-py3-none-any.whl")
    age(wheel, 45)
    live = cache.partition_dir(root, "bm", "abi-live")
    make_wheel(live, "new-1-py3-none-any.whl")

    for partition in (dead, live):
        for name in cache.HTTP_CACHE_DIRS:
            (partition / name).symlink_to(
                os.path.relpath(cache.shared_dir(root) / name, partition),
                target_is_directory=True,
            )
    age(dead, 45)

    result = cache.prune(30, root)

    assert result.partitions_removed == ["bm-abi-dead"]
    assert not dead.exists()
    assert live.exists()


def test_prune_keeps_a_freshly_created_partition(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    partition = Path(cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"])

    assert cache.prune(0.5, root).partitions_removed == []
    assert partition.is_dir()


def test_prune_no_cache(tmp_path):
    result = cache.prune(30, tmp_path / "nothing-here")
    assert result == cache.PruneResult()


def test_prune_rejects_a_negative_age(tmp_path):
    with pytest.raises(ValueError):
        cache.prune(-1, tmp_path)


# ---------------------------------------------------------------------- info


def test_info_of_a_missing_cache(tmp_path):
    result = cache.info(tmp_path / "nothing-here")
    assert not result.exists
    assert result.partitions == []
    assert "does not exist" in cache.format_info(result)


def test_info(stub, monkeypatch, tmp_path):
    root = tmp_path / "cache"
    stub.key(monkeypatch)
    host = Path(cache.pip_env(sys.executable, "host", root=root)["PIP_CACHE_DIR"])
    stub.key(monkeypatch)
    bench = Path(cache.pip_env(sys.executable, "bm", root=root)["PIP_CACHE_DIR"])

    make_wheel(host, "a-1-py3-none-any.whl")
    make_wheel(bench, "b-1-cp314-cp314-linux_x86_64.whl")
    download = cache.shared_dir(root) / "http-v2" / "0a" / "body"
    download.parent.mkdir(parents=True, exist_ok=True)
    download.write_text("x" * 500)
    cache.sweep_pure_wheels(root)

    result = cache.info(root)

    assert result.exists
    assert result.root == root
    assert result.shared_files == 1
    assert result.shared_size == 500
    assert result.pure_wheel_count == 1
    assert {partition.name for partition in result.partitions} == {
        host.name,
        bench.name,
    }
    assert {partition.kind for partition in result.partitions} == {"host", "bm"}
    for partition in result.partitions:
        assert partition.wheel_count == 1
        assert partition.abi == partition.name.split("-", 1)[1]

    # Every real file counted exactly once. The shared download is symlinked
    # into both partitions but only counted where it really lives; the pure
    # wheel is hardlinked into the wheelhouse, so it is counted twice.
    assert result.total_size == (
        500 + len("a-1-py3-none-any.whl") * 2 + len("b-1-cp314-cp314-linux_x86_64.whl")
    )

    text = cache.format_info(result)
    assert str(root) in text
    assert host.name in text and bench.name in text


def test_format_size():
    assert cache.format_size(0) == "0.0 B"
    assert cache.format_size(1536) == "1.5 kB"
    assert cache.format_size(3 * 1024 * 1024) == "3.0 MB"


# --------------------------------------------------------------------- misc


def test_clear(tmp_path):
    root = tmp_path / "cache"
    make_wheel(cache.partition_dir(root, "bm", "abi-one"), "a-1-py3-none-any.whl")

    assert cache.clear(root) is True
    assert not root.exists()
    assert cache.clear(root) is False


def test_is_enabled(monkeypatch):
    assert cache.is_enabled() is True
    assert cache.is_enabled(False) is False

    monkeypatch.setenv(cache.NO_CACHE_ENV_VAR, "1")
    assert cache.is_enabled() is False
    assert cache.is_enabled(True) is False


def test_is_offline(monkeypatch):
    assert cache.is_offline() is False
    assert cache.is_offline(True) is True

    monkeypatch.setenv(cache.OFFLINE_ENV_VAR, "yes")
    assert cache.is_offline() is True
    assert cache.is_offline(False) is True


def test_env_flag_is_not_fooled_by_a_false_value(monkeypatch):
    for value in ("0", "false", "no", "", "  "):
        monkeypatch.setenv(cache.NO_CACHE_ENV_VAR, value)
        assert cache.is_enabled() is True


# ------------------------------------------------- the [cache] config section


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """
    Write a bench_runner.toml with a [cache] section and chdir into it.

    The config is cached by path, so it has to be cleared on the way in and on
    the way out or one test's config leaks into the next.
    """
    from bench_runner import config as mconfig

    def configure(**settings):
        lines = [
            "[bases]",
            'versions = ["3.12.0"]',
            "",
            "[runners.testrunner]",
            'os = "linux"',
            'arch = "x86_64"',
            'hostname = "testhost"',
        ]
        if settings:
            lines += ["", "[cache]"]
            lines += [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
        path = tmp_path / "bench_runner.toml"
        path.write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        mconfig._load_config.cache_clear()
        return path

    mconfig._load_config.cache_clear()
    yield configure
    mconfig._load_config.cache_clear()


def test_cache_section_defaults():
    configured = cache.Cache()
    assert configured.enabled is True
    assert configured.dir == ""
    assert configured.abi_scope == "headers"
    assert configured.offline is False
    assert configured.share_pure_wheels is True
    assert configured.max_age_days == 30


def test_config_parses_the_cache_section(config_file):
    from bench_runner import config as mconfig

    cfgpath = config_file(
        enabled=False,
        dir="/tmp/somewhere",
        abi_scope="build",
        offline=True,
        share_pure_wheels=False,
        max_age_days=7,
    )

    configured = mconfig.get_config(cfgpath).cache

    assert isinstance(configured, cache.Cache)
    assert configured.enabled is False
    assert configured.dir == "/tmp/somewhere"
    assert configured.abi_scope == "build"
    assert configured.offline is True
    assert configured.share_pure_wheels is False
    assert configured.max_age_days == 7


def test_config_without_a_cache_section_gets_the_defaults(config_file):
    from bench_runner import config as mconfig

    configured = mconfig.get_config(config_file()).cache
    assert isinstance(configured, cache.Cache)
    assert configured == cache.Cache()


def test_an_invalid_abi_scope_is_refused_when_the_config_is_read(config_file):
    # Before a two-hour benchmark run, not at the first cache lookup inside it.
    from bench_runner import config as mconfig

    cfgpath = config_file(abi_scope="paranoid")
    with pytest.raises(ValueError, match="abi_scope"):
        mconfig.get_config(cfgpath)


def test_a_negative_max_age_is_refused(config_file):
    from bench_runner import config as mconfig

    cfgpath = config_file(max_age_days=-1)
    with pytest.raises(ValueError, match="max_age_days"):
        mconfig.get_config(cfgpath)


# ------------------------------------------------------- resolved settings


def test_get_settings_from_the_config(config_file, tmp_path):
    cfgpath = config_file(
        dir=str(tmp_path / "cache"),
        abi_scope="version",
        offline=True,
        share_pure_wheels=False,
        max_age_days=7,
    )

    settings = cache.get_settings(cfgpath)

    assert settings.enabled is True
    assert settings.root == tmp_path / "cache"
    assert settings.abi_scope == "version"
    assert settings.offline is True
    assert settings.share_pure_wheels is False
    assert settings.max_age_days == 7


def test_get_settings_without_a_config_file(tmp_path, monkeypatch):
    # The local-development paths this cache exists for are exactly the ones
    # run outside a results repository, so a missing config is not an error.
    from bench_runner import config as mconfig

    mconfig._load_config.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "bench_runner.toml").exists()
    try:
        settings = cache.get_settings()
        assert settings.enabled is True
        assert settings.abi_scope == "headers"
        assert settings.root == cache.get_cache_root()
    finally:
        mconfig._load_config.cache_clear()


def test_env_vars_override_the_config_section(config_file, tmp_path, monkeypatch):
    cfgpath = config_file(
        enabled=True,
        dir=str(tmp_path / "from-config"),
        abi_scope="build",
        offline=False,
    )
    monkeypatch.setenv(cache.CACHE_DIR_ENV_VAR, str(tmp_path / "from-env"))
    monkeypatch.setenv(cache.ABI_SCOPE_ENV_VAR, "version")
    monkeypatch.setenv(cache.OFFLINE_ENV_VAR, "1")
    monkeypatch.setenv(cache.NO_CACHE_ENV_VAR, "1")

    settings = cache.get_settings(cfgpath)

    assert settings.root == tmp_path / "from-env"
    assert settings.abi_scope == "version"
    assert settings.offline is True
    assert settings.enabled is False


def test_disabling_the_cache_in_the_config(config_file):
    settings = cache.get_settings(config_file(enabled=False))
    assert settings.enabled is False


def test_settings_pip_env(stub, monkeypatch, config_file, tmp_path):
    cfgpath = config_file(dir=str(tmp_path / "cache"), offline=True)
    stub.key(monkeypatch)

    env = cache.get_settings(cfgpath).pip_env(sys.executable, "bm")

    partition = Path(env["PIP_CACHE_DIR"])
    assert partition.parent == tmp_path / "cache" / "pip"
    assert partition.name.startswith("bm-")
    # Every setting applied, not just the one the caller was thinking about.
    assert env["PIP_NO_INDEX"] == "1"
    assert env["PIP_FIND_LINKS"] == str(tmp_path / "cache" / "wheelhouse" / "pure")


def test_settings_pip_env_honours_the_abi_scope(stub, monkeypatch, config_file):
    cfgpath = config_file(abi_scope="version")
    stub.key(monkeypatch)

    env = cache.get_settings(cfgpath).pip_env(sys.executable, "bm")

    assert Path(env["PIP_CACHE_DIR"]).name.endswith("-v")


def test_settings_pip_env_is_empty_when_disabled(stub, monkeypatch, config_file):
    cfgpath = config_file(enabled=False)
    stub.key(monkeypatch)

    # Nothing to merge into the environment, so pip behaves exactly as it did
    # before the cache existed.
    assert cache.get_settings(cfgpath).pip_env(sys.executable, "bm") == {}
