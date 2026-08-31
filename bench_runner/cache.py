"""
A cross-run package cache for the venvs that bench_runner and pyperformance
build.

This is pip's own cache, partitioned by an ABI key, with the download half
shared between all partitions:

    <root>/pip/_shared/http-v2/   downloaded artifacts, shared by everything
    <root>/pip/_shared/http/      the same, for pip < 23.3
    <root>/pip/host-<abi>/        PIP_CACHE_DIR for the outer venv
    <root>/pip/bm-<abi>/          PIP_CACHE_DIR for the benchmark venv
          http-v2 -> ../_shared/http-v2   (symlink)
          http    -> ../_shared/http      (symlink)
          wheels/                         (real directory, per-ABI)
    <root>/wheelhouse/pure/       *-none-any.whl, shared, passed as find-links

pip joins its cache directory with the literal names "http-v2" and "wheels", so
the symlinks are all it takes to share downloads while keeping built wheels
apart. That is not public pip API: if pip renames those directories the only
consequence is a cold download, never a wrong artifact.

Two layers keep the cache correct across Python versions. pip itself filters
candidate wheels by compatibility tag, which covers interpreter, abi and
platform differences. The ABI key covers what that cannot see: two builds of
the *same* X.Y with the same tag but different headers, which is a real hazard
on CPython main -- pip will happily serve a wheel built against 3.14.2 to
3.14.7.

Nothing here knows about `bench_runner.toml`; every knob is a keyword argument
so that the `[cache]` config section can be layered on top later. The
environment variables documented below always win over what a caller passes in.
"""

from __future__ import annotations


import dataclasses
import functools
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Iterable, Literal, Sequence


from .util import PathLike, smart_rmtree


# The scope of an ABI key: how much of the target interpreter's identity is
# mixed into the cache partition name.
#
#   headers  Every *.h under the interpreter's include directory, plus
#            pyconfig.h. A C extension's whole compile-time contract with
#            CPython lives in those files, so two builds that touch no header
#            can safely share built wheels. PGO/LTO/BOLT builds share; a
#            build with --enable-experimental-jit does not, because that
#            changes pyconfig.h.
#   build    The interpreter's full build identity, the same one pyperformance
#            uses to name a venv. Never shares across builds.
#   version  Nothing beyond X.Y, SOABI and platform. Fast, and only safe if
#            you know every dependency is pure Python.
AbiScope = Literal["headers", "build", "version"]
ABI_SCOPES: tuple[AbiScope, ...] = ("headers", "build", "version")
DEFAULT_ABI_SCOPE: AbiScope = "headers"

CACHE_DIR_ENV_VAR = "BENCH_RUNNER_CACHE_DIR"
ABI_SCOPE_ENV_VAR = "BENCH_RUNNER_CACHE_ABI_SCOPE"
OFFLINE_ENV_VAR = "BENCH_RUNNER_OFFLINE"
NO_CACHE_ENV_VAR = "BENCH_RUNNER_NO_CACHE"

# The two venvs a run installs into. They get separate partitions because they
# are usually different interpreters entirely.
PipKind = Literal["host", "bm"]
PIP_KINDS: tuple[PipKind, ...] = ("host", "bm")

# The names pip gives its download cache, newest first.
HTTP_CACHE_DIRS = ("http-v2", "http")

# How much of each sha256 ends up in a directory name.
DIGEST_LENGTH = 12

# How long an unused cache entry survives `bench_runner cache --prune`.
DEFAULT_MAX_AGE_DAYS = 30.0

# The environment variables pip_env() may produce. run_benchmarks.py has to
# allowlist these for --inherit-environ, because pyperformance strips
# everything else before calling pip.
PIP_ENV_VARS = (
    "PIP_CACHE_DIR",
    "PIP_FIND_LINKS",
    "PIP_NO_INDEX",
    "PIP_DISABLE_PIP_VERSION_CHECK",
)


class CacheError(RuntimeError):
    pass


@dataclasses.dataclass
class Cache:
    """
    The `[cache]` section of `bench_runner.toml`, exactly as written.

    These are the configured values, not the effective ones: the environment
    variables above override several of them. Call get_settings() to get what
    a run should actually do.
    """

    # Whether to cache at all. False restores the old behaviour of a cold
    # download and rebuild on every run.
    enabled: bool = True
    # Where the cache lives. Empty means the platform cache directory.
    dir: str = ""
    # How much of the interpreter's identity partitions the built-wheel cache.
    # See ABI_SCOPES above; "build" is the escape hatch if a C extension ever
    # misbehaves in a way the headers did not predict.
    abi_scope: str = DEFAULT_ABI_SCOPE
    # Whether to forbid pip from reaching the network at all.
    offline: bool = False
    # Whether to collect interpreter-independent wheels into one find-links
    # directory shared by every ABI.
    share_pure_wheels: bool = True
    # How long an unused entry survives a prune.
    max_age_days: float = DEFAULT_MAX_AGE_DAYS

    def __post_init__(self):
        # Fail when the config is read rather than when the cache is first
        # used, so a typo surfaces before a two-hour benchmark run.
        if self.abi_scope not in ABI_SCOPES:
            raise ValueError(
                f"Invalid `cache.abi_scope` {self.abi_scope!r} in "
                f"`bench_runner.toml`. Must be one of {', '.join(ABI_SCOPES)}."
            )
        if self.max_age_days < 0:
            raise ValueError(
                f"`cache.max_age_days` must not be negative, got {self.max_age_days}."
            )


def _is_windows() -> bool:
    # Not util.get_simple_platform(), which is cached and raises on platforms
    # we do not otherwise support. Getting a cache directory should never be
    # the thing that fails.
    return sys.platform.startswith("win")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_enabled(enabled: bool | None = None) -> bool:
    """
    Whether the cache should be used at all. `enabled` is the configured
    value, if any; BENCH_RUNNER_NO_CACHE=1 overrides it.
    """
    if _env_flag(NO_CACHE_ENV_VAR):
        return False
    return True if enabled is None else enabled


def is_offline(offline: bool | None = None) -> bool:
    """
    Whether pip should run with PIP_NO_INDEX. `offline` is the configured
    value, if any; BENCH_RUNNER_OFFLINE=1 overrides it.
    """
    if _env_flag(OFFLINE_ENV_VAR):
        return True
    return False if offline is None else offline


def get_cache_root(config_dir: PathLike | None = None) -> Path:
    """
    Where the cache lives. In order of precedence:

        $BENCH_RUNNER_CACHE_DIR
        `config_dir` (the `[cache].dir` setting)
        %LOCALAPPDATA%\\bench_runner on Windows, $XDG_CACHE_HOME/bench_runner
        ~/.cache/bench_runner

    The directory is not created here; pip_env() creates what it needs.
    """
    from_env = os.environ.get(CACHE_DIR_ENV_VAR, "").strip()
    if from_env:
        return Path(from_env).expanduser().absolute()

    if config_dir is not None and str(config_dir).strip():
        return Path(config_dir).expanduser().absolute()

    if _is_windows():
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data).expanduser().absolute() / "bench_runner"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg_cache_home:
            return Path(xdg_cache_home).expanduser().absolute() / "bench_runner"

    return Path.home().absolute() / ".cache" / "bench_runner"


def pip_root(root: PathLike) -> Path:
    return Path(root) / "pip"


def shared_dir(root: PathLike) -> Path:
    """The download cache every partition symlinks into."""
    return pip_root(root) / "_shared"


def partition_dir(root: PathLike, kind: PipKind, abi: str) -> Path:
    """One partition, used directly as a PIP_CACHE_DIR."""
    return pip_root(root) / f"{kind}-{abi}"


def pure_wheelhouse_dir(root: PathLike) -> Path:
    """The find-links directory of interpreter-independent wheels."""
    return Path(root) / "wheelhouse" / "pure"


def benchmark_venvs_dir(root: PathLike) -> Path:
    """
    Where pyperformance keeps the per-interpreter benchmark venvs.

    Outside the results repository on purpose. pyperformance's own default
    puts them at ./venv/<runid>, nested inside the outer venv, where the
    bootstrap's rebuild of that venv takes them with it -- and a sibling
    directory in the repository would instead risk being committed, since a
    results repository's .gitignore knows about `venv/` and nothing else.

    Sharing one directory between results repositories on a machine is safe:
    pyperformance names each venv after the interpreter's build identity, so
    two repositories benchmarking the same build want the same venv.
    """
    return Path(root) / "venvs"


# Run in the *target* interpreter, which may be a CPython that was built five
# minutes ago, so it can only use the standard library. Emits one JSON object
# on stdout. Keep it cheap: hashing every header is a few milliseconds, but it
# happens once per run per interpreter.
_PROBE = r"""
import hashlib
import json
import os
import sys
import sysconfig

scope = sys.argv[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fd:
        for chunk in iter(lambda: fd.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out = {
    "impl": sys.implementation.name,
    "version": "%d.%d" % sys.version_info[:2],
    "soabi": sysconfig.get_config_var("SOABI") or "",
    "platform": sysconfig.get_platform(),
    "headers": "",
    "header_count": 0,
    "build": "",
}

if scope == "headers":
    entries = []
    include = sysconfig.get_paths().get("include")
    if include and os.path.isdir(include):
        for dirpath, dirnames, filenames in os.walk(include):
            for name in filenames:
                if not name.endswith(".h"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, include).replace(os.sep, "/")
                try:
                    entries.append((rel, sha256_file(full)))
                except OSError:
                    pass
    config_h = sysconfig.get_config_h_filename()
    if config_h and os.path.isfile(config_h):
        try:
            entries.append(("<config_h>", sha256_file(config_h)))
        except OSError:
            pass
    h = hashlib.sha256()
    for rel, digest in sorted(entries):
        h.update(("%s\0%s\0" % (rel, digest)).encode("utf-8"))
    out["headers"] = h.hexdigest()
    out["header_count"] = len(entries)

if scope in ("headers", "build"):
    # The same identity pyperformance's _python.get_id() uses, minus
    # sys.executable so that moving an install does not repartition it.
    import importlib.util

    h = hashlib.sha256()
    for value in (
        sys.version,
        sys.implementation.name,
        ".".join(str(v) for v in sys.implementation.version),
        str(sys.api_version),
        importlib.util.MAGIC_NUMBER.hex(),
    ):
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    out["build"] = h.hexdigest()

sys.stdout.write(json.dumps(out))
"""


@functools.cache
def _probe(python_exe: str, scope: str) -> dict:
    """
    Ask an interpreter to describe itself. Cached, because a single run asks
    for the same interpreter's ABI key several times.
    """
    try:
        output = subprocess.check_output(
            [python_exe, "-c", _PROBE, scope],
            encoding="utf-8",
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise CacheError(
            f"Could not determine the ABI of {python_exe}:\n{exc.stderr}"
        ) from exc
    except OSError as exc:
        raise CacheError(f"Could not run {python_exe}: {exc}") from exc

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise CacheError(
            f"Could not parse the ABI probe output from {python_exe}: {output!r}"
        ) from exc


def resolve_abi_scope(scope: str | None = None) -> AbiScope:
    """
    `scope` is the configured value, if any; BENCH_RUNNER_CACHE_ABI_SCOPE
    overrides it.
    """
    from_env = os.environ.get(ABI_SCOPE_ENV_VAR, "").strip()
    resolved = from_env or scope or DEFAULT_ABI_SCOPE
    if resolved not in ABI_SCOPES:
        raise ValueError(
            f"Invalid cache ABI scope {resolved!r}. "
            f"Must be one of {', '.join(ABI_SCOPES)}."
        )
    return resolved  # pyright: ignore[reportReturnType]


def _sanitize(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+]+", "-", part).strip("-") or "unknown"


def abi_key(python_exe: PathLike, scope: str | None = None) -> str:
    """
    A directory-name-safe key naming everything about `python_exe` that a
    built wheel could depend on:

        <impl><X.Y>-<soabi>-<platform>-<digest>

    The digest is prefixed with the scope that produced it -- `h` for headers,
    `b` for build, `v` for version -- so partitions made under different
    scopes can never collide.
    """
    resolved = resolve_abi_scope(scope)
    info = _probe(str(python_exe), resolved)

    if resolved == "version":
        digest = "v"
    elif resolved == "build" or not info["header_count"]:
        # No headers to fingerprint means we cannot say anything about this
        # interpreter's ABI, so fall back to its build identity rather than
        # sharing a partition with every other headerless build.
        digest = "b" + info["build"][:DIGEST_LENGTH]
    else:
        digest = "h" + info["headers"][:DIGEST_LENGTH]

    parts = (
        f"{info['impl']}{info['version']}",
        info["soabi"] or "nosoabi",
        info["platform"],
        digest,
    )
    return "-".join(_sanitize(part) for part in parts)


def _link_shared(partition: Path, shared: Path, name: str) -> bool:
    """
    Point <partition>/<name> at <shared>/<name>. Returns False if the
    partition ended up with a real directory instead -- on Windows, symlinks
    need developer mode, and a per-ABI download cache is merely slower, not
    wrong.
    """
    target = shared / name
    target.mkdir(parents=True, exist_ok=True)

    link = partition / name
    if link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return True
        except OSError:
            pass
        link.unlink()
    elif link.exists():
        # A real directory, from an earlier fallback. Leave it alone: it is
        # still a valid download cache, just not a shared one.
        return False

    try:
        link.symlink_to(os.path.relpath(target, partition), target_is_directory=True)
    except OSError:
        link.mkdir(parents=True, exist_ok=True)
        return False
    return True


def pip_env(
    python_exe: PathLike,
    kind: PipKind = "host",
    *,
    scope: str | None = None,
    root: PathLike | None = None,
    config_dir: PathLike | None = None,
    offline: bool | None = None,
    share_pure_wheels: bool = True,
    extra_find_links: Sequence[PathLike] = (),
) -> dict[str, str]:
    """
    Create the cache partition for `python_exe` and return the environment
    variables that point pip at it. The result is meant to be merged into
    `os.environ` (or a subprocess's env) before installing anything.
    """
    if kind not in PIP_KINDS:
        raise ValueError(
            f"Invalid pip cache kind {kind!r}. Must be one of {', '.join(PIP_KINDS)}."
        )

    cache_root = Path(root) if root is not None else get_cache_root(config_dir)
    partition = partition_dir(cache_root, kind, abi_key(python_exe, scope))
    (partition / "wheels").mkdir(parents=True, exist_ok=True)

    shared = shared_dir(cache_root)
    for name in HTTP_CACHE_DIRS:
        _link_shared(partition, shared, name)

    env = {
        "PIP_CACHE_DIR": str(partition),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }

    find_links = []
    if share_pure_wheels:
        pure = pure_wheelhouse_dir(cache_root)
        pure.mkdir(parents=True, exist_ok=True)
        find_links.append(str(pure))
    find_links.extend(str(path) for path in extra_find_links)
    if find_links:
        # pip splits list-valued environment variables on whitespace.
        env["PIP_FIND_LINKS"] = " ".join(find_links)

    if is_offline(offline):
        env["PIP_NO_INDEX"] = "1"

    return env


def _partitions(root: PathLike) -> Iterable[Path]:
    pip_dir = pip_root(root)
    if not pip_dir.is_dir():
        return
    for path in sorted(pip_dir.iterdir()):
        if path.is_dir() and not path.is_symlink() and path.name != "_shared":
            yield path


def _is_pure_wheel(path: Path) -> bool:
    return path.name.lower().endswith("-none-any.whl")


def sweep_pure_wheels(
    root: PathLike | None = None,
    *,
    config_dir: PathLike | None = None,
    partitions: Iterable[str] | None = None,
) -> list[Path]:
    """
    Hardlink every `*-none-any.whl` out of the partitions' `wheels/`
    directories into the shared wheelhouse, and return the ones that were
    newly added.

    Only wheels pip had to *build* pass through `wheels/` -- a pre-built wheel
    off an index is installed straight out of the download cache -- so this
    collects exactly the pure-Python sdists that every ABI would otherwise
    rebuild for itself. That is what recovers cross-version sharing even under
    `abi_scope = "build"`, which by design shares nothing else.

    Safe because pip tag-filters find-links candidates exactly as it does its
    own cache: measured against pip 25.3, a py3-none-any wheel installs on both
    3.13 and 3.14, while a cp314-none-any one is offered to 3.14 and refused to
    3.13. So the wheelhouse cannot serve an incompatible wheel, and there is no
    need to restrict it to universal tags -- an interpreter-tagged pure wheel is
    still shared across every *build* of that interpreter, which is the case
    this exists for.

    Sweeping only from `wheels/` is also what keeps local dependencies out:
    pip refuses to persist a wheel built from an editable install, a plain
    local directory, or a mutable VCS ref, so a local dep's wheel never lands
    there and can never mask a later edit.
    """
    cache_root = Path(root) if root is not None else get_cache_root(config_dir)
    wanted = None if partitions is None else set(partitions)

    pure = pure_wheelhouse_dir(cache_root)
    added = []
    for partition in _partitions(cache_root):
        if wanted is not None and partition.name not in wanted:
            continue
        for wheel in sorted((partition / "wheels").glob("**/*.whl")):
            if not _is_pure_wheel(wheel) or not wheel.is_file():
                continue
            dest = pure / wheel.name
            if dest.exists():
                continue
            pure.mkdir(parents=True, exist_ok=True)
            try:
                os.link(wheel, dest)
            except OSError:
                # A different filesystem, or a filesystem without hardlinks.
                temp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
                shutil.copyfile(wheel, temp)
                temp.replace(dest)
            added.append(dest)
    return added


@dataclasses.dataclass
class PruneResult:
    files_removed: int = 0
    bytes_removed: int = 0
    partitions_removed: list[str] = dataclasses.field(default_factory=list)


def prune(
    max_age_days: float,
    root: PathLike | None = None,
    *,
    config_dir: PathLike | None = None,
    now: float | None = None,
) -> PruneResult:
    """
    Remove cached files that have not been used in `max_age_days`, then any
    partition that is left with no wheels at all.

    A file's age is the more recent of its mtime and its atime, so an entry
    that is being read every run survives even though nothing rewrites it.
    """
    if max_age_days < 0:
        raise ValueError(f"max_age_days must not be negative, got {max_age_days}")

    cache_root = Path(root) if root is not None else get_cache_root(config_dir)
    result = PruneResult()
    if not cache_root.is_dir():
        return result

    cutoff = (time.time() if now is None else now) - max_age_days * 86400

    # Record this before deleting anything: emptying a partition's wheels/
    # directory bumps the partition's own mtime, which would otherwise make
    # every partition look freshly created below.
    ages = {}
    for partition in _partitions(cache_root):
        try:
            ages[partition] = partition.stat().st_mtime
        except OSError:
            pass

    # Never remove the shared download roots themselves, even when they empty
    # out: the partitions symlink to them, and a dangling symlink is a pip
    # cache directory pip cannot create.
    protected = {shared_dir(cache_root) / name for name in HTTP_CACHE_DIRS}

    # topdown=False so directories are seen after their contents, and
    # followlinks defaults to False so the http-v2 symlinks do not lead us
    # into _shared twice.
    for dirpath, dirnames, filenames in os.walk(cache_root, topdown=False):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            if max(st.st_mtime, st.st_atime) >= cutoff:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            result.files_removed += 1
            result.bytes_removed += st.st_size

        for name in dirnames:
            path = Path(dirpath) / name
            if path.is_symlink() or path in protected:
                continue
            try:
                path.rmdir()
            except OSError:
                # Not empty, which is the common case.
                pass

    for partition in _partitions(cache_root):
        if ages.get(partition, cutoff) >= cutoff:
            # Freshly created, possibly by this very run.
            continue
        if any((partition / "wheels").glob("**/*.whl")):
            continue
        smart_rmtree(partition)
        result.partitions_removed.append(partition.name)

    return result


def clear(root: PathLike | None = None, *, config_dir: PathLike | None = None) -> bool:
    """
    Delete the whole cache. Returns whether there was anything to delete.
    """
    cache_root = Path(root) if root is not None else get_cache_root(config_dir)
    if not cache_root.exists():
        return False
    smart_rmtree(cache_root)
    return True


def _tree_size(path: Path) -> tuple[int, int]:
    """(files, bytes) of the real files under `path`, never following links."""
    files = 0
    total = 0
    if not path.is_dir() or path.is_symlink():
        return files, total
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                st = (Path(dirpath) / name).lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            files += 1
            total += st.st_size
    return files, total


@dataclasses.dataclass
class PartitionInfo:
    name: str
    kind: str
    abi: str
    path: Path
    size: int
    wheel_count: int
    last_used: float


@dataclasses.dataclass
class CacheInfo:
    root: Path
    exists: bool
    total_size: int = 0
    shared_files: int = 0
    shared_size: int = 0
    pure_wheel_count: int = 0
    pure_wheel_size: int = 0
    venv_count: int = 0
    venv_size: int = 0
    partitions: list[PartitionInfo] = dataclasses.field(default_factory=list)


def info(
    root: PathLike | None = None, *, config_dir: PathLike | None = None
) -> CacheInfo:
    """
    Describe what is in the cache, for `bench_runner cache --info`.

    Sizes count each real file once: the shared download cache is reported
    separately rather than once per partition that links to it.
    """
    cache_root = Path(root) if root is not None else get_cache_root(config_dir)
    result = CacheInfo(root=cache_root, exists=cache_root.is_dir())
    if not result.exists:
        return result

    _, result.total_size = _tree_size(cache_root)
    result.shared_files, result.shared_size = _tree_size(shared_dir(cache_root))
    result.pure_wheel_count, result.pure_wheel_size = _tree_size(
        pure_wheelhouse_dir(cache_root)
    )

    # Reported on their own line rather than folded into the partitions: they
    # are usually the largest thing under the root by far, and a total that
    # dwarfs the partitions with no explanation reads like a bug.
    venvs = benchmark_venvs_dir(cache_root)
    if venvs.is_dir():
        result.venv_count = sum(1 for path in venvs.iterdir() if path.is_dir())
        _, result.venv_size = _tree_size(venvs)

    for partition in _partitions(cache_root):
        kind, _, abi = partition.name.partition("-")
        wheels = sorted((partition / "wheels").glob("**/*.whl"))
        _, size = _tree_size(partition)
        try:
            last_used = max(
                (wheel.stat().st_mtime for wheel in wheels),
                default=partition.stat().st_mtime,
            )
        except OSError:
            last_used = 0.0
        result.partitions.append(
            PartitionInfo(
                name=partition.name,
                kind=kind,
                abi=abi,
                path=partition,
                size=size,
                wheel_count=len(wheels),
                last_used=last_used,
            )
        )

    return result


def format_size(nbytes: int) -> str:
    value = float(nbytes)
    unit = "B"
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1024.0:
            break
        value /= 1024.0
    return f"{value:.1f} {unit}"


def format_info(cache_info: CacheInfo) -> str:
    """A plain-text rendering of info(), for the command line."""
    lines = [f"Cache root: {cache_info.root}"]
    if not cache_info.exists:
        lines.append("  (does not exist yet)")
        return "\n".join(lines)

    lines.append(f"Total size: {format_size(cache_info.total_size)}")
    lines.append(
        f"Shared downloads: {cache_info.shared_files} files, "
        f"{format_size(cache_info.shared_size)}"
    )
    lines.append(
        f"Pure wheelhouse: {cache_info.pure_wheel_count} wheels, "
        f"{format_size(cache_info.pure_wheel_size)}"
    )
    lines.append(
        f"Benchmark venvs: {cache_info.venv_count}, "
        f"{format_size(cache_info.venv_size)}"
    )
    if not cache_info.partitions:
        lines.append("Partitions: none")
        return "\n".join(lines)

    lines.append(f"Partitions ({len(cache_info.partitions)}):")
    for partition in cache_info.partitions:
        age = (time.time() - partition.last_used) / 86400.0
        lines.append(
            f"  {partition.name}: {partition.wheel_count} wheels, "
            f"{format_size(partition.size)}, last used {age:.1f} days ago"
        )
    return "\n".join(lines)


@dataclasses.dataclass
class CacheSettings:
    """
    What this run should actually do: the `[cache]` section with the
    environment applied and the cache root resolved to a real path.
    """

    enabled: bool
    root: Path
    abi_scope: AbiScope
    offline: bool
    share_pure_wheels: bool
    max_age_days: float

    def pip_env(
        self,
        python_exe: PathLike,
        kind: PipKind = "host",
        *,
        extra_find_links: Sequence[PathLike] = (),
    ) -> dict[str, str]:
        """
        The environment that points one venv's pip at this cache, or nothing
        at all when the cache is turned off.

        This is the call B-2 and B-3 want: it applies every setting, so a
        caller cannot honour `abi_scope` while forgetting `offline`.
        """
        if not self.enabled:
            return {}
        return pip_env(
            python_exe,
            kind,
            scope=self.abi_scope,
            root=self.root,
            offline=self.offline,
            share_pure_wheels=self.share_pure_wheels,
            extra_find_links=extra_find_links,
        )


def get_settings(cfgpath: PathLike | None = None) -> CacheSettings:
    """
    The effective cache settings: `[cache]` overridden by the environment.

    A repository with no `bench_runner.toml` gets the defaults rather than an
    error, because the local-development paths this cache exists for are
    exactly the ones run outside a results repository.
    """
    from . import config as mconfig

    try:
        configured = mconfig.get_config(cfgpath).cache
    except FileNotFoundError:
        configured = Cache()

    return CacheSettings(
        enabled=is_enabled(configured.enabled),
        root=get_cache_root(configured.dir),
        abi_scope=resolve_abi_scope(configured.abi_scope),
        offline=is_offline(configured.offline),
        share_pure_wheels=configured.share_pure_wheels,
        max_age_days=configured.max_age_days,
    )
