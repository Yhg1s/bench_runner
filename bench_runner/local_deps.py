"""
Pointing the benchmarking stack at local checkouts of its own dependencies.

Normally every repository in the chain -- pyperformance, pyston-benchmarks,
pyperf -- arrives over the network at a pinned ref, which means iterating on
one of them requires pushing a commit first. A local dep replaces that pin with
a directory on this machine: the checkout is left alone rather than being
hard-reset to the pin, and the venvs install from the path instead.

Configured in `bench_runner.toml`:

    [dev.local_deps]
    pyperformance     = { path = "~/python/pyperformance", editable = true }
    pyperf            = { path = "~/python/pyperf" }
    pyston-benchmarks = { path = "~/python/python-macrobenchmarks" }

the equivalent environment variable, for one-offs and for repositories with no
`bench_runner.toml` at all:

    BENCH_RUNNER_LOCAL_DEPS="pyperf=~/python/pyperf:editable,pyperformance=~/python/pyperformance"

or a `local_deps` table on one runner, which is what a machine used for
development wants when the rest of the repository's runners are real ones.

The three sources do not merge. The most specific one that says anything at all
provides the whole set, the same rule the other per-runner settings follow, so
that a runner can say `local_deps = {}` to turn off deps configured repository
wide. Precedence is runner table, then environment variable, then
`[dev.local_deps]`.

A key is a distribution name (`pyperf`) or one of the
`benchmark_definitions.BENCHMARK_REPOS` directory names (`pyperformance`,
`pyston-benchmarks`) -- `pyperformance` happens to be both, since it is cloned
*and* installed.

Note that a local dep changes the benchmark hash: results built from an
unpushed tree are not comparable to results built from a pushed one, and must
not silently reuse each other's runs.
"""

from __future__ import annotations


import dataclasses
import os
from pathlib import Path


from . import benchmark_definitions
from .util import PathLike


ENV_VAR = "BENCH_RUNNER_LOCAL_DEPS"

# The suffixes an entry may carry, in both `BENCH_RUNNER_LOCAL_DEPS` and
# pyperformance's `--local-dep NAME=PATH[:editable]`.
EDITABLE_FLAGS = {"editable": True, "noneditable": False}


@dataclasses.dataclass
class LocalDep:
    # The distribution name, or a BENCHMARK_REPOS directory name.
    name: str
    # The local checkout. As written until get_local_deps() has resolved it,
    # an absolute path afterwards.
    path: str
    # Whether to `pip install -e`, which makes later edits live with no
    # reinstall at all. Everything in this dependency chain is pure Python, so
    # this defaults on; turn it off for a dependency whose extension modules
    # would need rebuilding anyway.
    editable: bool = True

    @classmethod
    def from_spec(cls, spec: str) -> LocalDep:
        """
        Parse one `NAME=PATH[:editable]` entry.

        The suffix is only taken as a suffix when it is one of the flag words,
        so a Windows path keeps its drive letter.
        """
        name, separator, path = spec.partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError(
                f"Invalid local dependency {spec.strip()!r}. "
                "Expected NAME=PATH, optionally followed by "
                f"{' or '.join(':' + flag for flag in EDITABLE_FLAGS)}."
            )

        editable = True
        head, separator, flag = path.rpartition(":")
        if separator and flag.strip().lower() in EDITABLE_FLAGS:
            editable = EDITABLE_FLAGS[flag.strip().lower()]
            path = head.strip()
            if not path:
                raise ValueError(
                    f"Local dependency {name!r} has no path, only {flag.strip()!r}."
                )

        return cls(name=name, path=path, editable=editable)

    def to_spec(self) -> str:
        """
        The `NAME=PATH[:editable]` form, as pyperformance's `--local-dep`
        takes it.
        """
        return f"{self.name}={self.path}" + (":editable" if self.editable else "")

    def resolved_path(self, repo_root: PathLike | None = None) -> Path:
        """
        The checkout as an absolute path. `~` is expanded and a relative path
        is taken against the results repository, which is the working
        directory for everything bench_runner runs.
        """
        path = Path(self.path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if repo_root is None else Path(repo_root)) / path
        return Path(os.path.normpath(path))

    @property
    def is_benchmark_repo(self) -> bool:
        """
        Whether this names a repository checkout_benchmarks() would otherwise
        clone and hard-reset over.
        """
        return self.name in {
            repo.dirname for repo in benchmark_definitions.BENCHMARK_REPOS
        }

    @property
    def is_installable(self) -> bool:
        """
        Whether pip could install this directory. False for a checkout that is
        only ever read in place, such as pyston-benchmarks.
        """
        path = self.resolved_path()
        return any(
            (path / name).is_file()
            for name in ("pyproject.toml", "setup.py", "setup.cfg")
        )

    def pip_args(self) -> list[str]:
        """The arguments that install this dependency."""
        return ["-e", self.path] if self.editable else [self.path]


@dataclasses.dataclass
class Dev:
    """The `[dev]` section of `bench_runner.toml`."""

    local_deps: dict[str, LocalDep] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.local_deps = coerce(self.local_deps)


def coerce(local_deps: dict) -> dict[str, LocalDep]:
    """
    Turn the tables tomllib hands over into LocalDeps, filling in each one's
    name from the key it was written under.
    """
    return {
        name: dep if isinstance(dep, LocalDep) else LocalDep(name=name, **dep)
        for name, dep in local_deps.items()
    }


def parse_env(value: str | None = None) -> dict[str, LocalDep] | None:
    """
    Parse `BENCH_RUNNER_LOCAL_DEPS`, or None when it says nothing.
    """
    if value is None:
        value = os.environ.get(ENV_VAR)
    if value is None or not value.strip():
        return None

    deps = {}
    for entry in value.split(","):
        if not entry.strip():
            continue
        dep = LocalDep.from_spec(entry)
        deps[dep.name] = dep
    return deps


def _from_runner(cfgpath: PathLike | None) -> dict[str, LocalDep] | None:
    from . import runners as mrunners

    try:
        runner = mrunners.get_current_runner(cfgpath)
    except FileNotFoundError:
        # No bench_runner.toml. Local deps are most useful in exactly that
        # situation, so this is "not configured", not an error.
        return None
    return runner.local_deps


def _from_config(cfgpath: PathLike | None) -> dict[str, LocalDep] | None:
    from . import config as mconfig

    try:
        local_deps = mconfig.get_config(cfgpath).dev.local_deps
    except FileNotFoundError:
        return None
    # An absent or empty section is the same as saying nothing: this is the
    # least specific source, so there is nothing below it to turn off.
    return dict(local_deps) or None


def _validate(dep: LocalDep, source: str, repo_root: PathLike | None) -> LocalDep:
    path = dep.resolved_path(repo_root)
    if not path.exists():
        raise ValueError(
            f"Local dependency {dep.name!r} from {source} points at {path}, "
            "which does not exist."
        )
    if not path.is_dir():
        raise ValueError(
            f"Local dependency {dep.name!r} from {source} points at {path}, "
            "which is not a directory."
        )
    return dataclasses.replace(dep, path=str(path))


def get_local_deps(
    cfgpath: PathLike | None = None, repo_root: PathLike | None = None
) -> dict[str, LocalDep]:
    """
    The local dependencies configured for this run, keyed by name, with every
    path resolved to an absolute one that exists.

    Empty when nothing configures any, which is the normal case and in
    particular the case on a real runner.
    """
    sources = (
        ("the runner's local_deps table", lambda: _from_runner(cfgpath)),
        (f"${ENV_VAR}", parse_env),
        ("[dev.local_deps]", lambda: _from_config(cfgpath)),
    )
    for source, get in sources:
        deps = get()
        if deps is not None:
            return {
                name: _validate(dep, source, repo_root) for name, dep in deps.items()
            }
    return {}
