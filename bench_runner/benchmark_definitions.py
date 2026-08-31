from __future__ import annotations


import dataclasses
import functools
import hashlib
from pathlib import Path
import subprocess


from . import git


@dataclasses.dataclass
class BenchmarkRepo:
    hash: str
    url: str
    dirname: str


BENCHMARK_REPOS = [
    BenchmarkRepo(
        "8b622be7de03bbb53ba9f41ddba833ce92523009",
        "https://github.com/Yhg1s/pyperformance.git",
        "pyperformance",
    ),
    BenchmarkRepo(
        "fd42e372fee15bd93a1d7f1a20a37884c933709d",
        "https://github.com/Yhg1s/python-macrobenchmarks.git",
        "pyston-benchmarks",
    ),
]

# Prefixes the contribution a local checkout makes to the benchmark hash, so
# that whatever else it hashes to, it can never be the value a pushed commit
# would have produced.
LOCAL_MARKER = "local:"

UNKNOWN_COMMIT = "unknown"


@dataclasses.dataclass
class LocalRepoState:
    """
    What a local checkout looked like when a run used it.

    `diff_sha` covers `git status --porcelain` and `git diff HEAD` together, so
    two runs of the same tree agree and any tracked edit between them does not.
    It does not cover edits to the *contents* of an untracked file: status
    names such a file but not what is in it. Adding or removing one is caught;
    editing one already there is not.
    """

    path: str
    commit: str
    dirty: bool
    diff_sha: str

    @property
    def hash_contribution(self) -> str:
        return f"{LOCAL_MARKER}{self.commit}:{self.diff_sha}"

    def as_metadata(self) -> dict:
        return {
            "path": self.path,
            "commit": self.commit,
            "dirty": self.dirty,
            "diff_sha": self.diff_sha,
        }


@functools.cache
def get_local_repo_state(path: str) -> LocalRepoState:
    """
    Fingerprint a local checkout: its HEAD plus everything not committed.

    Cached, because a run asks for the benchmark hash several times and the
    tree does not change underneath it.
    """
    try:
        commit = git.get_log("%H", path)
        status = git.get_status(path)
        diff = git.get_diff(path)
    except (OSError, subprocess.CalledProcessError):
        # Not a git checkout, or one with no commits. We cannot say what is in
        # it, so we say so -- and treat it as dirty, because we certainly
        # cannot claim it is clean.
        return LocalRepoState(
            path=str(path),
            commit=UNKNOWN_COMMIT,
            dirty=True,
            diff_sha=hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12],
        )

    digest = hashlib.sha256()
    digest.update(status.encode("utf-8"))
    digest.update(b"\0")
    digest.update(diff.encode("utf-8"))

    return LocalRepoState(
        path=str(path),
        commit=commit,
        dirty=bool(status.strip()) or bool(diff.strip()),
        diff_sha=digest.hexdigest()[:12],
    )


def get_local_repo_states(
    local: dict | None = None,
) -> dict[str, LocalRepoState]:
    """
    The state of every configured local dependency, keyed by name.

    Empty on a normal run, which is every run that has not opted in.
    """
    if local is None:
        from . import local_deps as mlocal_deps

        local = mlocal_deps.get_local_deps()
    return {name: get_local_repo_state(dep.path) for name, dep in local.items()}


def get_benchmark_hash(local: dict | None = None) -> str:
    """
    Identify the benchmark definitions a run used.

    A result is only reused or compared against another with the same hash, so
    anything that changes what the benchmarks *are* has to be in here. That
    includes a local checkout standing in for a pinned repository: its work is
    unpushed, so nobody else can reproduce it, and it must never land on the
    same hash as a commit somebody could fetch. The LOCAL_MARKER guarantees
    that much on its own; HEAD and the uncommitted diff then separate one local
    state from the next, so editing the tree and running again does not reuse
    the earlier result either.
    """
    states = get_local_repo_states(local)

    hash = hashlib.sha256()
    for repo in BENCHMARK_REPOS:
        if (state := states.get(repo.dirname)) is not None:
            hash.update(state.hash_contribution.encode("utf-8"))
            continue
        if Path(repo.dirname).is_dir():
            current_hash = git.get_git_hash(Path(repo.dirname))
        else:
            current_hash = repo.hash
        hash.update(current_hash.encode("ascii")[:7])

    # Dependencies that are not benchmark repositories, pyperf above all. It is
    # the harness that does the timing, so an edit to it moves the numbers just
    # as surely as an edit to a benchmark, and a run using one must not be
    # mistaken for a run that did not.
    repo_names = {repo.dirname for repo in BENCHMARK_REPOS}
    for name, state in sorted(states.items()):
        if name in repo_names:
            continue
        hash.update(f"{name}={state.hash_contribution}".encode("utf-8"))

    return hash.hexdigest()[:6]
