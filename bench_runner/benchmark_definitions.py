from __future__ import annotations


import dataclasses
import hashlib
from pathlib import Path


from . import git


@dataclasses.dataclass
class BenchmarkRepo:
    hash: str
    url: str
    dirname: str


BENCHMARK_REPOS = [
    BenchmarkRepo(
        "29575e852273c926a618e02be10423b06ff1bb34",
        "https://github.com/Yhg1s/pyperformance.git",
        "pyperformance",
    ),
    BenchmarkRepo(
        "fd42e372fee15bd93a1d7f1a20a37884c933709d",
        "https://github.com/Yhg1s/python-macrobenchmarks.git",
        "pyston-benchmarks",
    ),
]


def get_benchmark_hash() -> str:
    hash = hashlib.sha256()
    for repo in BENCHMARK_REPOS:
        if Path(repo.dirname).is_dir():
            current_hash = git.get_git_hash(Path(repo.dirname))
        else:
            current_hash = repo.hash
        hash.update(current_hash.encode("ascii")[:7])
    return hash.hexdigest()[:6]
