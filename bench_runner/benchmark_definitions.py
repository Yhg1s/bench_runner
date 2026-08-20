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
        "4a7d8df2720414625b330a67c69f97f40057cefd",
        "https://github.com/Yhg1s/pyperformance.git",
        "pyperformance",
    ),
    BenchmarkRepo(
        "265655e7f03ace13ec1e00e1ba299179e69f8a00",
        "https://github.com/pyston/python-macrobenchmarks.git",
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
