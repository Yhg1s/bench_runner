import json
from pathlib import Path
import subprocess


import pytest


from bench_runner import benchmark_definitions
from bench_runner import local_deps


def git_repo(path: Path, content: str = "original") -> Path:
    """A real git checkout with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "-q", "."], cwd=path)
    (path / "bench.py").write_text(content)
    subprocess.check_call(["git", "add", "-A"], cwd=path)
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=path,
    )
    return path


def dep(path: Path, name: str) -> dict[str, local_deps.LocalDep]:
    return {name: local_deps.LocalDep(name=name, path=str(path))}


@pytest.fixture(autouse=True)
def clear_state_cache():
    benchmark_definitions.get_local_repo_state.cache_clear()
    yield
    benchmark_definitions.get_local_repo_state.cache_clear()


@pytest.fixture(autouse=True)
def no_local_deps_in_the_environment(monkeypatch, tmp_path):
    # get_benchmark_hash() consults the configuration when it is not told
    # otherwise, and must not pick up the developer's own.
    monkeypatch.delenv(local_deps.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------- local repo state


def test_local_repo_state_of_a_clean_checkout(tmp_path):
    repo = git_repo(tmp_path / "pyperf")

    state = benchmark_definitions.get_local_repo_state(str(repo))

    assert state.path == str(repo)
    assert len(state.commit) == 40
    assert state.dirty is False


def test_local_repo_state_notices_an_uncommitted_edit(tmp_path):
    repo = git_repo(tmp_path / "pyperf")
    clean = benchmark_definitions.get_local_repo_state(str(repo))

    (repo / "bench.py").write_text("edited")
    benchmark_definitions.get_local_repo_state.cache_clear()
    dirty = benchmark_definitions.get_local_repo_state(str(repo))

    assert dirty.dirty is True
    assert dirty.commit == clean.commit  # Nothing was committed...
    assert dirty.diff_sha != clean.diff_sha  # ...but the tree is not the same.


def test_local_repo_state_notices_a_new_commit(tmp_path):
    repo = git_repo(tmp_path / "pyperf")
    before = benchmark_definitions.get_local_repo_state(str(repo))

    (repo / "bench.py").write_text("edited")
    subprocess.check_call(["git", "add", "-A"], cwd=repo)
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "second",
        ],
        cwd=repo,
    )
    benchmark_definitions.get_local_repo_state.cache_clear()
    after = benchmark_definitions.get_local_repo_state(str(repo))

    assert after.commit != before.commit
    assert after.dirty is False


def test_local_repo_state_notices_an_untracked_file(tmp_path):
    repo = git_repo(tmp_path / "pyperf")
    before = benchmark_definitions.get_local_repo_state(str(repo))

    (repo / "extra.py").write_text("new")
    benchmark_definitions.get_local_repo_state.cache_clear()
    after = benchmark_definitions.get_local_repo_state(str(repo))

    assert after.dirty is True
    assert after.diff_sha != before.diff_sha


def test_local_repo_state_of_something_that_is_not_a_checkout(tmp_path):
    # An unpacked tarball, say. We cannot describe it, so we say so rather
    # than claiming it is clean.
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    state = benchmark_definitions.get_local_repo_state(str(plain))

    assert state.commit == benchmark_definitions.UNKNOWN_COMMIT
    assert state.dirty is True
    assert state.diff_sha


def test_local_repo_state_is_cached(tmp_path):
    repo = git_repo(tmp_path / "pyperf")
    first = benchmark_definitions.get_local_repo_state(str(repo))
    # A run asks for the benchmark hash several times; the tree does not change
    # underneath it.
    assert benchmark_definitions.get_local_repo_state(str(repo)) is first


# ------------------------------------------------------------ benchmark hash


def test_benchmark_hash_without_local_deps_is_unchanged(tmp_path):
    # A normal run must produce exactly the hash it always did, or every
    # existing result in every results repository is orphaned.
    assert benchmark_definitions.get_benchmark_hash({}) == (
        benchmark_definitions.get_benchmark_hash()
    )
    assert len(benchmark_definitions.get_benchmark_hash({})) == 6


def test_a_local_benchmark_repo_changes_the_hash(tmp_path):
    repo = git_repo(tmp_path / "pyperformance")
    pinned = benchmark_definitions.get_benchmark_hash({})

    assert (
        benchmark_definitions.get_benchmark_hash(dep(repo, "pyperformance")) != pinned
    )


def test_a_local_pyperf_changes_the_hash(tmp_path):
    # pyperf is not a benchmark repository, but it is the harness that does the
    # timing: an edit to it moves the numbers just as surely as an edit to a
    # benchmark, so a run using one must not be mistaken for one that did not.
    repo = git_repo(tmp_path / "pyperf")
    pinned = benchmark_definitions.get_benchmark_hash({})

    assert benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf")) != pinned


def test_a_dirty_tree_gets_its_own_hash(tmp_path):
    repo = git_repo(tmp_path / "pyperf")
    clean = benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf"))

    (repo / "bench.py").write_text("edited")
    benchmark_definitions.get_local_repo_state.cache_clear()
    dirty = benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf"))

    assert dirty != clean


def test_the_same_tree_hashes_the_same_twice(tmp_path):
    # Otherwise a second run would never reuse the first one's result.
    repo = git_repo(tmp_path / "pyperf")
    first = benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf"))
    benchmark_definitions.get_local_repo_state.cache_clear()

    assert benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf")) == first


def test_a_local_checkout_never_collides_with_the_pushed_one(tmp_path, monkeypatch):
    """
    The point of the whole exercise. A local pyperformance checked out at
    exactly the pinned commit still must not produce the pinned hash, because
    what is in the working tree is not what the pin names.
    """
    repo = git_repo(tmp_path / "pyperformance")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, encoding="utf-8"
    ).strip()

    # Pretend the pin is exactly this commit, and that it is checked out where
    # checkout_benchmarks() would have put it.
    monkeypatch.setattr(
        benchmark_definitions,
        "BENCHMARK_REPOS",
        [benchmark_definitions.BenchmarkRepo(head, "https://example.com/x", "pyperf")],
    )

    as_pinned = benchmark_definitions.get_benchmark_hash({})
    as_local = benchmark_definitions.get_benchmark_hash(dep(repo, "pyperf"))

    assert as_local != as_pinned


def test_different_local_deps_give_different_hashes(tmp_path):
    one = git_repo(tmp_path / "one", "one")
    two = git_repo(tmp_path / "two", "two")

    assert benchmark_definitions.get_benchmark_hash(
        dep(one, "pyperf")
    ) != benchmark_definitions.get_benchmark_hash(dep(two, "pyperf"))


def test_the_hash_reads_the_configuration_when_not_told(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "pyperf")
    pinned = benchmark_definitions.get_benchmark_hash({})

    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={repo}")

    assert benchmark_definitions.get_benchmark_hash() != pinned


# ---------------------------------------------------------------- metadata


def test_local_repo_states_from_the_configuration(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "pyperf")
    monkeypatch.setenv(local_deps.ENV_VAR, f"pyperf={repo}")

    states = benchmark_definitions.get_local_repo_states()

    assert sorted(states) == ["pyperf"]
    assert states["pyperf"].path == str(repo)


def test_as_metadata_shape(tmp_path):
    repo = git_repo(tmp_path / "pyperf")

    recorded = benchmark_definitions.get_local_repo_state(str(repo)).as_metadata()

    assert sorted(recorded) == ["commit", "diff_sha", "dirty", "path"]
    # Has to survive a round trip into the results file.
    assert json.loads(json.dumps(recorded)) == recorded
