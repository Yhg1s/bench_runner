import subprocess

from bench_runner import git


def _run(*args, cwd):
    subprocess.check_call(args, cwd=cwd)


def _make_origin(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=origin)
    _run("git", "config", "user.email", "t@example.com", cwd=origin)
    _run("git", "config", "user.name", "T", cwd=origin)
    (origin / "f.txt").write_text("first\n")
    _run("git", "add", "f.txt", cwd=origin)
    _run("git", "commit", "-q", "-m", "first", cwd=origin)
    return origin


def _head(repo):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, encoding="utf-8"
    ).strip()


def test_clone_creates_a_checkout(tmp_path):
    origin = _make_origin(tmp_path)
    dest = tmp_path / "dest"
    git.clone(dest, str(origin), branch="main")
    assert (dest / "f.txt").read_text() == "first\n"


def test_clone_refreshes_an_existing_checkout(tmp_path):
    """
    Regression: an existing checkout used to be left alone apart from `git
    checkout <branch>`, which moves to the stale local branch. Benchmarking
    then ran old code under the new commit's name.
    """
    origin = _make_origin(tmp_path)
    dest = tmp_path / "dest"
    git.clone(dest, str(origin), branch="main")
    assert (dest / "f.txt").read_text() == "first\n"

    (origin / "f.txt").write_text("second\n")
    _run("git", "commit", "-q", "-am", "second", cwd=origin)
    expected = _head(origin)

    git.clone(dest, str(origin), branch="main")

    assert (dest / "f.txt").read_text() == "second\n"
    assert _head(dest) == expected


def test_clone_can_select_a_specific_hash(tmp_path):
    origin = _make_origin(tmp_path)
    first = _head(origin)
    (origin / "f.txt").write_text("second\n")
    _run("git", "commit", "-q", "-am", "second", cwd=origin)

    dest = tmp_path / "dest"
    git.clone(dest, str(origin), branch="main")
    git.clone(dest, str(origin), branch=first)

    assert _head(dest) == first
    assert (dest / "f.txt").read_text() == "first\n"


def test_clone_fetches_a_pinned_hash_into_a_fresh_directory(tmp_path):
    """
    Benchmark repos are pinned to an exact commit. `git clone --branch` rejects
    a SHA, so a fresh checkout of a pin has to go through init + fetch.
    """
    origin = _make_origin(tmp_path)
    first = _head(origin)
    (origin / "f.txt").write_text("second\n")
    _run("git", "commit", "-q", "-am", "second", cwd=origin)

    dest = tmp_path / "fresh"
    git.clone(dest, str(origin), branch=first, depth=1)

    assert _head(dest) == first
    assert (dest / "f.txt").read_text() == "first\n"


def test_clone_without_a_branch_still_clones(tmp_path):
    origin = _make_origin(tmp_path)
    dest = tmp_path / "dest"
    git.clone(dest, str(origin))
    assert (dest / "f.txt").read_text() == "first\n"
