import sys


import pytest


from bench_runner import cache
from bench_runner.scripts import cache as cache_command


@pytest.fixture
def cache_config(tmp_path, monkeypatch):
    """A results repository whose [cache] points somewhere disposable."""
    from bench_runner import config as mconfig

    def configure(**settings):
        settings.setdefault("dir", str(tmp_path / "cache"))
        lines = [
            "[bases]",
            'versions = ["3.12.0"]',
            "",
            "[runners.testrunner]",
            'os = "linux"',
            'arch = "x86_64"',
            'hostname = "testhost"',
            "",
            "[cache]",
        ]
        for key, value in settings.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        (tmp_path / "bench_runner.toml").write_text("\n".join(lines) + "\n")
        monkeypatch.chdir(tmp_path)
        mconfig._load_config.cache_clear()
        return tmp_path / "cache"

    for name in (
        cache.CACHE_DIR_ENV_VAR,
        cache.ABI_SCOPE_ENV_VAR,
        cache.OFFLINE_ENV_VAR,
        cache.NO_CACHE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    mconfig._load_config.cache_clear()
    yield configure
    mconfig._load_config.cache_clear()


def populate(root, *, wheels=("chameleon-4.0-py3-none-any.whl",)):
    """A cache with one partition, one download and some built wheels."""
    partition = cache.partition_dir(root, "bm", "abi-one")
    wheel_dir = partition / "wheels" / "0a"
    wheel_dir.mkdir(parents=True)
    for name in wheels:
        (wheel_dir / name).write_bytes(b"w" * 1000)
    download = cache.shared_dir(root) / "http-v2" / "9a" / "body"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"z" * 500)
    return partition


def run(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["cache", *args])
    return cache_command.main()


def test_info(cache_config, monkeypatch, capsys):
    root = cache_config()
    partition = populate(root)

    assert run(monkeypatch) == 0

    out = capsys.readouterr().out
    assert str(root) in out
    assert partition.name in out
    assert "1 wheels" in out
    assert "Shared downloads: 1 files" in out


def test_info_is_the_default(cache_config, monkeypatch, capsys):
    root = cache_config()
    populate(root)

    run(monkeypatch)
    default = capsys.readouterr().out
    run(monkeypatch, "--info")

    assert capsys.readouterr().out == default


def test_info_on_a_cache_that_does_not_exist(cache_config, monkeypatch, capsys):
    cache_config()
    assert run(monkeypatch) == 0
    assert "does not exist yet" in capsys.readouterr().out


def test_info_says_when_caching_is_off(cache_config, monkeypatch, capsys):
    cache_config(enabled=False)
    run(monkeypatch)
    assert "disabled" in capsys.readouterr().out


def test_prune_uses_the_configured_max_age(cache_config, monkeypatch, capsys):
    import os
    import time

    root = cache_config(max_age_days=30)
    partition = populate(root)
    old = time.time() - 45 * 86400
    for path in partition.rglob("*"):
        if path.is_file() and not path.is_symlink():
            os.utime(path, (old, old))

    assert run(monkeypatch, "--prune") == 0

    out = capsys.readouterr().out
    assert "Removed 1 file(s) unused for 30 days" in out


def test_prune_keeps_recent_entries(cache_config, monkeypatch, capsys):
    root = cache_config(max_age_days=30)
    partition = populate(root)

    run(monkeypatch, "--prune")

    assert "Removed 0 file(s)" in capsys.readouterr().out
    assert list((partition / "wheels").rglob("*.whl"))


def test_prune_max_age_override(cache_config, monkeypatch, capsys):
    import os
    import time

    root = cache_config(max_age_days=30)
    partition = populate(root)
    somewhat_old = time.time() - 10 * 86400
    for path in partition.rglob("*"):
        if path.is_file() and not path.is_symlink():
            os.utime(path, (somewhat_old, somewhat_old))

    # Younger than the configured 30 days, older than an explicit 5.
    run(monkeypatch, "--prune", "--max-age-days", "5")

    assert "unused for 5 days" in capsys.readouterr().out
    assert not list((partition / "wheels").rglob("*.whl"))


def test_clear(cache_config, monkeypatch, capsys):
    root = cache_config()
    populate(root)

    assert run(monkeypatch, "--clear") == 0

    assert not root.exists()
    assert "Removed" in capsys.readouterr().out


def test_clear_is_idempotent(cache_config, monkeypatch, capsys):
    root = cache_config()
    run(monkeypatch, "--clear")
    capsys.readouterr()

    assert run(monkeypatch, "--clear") == 0
    assert "Nothing to remove" in capsys.readouterr().out
    assert not root.exists()


def test_the_actions_are_mutually_exclusive(cache_config, monkeypatch):
    cache_config()
    with pytest.raises(SystemExit):
        run(monkeypatch, "--prune", "--clear")


def test_max_age_days_needs_prune(cache_config, monkeypatch):
    cache_config()
    with pytest.raises(SystemExit):
        run(monkeypatch, "--max-age-days", "5")


def test_the_command_follows_the_cache_dir_env_var(
    cache_config, monkeypatch, tmp_path, capsys
):
    # The command has to talk about the cache a run would actually use.
    cache_config()
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(cache.CACHE_DIR_ENV_VAR, str(elsewhere))
    populate(elsewhere)

    run(monkeypatch)

    assert str(elsewhere) in capsys.readouterr().out


def test_paths_are_not_treated_as_console_markup(
    cache_config, monkeypatch, tmp_path, capsys
):
    # rich.print would read "[bold]" in a path as a style tag and swallow it,
    # so the report is written with plain print.
    awkward = tmp_path / "[bold]cache"
    cache_config(dir=str(awkward))
    populate(awkward)

    run(monkeypatch)

    assert str(awkward) in capsys.readouterr().out


def test_the_command_is_registered():
    from bench_runner import __main__

    assert "cache" in __main__.COMMANDS
