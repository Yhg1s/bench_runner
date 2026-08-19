from pathlib import Path


import pytest


from bench_runner import groups


DATA_PATH = Path(__file__).parent / "data"


def _write_config(tmp_path, body):
    cfgpath = tmp_path / "bench_runner.toml"
    cfgpath.write_text((DATA_PATH / "bench_runner.toml").read_text() + body)
    groups.get_groups.cache_clear()
    return cfgpath


def test_group_expands_to_its_runners(tmp_path):
    cfgpath = _write_config(
        tmp_path,
        """
[groups.linuxes]
runners = ["pyperf"]
""",
    )
    result = groups.get_groups(cfgpath)
    groups.get_groups.cache_clear()
    assert "pyperf" in result["linuxes"].runners


def test_group_can_include_another_group(tmp_path):
    cfgpath = _write_config(
        tmp_path,
        """
[groups.inner]
runners = ["pyperf"]

[groups.outer]
runners = ["inner"]
""",
    )
    result = groups.get_groups(cfgpath)
    groups.get_groups.cache_clear()
    assert result["outer"].runners == result["inner"].runners


def test_mutually_recursive_groups_are_rejected(tmp_path):
    """
    A cycle used to build the ValueError and drop it on the floor, so the
    recursion ran until something further down failed with an unrelated error.
    """
    cfgpath = _write_config(
        tmp_path,
        """
[groups.a]
runners = ["b"]

[groups.b]
runners = ["a"]
""",
    )
    with pytest.raises(ValueError, match="Circular inclusion"):
        groups.get_groups(cfgpath)
    groups.get_groups.cache_clear()


def test_self_referential_group_is_rejected(tmp_path):
    cfgpath = _write_config(
        tmp_path,
        """
[groups.loop]
runners = ["loop"]
""",
    )
    with pytest.raises(ValueError, match="Circular inclusion"):
        groups.get_groups(cfgpath)
    groups.get_groups.cache_clear()
