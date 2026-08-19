from pathlib import Path
import socket


from bench_runner import runners


DATA_PATH = Path(__file__).parent / "data"


def test_get_runner_for_hostname(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "pyperf")

    runner = runners.get_runner_for_hostname(cfgpath=DATA_PATH / "bench_runner.toml")

    assert runner.os == "linux"
    assert runner.arch == "x86_64"
    assert runner.hostname == "pyperf"


def test_interactive_plots_defaults_to_no_base_url():
    from bench_runner import config

    assert config.InteractivePlots().base_url == ""


def test_interactive_plots_appends_missing_trailing_slash():
    from bench_runner import config

    # Callers join paths straight onto base_url, so it has to end in a slash.
    cfg = config.InteractivePlots(base_url="https://myorg.github.io/repo")
    assert cfg.base_url == "https://myorg.github.io/repo/"


def test_interactive_plots_keeps_existing_trailing_slash():
    from bench_runner import config

    cfg = config.InteractivePlots(base_url="https://myorg.github.io/repo/")
    assert cfg.base_url == "https://myorg.github.io/repo/"


def test_interactive_plots_empty_base_url_is_left_alone():
    from bench_runner import config

    # An empty string must not become a bare "/", which would look configured.
    assert config.InteractivePlots(base_url="").base_url == ""


def test_config_coerces_interactive_plots_table(tmp_path):
    from bench_runner import config

    # tomllib hands the section over as a plain dict, which Config must coerce.
    toml = (DATA_PATH / "bench_runner.toml").read_text()
    cfgpath = tmp_path / "bench_runner.toml"
    cfgpath.write_text(
        toml + '\n[interactive_plots]\nbase_url = "https://example.com/x"\n'
    )

    cfg = config.get_config(cfgpath)
    assert isinstance(cfg.interactive_plots, config.InteractivePlots)
    assert cfg.interactive_plots.base_url == "https://example.com/x/"


def test_config_without_interactive_plots_section_gets_default():
    from bench_runner import config

    # The stock test config has no [interactive_plots] table at all.
    cfg = config.get_config(DATA_PATH / "bench_runner.toml")
    assert isinstance(cfg.interactive_plots, config.InteractivePlots)
    assert cfg.interactive_plots.base_url == ""
