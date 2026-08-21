from __future__ import annotations


import dataclasses
import os
import socket
from typing import Literal


from .util import PathLike


@dataclasses.dataclass
class PlotConfig:
    # The name of the runner in the plot legend
    name: str
    # A matplotlib color to use for this runner in plots
    color: str = "C0"
    # A matplotlib line style to use for this runner in plots
    style: str = "-"
    # A matplotlib marker to use for this runner in plots
    marker: str = "s"


@dataclasses.dataclass
class Runner:
    # The short "nickname" of the runner
    nickname: str
    # The OS of the runner
    os: Literal["linux", "darwin", "windows", "unknown"]
    # The architecture of the runner, e.g. "x86_64", "arm64"
    arch: str
    # The hostname of the runner, used to identify which runner we are running on
    hostname: str
    # Whether the runner is available for benchmarking (e.g. not a VM)
    available: bool = True
    # Environment variables to set for the benchmark
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    # The name of the Github runner to use for this machine, only required when
    # this runner needs to map to another physical machine.
    github_runner_name: str | None = None
    # Whether to include this runner in the "all" choice
    include_in_all: bool = True
    # The plot configuration for this runner
    plot: PlotConfig | None = None
    # The number of cores to use to compile CPython. If not provided, `make -j`
    # will be used.
    use_cores: int | None = None
    # The loops table to use on this runner, overriding
    # PYPERFORMANCE_LOOPS_FILE. Loop counts describe the machine they were
    # measured on, so a repo benchmarking on several runners needs one table
    # per runner rather than one for all of them.
    loops_table_file: str | None = None
    # Whether this runner refuses to calibrate, overriding
    # PYPERFORMANCE_NO_CALIBRATE. None means "not configured", which is what
    # lets an explicit `no_calibrate = false` here turn the variable off for
    # one runner while it stays on everywhere else.
    no_calibrate: bool | None = None
    # The groups this runner belongs to. Not configured per-runner: filled in
    # from the [groups] sections by groups.get_groups().
    groups: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self):
        if self.github_runner_name is None:
            self.github_runner_name = self.name
        if self.plot is None:
            self.plot = PlotConfig(name=self.nickname)
        else:
            self.plot = PlotConfig(**self.plot)  # pyright: ignore[reportCallIssue]

    @property
    def name(self) -> str:
        return f"{self.os}-{self.arch}-{self.nickname}"

    @property
    def display_name(self) -> str:
        return f"{self.os} {self.arch} ({self.nickname})"


unknown_runner = Runner("unknown", "unknown", "unknown", "unknown", False, {}, None)


def get_runners_by_hostname(cfgpath: PathLike | None = None) -> dict[str, Runner]:
    from . import config

    return {x.hostname: x for x in config.get_config(cfgpath).runners.values()}


def get_runners_by_nickname(cfgpath: PathLike | None = None) -> dict[str, Runner]:
    from . import config
    from . import groups as mgroups

    runners = config.get_config(cfgpath).runners
    # Groups are resolved to nicknames, so this is where a runner learns which
    # groups it is in. Doing it here, against the one config cached for
    # cfgpath, keeps every caller looking at the same Runner objects.
    for group in mgroups.get_groups(cfgpath).values():
        group.update_runners(runners)
    return runners


def get_nickname_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> str:
    # The envvar BENCHMARK_MACHINE_NICKNAME is used to override the machine that
    # results are reported for.
    if "BENCHMARK_MACHINE_NICKNAME" in os.environ:
        return os.environ["BENCHMARK_MACHINE_NICKNAME"]
    return get_runner_for_hostname(hostname, cfgpath).nickname


def get_runner_by_nickname(nickname: str, cfgpath: PathLike | None = None) -> Runner:
    from . import config

    return config.get_config(cfgpath).runners.get(nickname, unknown_runner)


def get_runner_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> Runner:
    if hostname is None:
        hostname = socket.gethostname()
    return get_runners_by_hostname(cfgpath).get(hostname, unknown_runner)


def get_current_runner(cfgpath: PathLike | None = None) -> Runner:
    """
    The runner this process is running as.

    One machine can host more than one runner, in which case the hostname does
    not say which one this is, so an explicitly set nickname wins over it.
    """
    if nickname := os.environ.get("BENCHMARK_MACHINE_NICKNAME"):
        return get_runner_by_nickname(nickname, cfgpath)
    return get_runner_for_hostname(cfgpath=cfgpath)


def get_runners_from_nicknames_and_groups(
    nicknames: list[str], cfgpath: PathLike | None = None
) -> list[Runner]:
    from . import groups as mgroups

    groups = mgroups.get_groups(cfgpath)
    runners = get_runners_by_nickname(cfgpath)
    # Keyed by nickname rather than a set of runners: Runner is a dataclass,
    # so it is unhashable.
    result: dict[str, Runner] = {}
    for nickname in nicknames:
        if nickname in groups:
            for n in groups[nickname].runners:
                result[n] = runners[n]
        else:
            if nickname not in runners:
                raise ValueError(f"Runner {nickname} not found in bench_runner.toml")
            result[nickname] = runners[nickname]
    return sorted(result.values(), key=lambda r: r.nickname)
