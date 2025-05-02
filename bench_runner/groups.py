from __future__ import annotations

import functools

from . import config
from . import runners as mrunners
from .util import PathLike


class Group:
    def __init__(
        self,
        name: str,
        runners: set[str],
        displayname: str = "",
        collapsed: bool = False,
    ):
        self.name = name
        self.runners = runners
        self.displayname = displayname
        self.collapsed = collapsed

    def update_runners(self, runners_by_nickname: dict[str, mrunners.Runner]):
        for nickname in self.runners:
            if nickname not in runners_by_nickname:
                raise ValueError(
                    f"Runner {nickname} in group {self.name}"
                    + " not found in bench_runner.toml"
                )

            runners_by_nickname[nickname].groups.add(self.name)


@functools.cache
def get_groups(cfgpath: PathLike | None = None) -> dict[str, Group]:
    groupcfgs = config.get_config(cfgpath).groups
    groups: dict[str, Group] = {}
    processing = set()

    def process_group(name):
        if name in groups:
            # Already processed.
            return
        processing.add(name)
        groupcfg = groupcfgs[name]
        runners = set()
        for nickname in groupcfg.runners:
            if nickname in groups:
                runners.update(groups[nickname].runners)
            elif nickname in groupcfgs:
                if nickname in processing:
                    ValueError(
                        f"Circular inclusion of groups {name!r} and {nickname!r}"
                    )
                process_group(nickname)
                runners.update(groups[nickname].runners)
            else:
                runners.add(nickname)
        groups[name] = Group(
            name=name,
            runners=runners,
            displayname=groupcfg.displayname,
            collapsed=groupcfg.collapsed,
        )
        processing.remove(name)

    for name in groupcfgs:
        process_group(name)

    assert len(groups) == len(groupcfgs)
    return groups
