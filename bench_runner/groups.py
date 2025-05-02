from __future__ import annotations

import functools

from . import config
from . import runners as mrunners
from .util import PathLike


class Group:
    def __init__(
        self,
        name: str,
        runners: list[mrunners.Runner],
        displayname: str = "",
        collapsed: bool = False,
    ):
        self.name = name
        self.runners = runners
        self.displayname = displayname
        self.collapsed = collapsed
        self.update_runners()

    def update_runners(self):
        for runner in self.runners:
            runner.groups.add(self.name)


@functools.cache
def get_groups(cfgpath: PathLike | None = None) -> dict[str, dict]:
    groupcfgs = config.get_bench_runner_config(cfgpath).get("groups", {})
    all_runners = {r.nickname: r for r in mrunners._get_runners_without_groups(cfgpath)}
    groups = {}
    processing = set()

    def process_group(name):
        if name in groups:
            # Already processed.
            return
        processing.add(name)
        groupcfg = groupcfgs[name].copy()
        nicknames = groupcfg.pop("runners")
        runners = []
        for nickname in nicknames:
            if nickname in groups:
                runners.extend(groups[nickname].runners)
            elif nickname in groupcfgs:
                if nickname in processing:
                    ValueError(
                        f"Circular inclusion of groups {name!r} and {nickname!r}"
                    )
                process_group(nickname)
                runners.extend(groups[nickname].runners)
            elif nickname in all_runners:
                runners.append(all_runners[nickname])
            else:
                raise ValueError(
                    f"Runner {nickname} in group {name} not found in bench_runner.toml"
                )
        groups[name] = Group(name=name, runners=runners, **groupcfg)
        processing.remove(name)

    for name in groupcfgs:
        process_group(name)

    assert len(groups) == len(groupcfgs)
    return groups
