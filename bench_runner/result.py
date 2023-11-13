# Utilities to manage a results file
from __future__ import annotations


import functools
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any, Iterable, Optional


from . import git
from . import hpt
from . import runners


def _clean(string: str) -> str:
    """
    Clean an arbitrary string to be suitable for a filename.

    It can't contain dashes, since dashes are used as a delimiter.
    """
    return string.replace("-", "_")


def _clean_for_url(string: str) -> str:
    """
    Clean an arbitrary string to be suitable for a filename, being careful to
    create something that can be re-used as a URL.

    It can't contain dashes, since dashes are used as a delimiter.
    """
    return string.replace("-", "%2d")


def _get_platform_value(python: Path, item: str) -> str:
    """
    Get a value from the platform module of the given Python interpreter.
    """
    output = subprocess.check_output(
        [python, "-c", f"import platform; print(platform.{item}())"], encoding="utf-8"
    )
    return output.strip().lower()


class Comparison:
    def __init__(self, ref: "Result", head: "Result", base: str):
        self.ref = ref
        self.head = head
        self.base = base

    def copy(self):
        return type(self)(self.ref, self.head, self.base)

    @property
    def filename(self) -> Optional[Path]:
        if self.ref == self.head:
            return None

        if (
            self.ref.cpython_hash == self.head.cpython_hash
            and self.ref.flags == self.head.flags
        ):
            return None

        return self.head.filename.parent / (
            f"{self.head.filename.stem}-vs-{self.base}.txt"
        )

    @functools.cached_property
    def contents(self) -> Optional[str]:
        if self.filename is None:
            return None

        if self.filename.with_suffix(".md").is_file():
            with open(self.filename.with_suffix(".md"), "r", encoding="utf-8") as fd:
                return fd.read()
        else:
            return self._generate_contents()

    @property
    def contents_lines(self) -> Iterable[str]:
        contents = self.contents
        if contents is None:
            return []
        else:
            return contents.splitlines()

    def _generate_contents(self) -> Optional[str]:
        return None


class BenchmarkComparison(Comparison):
    @functools.cached_property
    def geometric_mean(self) -> str:
        if self.ref == self.head:
            return ""

        lines = self.contents_lines

        if (
            self.head.benchmark_hash is None
            or self.ref.benchmark_hash != self.head.benchmark_hash
        ):
            suffix = r" \*"
        else:
            suffix = ""

        geometric_mean = None
        for line in lines:
            # We want to get the *last* geometric mean in the file, in case
            # it's divided by tags
            if "Geometric mean" in line:
                geometric_mean = line.split("|")[3].strip() + suffix

        if geometric_mean is None:
            geometric_mean = "not sig"

        return geometric_mean

    @functools.cached_property
    def hpt_reliability(self) -> Optional[str]:
        if self.ref == self.head:
            return None

        lines = self.contents_lines

        for line in lines:
            m = re.match(r"- Reliability score: (\S+)", line)
            if m is not None:
                return m.group(1)

        return None

    def hpt_percentile(self, percentile: int) -> Optional[str]:
        if self.ref == self.head:
            return None

        lines = self.contents_lines

        for line in lines:
            m = re.match(r"- ([0-9]+)% likely to have a (\S+) of (\S+)", line)
            if m is not None:
                if int(m.group(1)) == percentile:
                    if m.group(2) == "slowdown":
                        suffix = "slower"
                    else:
                        suffix = "faster"
                    return f"{m.group(3)} {suffix}"

        return None

    def hpt_percentile_float(self, percentile: int) -> Optional[float]:
        result = self.hpt_percentile(percentile)
        if result is not None:
            num = float(result.split()[0][:-1])
            if result.endswith("slower"):
                return 1.0 - (num - 1.0)
            else:
                return num
        else:
            return None

    @property
    def summary(self) -> str:
        if self.ref == self.head:
            return ""

        result = self.geometric_mean
        reliability = self.hpt_reliability
        if reliability is not None:
            reliability = reliability[:-4]
            result += f" ({reliability}% rel.)"

        return result

    @property
    def long_summary(self) -> str:
        if self.ref == self.head:
            return ""

        result = f"Geometric mean: {self.geometric_mean}"
        reliability = self.hpt_reliability
        if reliability is not None:
            subresult = f"HPT: reliability of {reliability}"
            percentile = self.hpt_percentile(99)
            if percentile is not None:
                subresult += f", {percentile} at 99th %ile"
            result += f" ({subresult})"

        return result

    def _generate_contents(self) -> Optional[str]:
        return (
            subprocess.check_output(
                [
                    "pyperf",
                    "compare_to",
                    "-G",
                    "--table",
                    "--table-format",
                    "md",
                    self.ref.filename,
                    self.head.filename,
                ],
                encoding="utf-8",
            )
            + "\n\n"
            + hpt.make_report(self.ref.filename, self.head.filename)
        )


class PystatsComparison(Comparison):
    def _generate_contents(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                [
                    sys.executable,
                    Path("cpython") / "Tools" / "scripts" / "summarize_stats.py",
                    self.ref.filename,
                    self.head.filename,
                ],
                encoding="utf-8",
            )
        except subprocess.CalledProcessError:
            return None


def comparison_factory(ref: "Result", head: "Result", base: str) -> Comparison:
    if head.result_info[0] == "raw results":
        return BenchmarkComparison(ref, head, base)
    elif head.result_info[0] == "pystats raw":
        return PystatsComparison(ref, head, base)

    raise ValueError(f"Can not compare result of type {head.result_info[0]}")


class Result:
    """
    Stores information about a single result file.
    """

    def __init__(
        self,
        nickname: str,
        machine: str,
        fork: str,
        ref: str,
        version: str,
        cpython_hash: str,
        extra: list[str] = [],
        suffix: str = ".json",
        commit_datetime: Optional[str] = None,
        flags: list[str] = [],
    ):
        self.nickname = nickname
        if nickname not in runners.get_runners_by_nickname():
            raise ValueError(f"Unknown runner {nickname}")
        self.machine = machine
        self.fork = fork
        self.ref = ref
        self.version = version
        self.cpython_hash = cpython_hash
        self.extra = extra
        self.suffix = suffix
        self.flags = sorted(set(flags))
        self._commit_datetime = commit_datetime
        self._filename = None
        self.bases = {}

    @classmethod
    def from_filename(cls, filename: Path) -> "Result":
        (
            name,
            date,
            nickname,
            machine,
            fork,
            ref,
            version,
            cpython_hash,
            *extra,
        ) = filename.stem.split("-")
        assert name == "bm"
        (name, _, _, _, *flags) = filename.parent.name.split("-")
        assert name == "bm"
        assert len(flags) <= 1
        if len(flags) == 1:
            flags = flags[0].split(",")
        obj = cls(
            nickname=nickname,
            machine=machine,
            fork=fork,
            ref=ref,
            version=version,
            cpython_hash=cpython_hash,
            extra=extra,
            suffix=filename.suffix,
            flags=flags,
        )
        obj._filename = filename
        return obj

    @classmethod
    def from_scratch(
        cls,
        python: Path,
        fork: str,
        ref: str,
        extra: list[str] = [],
        flags: list[str] = [],
    ) -> "Result":
        result = cls(
            _clean(runners.get_nickname_for_hostname(socket.gethostname())),
            _clean(_get_platform_value(python, "machine")),
            _clean_for_url(fork),
            _clean(ref[:20]),
            _clean(_get_platform_value(python, "python_version")),
            git.get_git_hash(Path("cpython"))[:7],
            extra,
            ".json",
            commit_datetime=git.get_git_commit_date(Path("cpython")),
            flags=flags,
        )
        return result

    @property
    def filename(self) -> Path:
        if self._filename is None:
            date = self.commit_date.replace("-", "")
            if self.extra:
                extra = ["-".join(self.extra)]
            else:
                extra = []
            if self.flags:
                flags = [",".join(self.flags)]
            else:
                flags = []
            self._filename = (
                Path("results")
                / "-".join(["bm", date, self.version, self.cpython_hash, *flags])
                / (
                    "-".join(
                        [
                            "bm",
                            date,
                            self.nickname,
                            self.machine,
                            self.fork,
                            self.ref,
                            self.version,
                            self.cpython_hash,
                            *extra,
                        ]
                    )
                    + self.suffix
                )
            )
        return self._filename

    @functools.cached_property
    def result_info(self) -> tuple[Optional[str], Optional[str]]:
        if self.extra == [] and self.suffix == ".json":
            return ("raw results", None)
        elif self.extra[0] == "pystats":
            if len(self.extra) == 3 and self.extra[1] == "vs" and self.suffix == ".md":
                return ("pystats diff", self.extra[2])
            elif self.suffix == ".md":
                if self.filename.name.endswith("pystats.md"):
                    return ("pystats table", None)
                else:
                    return (None, None)
            elif self.suffix == ".json":
                return ("pystats raw", None)
        elif len(self.extra) == 2 and self.extra[0] == "vs":
            if self.suffix == ".md":
                return ("table", self.extra[1])
            elif self.suffix == ".png":
                return ("plot", self.extra[1])
        raise ValueError("Unknown result type")

    @property
    def fast_contents(self) -> dict[str, Any]:
        """
        Gets just a portion of the JSON contents when the whole set isn't needed.
        """
        if hasattr(self, "_full_contents"):
            return self._full_contents
        if hasattr(self, "_fast_contents"):
            return self._fast_contents

        try:
            import ijson
        except ImportError:
            return self.contents

        fast_contents = {"metadata": {}, "benchmarks": []}
        with open(self.filename, "rb") as fd:
            parser = ijson.parse(fd)
            for prefix, _, value in parser:
                if prefix == "benchmarks.item.metadata.name":
                    fast_contents["benchmarks"].append({"metadata": {"name": value}})
                elif prefix == "benchmarks.item.runs.item.metadata.date":
                    if len(fast_contents["benchmarks"]) == 0:
                        fast_contents["benchmarks"].append({})
                    if len(fast_contents["benchmarks"]) == 1:
                        fast_contents["benchmarks"][-1]["runs"] = [
                            {"metadata": {"date": value}}
                        ]
                elif prefix.startswith("metadata."):
                    fast_contents["metadata"][prefix[9:]] = value

        self._fast_contents = fast_contents
        return fast_contents

    @property
    def contents(self) -> dict[str, Any]:
        if hasattr(self, "_full_contents"):
            return self._full_contents
        with open(self.filename, "rb") as fd:
            self._full_contents = json.load(fd)
        return self._full_contents

    @property
    def metadata(self) -> dict[str, Any]:
        return self.fast_contents.get("metadata", {})

    @property
    def commit_datetime(self) -> str:
        if self._commit_datetime is not None:
            return self._commit_datetime
        return self.metadata.get("commit_date", "<unknown>")

    @property
    def commit_date(self) -> str:
        return self.commit_datetime[:10]

    @property
    def run_datetime(self) -> str:
        return self.fast_contents["benchmarks"][0]["runs"][0]["metadata"]["date"]

    @property
    def run_date(self) -> str:
        return self.run_datetime[:10]

    @property
    def commit_merge_base(self) -> str:
        return self.metadata.get("commit_merge_base", None)

    @property
    def benchmark_hash(self) -> str:
        return self.metadata.get("benchmark_hash", None)

    @property
    def hostname(self) -> str:
        return self.metadata.get("hostname", "unknown host")

    @property
    def system(self) -> str:
        return runners.get_runner_by_nickname(self.nickname).os

    @property
    def runner(self) -> str:
        return f"{self.system} {self.machine} ({self.nickname})"

    @property
    def cpu_model_name(self) -> str:
        return self.metadata.get("cpu_model_name", "missing")

    @property
    def platform(self) -> str:
        return self.metadata.get("platform", "missing")

    @property
    def github_action_url(self) -> Optional[str]:
        return self.metadata.get("github_action_url", None)

    @functools.cached_property
    def benchmark_names(self) -> set[str]:
        contents = self.fast_contents
        names = set()
        for benchmark in contents["benchmarks"]:
            if "metadata" in benchmark:
                names.add(benchmark["metadata"]["name"])
            else:
                names.add(contents["metadata"]["name"])
        return names

    @functools.cached_property
    def parsed_version(self):
        from packaging import version as pkg_version

        return pkg_version.parse(self.version.replace("+", "0"))

    def match_to_bases(self, bases: list[str], results: Iterable["Result"]) -> None:
        loose_results = [
            ref
            for ref in results
            if (
                ref != self
                and ref.nickname == self.nickname
                and ref.fork == "python"
                and ref.extra == self.extra
            )
        ]

        if self.benchmark_hash is not None:
            exact_results = [
                ref
                for ref in loose_results
                if ref.benchmark_hash == self.benchmark_hash
            ]
        else:
            exact_results = []

        def find_match(base, func):
            # Try for an exact match (same benchmark_hash) first,
            # then fall back to less exact.
            for result_set in [exact_results, loose_results]:
                for ref in result_set:
                    if func(ref):
                        self.bases[base] = comparison_factory(ref, self, base)
                        return True
            return False

        for base in bases:
            find_match(base, lambda ref: ref.version == base)

        merge_base = self.commit_merge_base
        if merge_base is not None:
            if (
                not find_match(
                    "base",
                    lambda ref: (
                        merge_base.startswith(ref.cpython_hash)
                        and ref.flags == self.flags
                    ),
                )
                and self.fork == "python"
            ):
                # Compare Tier 1 and Tier 2 of the same commit
                find_match("base", lambda ref: (ref.cpython_hash == self.cpython_hash))


def has_result(
    results_dir: Path, commit_hash: str, machine: str, pystats: bool, flags: list[str]
) -> Optional[Result]:
    if machine == "all":
        nickname = None
    else:
        _, _, nickname = machine.split("-")

    results = load_all_results([], results_dir, False)

    if pystats:
        for result in results:
            if (
                commit_hash.startswith(result.cpython_hash)
                and result.result_info[0] == "pystats raw"
                and result.flags == flags
            ):
                return result
    else:
        for result in results:
            if (
                commit_hash.startswith(result.cpython_hash)
                and (nickname is None or result.nickname == nickname)
                and result.flags == flags
            ):
                return result

    return None


def load_all_results(
    bases: Optional[list[str]], results_dir: Path, sorted: bool = True
) -> list[Result]:
    results = []

    for entry in results_dir.glob("**/*.json"):
        result = Result.from_filename(entry)
        if result.result_info[0] not in ["raw results", "pystats raw"]:
            continue
        results.append(Result.from_filename(entry))
    if len(results) == 0:
        raise ValueError("Didn't find any results.  That seems fishy.")

    if bases is not None:
        for result in results:
            result.match_to_bases(bases, results)

    if sorted:
        results.sort(
            key=lambda x: (
                x.parsed_version,
                x.commit_datetime,
            ),
            reverse=True,
        )

    return results
