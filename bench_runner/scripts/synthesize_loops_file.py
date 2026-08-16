import argparse
import collections
import errno
import json
import pathlib
import platform
import socket
import sys
import time
from typing import Iterable

import rich_argparse

# pyperf owns the loops table format, so prefer pyperf's own writer: that way
# the file cannot drift out of step with the pyperf that will read it. The
# feature is newer than the pyperf this package pins, though, so fall back to
# writing the same layout here. Keep FALLBACK_TABLE_VERSION equal to pyperf's
# TABLE_VERSION -- a table whose version pyperf does not recognise is refused
# rather than misread, so a mismatch fails loudly.
try:
    from pyperf._loops_table import build_table as _pyperf_build_table
except ImportError:  # pragma: no cover - depends on the installed pyperf
    _pyperf_build_table = None

FALLBACK_TABLE_VERSION = 1

# pyperf's own default for --min-time. Loop counts are only comparable between
# runs that calibrated against the same target, so the table records it.
DEFAULT_MIN_TIME = 0.1

# ...but it cannot be recovered from the results. pyperf records `loops` in
# each benchmark's metadata and does NOT record `min_time` anywhere, so a
# result file cannot say what it calibrated against. The value written here is
# therefore an assumption -- pyperf's default, which is what bench_runner gets
# because it never passes --min-time -- unless the caller states otherwise with
# --min-time. Writing an unverified value into a provenance field is worse than
# admitting it is assumed, hence this note and the flag.
MIN_TIME_IS_ASSUMED = (
    "pyperf does not record --min-time in its results, so it cannot be read "
    "back from them; pass --min-time to record the value the runs actually "
    f"used. Assuming pyperf's default of {DEFAULT_MIN_TIME}."
)


def build_table(loops, min_time, machine=None):
    if _pyperf_build_table is not None:
        table = _pyperf_build_table(loops, min_time)
    else:
        table = {
            "table_version": FALLBACK_TABLE_VERSION,
            "min_time": min_time,
            "machine": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "loops": dict(sorted(loops.items())),
        }
    if machine:
        # pyperf stamps the machine the table is generated ON, because that is
        # also the machine it benchmarks on. Here it is not: this runs wherever
        # the results happen to be collected, so the counts describe the runner
        # they were measured on, not this one. `machine` is the field pyperf
        # tells you to use to decide whether a table applies, so it has to name
        # the machine the counts came from.
        table["machine"] = {**table["machine"], **machine}
    return table


def load_table(loops_file):
    """
    Read an existing loops table, for -u/--update.
    """
    with loops_file.open() as f:
        data = json.load(f)
    if "loops" not in data:
        raise SystemExit(
            f"{loops_file} is not a loops table. Files written before "
            "--loops-table replaced --same-loops held benchmark results "
            "instead; regenerate with -f rather than -u."
        )
    return data


def parse_result(results_file, benchmark_data, machines):
    with results_file.open() as f:
        result = json.load(f)
    bms = result["benchmarks"]
    if len(bms) == 1 and "metadata" not in bms[0]:
        # Sometimes a .json file contains just a single benchmark.
        bms = [result]
    # Suite-level metadata describes the run, and so the machine that produced
    # these loop counts. pyperf does not record --min-time anywhere, which is
    # why it cannot be recovered here; see MIN_TIME_IS_ASSUMED.
    suite_metadata = result.get("metadata", {})
    if hostname := suite_metadata.get("hostname"):
        machines["hostname"].add(hostname)
    if plat := suite_metadata.get("platform"):
        # Collected separately from the hostname: one machine reports different
        # platform strings over its life (a kernel upgrade is enough), so a
        # difference here does not mean a different machine.
        machines["platform"].add(plat)
    for bm in bms:
        if "metadata" not in bm:
            raise RuntimeError(f"Invalid data {bm.keys()!r} in {results_file}")
        metadata = bm["metadata"]
        # Keyed by the name the benchmark REPORTS. That is what a loops table
        # is keyed by, and it is the only key that works for a script
        # reporting several benchmarks under names of its own choosing.
        benchmark_data[metadata["name"]].append(metadata["loops"])


def _main(
    loops_file: pathlib.Path,
    update: bool,
    overwrite: bool,
    merger: str,
    results: Iterable[pathlib.Path],
    min_time: float | None = None,
):
    if not update and not overwrite and loops_file.exists():
        raise OSError(
            errno.EEXIST,
            f"{loops_file} exists (use -f to overwrite, -u to merge data)",
        )
    benchmark_data = collections.defaultdict(list)
    machines = {"hostname": set(), "platform": set()}
    existing_min_time = None
    if update and loops_file.exists():
        existing = load_table(loops_file)
        for name, loops in existing["loops"].items():
            benchmark_data[name].append(loops)
        existing_min_time = existing.get("min_time")
    for result_file in results:
        parse_result(result_file, benchmark_data, machines)

    merge_func = {
        "max": max,
        "min": min,
    }[merger]

    # Loop counts calibrated against different --min-time targets are not
    # comparable, so a table cannot honestly describe both. Refuse rather than
    # pick one: writing a table that is wrong for some of its own entries
    # defeats the point of pinning loop counts at all.
    if (
        min_time is not None
        and existing_min_time is not None
        and min_time != existing_min_time
    ):
        raise SystemExit(
            f"--min-time={min_time} disagrees with the {existing_min_time} "
            f"recorded in {loops_file}. Loop counts calibrated against "
            "different targets are not comparable, so they cannot share a "
            "table; regenerate with -f instead of merging with -u."
        )
    if min_time is None:
        min_time = existing_min_time
    if min_time is None:
        min_time = DEFAULT_MIN_TIME
        print(f"NOTE: {MIN_TIME_IS_ASSUMED}", file=sys.stderr)

    # Same argument for the machine: a table describes the machine its counts
    # were measured on, and counts from two machines cannot both be described.
    hostnames = machines["hostname"]
    if len(hostnames) > 1:
        raise SystemExit(
            "results come from more than one machine "
            f"({', '.join(sorted(hostnames))}); their loop counts are not "
            "comparable and cannot share a table. "
            "Generate one table per machine."
        )
    machine = None
    if hostnames:
        machine = {"hostname": hostnames.copy().pop()}
        platforms = machines["platform"]
        # One host legitimately spans several platform strings over time, so
        # record them all rather than silently picking one.
        if platforms:
            machine["platform"] = " | ".join(sorted(platforms))

    loops = {bm: merge_func(values) for bm, values in benchmark_data.items()}
    with loops_file.open("w") as f:
        json.dump(build_table(loops, min_time, machine), f, sort_keys=True, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Synthesize a pyperf loops table from one or more benchmark results, for
        use with `pyperformance`'s `--loops-table` (or the
        `PYPERFORMANCE_LOOPS_FILE` environment variable, which bench_runner
        passes through).

        The table is keyed by the name each benchmark reports, and records the
        machine the results came from and the --min-time they are assumed to
        have been calibrated against.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-o", "--loops_file", help="loops file to write to", required=True
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-u", "--update", action="store_true", help="add to existing loops file"
    )
    group.add_argument(
        "-f", "--overwrite", action="store_true", help="replace loops file"
    )
    parser.add_argument(
        "-s",
        "--select",
        choices=("max", "min"),
        default="max",
        help="how to merge multiple runs",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="the --min-time the results were calibrated against. pyperf does "
        "not record this, so it cannot be read back from the results; without "
        f"it the table records pyperf's default of {DEFAULT_MIN_TIME}",
    )
    parser.add_argument("results", nargs="+", help="benchmark results to parse")
    args = parser.parse_args()

    _main(
        pathlib.Path(args.loops_file),
        args.update,
        args.overwrite,
        args.select,
        [pathlib.Path(r) for r in args.results],
        min_time=args.min_time,
    )


if __name__ == "__main__":
    main()
