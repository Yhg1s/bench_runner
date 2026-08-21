import argparse
import copy
import os

import rich_argparse

from bench_runner import runners
from bench_runner import flags as mflags
from bench_runner.scripts import workflow


def set_environment_for(machine):
    _, _, nickname = machine.split("-")
    runner = runners.get_runner_by_nickname(nickname)
    if runner is runners.unknown_runner:
        raise ValueError(f"Invalid runner {machine}")
    vars = copy.copy(runner.env)
    vars["BENCHMARK_MACHINE_NICKNAME"] = runner.nickname
    vars["BENCHMARK_RUNNER_NAME"] = runner.name
    os.environ.update(vars)


def _main(fork: str, ref: str, machine: str, *args, **kwargs):
    set_environment_for(machine)
    workflow._main(fork, ref, machine, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow with info from bench_runner.toml.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument("machine", help="The runner config to use")
    parser.add_argument("benchmarks", help="The benchmarks to run")
    parser.add_argument("flags", help="Configuration flags")
    parser.add_argument("--force", action="store_true", help="Force a re-run")
    parser.add_argument(
        "--pgo",
        action="store_true",
        help="Build with profiling guided optimization",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Collect Linux perf profiling data (Linux only)",
    )
    parser.add_argument(
        "--pystats",
        action="store_true",
        help="Enable Pystats (Linux only)",
    )
    parser.add_argument(
        "--32bit",
        action="store_true",
        dest="force_32bit",
        help="Do a 32-bit build (Windows only)",
    )
    parser.add_argument(
        "--_fast", action="store_true", help="Use fast mode, for testing"
    )
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.machine,
        args.benchmarks,
        mflags.parse_flags(args.flags),
        args.force,
        args.pgo,
        args.perf,
        args.pystats,
        args.force_32bit,
        args._fast,
    )
