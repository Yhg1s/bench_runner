"""
Inspect and maintain the cross-run package cache.

Named `cache` because that is the subcommand; the module it drives is
`bench_runner.cache`, imported below as `mcache`.
"""

from __future__ import annotations


import argparse
import sys


import rich
import rich_argparse


from bench_runner import cache as mcache


# Plain print, not rich.print, for anything containing a path or a partition
# name: rich would rewrap the aligned report, and would swallow a `[...]` in a
# cache directory someone configured as console markup.


def show_info(settings: mcache.CacheSettings) -> None:
    print(mcache.format_info(mcache.info(settings.root)))
    if not settings.enabled:
        print()
        rich.print(
            "[yellow]Caching is currently disabled[/yellow], so nothing new is "
            "being written here."
        )


def do_prune(settings: mcache.CacheSettings, max_age_days: float | None) -> None:
    if max_age_days is None:
        max_age_days = settings.max_age_days

    result = mcache.prune(max_age_days, settings.root)

    print(
        f"Removed {result.files_removed} file(s) unused for "
        f"{max_age_days:g} days, freeing "
        f"{mcache.format_size(result.bytes_removed)}."
    )
    for name in result.partitions_removed:
        print(f"  Removed empty partition {name}")


def do_clear(settings: mcache.CacheSettings) -> None:
    existing = mcache.info(settings.root)
    if not mcache.clear(settings.root):
        print(f"Nothing to remove: {settings.root} does not exist.")
        return
    print(
        f"Removed {settings.root}, freeing "
        f"{mcache.format_size(existing.total_size)}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="""
        Inspect and maintain the cross-run package cache: the downloads and
        built wheels that bench_runner and pyperformance reuse between runs
        instead of fetching again.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--info",
        action="store_true",
        help="Report where the cache is and what is in it. The default.",
    )
    action.add_argument(
        "--prune",
        action="store_true",
        help="Remove entries that have not been used recently, and any cache "
        "partition left with no wheels at all.",
    )
    action.add_argument(
        "--clear",
        action="store_true",
        help="Remove the whole cache. The next run starts cold.",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        # No square brackets in help text: rich_argparse reads them as console
        # markup, so a literal "[cache].max_age_days" prints as ".max_age_days".
        help="How long an unused entry survives --prune. Defaults to the "
        "max_age_days setting in the cache section of bench_runner.toml.",
    )

    args = parser.parse_args()

    if args.max_age_days is not None and not args.prune:
        parser.error("--max-age-days only means something with --prune")

    # Reads [cache] and applies the environment, so this command talks about
    # the same cache a run would use, including any BENCH_RUNNER_CACHE_DIR.
    settings = mcache.get_settings()

    if args.prune:
        do_prune(settings, args.max_age_days)
    elif args.clear:
        do_clear(settings)
    else:
        show_info(settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
