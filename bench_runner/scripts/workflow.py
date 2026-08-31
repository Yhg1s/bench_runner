from __future__ import annotations


import argparse
import contextlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


import rich_argparse


from bench_runner import benchmark_definitions
from bench_runner import cache as mcache
from bench_runner import flags as mflags
from bench_runner import git
from bench_runner import local_deps as mlocal_deps
from bench_runner.result import has_result
from bench_runner import runners
from bench_runner import util
from bench_runner.util import log_group, PathLike


from bench_runner.scripts import run_benchmarks as mrun_benchmarks


VENV_ENV_VAR = "BENCH_RUNNER_VENV"
ALLOW_LOCAL_DEPS_ENV_VAR = "BENCH_RUNNER_ALLOW_LOCAL_DEPS"


def in_github_actions() -> bool:
    """
    Whether this is a CI run.

    Any non-empty value counts, not just the "true" GitHub sets. This gates a
    safety check, so the failure to be avoided is deciding we are not in CI
    when we are.
    """
    return bool(os.environ.get("GITHUB_ACTIONS", "").strip())


def get_venv_dir() -> Path:
    """
    The outer venv: the one bench_runner, pyperformance and pyperf live in.

    In CI the bootstrap creates ./venv and runs this workflow inside it, so the
    hardcoded relative path is right and is kept exactly. Locally there is no
    reason the outer venv should have to sit at <results-repo>/venv, so the
    default is the venv this process is already running in.
    """
    if override := os.environ.get(VENV_ENV_VAR, "").strip():
        return Path(override).expanduser()

    # sys.prefix is only the venv when there IS one. Without this check, a
    # bench_runner installed into a system Python would have the workflow try
    # to pip install into system site-packages.
    if not in_github_actions() and sys.prefix != sys.base_prefix:
        return Path(sys.prefix)

    return Path("venv")


def check_local_deps_allowed(
    local: dict[str, mlocal_deps.LocalDep], exempt: bool = False
) -> None:
    """
    Refuse to run a CI benchmark against unpushed code.

    Results built from a local checkout are not reproducible by anyone else,
    and CI is what publishes the results corpus. A-3 makes such a result
    impossible to confuse with a pushed one, and this makes it impossible to
    produce by accident in the first place.

    `exempt` is for local_workflow, which is the deliberate local-development
    entry point and stays usable even when run on a runner.
    """
    if exempt or not local or not in_github_actions():
        return
    if os.environ.get(ALLOW_LOCAL_DEPS_ENV_VAR, "").strip() not in ("", "0"):
        print(
            f"*** {ALLOW_LOCAL_DEPS_ENV_VAR} is set: benchmarking unpushed code "
            "in CI."
        )
        return

    named = ", ".join(f"{name} ({dep.path})" for name, dep in sorted(local.items()))
    raise RuntimeError(
        f"Local dependencies are configured ({named}), but this is a CI run. "
        "Results built from an unpushed checkout cannot be reproduced from the "
        "results repository, so they must not be produced here by accident. "
        "Remove the [dev.local_deps] configuration, or set "
        f"{ALLOW_LOCAL_DEPS_ENV_VAR}=1 if this is deliberate."
    )


def get_windows_build_dir(force_32bit: bool) -> Path:
    if force_32bit:
        return Path("PCbuild") / "win32"
    return Path("PCbuild") / "amd64"


def get_exe_path(cpython: Path, flags: list[str], force_32bit: bool) -> Path:
    match util.get_simple_platform():
        case "linux":
            return cpython / "python"
        case "macos":
            return cpython / "python.exe"
        case "windows":
            build_dir = cpython / get_windows_build_dir(force_32bit)
            if "NOGIL" in flags:
                exe = next(build_dir.glob("python3.*.exe"))
            else:
                exe = build_dir / "python.exe"
            return exe


def venv_python(venv: PathLike) -> Path:
    """The interpreter inside a venv."""
    venv = Path(venv)

    if util.get_simple_platform() == "windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run_in_venv(
    venv: PathLike,
    module: str,
    cmd: list[str],
    sudo: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    args = [
        str(venv_python(venv)),
        "-m",
        module,
        *cmd,
    ]

    if sudo:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        args = ["sudo", f"LD_LIBRARY_PATH={ld_library_path}"] + args

    print("Running command:", " ".join(args))
    # `env` adds to the environment rather than replacing it: pip needs the
    # rest of it (HOME, proxies, certificates) to work at all.
    subprocess.check_call(args, env={**os.environ, **env} if env else None)


def should_run(
    force: bool,
    fork: str,
    ref: str,
    nickname: str,
    pystats: bool,
    flags: list[str],
    cpython: Path = Path("cpython"),
    results_dir: Path = Path("results"),
) -> bool:
    try:
        commit_hash = git.get_git_hash(cpython)
    except subprocess.CalledProcessError:
        # This will fail if the cpython checkout failed for some reason. Print
        # a nice error message since the one the checkout itself gives is
        # totally inscrutable.
        print("The checkout of cpython failed.", file=sys.stderr)
        print(f"You specified fork {fork!r} and ref {ref!r}.", file=sys.stderr)
        print("Are you sure you entered the fork and ref correctly?", file=sys.stderr)
        # Fail the rest of the workflow
        sys.exit(1)

    found_result = has_result(
        results_dir,
        commit_hash,
        nickname,
        pystats,
        flags,
        benchmark_definitions.get_benchmark_hash(),
        progress=False,
        pattern=f"*{nickname}*{commit_hash[:7]}*",
    )

    if force:
        if found_result is not None:
            for filepath in found_result.filename.parent.iterdir():
                if filepath.suffix != ".json":
                    git.remove(results_dir.parent, filepath)
        should_run = True
    else:
        should_run = (nickname in ("__really_all", "all")) or found_result is None

    return should_run


def checkout_cpython(fork: str, ref: str, cpython: PathLike = Path("cpython")):
    git.clone(cpython, f"https://github.com/{fork}/cpython.git", branch=ref, depth=50)


def checkout_benchmarks(local: dict[str, mlocal_deps.LocalDep] | None = None):
    """
    Fetch each benchmark repository at its pinned hash, except any that a
    local checkout stands in for.

    Skipping matters because git.clone() hard-resets an existing directory to
    the pin: for a repository someone is editing, "checking out" means
    destroying the very changes the run is meant to measure.
    """
    local = local or {}

    for repo in benchmark_definitions.BENCHMARK_REPOS:
        if (dep := local.get(repo.dirname)) is not None:
            # Loudly, because results built from an unpushed tree are not
            # comparable to results built from a pushed one.
            print(f"*** USING LOCAL {repo.dirname}: {dep.path}")
            print(f"***   not fetching {repo.url} at {repo.hash[:10]}")
            continue
        git.clone(
            Path(repo.dirname),
            repo.url,
            branch=repo.hash,
            depth=1,
        )


def compile_unix(
    cpython: PathLike,
    flags: list[str],
    pgo: bool,
    pystats: bool,
    reconfigure: bool = True,
) -> None:
    cpython = Path(cpython)
    runner = runners.get_current_runner()

    env = os.environ.copy()

    if util.get_simple_platform() == "macos":
        openssl_prefix = util.get_brew_prefix("openssl@1.1")
        env["PKG_CONFIG_PATH"] = f"{openssl_prefix}/lib/pkgconfig"

    args = []
    if pystats:
        args.append("--enable-pystats")
    if pgo:
        args.extend(["--enable-optimizations", "--with-lto=full"])
    if "PYTHON_UOPS" in flags:
        assert "JIT" not in flags
        args.append("--enable-experimental-jit=interpreter")
    if "JIT" in flags:
        assert "PYTHON_UOPS" not in flags
        args.append("--enable-experimental-jit=yes")
    if "NOGIL" in flags:
        args.append("--disable-gil")
    if "TAILCALL" in flags:
        args.append("--with-tail-call-interp")
    args.append("--enable-option-checking=fatal")
    if configure_flags := os.environ.get("PYTHON_CONFIGURE_FLAGS"):
        args.extend(shlex.split(configure_flags))

    make_args = []
    if (cores := runner.use_cores) is not None:
        make_args.extend(["-j", str(cores)])
    else:
        make_args.extend(["-j"])

    with contextlib.chdir(cpython):
        if reconfigure:
            subprocess.check_call(["./configure", *args], env=env)
            # Configuring again leaves the old PGO results in place, which the
            # new build must not pick up. Only on a reconfigure: the callers
            # that skip it are after an incremental build.
            subprocess.check_call(["make", *make_args, "clean"], env=env)
        subprocess.check_call(["make", *make_args], env=env)


def compile_windows(
    cpython: PathLike, flags: list[str], pgo: bool, force_32bit: bool
) -> None:
    cpython = Path(cpython)

    args = ["--%"]  # This is the PowerShell "stop parsing" flag
    if force_32bit:
        args.extend(["-p", "win32"])
    if pgo:
        args.append("--pgo")
    else:
        args.extend(["-c", "Release"])
    if "JIT" in flags:
        args.append("--experimental-jit")
    if "PYTHON_UOPS" in flags:
        args.append("--experimental-jit-interpreter")
    if "NOGIL" in flags:
        args.append("--disable-gil")
    if "TAILCALL" in flags:
        args.append("--tail-call-interp")
    if configure_flags := os.environ.get("PYTHON_CONFIGURE_FLAGS"):
        args.append(configure_flags)

    with contextlib.chdir(cpython):
        subprocess.check_call(
            [
                "powershell.exe",
                Path("PCbuild") / "build.bat",
                *args,
            ],
        )
        shutil.copytree(get_windows_build_dir(force_32bit), "libs", dirs_exist_ok=True)


def clear_pip_cache(venv: PathLike) -> None:
    """
    Throw away pip's cache for the whole machine.

    This is the pre-cache behaviour, kept for the two cases that still want it:
    `[cache].enabled = false` and an explicit `--purge-pip-cache`. It used to
    run on every workflow, which is why nothing was ever cached between runs --
    pip's cache is per-user, not per-venv, so one benchmark run cleared the
    downloads and built wheels of everything else on the machine.

    Deliberately run without the cache environment. `pip cache purge` empties
    whatever PIP_CACHE_DIR points at, and since our partitions share one
    download directory by symlink, pointing this at a partition would throw
    away every ABI's downloads, not just that partition's wheels. Use
    `bench_runner cache --clear` to clear the bench_runner cache.
    """
    run_in_venv(venv, "pip", ["cache", "purge"])


def install_pyperformance(
    venv: PathLike,
    env: dict[str, str] | None = None,
    local: mlocal_deps.LocalDep | None = None,
) -> None:
    """
    Install pyperformance into the outer venv, from ./pyperformance unless a
    local checkout stands in for it.

    ./pyperformance is where checkout_benchmarks() puts the pinned clone, and
    that directory does not exist at all when the repository is overridden --
    nothing else in the run needs it, which is what makes skipping the clone
    safe.
    """
    args = ["./pyperformance"] if local is None else local.pip_args()
    if local is not None:
        print(f"*** Installing pyperformance from {local.path}")
    run_in_venv(venv, "pip", ["install", *args], env=env)


def install_local_outer_deps(
    venv: PathLike,
    local: dict[str, mlocal_deps.LocalDep],
    env: dict[str, str] | None = None,
) -> list[str]:
    """
    Install the local dependencies that are not benchmark repositories -- in
    practice pyperf, which otherwise reaches the outer venv only as a pinned
    transitive dependency.

    Must run after install_pyperformance(): pyperformance's own metadata
    requires `pyperf@git+...`, and pip reinstalls a PEP 508 direct reference
    even when something is already there, so installing a local pyperf first
    would be silently undone.

    pyperformance is excluded because install_pyperformance() handles it, and
    pyston-benchmarks because it is read in place and never installed.
    """
    installed = []
    for name, dep in sorted(local.items()):
        if dep.is_benchmark_repo:
            continue
        if not dep.is_installable:
            raise ValueError(
                f"Local dependency {name!r} points at {dep.path}, which pip "
                "cannot install: it has no pyproject.toml, setup.py or "
                "setup.cfg."
            )
        print(f"*** Installing {name} from {dep.path}")
        run_in_venv(venv, "pip", ["install", *dep.pip_args()], env=env)
        installed.append(name)
    return installed


def setup_pip_cache(venv: PathLike, purge: bool) -> dict[str, str]:
    """
    Point the outer venv's pip at the cross-run package cache, and return the
    environment that does it.

    Empty when the cache is turned off, in which case pip falls back to its own
    default cache and behaves exactly as it did before any of this existed.
    """
    settings = mcache.get_settings()

    if not settings.enabled:
        print("Package caching is disabled.")
    if purge or not settings.enabled:
        with log_group("Purging the pip cache"):
            clear_pip_cache(venv)
    if not settings.enabled:
        return {}

    env = settings.pip_env(venv_python(venv), "host")
    print(
        f"Caching packages in {settings.root} "
        f"(abi_scope={settings.abi_scope}, offline={settings.offline})"
    )
    print(f"Outer venv pip cache: {env['PIP_CACHE_DIR']}")
    return env


def export_benchmark_pip_cache(python_exe: PathLike) -> dict[str, str]:
    """
    Point the benchmark venvs' pip at the cache, through the environment.

    The outer venv gets its cache handed over as an argument, but these venvs
    are built by pyperformance, several layers down, so there is nothing to
    hand it to: the variables have to be in the environment for pyperformance
    to inherit. They still only reach pip because run_benchmarks names them in
    --inherit-environ -- pyperformance strips everything else.

    A separate partition from the outer venv's, keyed on the CPython that was
    just built rather than on the system Python that runs the workflow. That is
    the whole point of the ABI key: these are usually different interpreters,
    and between two commits of main they can be differently built ones.
    """
    settings = mcache.get_settings()
    env = settings.pip_env(python_exe, "bm")
    if env:
        os.environ.update(env)
        print(f"Benchmark venv pip cache: {env['PIP_CACHE_DIR']}")
    return env


def collect_pure_wheels(benchmark_pip_env: dict[str, str]) -> list[Path]:
    """
    Hardlink the pure-Python wheels this run built into the shared wheelhouse,
    where every other ABI's pip will find them.

    Only the partition this run actually used is swept, and only its `wheels/`
    directory, which is what keeps a locally-installed dependency out: pip
    never persists a wheel built from an editable install or a plain path, so
    one cannot be there to sweep, and a later edit can never be masked by a
    stale copy.

    The wheelhouse is already on PIP_FIND_LINKS for both venvs -- pip_env() put
    it there -- so anything collected here is picked up by the next run without
    further wiring.
    """
    settings = mcache.get_settings()
    if not settings.enabled or not settings.share_pure_wheels:
        return []

    partition = benchmark_pip_env.get("PIP_CACHE_DIR")
    if partition is None:
        return []

    added = mcache.sweep_pure_wheels(settings.root, partitions=[Path(partition).name])
    if added:
        print(
            f"Shared {len(added)} pure wheel(s) into "
            f"{mcache.pure_wheelhouse_dir(settings.root)}:"
        )
        for wheel in added:
            print(f"  {wheel.name}")
    return added


def tune_system(venv: PathLike, perf: bool) -> None:
    # System tuning is Linux only
    if util.get_simple_platform() != "linux":
        return

    args = ["system", perf and "reset" or "tune"]
    if cpu_affinity := os.environ.get("CPU_AFFINITY"):
        args.append(f"--affinity={cpu_affinity}")

    try:
        run_in_venv(venv, "pyperf", args, sudo=True)
    except subprocess.CalledProcessError:
        # pyperf can fail to set IRQs, which is not a problem.
        pass

    if perf:
        subprocess.check_call(
            [
                "sudo",
                "bash",
                "-c",
                "echo 100000 > /proc/sys/kernel/perf_event_max_sample_rate",
            ]
        )


def reset_system(venv: PathLike) -> None:
    # System tuning is Linux only
    if util.get_simple_platform() != "linux":
        return

    try:
        run_in_venv(
            venv,
            "pyperf",
            ["system", "reset"],
            sudo=True,
        )
    except subprocess.CalledProcessError:
        # pyperf can fail to reset IRQs, which is not a problem.
        pass


def nickname_for_machine(machine: str) -> str:
    """
    The runner nickname within a machine identifier.

    Machine identifiers are os-arch-nickname, and the nickname is whatever the
    runner is called in bench_runner.toml, so it may itself contain dashes.
    Only the two known leading fields are split off.
    """
    if machine in ("all", "__really_all"):
        return machine
    _, _, nickname = machine.split("-", 2)
    return nickname


def _main(
    fork: str,
    ref: str,
    machine: str,
    benchmarks: str,
    flags: list[str],
    force: bool,
    pgo: bool,
    perf: bool,
    pystats: bool,
    force_32bit: bool,
    run_id: str | None = None,
    fast: bool = False,
    generate_loops: bool = False,
    purge_pip_cache: bool = False,
    allow_local_deps: bool = False,
):
    nickname = nickname_for_machine(machine)

    venv = get_venv_dir()
    cpython = Path("cpython")
    platform = util.get_simple_platform()

    # Resolved once, up front, so a misconfigured path fails before an hour of
    # compiling rather than after it.
    local = mlocal_deps.get_local_deps()
    check_local_deps_allowed(local, allow_local_deps)
    if local:
        print(f"Using local checkouts for: {', '.join(sorted(local))}")
    print(f"Outer venv: {venv}")

    if force_32bit and platform != "windows":
        raise RuntimeError("32-bit builds are only supported on Windows")
    if perf and platform != "linux":
        raise RuntimeError("perf profiling is only supported on Linux")
    if pystats and platform != "linux":
        raise RuntimeError("Pystats is only supported on Linux")

    with log_group("Checking out CPython"):
        checkout_cpython(fork, ref, cpython)

    with log_group("Determining if we need to run benchmarks"):
        if not fast and not should_run(
            force, fork, ref, nickname, False, flags, cpython=cpython
        ):
            print("No need to run benchmarks.  Skipping...")
            return

    with log_group("Checking out benchmarks"):
        checkout_benchmarks(local)

    with log_group("Compiling CPython"):
        match platform:
            case "linux" | "macos":
                compile_unix(cpython, flags, pgo, pystats)
            case "windows":
                compile_windows(cpython, flags, pgo, force_32bit)

        # Print out the version of Python we built just so we can confirm it's the
        # right thing in the logs
        subprocess.check_call([get_exe_path(cpython, flags, force_32bit), "-VV"])

    with log_group("Setting up the package cache"):
        pip_env = setup_pip_cache(venv, purge_pip_cache)

    with log_group("Installing pyperformance"):
        install_pyperformance(venv, pip_env, local.get("pyperformance"))
        install_local_outer_deps(venv, local, pip_env)

    # Only once the outer venv has finished installing. These go into the
    # environment rather than being passed as an argument, so doing it earlier
    # would leave the outer venv's pip one accidental edit away from writing
    # into the benchmark interpreter's partition.
    with log_group("Setting up the benchmark package cache"):
        benchmark_pip_env = export_benchmark_pip_cache(
            get_exe_path(cpython, flags, force_32bit)
        )

    if not fast:
        with log_group("Tuning system"):
            tune_system(venv, perf)

    try:
        if Path(".debug").exists():
            shutil.rmtree(".debug")

        pystats_dir = Path("/tmp") / "py_stats"
        if pystats:
            shutil.rmtree(pystats_dir, ignore_errors=True)
            pystats_dir.mkdir(parents=True)

        if perf:
            mode = "perf"
        elif pystats:
            mode = "pystats"
        else:
            mode = "benchmark"

        with log_group("Running benchmarks"):
            mrun_benchmarks._main(
                mode,
                get_exe_path(cpython, flags, force_32bit),
                fork,
                ref,
                benchmarks,
                flags=flags,
                run_id=run_id,
                test_mode=fast,
                individual=pystats,
                generate_loops=generate_loops,
            )
    finally:
        # In the `finally` because a run that failed part way through still
        # built real wheels, and they are worth keeping.
        with log_group("Sharing pure wheels"):
            collect_pure_wheels(benchmark_pip_env)

        if not fast:
            reset_system(venv)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument(
        "machine",
        help="The machine to run the benchmarks on.",
    )
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
        "--generate-loops",
        action="store_true",
        help="Calibrate the benchmarks first and hold the loop counts fixed "
        "for this run, instead of calibrating again inside it",
    )
    parser.add_argument(
        "--purge-pip-cache",
        action="store_true",
        help="Empty pip's cache for the whole machine before installing, the "
        "way every run used to. Not needed to keep builds apart: the package "
        "cache is already partitioned by ABI.",
    )
    parser.add_argument("--run_id", default=None, type=str, help="The github run id")
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
        args.run_id,
        args._fast,
        generate_loops=args.generate_loops,
        purge_pip_cache=args.purge_pip_cache,
    )


if __name__ == "__main__":
    main()
