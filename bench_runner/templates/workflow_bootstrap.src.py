# This script may only use the standard library, since it bootstraps setting up
# the virtual environment to run the full bench_runner.


# NOTE: This file should import in Python 3.9 or later so it can at least print
# the error message that the version of Python is too old.


import os
from pathlib import Path
import shutil
import subprocess
import sys


# Set this to reuse an existing venv instead of building a fresh one each run.
# Off by default: a run that starts from nothing is the one whose results are
# easiest to trust, and that is what a benchmarking machine wants. It is worth
# turning on while iterating locally, where rebuilding the venv (and, unless
# pyperformance is told otherwise with --venvs-dir, every benchmark venv nested
# inside it) dominates the time a short run takes.
REUSE_VENV_ENV_VAR = "BENCH_RUNNER_REUSE_VENV"


def get_reuse_venv() -> bool:
    return os.environ.get(REUSE_VENV_ENV_VAR, "") not in ("", "0", "false")


def venv_python(venv: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def create_venv(venv: Path) -> None:
    if get_reuse_venv() and venv_python(venv).exists():
        # Left exactly as it is, rather than rebuilt in place: the point is to
        # keep whatever is already installed here, which includes any benchmark
        # venvs underneath it. install_requirements() still runs, so
        # bench_runner itself is brought up to date either way.
        print(f"Reusing the existing venv at {venv} ({REUSE_VENV_ENV_VAR} is set)")
        return

    if venv.exists():
        shutil.rmtree(venv)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "venv",
            str(venv),
        ]
    )


def run_in_venv(
    venv: Path, module: str, cmd: list[str], prefix: list[str] = []
) -> None:
    args = [
        *prefix,
        str(venv_python(Path(venv))),
        "-m",
        module,
        *cmd,
    ]

    print("Running command:", " ".join(args))
    subprocess.check_call(args)


def install_requirements(venv: Path) -> None:
    run_in_venv(venv, "pip", ["install", "--upgrade", "pip"])
    run_in_venv(venv, "pip", ["install", "-r", "requirements.txt"])

    # To facilitate coverage testing
    if "--_fast" in sys.argv:
        run_in_venv(venv, "pip", ["install", "pytest-cov"])


def main():
    venv = Path("venv")
    print("::group::Creating venv", file=sys.stderr)
    create_venv(venv)
    print("::endgroup::", file=sys.stderr)
    print("::group::Installing requirements", file=sys.stderr)
    install_requirements(venv)
    print("::endgroup::", file=sys.stderr)

    # Now that we've installed the full bench_runner library,
    # continue on in a new process...

    last_arg = sys.argv.index("workflow_bootstrap.py")
    if last_arg == -1:
        raise ValueError("Couldn't parse command line")

    run_in_venv(venv, "bench_runner", ["workflow", *sys.argv[last_arg + 1 :]])


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 11):
        print(
            "The benchmarking infrastructure requires Python 3.11 or later.",
            file=sys.stderr,
        )
        sys.exit(1)

    main()
