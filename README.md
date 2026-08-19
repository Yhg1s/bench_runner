# Benchmarking tools for the Faster CPython project

## Usage

This code lets you set up your own Github repo to run pyperformance benchmarks on your own self-hosted Github Action runners.

For example, you can see [the Faster CPython team's benchmarking results](https://github.com/faster-cpython/benchmarking-public).

### Set up the repo

Create a new empty repository on Github and clone it locally.

Add bench_runner to your `requirements.txt`.

```text
bench_runner=={VERSION}
```

Replace the {VERSION} above with the latest version tag of `bench_runner`.

Create a virtual environment and install your requirements to it, for example:

```bash session
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Generate workflows

Run the install script to generate the files to make the Github Actions work (from the root of your repo):

```bash session
python -m bench_runner install
```

This will create some files in `.github/workflows` as well as some configuration files at the root of your repo.
Commit them to your repository, and push up to Github.

```bash session
git commit -a -m "Initial commit"
git push origin main
```

The `bench_runner.toml` file created at the root of your repository contains configuration specific to your instance.
More details about this configuration are below.
Every time you make a change to the `bench_runner.toml` file, you will need to rerun `python -m bench_runner install` to have the changes reflected.

### Add some self-hosted runners

Provision the machine to have the build requirements for CPython and the base requirements for Github Actions according to the [provisioning instructions](PROVISIONING.md).

Then, add it to the pool of runners by following the instructions on Github's `Settings -> Actions -> Runners -> Add New Runner` to add a new runner.

The default responses to all questions should be fine _except_ pay careful attention to set the labels correctly.
Each runner must have the following labels:

- One of `linux`, `macos` or `windows`.
- `bare-metal` (to distinguish it from VMs in the cloud).
- `$os-$arch-$nickname`, where:
  - `$os` is one of `linux`, `macos`, `windows`
  - `$arch` is one of `x86_64` or `arm64` (others may be supported in future)
  - `$nickname` is a unique nickname for the runner.

Once the runner is set up, [enable it as a service](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service) so it will start automatically on boot.

In addition, the metadata about the runner(s) must be added to `runners` section in `bench_runner.toml`, for example:

```toml
[runners.linux]
os = "linux"
arch = "x86_64"
hostname = "pyperf"
```

You may also want to add a specific machine to collect pystats.
Since this machine doesn't need to run on bare metal to have accurate timings, it usually is a cloud instance.
Give this machine the special label `cloud` to indicate it is available for collecting pystats.
Additionally, flag it as `available = false` in its configuration so it won't be used to collect timings.

```
[runners.pystats]
os = "linux"
arch = "x86_64"
hostname = "cpython-benchmarking-azure"
available = false
```

If you don't want a machine to be included when the user selects "machine == 'all'", add:

```
include_in_all = false
```

You may limit the number of cores used to build Python with the `use_cores` option. This may be necessary, for example, on cloud VMs with limited RAM.

```
use_cores = 2
```

### Try a benchmarking run

There are instructions for running a benchmarking action already in the `README.md` of your repo. Look there and give it a try!

### Additional configuration

#### Set of benchmarks

By default, all of the benchmarks in `pyperformance` and `python-macrobenchmarks` are run. To configure the set of benchmarks, or add more, edit the `benchmarks.manifest` file.
The format of this file is documented with `pyperformance`.

You can also exclude specific benchmarks by adding them to the `benchmarks/excluded_benchmarks` value in your `bench_runner.toml` file.

#### Reference versions

All benchmarked commits are automatically compared to key "reference" versions, as well as their merge base, if available.
The reference versions are defined in the `bases/versions` value in your `bench_runner.toml` file.
Don't forget to actually collect benchmark data for those tags -- it's doesn't happen automatically.

By default, pyperformance will determine the number of times to run each benchmark dynamically at runtime, by choosing a number at which the timing
measurement becomes stable.
However, this can make comparing benchmark runs less accurate.
It is recommended to specify one of your base benchmarking runs as the source of a hardcoded number of loops.
To do so, generate a *loops table* from that run and save it as `loops.json` in the root of your repository:

```sh
python -m bench_runner synthesize_loops_file -o loops.json \
    results/bm-20231002-3.12.0-0fb18b0/bm-20231002-linux-x86_64-python-v3.12.0-3.12.0-0fb18b0.json
```

The table is keyed by the name each benchmark function reports.
Add `-u` to merge further runs into an existing table, or `-f` to replace it.
Generate one table per machine; `synthesize_loops_file` refuses to merge results from different machines.

> [!IMPORTANT]
> **If you already have a `loops.json`, regenerate it.** Earlier versions of `bench_runner` passed this file to `pyperformance --same-loops`, which took a benchmark *results* file, and the documented setup was a symlink to one:
>
> ```sh
> # No longer works -- a results file is not a loops table.
> ln -s results/bm-.../bm-....json loops.json
> ```
>
> `--same-loops` has been removed in favour of pyperf's `--loops-table`, which reads a different format. An old `loops.json` will be rejected; run the `synthesize_loops_file` command above on the same results file to convert it.

> [!NOTE]
> This requires a `pyperformance` new enough to accept `--loops-table`, and a `pyperf` new enough to implement it. Both are newer than the versions currently pinned in `benchmark_definitions.py` and `pyproject.toml`, so this feature does not work until those pins are bumped — see [Loops table support](#loops-table-support).

##### Loops table support

The loops table spans three repositories, and they have to be updated in order:

1. **pyperf** implements `--loops-table` and the table format. The newest release, 2.10.0, does not have it; `pyproject.toml` pins `pyperf==2.9.0`.
2. **pyperformance** forwards `--loops-table` through to pyperf. Upstream does not have it either, and `benchmark_definitions.py` pins a `pyperformance` commit that predates it.
3. **bench_runner** passes the flag when `PYPERFORMANCE_LOOPS_FILE` is set, which is what this section describes.

Until (1) and (2) are released and both pins here are bumped, setting `PYPERFORMANCE_LOOPS_FILE` makes `pyperformance` exit with an unrecognised-argument error before any benchmark runs.
`bench_runner` checks the file itself and fails early with a clear message, but it cannot detect an old `pyperformance`, so leave the variable unset until the pins move.

#### Plot configuration

`bench_runner` will produce longitudinal plots comparing versions in a series to a specific base version, as well as showing the effect of various flags on the same commits over time.

##### Runner plot styles

For each runner in your `bench_runner.toml`, you can specify a `plot` table with the following keys to control how that runner is rendered in the longitudinal plots:

- `name`: A human-friendly name to display in the plot legend
- `style`: A [matplotlib line style](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle)
- `marker`: A [matplotlib marker](https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers)
- `color`: A [matplotlib color](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def)

##### Longitudinal plot configuration

The longitudinal plot shows the change of a version branch over time against a specified base version. It is made up of multiple subplots, each with its own head and base, and optionally configuration flags.

In `bench_runner.toml`, the `longitudinal_plot` table has a `subplots` key which is an array of tables with the following keys:

- `base`: The base version to compare to. Should be a fully-specified version, e.g. "3.13.0".
- `version`: The version series to use as a head. Should be a 2-part version, e.g. "3.14"
- `flags`: (optional) A list of flags to match to for the head versions
- `runners`: (optional) A list of nicknames of runners to plot. Defaults to all runners.

For example:

```toml
[longitudinal_plot]
subplots = [
    { base = "3.10.4", version = "3.11" },
    { base = "3.12.0", version = "3.13" },
    { base = "3.13.0", version = "3.14", runners = ["linux1", "linux2"] },
    { base = "3.13.0", version = "3.14", runners = ["windows1", "macos1"] },
    { base = "3.13.0", version = "3.14", flags = ["JIT"] }
]
```

##### Flag effect plot configuration

The flag effect plot shows the effect of specified configuration flags against a base with the same commit hash, but different configuration flags.

In `bench_runner.toml`, the `flag_effect_plot` table has a `subplots` key which is an array of tables with the following keys:

- `name`: The description of the flags to use in the title.
- `version`: The version series to compare. Should be a 2-part version, e.g. "3.14"
- `head_flags`: A list of flags to use as the head.
- `base_flags`: (optional) A list of flags to use as the base. By default, this is a default build, i.e. no flags.
- `runners`: (optional) A list of nicknames of runners to plot. Defaults to all runners.
- `runner_map`: (optional) If you need to map a runner to a base in a
  different runner, you can provide that mapping here. For example, with
  tail-calling, you may want to compare runners configured to use clang
  against runners configured with the "default compiler" for a given
  platform. The mapping is from the "head" runner nickname to the "base"
  runner nickname. If `runner_map` is not empty, only the "head" runners in
  the map are plotted.

For example:

```toml
[[flag_effect_plot.subplots]]
name = "JIT"
version = "3.14"
head_flags = ["JIT"]

[[flag_effect_plot.subplots]]
name = "Tail calling interpreter"
version = "3.14"
head_flags = ["TAILCALL"]
runners = ["linux_clang"]
runner_map = { linux_clang = "linux" }
```

##### Benchmark longitudinal plot configuration

The benchmark longitudinal plot shows the change over time, per benchmark. The configuration consists of the following keys:

- `base`: The base version
- `version`: The version to track
- `runners`: The runners to show
- `head_flags`: (optional) The flags to use for the head commits
- `base_flags`: (optional) The flags to use for the base commits

#### Defining groups

Runners can be organised in groups, and a group name can appear anywhere
more than one runner nickname can appear: in the "runners" list in "weekly"
runs, the "runners" list in longitudinal plots, as keys in the "runner_map"
mapping in flag effect plots, and in other group definitions. They can also
be selected from the workflow dropdown, and it schedules runs on all the
runners in the group.

```toml
[groups.linux_gcc]
runners = ["linux1_gcc11", "linux2-gcc12"]

[groups.linux_clang]
runners = ["linux1_clang14", "linux1_clang19"]

[groups.all_clang]
runners = ["linux_clang", "darwin_xcode"]

[weekly.tailcall]
flags = ["TAILCALL"]
runners = ["all_clang"]
```

#### Viewing the interactive charts

Alongside the `.svg` plots, `bench_runner` writes an interactive version of each chart as a `.html` file: the top-level `longitudinal.html`, `configs.html`, `benchmarks.html`, `memory_long.html` and `memory_configs.html`, plus a `-vs-{base}.html` next to every comparison. These can be zoomed and panned, individual runners and benchmarks can be toggled from the legend, and the longitudinal charts can overlay smoothed lines and a rolling average band.

**These do not work on github.com without one extra step.** GitHub serves `.html` files out of a repository as source code, so following a link to one shows you a page of markup rather than a chart. To make them viewable, serve the repository as a website and tell `bench_runner` where that is.

The recommended way is GitHub Pages, which serves the committed files directly and needs no extra workflow:

1. In your results repository, go to `Settings -> Pages`.
2. Under "Build and deployment", set "Source" to **Deploy from a branch**, and pick the `main` branch and the `/ (root)` folder.
3. Wait for the first deployment, then note the URL GitHub shows you. It will look like `https://myorg.github.io/my-benchmarking-repo/`.
4. Put that URL in your `bench_runner.toml`:

   ```toml
   [interactive_plots]
   base_url = "https://myorg.github.io/my-benchmarking-repo/"
   ```

5. Re-run the `_generate` workflow (or `python -m bench_runner generate_results --force`) so the indices are rewritten with links to the served copies.

A few things to know:

- `python -m bench_runner install` creates an empty `.nojekyll` file at the root of your repository. Commit it. Without it, Pages runs Jekyll over every file in the repository, which is slow on a large results repository and skips paths beginning with an underscore.
- If you publish to a public mirror (see `publish_mirror`), enable Pages on the **public** repository and use its URL. GitHub Pages on a private repository requires a paid plan.
- The charts load plotly.js from a CDN (with a Subresource Integrity hash), so they need network access to display, and they will not render in an offline clone. That default is `PLOTLYJS_MODE` in `bench_runner/interactive_plot.py`: set it to `True` for self-contained files that work offline, or to `"directory"` to share one `plotly.min.js` per directory. Note that `True` inlines roughly 3MB into every generated `.html`, which adds up quickly in a repository that keeps one chart per result per base.

If you don't want to enable Pages, leave `base_url` empty. The links stay relative, which means they work when you open the markdown from a local clone, but not on github.com. Third-party viewers such as `htmlpreview.github.io` or `raw.githack.com` can render a single file from a public repository if you paste its URL in, and either can be used as `base_url`, but both are rate-limited free services with no availability guarantee, so neither is a good default for a repository you expect people to browse.

#### Purging old data

With a local checkout of your results repository you can perform some maintenance tasks.

Clone your results repository, and then install the correct version of bench_runner into a virtual environment to use the bench_runner command:

```
git clone {YOUR_RESULTS_REPO}
cd {YOUR_RESULTS_REPO}
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Periodically, you will want to run the `purge` command to remove old results that aren't of a tagged Python release.

```
python -m bench_runner purge
```

To see more options that control what is deleted, run `python -m bench_runner purge --help`.

After purging the results, you will usually want to squash the git history down to a single commit to save space in your repository. **NOTE THAT THIS IS A DESTRUCTIVE OPERATION THAT WILL DELETE OLD DATA.**

```
git checkout --orphan new-main main
git commit -m "Purging old results on ..."

# Overwrite the old master branch reference with the new one
git branch -M new-main main
git push -f origin main
```

### Running

## Developer

To learn how to hack on this project, see the full [developer documentation](DEVELOPER.md).
