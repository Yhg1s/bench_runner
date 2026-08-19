import json
from pathlib import Path
import platform
import shutil
import socket
import sys


import pytest


from bench_runner import result as mod_result


DATA_PATH = Path(__file__).parent / "data"


def _copy_results(tmp_path):
    results_path = tmp_path / "results"
    shutil.copyfile(DATA_PATH / "bench_runner.toml", tmp_path / "bench_runner.toml")
    shutil.copytree(DATA_PATH / "results", tmp_path / "results")
    return results_path


def test_load_all_results(tmp_path, monkeypatch):
    results_path = _copy_results(tmp_path)
    monkeypatch.chdir(tmp_path)

    results = mod_result.load_all_results(["3.10.4", "3.11.0b3"], results_path)

    assert len(results) == 11

    by_version = {x.version: x for x in results}

    result_310 = by_version["3.10.4"]

    for result in results:
        if result is not result_310:
            comparison = result.bases["3.10.4"]
            assert comparison.ref is result_310
            assert comparison.head is result
            assert comparison.base == "3.10.4"

    assert result_310.commit_datetime == "2022-03-23T20:12:04+00:00"
    assert result_310.commit_date == "2022-03-23"
    assert result_310.commit_merge_base is None
    assert result_310.benchmark_hash == "215d35"


def test_merge_base(tmp_path, monkeypatch):
    monkeypatch.chdir(DATA_PATH)

    results_path = _copy_results(tmp_path)

    # Hack up so one of the results has an explicit commit_merge_base
    result_with_base = (
        results_path
        / "bm-20221119-3.12.0a3+-b0e1f9c"
        / "bm-20221119-linux-x86_64-python-main-3.12.0a3+-b0e1f9c.json"
    )
    with open(result_with_base) as fd:
        contents = json.load(fd)
    contents["metadata"][
        "commit_merge_base"
    ] = "9d38120e335357a3b294277fd5eff0a10e46e043"
    with open(result_with_base, "w") as fd:
        json.dump(contents, fd)
    # End hack

    results = mod_result.load_all_results([], results_path)

    by_hash = {x.cpython_hash: x for x in results}

    head = by_hash["b0e1f9c"]
    comparison = head.bases["base"]

    assert head.commit_merge_base == "9d38120e335357a3b294277fd5eff0a10e46e043"
    assert comparison.ref.version == "3.10.4"
    assert comparison.head is head
    assert comparison.geometric_mean == "1.702x faster"


def test_from_scratch(monkeypatch):
    monkeypatch.chdir(DATA_PATH)

    python = sys.executable

    def get_git_hash(*args):
        return "b7e4f1d97c6e784d2dee182d2b81541ddcff5751"

    monkeypatch.setattr(mod_result.git, "get_git_hash", get_git_hash)

    def get_git_commit_date(*args):
        return "2022-11-19T20:47:09+00:00"

    monkeypatch.setattr(mod_result.git, "get_git_commit_date", get_git_commit_date)

    def gethostname(*args):
        return "pyperf"

    monkeypatch.setattr(socket, "gethostname", gethostname)

    result = mod_result.Result.from_scratch(
        python, "my-fork", "9d38120e335357a3b294277fd5eff0a10e46e043"
    )

    assert result.filename == Path(
        f"results/bm-20221119-{platform.python_version()}-b7e4f1d/"
        f"bm-20221119-{platform.system().lower()}-{platform.machine().lower()}"
        f"-my%2dfork-9d38120e335357a3b294-{platform.python_version()}-b7e4f1d.json"
    )

    assert result.runner == "linux x86_64 (linux)"
    assert result.system == "linux"

    result = mod_result.Result.from_scratch(
        python,
        "my-fork",
        "9d38120e335357a3b294277fd5eff0a10e46e043",
        flags=["PYTHON_UOPS", "BAR"],
    )

    assert result.filename == Path(
        f"results/bm-20221119-{platform.python_version()}-b7e4f1d-BAR,PYTHON_UOPS/"
        f"bm-20221119-{platform.system().lower()}-{platform.machine().lower()}"
        f"-my%2dfork-9d38120e335357a3b294-{platform.python_version()}-b7e4f1d.json"
    )


# ---------------------------------------------------------------------------
# Bounding the ref/head ratio cross product
# ---------------------------------------------------------------------------


def test_quantile_grid_shorter_than_count_is_unchanged():
    import numpy as np

    values = np.arange(10.0)
    np.testing.assert_array_equal(mod_result._quantile_grid(values, 200), values)


def test_quantile_grid_returns_exactly_count():
    import numpy as np

    values = np.arange(10_000.0)
    assert len(mod_result._quantile_grid(values, 200)) == 200


def test_quantile_grid_preserves_min_and_max_exactly():
    import numpy as np

    values = np.sort(np.random.default_rng(0).normal(size=10_000))
    grid = mod_result._quantile_grid(values, 200)
    # The extremes are the whole point of the violin's tails.
    assert grid[0] == values[0]
    assert grid[-1] == values[-1]


def test_quantile_grid_preserves_distribution_shape():
    import numpy as np

    values = np.sort(np.random.default_rng(0).lognormal(0, 0.3, 20_000))
    grid = mod_result._quantile_grid(values, 200)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    assert np.allclose(
        np.quantile(values, quantiles), np.quantile(grid, quantiles), atol=1e-3
    )


def test_analytic_mean_matches_the_full_cross_product():
    import numpy as np

    # calculate_diffs computes mean(a) * mean(1/b) instead of materializing
    # outer(a, 1/b); the two must agree to floating point.
    rng = np.random.default_rng(0)
    for n_ref, n_head in [(120, 120), (400, 400), (37, 401)]:
        ref = rng.lognormal(0, 0.3, n_ref)
        head = rng.lognormal(0, 0.3, n_head)
        brute = np.outer(ref, 1.0 / head).flatten().mean()
        analytic = float(np.mean(ref) * np.mean(1.0 / head))
        assert analytic == pytest.approx(brute, rel=1e-12)


def test_timing_diff_is_bounded_to_violin_points(tmp_path, monkeypatch):
    import numpy as np

    results_path = _copy_results(tmp_path)
    monkeypatch.chdir(tmp_path)

    results = mod_result.load_all_results(None, results_path, sorted=True, match=False)
    by_hash = {}
    for result in results:
        by_hash.setdefault(result.cpython_hash[:7], result)

    comparison = mod_result.BenchmarkComparison(
        by_hash["9d38120"], by_hash["eb0004c"], "", force_valid=True
    )
    for _, values, _ in comparison.get_timing_diff():
        if values is not None:
            assert len(values) <= mod_result.VIOLIN_POINTS
            # Already sorted, which _subsample relies on.
            assert np.all(np.diff(values) >= 0)


def test_cross_product_guard_does_not_move_the_reported_mean(tmp_path, monkeypatch):
    # The guard never trips on pyperformance data (at most ~160k pairs against
    # a 1,000,000 limit), so force it with a tiny limit and confirm that
    # clamping the cross product leaves the reported mean untouched -- that is
    # what makes the subsampling safe.
    results_path = _copy_results(tmp_path)
    monkeypatch.chdir(tmp_path)

    results = mod_result.load_all_results(None, results_path, sorted=True, match=False)
    by_hash = {}
    for result in results:
        by_hash.setdefault(result.cpython_hash[:7], result)
    ref, head = by_hash["9d38120"], by_hash["eb0004c"]

    def means_with_limit(limit):
        monkeypatch.setattr(mod_result, "MAX_CROSS_PRODUCT", limit)
        mod_result.clear_contents_cache()
        comparison = mod_result.BenchmarkComparison(ref, head, "", force_valid=True)
        return {name: mean for name, _, mean in comparison.get_timing_diff()}

    unclamped = means_with_limit(1_000_000)
    clamped = means_with_limit(100)

    assert set(unclamped) == set(clamped)
    for name in unclamped:
        assert clamped[name] == pytest.approx(unclamped[name], rel=1e-12)


def test_cross_product_guard_still_bounds_the_distribution(tmp_path, monkeypatch):
    results_path = _copy_results(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod_result, "MAX_CROSS_PRODUCT", 100)

    results = mod_result.load_all_results(None, results_path, sorted=True, match=False)
    by_hash = {}
    for result in results:
        by_hash.setdefault(result.cpython_hash[:7], result)

    comparison = mod_result.BenchmarkComparison(
        by_hash["9d38120"], by_hash["eb0004c"], "", force_valid=True
    )
    for _, values, _ in comparison.get_timing_diff():
        if values is not None:
            assert len(values) <= mod_result.VIOLIN_POINTS


# ---------------------------------------------------------------------------
# benchmark_hash gating in has_result
# ---------------------------------------------------------------------------


def _a_raw_result(tmp_path, monkeypatch):
    results_path = _copy_results(tmp_path)
    monkeypatch.chdir(tmp_path)
    results = mod_result.load_all_results(None, results_path, sorted=True, match=False)
    raw = [r for r in results if r.result_info[0] == "raw results"]
    assert raw, "no raw results in the test data"
    return results_path, raw[0]


def test_has_result_finds_a_matching_result(tmp_path, monkeypatch):
    results_path, result = _a_raw_result(tmp_path, monkeypatch)
    found = mod_result.has_result(
        results_path,
        result.cpython_hash,
        result.nickname,
        False,
        result.flags,
        result.benchmark_hash,
        progress=False,
    )
    assert found is not None


def test_has_result_rejects_a_different_benchmark_hash(tmp_path, monkeypatch):
    """
    A result produced with a different benchmark corpus measures a different
    workload, so reusing it compares two different things.
    """
    results_path, result = _a_raw_result(tmp_path, monkeypatch)
    found = mod_result.has_result(
        results_path,
        result.cpython_hash,
        result.nickname,
        False,
        result.flags,
        "0000000000",
        progress=False,
    )
    assert found is None


def test_has_result_can_be_told_to_ignore_the_benchmark_hash(tmp_path, monkeypatch):
    # Updating the benchmarks invalidates every result, which is sometimes more
    # regeneration than is wanted -- but it has to be asked for explicitly.
    results_path, result = _a_raw_result(tmp_path, monkeypatch)
    monkeypatch.setenv(mod_result.IGNORE_BENCHMARK_HASH_ENV_VAR, "1")
    found = mod_result.has_result(
        results_path,
        result.cpython_hash,
        result.nickname,
        False,
        result.flags,
        "0000000000",
        progress=False,
    )
    assert found is not None


@pytest.mark.parametrize("value", ["", "0", "false"])
def test_ignore_benchmark_hash_is_off_for_falsey_values(monkeypatch, value):
    monkeypatch.setenv(mod_result.IGNORE_BENCHMARK_HASH_ENV_VAR, value)
    assert mod_result.ignore_benchmark_hash() is False


def test_ignore_benchmark_hash_is_off_by_default(monkeypatch):
    monkeypatch.delenv(mod_result.IGNORE_BENCHMARK_HASH_ENV_VAR, raising=False)
    assert mod_result.ignore_benchmark_hash() is False
