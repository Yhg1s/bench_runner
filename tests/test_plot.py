import numpy as np
import pytest


from bench_runner import plot


# ---------------------------------------------------------------------------
# plot.plot_diff_pair: the violin data collection
# ---------------------------------------------------------------------------


def test_plot_diff_pair_empty_data():
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()
    try:
        assert list(plot.plot_diff_pair(ax, [])) == []
    finally:
        plt.close("all")


def test_plot_diff_pair_concatenates_all_samples():
    from matplotlib import pyplot as plt

    data = [
        ("a", np.array([0.9, 1.0, 1.1]), 1.0),
        ("b", np.array([1.2, 1.3]), 1.25),
    ]
    _, ax = plt.subplots()
    try:
        all_data = plot.plot_diff_pair(ax, data)
    finally:
        plt.close("all")

    # One combined array holding every sample, not a boxed Python list.
    assert isinstance(all_data, np.ndarray)
    assert len(all_data) == 5
    assert sorted(all_data) == pytest.approx([0.9, 1.0, 1.1, 1.2, 1.3])


def test_plot_diff_pair_handles_insignificant_benchmarks():
    from matplotlib import pyplot as plt

    # A None distribution means "insignificant"; it contributes a single 1.0
    # so the benchmark still gets a row.
    data = [
        ("a", np.array([0.9, 1.1]), 1.0),
        ("insignificant", None, 1.0),
    ]
    _, ax = plt.subplots()
    try:
        all_data = plot.plot_diff_pair(ax, data)
    finally:
        plt.close("all")

    assert isinstance(all_data, np.ndarray)
    assert len(all_data) == 3
    assert 1.0 in all_data
