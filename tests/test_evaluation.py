import numpy as np
import pandas as pd
import pytest
import yaml

from src import evaluation as E


def test_configured_development_and_holdout_target_dates():
    cfg = E.load_config()
    splits = cfg["splits"]
    dates = pd.date_range("2004-12-30", "2015-01-03", freq="D")
    dev = E.target_positions(dates, splits["dev_target_start"], splits["dev_target_end"])
    holdout = E.target_positions(
        dates, splits["test_target_start"], splits["test_target_end"] or dates[-1]
    )
    np.testing.assert_array_equal(dates[dev[[0, -1]]].strftime("%Y-%m-%d"),
                                  ["2005-01-01", "2014-12-31"])
    np.testing.assert_array_equal(dates[holdout[[0, -1]]].strftime("%Y-%m-%d"),
                                  ["2015-01-01", "2015-01-03"])


def test_rolling_forecasts_use_window_and_next_day_target():
    values = np.arange(60, dtype=float).reshape(10, 6)
    origins = np.array([2, 4, 7])
    seen = []

    def predictor(window):
        seen.append(window.copy())
        return window[-1][None, :]

    forecasts, targets = E.rolling_forecasts(values, origins, 3, predictor)
    assert forecasts.shape == (3, 1, 6)
    np.testing.assert_array_equal(targets, values[origins + 1])
    np.testing.assert_array_equal(seen[0], values[0:3])
    np.testing.assert_array_equal(seen[-1], values[5:8])

def test_target_positions_include_endpoints_and_allow_open_end():
    dates = pd.date_range("2004-12-30", periods=5)
    np.testing.assert_array_equal(E.target_positions(dates, dates[1], dates[3]), [1, 2, 3])
    np.testing.assert_array_equal(E.target_positions(dates, dates[3]), [3, 4])
    assert E.target_positions(dates, "2006-01-01").size == 0


def test_losses_preserve_date_and_factor_axes():
    actual = np.tile(np.arange(1, 7), (2, 1))
    errors = E.squared_errors(actual, np.zeros((2, 6)))
    np.testing.assert_array_equal(errors, [[1, 4, 9, 16, 25, 36]] * 2)
    np.testing.assert_array_equal(E.pooled_loss(errors, [1, 4, 9, 16, 25, 36]), [1, 1])
    np.testing.assert_array_equal(E.oos_r2(errors, errors * 4), np.full(6, 0.75))
    assert E.pooled_oos_r2([1, 3], [4, 12]) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "actual, forecast",
    [
        (np.ones((2, 6)), np.ones(6)),
        (np.ones((2, 6)), np.ones((1, 6))),
        (np.ones((2, 6)), np.ones((2, 1, 6))),
        (np.ones((2, 5)), np.ones((2, 5))),
    ],
)
def test_squared_errors_reject_accidental_broadcasting(actual, forecast):
    with pytest.raises(ValueError, match="shape"):
        E.squared_errors(actual, forecast)


@pytest.mark.parametrize("scales", [1, np.ones((1, 6)), np.ones(5), [1, 1, 1, 1, 1, 0]])
def test_pooled_loss_requires_six_positive_scales(scales):
    with pytest.raises(ValueError, match="scales2"):
        E.pooled_loss(np.ones((2, 6)), scales)


def test_scores_reject_misaligned_errors_and_pooled_losses():
    with pytest.raises(ValueError, match="shape"):
        E.pooled_loss(np.ones((2, 1, 6)), np.ones(6))
    with pytest.raises(ValueError, match="shape"):
        E.oos_r2(np.ones((2, 6)), np.ones((1, 6)))
    with pytest.raises(ValueError, match="shape"):
        E.pooled_oos_r2(np.ones(2), np.ones(1))
    with pytest.raises(ValueError, match="shape"):
        E.pooled_oos_r2(np.ones((2, 6)), np.ones((2, 6)))


def test_bootstrap_and_selection_are_deterministic():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    taus = E.tau_grid(cfg)
    bootstrap = cfg["bootstrap"]
    block = cfg["selection"]["block_length"]
    n_boot = bootstrap["n_boot"]
    seed = bootstrap["seed"]
    rng = np.random.default_rng(10)
    losses = rng.random((240, len(taus)))
    first = E.block_bootstrap_indices(240, block, n_boot, np.random.default_rng(seed))
    second = E.block_bootstrap_indices(240, block, n_boot, np.random.default_rng(seed))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (n_boot, 240)
    # Each configured bootstrap row consists of circularly contiguous date blocks;
    # the same row indices can then be applied to every factor/model loss series.
    for configured_block in bootstrap["block_lengths"]:
        sampled = E.block_bootstrap_indices(
            240, configured_block, n_boot, np.random.default_rng(seed)
        )
        for row in sampled:
            for start in range(0, len(row), configured_block):
                chunk = row[start:start + configured_block]
                np.testing.assert_array_equal(
                    np.diff(chunk) % 240, np.ones(len(chunk) - 1)
                )
    np.testing.assert_allclose(taus, np.logspace(-2, 3, 12))
    assert cfg["bootstrap"]["block_lengths"] == [5, 20, 60]
    ci1 = E.mean_ci(losses[:, 0], block, n_boot=n_boot, seed=seed)
    ci2 = E.mean_ci(losses[:, 0], block, n_boot=n_boot, seed=seed)
    assert ci1 == ci2
    selected1 = E.one_se_rule(losses, taus, block, n_boot=n_boot, seed=seed)
    selected2 = E.one_se_rule(losses, taus, block, n_boot=n_boot, seed=seed)
    assert selected1["chosen_idx"] == selected2["chosen_idx"]
    table = selected1["table"]
    best = selected1["best_idx"]
    for k, row in table.iterrows():
        paired_diff = losses[:, k] - losses[:, best]
        assert row["diff"] == pytest.approx(paired_diff.mean())
        paired_boot = paired_diff[first]
        expected_se = paired_boot.mean(axis=1).std(ddof=1)
        assert row["se"] == pytest.approx(expected_se)
        assert bool(row["eligible"]) == (row["diff"] <= row["se"])
    assert selected1["chosen_idx"] == int(table.index[table["eligible"]].max())
    assert selected1["tau"] == pytest.approx(taus[selected1["chosen_idx"]])


def test_detectable_improvement_is_nonnegative():
    d = np.linspace(-0.2, 0.3, 100)
    assert E.min_detectable_improvement(d, 1.0, 5, n_boot=50, seed=3) >= 0


def test_one_se_rule_selects_largest_eligible_tau_and_reports_boundaries():
    losses = np.tile([1.0, 1.0, 2.0], (8, 1))
    selected = E.one_se_rule(losses, [0.1, 1, 10], 3, n_boot=20, seed=4)
    assert selected["best_idx"] == 0
    assert selected["chosen_idx"] == 1
    assert selected["tau"] == 1
    assert selected["best_at_boundary"]
    assert not selected["chosen_at_boundary"]
    np.testing.assert_array_equal(selected["table"]["eligible"], [True, True, False])


def test_constant_paired_differences_have_zero_uncertainty():
    ci = E.mean_ci(np.full(8, 2.0), 3, n_boot=20, seed=4)
    assert ci == {"mean": 2.0, "se": 0.0, "lo": 2.0, "hi": 2.0}
    assert E.min_detectable_improvement(np.full(8, 2.0), 1, 3, 20, 4) == 0
