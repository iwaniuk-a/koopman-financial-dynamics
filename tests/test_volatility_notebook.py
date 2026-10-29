"""Focused contracts for the merged volatility notebook and its shared APIs."""

import ast
from pathlib import Path

import marimo as mo
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    load_config,
    mean_ci,
    pooled_loss,
    rolling_forecasts,
    target_positions,
)
from src.models import (
    fit_ridge_var,
    make_ewma_predictor,
    make_har_predictor,
    make_predictor,
    multiscale_features,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/06_volatility_forecasting.py"


def run_notebook_cell(defines, **inputs):
    """Execute the actual notebook cell with controlled inputs, without a server."""
    tree = ast.parse(NOTEBOOK.read_text())
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        if any(
            (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == defines)
            or (isinstance(node, ast.FunctionDef) and node.name == defines)
            for node in ast.walk(cell)
        ):
            body = [node for node in cell.body if not isinstance(node, ast.Return)]
            namespace = {"np": np, "pd": pd, "mo": mo, **inputs}
            # Execute trusted repository code so tests exercise the actual cell.
            exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
            return namespace
    pytest.fail(f"No notebook cell defines {defines}")


@pytest.fixture
def run_forecasts():
    return run_notebook_cell(
        "run_forecasts", make_ewma_predictor=make_ewma_predictor,
        make_har_predictor=make_har_predictor, make_predictor=make_predictor,
        rolling_forecasts=rolling_forecasts,
    )["run_forecasts"]


def test_feature_shapes_and_trailing_observables():
    values = np.tile([1., -1.], 15)[:, None] * np.arange(1., 7.)[None, :]
    symmetric = multiscale_features(values)
    asymmetric = multiscale_features(values, asymmetric=True)
    assert symmetric.shape == (30, 18)
    assert asymmetric.shape == (30, 36)
    np.testing.assert_array_equal(symmetric[-1], np.tile(np.arange(1., 7.) ** 2, 3))
    np.testing.assert_array_equal(asymmetric[:, :18], symmetric)
    np.testing.assert_allclose(
        asymmetric[-1, 18:],
        np.concatenate([np.arange(1., 7.) ** 2 * share for share in (1., .6, .5)]),
    )
    assert np.isnan(symmetric[:21, 12:]).all()
    assert np.isfinite(asymmetric[21:]).all()


def test_target_dates_cross_year_and_weekend_boundaries():
    dates = pd.to_datetime([
        "2004-12-30",
        "2004-12-31",
        "2005-01-03",
        "2005-01-04",
        "2014-12-31",
        "2015-01-02",
        "2015-01-05",
    ])

    cfg = load_config()
    splits = cfg["splits"]

    np.testing.assert_array_equal(
        target_positions(
            dates,
            splits["dev_target_start"],
            splits["dev_target_end"],
        ),
        [2, 3, 4],
    )

    np.testing.assert_array_equal(
        target_positions(
            dates,
            splits["test_target_start"],
            splits["test_target_end"],
        ),
        [5, 6],
    )


@pytest.mark.parametrize("name, settings", [
    ("Historical mean", (None,)),
    ("Squared VAR", (.1, 1.)),
    ("Symmetric HAR", (.1, 1.)),
    ("Asymmetric HAR", (.1, 1.)),
    ("EWMA", (.90, .99)),
])
def test_forecasts_ignore_future_and_align_to_next_day_squares(name, settings, run_forecasts):
    values = np.random.default_rng(72).normal(0, .01, (100, 6))
    origins = np.array([75])
    predictions, targets = run_forecasts(values, origins, {name: settings}, 60)
    changed = values.copy()
    changed[76:] = .8
    after, after_targets = run_forecasts(changed, origins, {name: settings}, 60)
    np.testing.assert_array_equal(predictions[name], after[name])
    assert predictions[name].shape == (1, len(settings), 6)
    np.testing.assert_array_equal(targets, values[[76]] ** 2)
    np.testing.assert_array_equal(after_targets, changed[[76]] ** 2)


def test_historical_mean_includes_origin_and_excludes_outside_window(run_forecasts):
    values = np.full((6, 6), 99.)
    values[1:4] = np.array([1., 2., 4.])[:, None]
    forecasts, _ = run_forecasts(values, [3], {"Historical mean": (None,)}, window=3)
    np.testing.assert_array_equal(forecasts["Historical mean"], np.full((1, 1, 6), 7.))


def test_koopman_spectrum_preserves_original_ridge_and_pretest_sample():
    values = np.random.default_rng(35).normal(size=(120, 6))
    dates = pd.bdate_range(end="2004-12-31", periods=100).append(
        pd.bdate_range("2005-01-03", periods=20)
    )
    cfg = load_config()

    inputs = {
        "cfg": cfg,
        "factors": pd.DataFrame(values, index=dates),
        "returns_array": values,
        "multiscale_features": multiscale_features,
        "fit_ridge_var": fit_ridge_var,
    }
    result = run_notebook_cell("koopman_summary", **inputs)
    features = multiscale_features(values[:100])[21:]
    standardized = (features - features.mean(0)) / features.std(0, ddof=1)
    x, y = standardized[:-1], standardized[1:]
    x, y = x - x.mean(0), y - y.mean(0)
    spectral_tau = float(cfg["spectral_tau"])

    transition = np.linalg.solve(
        x.T @ x / len(x) + spectral_tau * np.eye(18),
        x.T @ y / len(x),
    )
    expected_radius = np.abs(np.linalg.eigvals(transition)).max()
    row = result["koopman_summary"].iloc[0]
    assert row["largest_eigenvalue_modulus"] == pytest.approx(expected_radius, abs=5e-7)
    assert row["training_transitions"] == 78
    changed = values.copy()
    changed[100:] = 50.
    after = run_notebook_cell("koopman_summary", **{**inputs, "returns_array": changed})
    pd.testing.assert_frame_equal(result["koopman_summary"], after["koopman_summary"])


def test_holdout_uncertainty_uses_paired_dates_and_configured_bootstrap():
    losses = {
        "Historical mean": np.array([1., 3., 2., 6., 5., 9., 4., 7.]),
        "Symmetric HAR": np.array([2., 1., 3., 4., 7., 5., 3., 2.]),
        "Squared VAR": np.array([3., 4., 1., 5., 2., 8., 6., 3.]),
        "Asymmetric HAR": np.array([1., 2., 4., 3., 6., 7., 5., 4.]),
        "EWMA": np.array([2., 3., 1., 4., 5., 6., 7., 8.]),
    }
    settings = {"block_lengths": [2, 3], "n_boot": 47, "seed": 921}
    result = run_notebook_cell(
        "holdout_uncertainty", holdout_losses=losses,
        garch_holdout_loss=np.arange(1., 9.), bootstrap_settings=settings, mean_ci=mean_ci,
    )["holdout_uncertainty"]
    pairs = [
        ("Symmetric HAR", "Historical mean"), ("Symmetric HAR", "Squared VAR"),
        ("Symmetric HAR", "Asymmetric HAR"), ("EWMA", "Historical mean"),
        ("GARCH(1,1)", "Historical mean"),
    ]
    all_losses = {**losses, "GARCH(1,1)": np.arange(1., 9.)}
    assert len(result) == len(pairs) * len(settings["block_lengths"])
    for model, benchmark in pairs:
        for block in settings["block_lengths"]:
            row = result.loc[
                (result["comparison"] == f"{model} vs {benchmark}")
                & (result["block_length"] == block)
            ].iloc[0]
            expected = mean_ci(
                all_losses[benchmark] - all_losses[model], block=block,
                n_boot=settings["n_boot"], seed=settings["seed"],
            )
            for column, key in [("mean_loss_improvement", "mean"), ("bootstrap_se", "se"),
                                ("ci_low", "lo"), ("ci_high", "hi")]:
                assert row[column] == pytest.approx(expected[key], abs=5e-7)


@pytest.mark.parametrize("period", ["dev", "holdout"])
def test_garch_status_counts_are_labeled_as_forecasts(period):
    targets = np.full((21, 6), 2.)
    forecasts = np.ones((21, 6))
    statuses = np.zeros((21, 6), dtype=int)
    historical = {"Historical mean": np.full(21, 2.)}
    inputs = {
        f"{period}_targets": targets, f"garch_{period}": forecasts,
        f"garch_{period}_status": statuses, "pooled_loss": pooled_loss,
        "scales2": np.ones(6), "selected_losses": historical, "holdout_losses": historical,
    }
    summary = "garch_summary" if period == "dev" else "garch_holdout_summary"
    result = run_notebook_cell(summary, **inputs)[summary]
    assert result.loc[0, "successful_forecasts"] == 126
    assert "converged_fits" not in result.columns
