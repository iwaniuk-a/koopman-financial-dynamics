"""Configuration, rolling evaluation, losses, and paired bootstrap inference.

Return panels and targets have shape (T, 6). Rolling predictors return (K, 6)
per origin, stacked as (T, K, 6); score one model at a time with (T, 6) errors
and (T,) pooled losses.
"""

import numpy as np
import pandas as pd
import yaml


# Configuration and forecast information sets.
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def tau_grid(cfg):
    grid = cfg["tau_grid"]
    return np.logspace(grid["min_exp"], grid["max_exp"], grid["num"])


def target_positions(index, start, end=None):
    """Integer positions of target dates in the inclusive interval [start, end]."""
    mask = np.asarray(index >= pd.Timestamp(start))
    if end is not None:
        mask &= np.asarray(index <= pd.Timestamp(end))
    return np.flatnonzero(mask)


def rolling_forecasts(returns, origins, window, predict_fn):
    """Fit on returns[t-window+1:t+1] and forecast returns[t+1] at each origin.

    Origins may be non-consecutive: each prediction is fitted independently.
    Return forecasts (T, K, 6) and targets (T, 6) in the supplied origin order.
    """
    returns = np.asarray(returns, float)
    if returns.ndim != 2 or returns.shape[1] != 6:
        raise ValueError("returns must have shape (T, 6)")
    origins = np.asarray(origins)
    if origins.ndim != 1:
        raise ValueError("origins must be one-dimensional")
    if origins.size == 0:
        raise ValueError("origins must be non-empty")
    if not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("origins must contain integer positions")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if origins.min() < window - 1:
        raise ValueError("an origin has fewer than `window` past observations")
    if origins.max() > len(returns) - 2:
        raise ValueError("an origin has no next-day target")

    predictions = []
    for origin in origins:
        forecast = np.asarray(predict_fn(returns[origin - window + 1:origin + 1]), float)
        if forecast.ndim != 2 or forecast.shape[0] == 0 or forecast.shape[1] != 6:
            raise ValueError("predict_fn must return shape (K, 6) with K >= 1")
        if predictions and forecast.shape != predictions[0].shape:
            raise ValueError("predict_fn must return the same shape at every origin")
        if not np.isfinite(forecast).all():
            raise FloatingPointError("non-finite forecast")
        predictions.append(forecast)
    return np.stack(predictions), returns[origins + 1]


# Fixed historical scales and scores for one model at a time.
def _factor_panel(values, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{name} must have shape (T, 6)")
    return values


def fixed_scales(returns_df, end="2004-12-31"):
    """Per-factor sample variances using only dates through the inclusive cutoff."""
    return returns_df.loc[:end].var(ddof=1).to_numpy()


def squared_errors(actual, forecast):
    """Elementwise squared errors for matching (T, 6) targets and forecasts."""
    actual = _factor_panel(actual, "actual")
    forecast = _factor_panel(forecast, "forecast")
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape (T, 6)")
    return (actual - forecast) ** 2


def pooled_loss(sq_err, scales2):
    """Average six variance-normalized factor errors into daily losses (T,)."""
    sq_err = _factor_panel(sq_err, "sq_err")
    scales2 = np.asarray(scales2)
    if scales2.shape != (6,) or not np.isfinite(scales2).all() or np.any(scales2 <= 0):
        raise ValueError("scales2 must have shape (6,) with positive finite variances")
    return (sq_err / scales2).mean(axis=1)


def oos_r2(sq_err_model, sq_err_bench):
    """Per-factor OOS R² from matching (T, 6) squared errors."""
    sq_err_model = _factor_panel(sq_err_model, "sq_err_model")
    sq_err_bench = _factor_panel(sq_err_bench, "sq_err_bench")
    if sq_err_model.shape != sq_err_bench.shape:
        raise ValueError("model and benchmark errors must have the same shape (T, 6)")
    return 1.0 - np.sum(sq_err_model, axis=0) / np.sum(sq_err_bench, axis=0)


def pooled_oos_r2(loss_model, loss_bench):
    """Pooled OOS R² from matching daily loss vectors (T,)."""
    loss_model, loss_bench = np.asarray(loss_model), np.asarray(loss_bench)
    if loss_model.ndim != 1 or loss_model.shape != loss_bench.shape:
        raise ValueError("model and benchmark losses must have the same shape (T,)")
    return 1.0 - np.sum(loss_model) / np.sum(loss_bench)


# Paired circular date-block bootstrap and development-period selection.
def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def block_bootstrap_indices(n, block, n_boot, rng):
    """Circular date indices (n_boot, n), shared across factors and models."""
    for value, name in [(n, "n"), (block, "block"), (n_boot, "n_boot")]:
        _positive_integer(value, name)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return indices.reshape(n_boot, -1)[:, :n]


def mean_ci(d, block, n_boot=2000, seed=0, level=0.95):
    """Bootstrap mean, sample SE, and percentile interval of daily differences."""
    d = np.asarray(d, float)
    if d.ndim != 1 or d.size == 0 or not np.isfinite(d).all():
        raise ValueError("d must be a non-empty, one-dimensional finite series")
    _positive_integer(n_boot, "n_boot")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2 to estimate a standard error")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    indices = block_bootstrap_indices(len(d), block, n_boot, np.random.default_rng(seed))
    means = d[indices].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {
        "mean": float(d.mean()), "se": float(means.std(ddof=1)),
        "lo": float(lo), "hi": float(hi),
    }


def one_se_rule(losses, taus, block, n_boot=2000, seed=0):
    """Select the largest ascending tau within one paired-difference SE of best.

    losses has shape (T, K), one pooled daily loss column per tau. Every tau
    uses the same resampled date indices so comparisons remain paired.
    """
    losses = np.asarray(losses, float)
    taus = np.asarray(taus, float)
    if losses.ndim != 2 or 0 in losses.shape or not np.isfinite(losses).all():
        raise ValueError("losses must be a non-empty finite array of shape (T, K)")
    if taus.ndim != 1 or len(taus) != losses.shape[1] or not np.isfinite(taus).all():
        raise ValueError("taus must be a finite vector matching the K loss columns")
    if not np.all(np.diff(taus) > 0):
        raise ValueError("taus must be strictly ascending")
    _positive_integer(n_boot, "n_boot")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2 to estimate a standard error")
    n_dates, n_taus = losses.shape
    means = losses.mean(axis=0)
    best = int(np.argmin(means))
    indices = block_bootstrap_indices(n_dates, block, n_boot, np.random.default_rng(seed))
    rows = []
    for k in range(n_taus):
        difference = losses[:, k] - losses[:, best]
        se = float(difference[indices].mean(axis=1).std(ddof=1))
        rows.append({
            "tau": taus[k], "mean_loss": means[k],
            "diff": float(difference.mean()), "se": se,
        })
    table = pd.DataFrame(rows)
    table["eligible"] = table["diff"] <= table["se"]
    chosen = int(table.index[table["eligible"]].max())
    return {
        "table": table,
        "best_idx": best, "best_tau": float(taus[best]),
        "chosen_idx": chosen, "tau": float(taus[chosen]),
        "best_at_boundary": best in (0, n_taus - 1),
        "chosen_at_boundary": chosen in (0, n_taus - 1),
    }


def min_detectable_improvement(d, ref_mean_loss, block, n_boot=2000, seed=0):
    """Relative loss reduction at ~80% power and one-sided 5%: 2.487 SE / loss."""
    if not np.isfinite(ref_mean_loss) or ref_mean_loss <= 0:
        raise ValueError("ref_mean_loss must be positive and finite")
    se = mean_ci(d, block, n_boot, seed)["se"]
    return 2.487 * se / ref_mean_loss
