import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import sys
    from pathlib import Path
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data import load_factors
    from src.evaluation import (
        fixed_scales,
        load_config,
        one_se_rule,
        oos_r2,
        pooled_loss,
        rolling_forecasts,
        squared_errors,
        target_positions,
        tau_grid,
    )
    from src.models import make_predictor

    return (
        fixed_scales,
        load_config,
        load_factors,
        make_predictor,
        mo,
        np,
        one_se_rule,
        oos_r2,
        pd,
        plt,
        pooled_loss,
        rolling_forecasts,
        squared_errors,
        target_positions,
        tau_grid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear signed-return baselines

    We compare zero, rolling mean, VAR(1), and ridge VAR on development targets
    from 2005–2014. `config.yaml` fixes the window, grid, loss scales, and
    bootstrap seed. Each fit sees only the window ending at its forecast origin.
    """)
    return


@app.cell
def _(fixed_scales, load_config, load_factors, target_positions, tau_grid):
    cfg = load_config() # loads paths, split dates, window lengths, and model hyperparameters

    factors = load_factors(cfg["data"]["path"]).loc[:cfg["splits"]["dev_target_end"]] # loads data with aproperiate period
    returns = factors.to_numpy()

    window = cfg["window"] # numer of observations used in each rolling/window estimation

    scales2 = fixed_scales(factors, cfg["splits"]["scale_estimation_end"]) # estimates normalization scales

    taus = tau_grid(cfg)

    # positions of the observations that we forecast
    targets = target_positions(
        factors.index, cfg["splits"]["dev_target_start"], cfg["splits"]["dev_target_end"]
    )

    # forecast origin for each target, to predict x[t+1], the model is fitted
    # or evaluated using information available at x[t]
    origins = targets - 1
    return cfg, factors, origins, returns, scales2, taus, window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rolling mean and zero

    For a window of $W$ observations through $t$, let $n=W-1$. The mean uses
    the same transition targets as VAR:
    \[
    \hat x_{t+1}=\hat\mu_t=\frac1n\sum_{s=t-W+2}^{t}x_s,
    \qquad \hat x^{\mathrm{zero}}_{t+1}=0.
    \]
    """)
    return


@app.cell
def _(make_predictor, origins, returns, rolling_forecasts, window):
    mean_forecasts, actual = rolling_forecasts(returns, origins, window, make_predictor("mean"))
    zero_forecasts, _ = rolling_forecasts(returns, origins, window, make_predictor("zero"))
    return actual, mean_forecasts, zero_forecasts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VAR(1)

    \[
    \hat x_{t+1}=c+Ax_t,\qquad
    \min_{c,A}\sum_s\|x_{s+1}-c-Ax_s\|_2^2.
    \]
    We fit the $n$ adjacent pairs within each window. The shared predictor uses
    centered and standardized coordinates with zero ridge penalty. It is
    equivalent to intercept OLS in return units.
    """)
    return


@app.cell
def _(make_predictor, origins, returns, rolling_forecasts, window):
    ols_forecasts, _ = rolling_forecasts(returns, origins, window, make_predictor("ols"))
    return (ols_forecasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridge VAR

    We standardize each factor using only its current window. With standardized
    pairs as rows, we subtract their separate means to obtain $X_c,Y_c$ and solve
    \[
    \min_B\frac1n\|Y_c-X_cB\|_F^2+\tau\|B\|_F^2,
    \qquad \tau=\lambda/n.
    \]
    Then we predict $\hat z_{t+1}=\bar y+(z_t-\bar x)B$ and transform back to original returns.
    The intercept is unpenalized. We fit every $\tau$ on identical
    windows before selecting one based on losses.
    """)
    return


@app.cell
def _(make_predictor, origins, returns, rolling_forecasts, taus, window):
    ridge_forecasts, _ = rolling_forecasts(returns, origins, window, make_predictor("ridge", taus))
    return (ridge_forecasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  Estimator selection

    We pool six squared errors using fixed pre-2005 variances $s_j^2$:
    \[
    \ell_t=\frac16\sum_{j=1}^6\frac{(x_{t,j}-\hat x_{t,j})^2}{s_j^2}.
    \]
    The one-SE rule chooses the largest $\tau$ whose mean loss difference from
    the best grid point is at most one paired block-bootstrap standard error.
    """)
    return


@app.cell
def _(
    actual,
    cfg,
    mean_forecasts,
    np,
    ols_forecasts,
    one_se_rule,
    pooled_loss,
    ridge_forecasts,
    scales2,
    squared_errors,
    taus,
    zero_forecasts,
):
    baseline_errors = {
        "zero": squared_errors(actual, zero_forecasts[:, 0]),
        "mean": squared_errors(actual, mean_forecasts[:, 0]),
        "ols": squared_errors(actual, ols_forecasts[:, 0]),
    }

    baseline_losses = {
        _name: pooled_loss(_errors, scales2) for _name, _errors in baseline_errors.items()
    }

    ridge_losses = np.stack([
        pooled_loss(squared_errors(actual, ridge_forecasts[:, _k]), scales2)
        for _k in range(len(taus))
    ], axis=1)

    selection = one_se_rule(
        ridge_losses, taus, cfg["selection"]["block_length"],
        cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"],
    )

    ridge_tuning_table = selection["table"].assign(**{
        "rel_to_mean_%": (selection["table"]["mean_loss"] / baseline_losses["mean"].mean() - 1) * 100
    }).round(4)
    ridge_tuning_table
    return baseline_errors, baseline_losses, ridge_tuning_table, selection


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The ridge model performs best at \(\tau \approx 5.34\), reducing the mean normalized loss by only about 0.17% relative to the rolling-mean benchmark. Nearby values perform almost identically, so the result is not very sensitive to the exact tuning parameter.

    The one-standard-error rule selects the larger value \(\tau \approx 15.20\). This gives up a very small amount of in-sample fit in exchange for a more conservative and stable model. At very large \(\tau\), performance moves back toward the mean benchmark, suggesting that excessive regularization removes useful dynamics.
    """)
    return


@app.cell
def _(
    actual,
    baseline_errors,
    factors,
    oos_r2,
    pd,
    ridge_forecasts,
    selection,
    squared_errors,
):
    selected_errors = {
        **baseline_errors,
        "ridge_1se": squared_errors(actual, ridge_forecasts[:, selection["chosen_idx"]]),
    }
    baseline_summary = pd.DataFrame({
        "factor": factors.columns,
        **{f"{_name}_mse": _errors.mean(axis=0) for _name, _errors in selected_errors.items()},
        **{
            f"{_name}_r2_vs_mean": oos_r2(_errors, baseline_errors["mean"])
            for _name, _errors in selected_errors.items() if _name != "mean"
        },
    }).round(4)
    baseline_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Out of sample, the one-standard-error ridge model delivers only very small gains over the mean benchmark. Its \(R^2\) is positive for MKT, RMW, CMA, and MOM, but slightly negative for SMB and HML.

    The OLS benchmark performs worse than the mean for every factor, suggesting that the unregularized model overfits the sample. Overall, regularization improves stability, but the small \(R^2\) values indicate that daily factor returns contain little reliable directional predictability.
    """)
    return


@app.cell
def _(baseline_losses, mo, plt, ridge_tuning_table, selection, taus):
    tuning_figure, _ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    _ax.semilogx(taus, ridge_tuning_table["rel_to_mean_%"], marker="o", label="ridge VAR")
    _ax.axhline(0, color="k", lw=0.8, label="rolling mean")
    for _name, _style in [("zero", "--"), ("ols", ":")]:
        _ax.axhline(
            (baseline_losses[_name].mean() / baseline_losses["mean"].mean() - 1) * 100,
            ls=_style, color=f"C{1 if _name == 'zero' else 2}", label=_name,
        )
    _ax.axvline(selection["tau"], color="gray", alpha=0.5)
    _ax.set(xlabel=r"$\tau=\lambda/n$", ylabel="Pooled loss vs mean (%)",
            title="Development tuning (selection diagnostic)")
    _ax.legend()
    mo.output.append(tuning_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    For small \(\tau\), the ridge VAR performs worse than the rolling-mean benchmark. Its loss decreases as regularization increases, reaching a minimum around \(\tau \approx 5\).

    The zero estimator already performs slightly better than the rolling mean, while OLS performs much worse, suggesting overfitting. At very large \(\tau\), the ridge model converges back toward the benchmark.
    """)
    return


if __name__ == "__main__":
    app.run()
