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

    from src.evaluation import (
        fixed_scales,
        load_config,
        pooled_loss,
        pooled_oos_r2,
        rolling_forecasts,
        squared_errors,
        target_positions,
        tau_grid,
    )
    from src.models import (
        dictionary,
        fit_edmd,
        make_edmd_predictor,
        make_edmd_second_moment_predictor,
        make_predictor,
    )

    return (
        dictionary,
        fit_edmd,
        fixed_scales,
        load_config,
        make_edmd_predictor,
        make_edmd_second_moment_predictor,
        make_predictor,
        mo,
        np,
        pd,
        plt,
        pooled_loss,
        pooled_oos_r2,
        rolling_forecasts,
        squared_errors,
        target_positions,
        tau_grid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polynomial EDMD: signed and squared factor returns

    The notebook studies whether polynomial EDMD observables improve one-day forecasts of six daily financial factors:
    \[
    r_t =
    \begin{bmatrix}
    r_t^{MKT} &
    r_t^{SMB} &
    r_t^{HML} &
    r_t^{RMW} &
    r_t^{CMA} &
    r_t^{MOM}
    \end{bmatrix}^{\top}.
    \]
    It compares:
    - Ridge VAR: a linear model
    - Squares EDMD: returns plus individual squared returns
    - Full EDMD: returns, squares, and pairwise interactions
    """)
    return


@app.cell
def _(fixed_scales, load_config, pd, target_positions, tau_grid):
    # loads configurations

    cfg = load_config()

    # load data with the approperiate time period
    factors = pd.read_csv(cfg["data"]["path"], index_col="Date", parse_dates=True)
    factors = factors.loc[:cfg["splits"]["dev_target_end"]]
    returns = factors.to_numpy()

    # data for estimating scaling parameters
    train = factors.loc[:cfg["splits"]["scale_estimation_end"]] 

    # window length and tau grid
    window = cfg["window"] 
    taus = tau_grid(cfg)

    # positions of development period targets
    positions = target_positions(
        factors.index, cfg["splits"]["dev_target_start"], cfg["splits"]["dev_target_end"]
    )

    # information available immediately before each forecast target
    origins = positions - 1

    # scales for returns
    scales2 = fixed_scales(factors, cfg["splits"]["scale_estimation_end"])
    second_moment_scales2 = fixed_scales(factors**2, cfg["splits"]["scale_estimation_end"])
    return (
        cfg,
        origins,
        returns,
        scales2,
        second_moment_scales2,
        taus,
        train,
        window,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polynomial dictionaries

    EDMD does not model the original state directly. Instead, it maps the state into a larger feature space using observables.
    For \(d=6\), the linear dictionary is
    \[
    \psi_{\text{linear}}(z)
    =
    (z_1,\ldots,z_6).
    \]
    It contains 6 features.
    The squares dictionary is
    \[
    \psi_{\text{squares}}(z)
    =
    (z_1,\ldots,z_6,z_1^2,\ldots,z_6^2).
    \]
    It contains $6+6=12$ features.

    The full quadratic dictionary adds all pairwise interactions:
    \[
    \psi_{\text{full}}(z)
    =
    \left(
    z_i,\ z_i^2,\ z_i z_j
    \right)_{i<j}.
    \]
    The number of quadratic monomials is
    $\frac{d(d+1)}{2}=\frac{6\cdot 7}{2}=21.$

    Therefore, the full dictionary has $27$ nonconstant features.

    The additional features allow the model to represent nonlinear effects such as:
    - dependence on the magnitude of returns
    - volatility-like behaviour through \(z_j^2\)
    - interactions between factors through \(z_i z_j\)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridge EDMD
    Let \(X\) contain the lifted states in a rolling window and \(Y\) the corresponding next-period states. With row-oriented snapshots,
    \[
    Y\approx XK.
    \]
    After centering the snapshots,
    \[
    X_c=X-\bar X,
    \qquad
    Y_c=Y-\bar Y,
    \]
    the ridge EDMD estimator is
    \[
    \widehat K_\tau
    =
    \left(
    \frac{X_c^\top X_c}{n}
    +
    \tau I
    \right)^{-1}
    \frac{X_c^\top Y_c}{n},
    \]
    where \(n=W-1\) and \(\tau\) controls regularization.

    The fitted prediction is
    \[
    \widehat P_{t+1}
    =
    \bar Y+(P_t-\bar X)\widehat K_\tau.
    \]
    The result is then transformed back from standardized feature space into return units. For squared returns,
    \[
    r^2=(\mu+\sigma z)^2
    =
    \mu^2+2\mu\sigma z+\sigma^2z^2,
    \]
    so both the predicted linear and squared coordinates are needed.
    """)
    return


@app.cell
def _(cfg, dictionary, fit_edmd, np, pd, train, window):
    # Training only: compare window lengths at common endpoints and fixed tau.
    spectral_windows = sorted({window, *cfg["window_sensitivity"]})
    spectral_tau = 0.1

    feature_shapes = {}
    _sample = train.to_numpy()[-window:]

    _z = (_sample - _sample.mean(0)) / _sample.std(0, ddof=1)

    for _kind, _size in [("linear", 6), ("squares", 12), ("full", 27)]:
        feature_shapes[_kind] = dictionary(_z, _kind).shape

    _rows = []
    for _end in range(max(spectral_windows), len(train) + 1, 63):
        for _width in spectral_windows:
        
            _fit = fit_edmd(train.iloc[_end - _width:_end].to_numpy(), [spectral_tau], "full")
        
            _radius = np.abs(np.linalg.eigvals(_fit["K"][0])).max()
        
            _rows.append({"date": train.index[_end - 1], "window": _width, "radius": _radius})

    training_spectra = pd.DataFrame(_rows)
    return spectral_tau, training_spectra


@app.cell(hide_code=True)
def _(mo, plt, spectral_tau, training_spectra):
    spectral_figure, _ax = plt.subplots(figsize=(9, 3.5))
    for _width, _group in training_spectra.groupby("window"):
        _ax.plot(_group["date"], _group["radius"], label=f"{_width} observations")
    _ax.set(xlabel="Training-window end", ylabel="Nonconstant spectral radius")
    _ax.grid(alpha=0.3)
    _ax.legend()
    spectral_figure.tight_layout()
    mo.vstack([
        mo.md(f"Training-window sensitivity at fixed tau = {spectral_tau:g}. "
              "Endpoints are 63 trading observations apart, all end by 2004. "
              "Overlapping windows are dependent."),
        spectral_figure,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The estimated spectral radius changes substantially over time, with noticeable spikes around the late 1970s, late 1980s, and late 1990s. This suggests that the estimated dynamics are regime-dependent rather than constant throughout the sample.

    The shorter 504-observation window is more volatile, while the longer windows produce smoother estimates. However, all spectral radii remain below one, so the fitted operators are stable. We hav to keep in mind that the windows overlap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signed-return validation

    For each forecast date, the normalized pooled loss is
    $$
    \ell_t=\frac{1}{6}\sum_{j=1}^{6}\frac{(r_{t,j}-\widehat r_{t,j})^2}{s_j^2}.
    $$
    The normalization prevents the most volatile factor from dominating the comparison.
    Performance is reported relative to the historical-mean benchmark:
    \[
    R^2_{\mathrm{OOS}}
    =
    1-
    \frac{
    \sum_t\ell_t^{\mathrm{model}}
    }{
    \sum_t\ell_t^{\mathrm{benchmark}}
    }.
    \]
    Positive \(R^2_{\mathrm{OOS}}\) means the model improves on the benchmark, negative values mean it performs worse.
    """)
    return


@app.cell
def _(pd, pooled_loss, pooled_oos_r2, squared_errors, taus):
    def compare_forecasts(predictions, actual, benchmark, scales, *, second_moment=False):
        benchmark_loss = pooled_loss(squared_errors(actual, benchmark), scales)
        columns = {"tau": taus}
        for name, forecasts in predictions.items():
            losses = [
                pooled_loss(squared_errors(actual, forecasts[:, j, :]), scales)
                for j in range(len(taus))
            ]
            columns[f"{name} loss"] = [loss.mean() for loss in losses]
            columns[f"{name} R²"] = [pooled_oos_r2(loss, benchmark_loss) for loss in losses]
            if second_moment:
                columns[f"{name} negative fraction"] = (forecasts < 0).mean(axis=(0, 2))
        return pd.DataFrame(columns), benchmark_loss

    return (compare_forecasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signed-return experiment
    The signed-return experiment compares ridge VAR, squares EDMD, and full EDMD for forecasting \(r_{t+1}\).
    The purpose is to test whether nonlinear observables provide useful directional information. Because daily factor returns are usually difficult to predict, even a positive improvement is likely to be small.
    """)
    return


@app.cell
def _(
    compare_forecasts,
    make_edmd_predictor,
    make_predictor,
    origins,
    returns,
    rolling_forecasts,
    scales2,
    taus,
    window,
):
    mean_forecasts, signed_targets = rolling_forecasts(
        returns, origins, window, make_predictor("mean")
    )
    signed_predictions = {}
    for _name, _predictor in [
        ("Ridge VAR", make_predictor("ridge", taus)),
        ("Squares EDMD", make_edmd_predictor(taus, "squares")),
        ("Full EDMD", make_edmd_predictor(taus, "full")),
    ]:
        _forecasts, _actual = rolling_forecasts(returns, origins, window, _predictor)
    
        signed_predictions[_name] = _forecasts

    signed_comparison, signed_benchmark_loss = compare_forecasts(
        signed_predictions, signed_targets, mean_forecasts[:, 0, :], scales2
    )
    signed_comparison.round(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The ridge VAR performs best around \(\tau\approx 5.3\), with an out-of-sample \(R^2\) of about \(0.17\%\) relative to the historical-mean benchmark. Squares EDMD is very similar, reaching about \(0.14\%\), while full EDMD provides almost no improvement.

    Weak regularization performs poorly, especially for full EDMD, indicating overfitting from the additional nonlinear features. As \(\tau\) increases, the models become more stable but gradually move back toward benchmark performance.

    Overall, nonlinear observables provide little additional directional information. The main benefit comes from regularization, not from expanding the dictionary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Squared-return validation and negative forecasts

    The second experiment forecasts
    \[
    r_{t+1}^2.
    \]
    This focuses on the magnitude of returns rather than their direction. Persistent squared-return autocorrelation suggests that this problem may contain more predictable structure than signed-return forecasting.
    The squared-return models are compared using variances of squared returns as the normalization scales.

    A true conditional second moment must satisfy
    \[
    \mathbb{E}[r_{t+1}^2\mid\mathcal I_t]\geq 0.
    \]
    The unconstrained ridge and EDMD regressions do not enforce this restriction. Therefore, we rocord the fraction of forecasts below zero.
    A model with lower loss but many negative forecasts may fit numerically while producing invalid volatility estimates. Loss and validity must therefore be interpreted together.
    """)
    return


@app.cell
def _(
    compare_forecasts,
    make_edmd_second_moment_predictor,
    make_predictor,
    origins,
    returns,
    rolling_forecasts,
    second_moment_scales2,
    taus,
    window,
):
    squared_returns = returns**2
    historical_second_moment, squared_targets = rolling_forecasts(
        squared_returns, origins, window, make_predictor("mean")
    )
    squared_predictions = {}
    for _name, _panel, _predictor in [
        ("Squared VAR", squared_returns, make_predictor("ridge", taus)),
        ("Squares EDMD", returns, make_edmd_second_moment_predictor(taus, "squares")),
        ("Full EDMD", returns, make_edmd_second_moment_predictor(taus, "full")),
    ]:
        _forecasts, _actual = rolling_forecasts(_panel, origins, window, _predictor)
        squared_predictions[_name] = _forecasts

    squared_comparison, squared_benchmark_loss = compare_forecasts(
        squared_predictions, squared_targets, historical_second_moment[:, 0, :],
        second_moment_scales2, second_moment=True,
    )
    squared_comparison.round(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Forecasting squared returns is more successful than forecasting signed returns. The squared VAR achieves the best result, with an out-of-sample \(R^2\) of approximately \(12.1\%\) at \(\tau\approx0.66\). Squares EDMD reaches about \(11.4\%\), while full EDMD reaches about \(10.0\%\).

    The negative forecast fraction decreases as regularization increases. This shows the trade-off clearly, weakly regularized models achieve more flexible fits but occasionally produce invalid negative second-moment forecasts, while stronger regularization improves validity at the cost of forecast accuracy.

    These results suggest that the main predictable structure in the factors is volatility clustering rather than directional return predictability. However, a dedicated nonnegative volatility model would be preferable for production use.
    """)
    return


if __name__ == "__main__":
    app.run()
