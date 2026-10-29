import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data import load_factors
    from src.evaluation import (
        load_config,
        mean_ci,
        pooled_loss,
        rolling_forecasts,
        target_positions,
        tau_grid
    )
    from src.models import (
        fit_ridge_var,
        make_ewma_predictor,
        make_har_predictor,
        make_predictor,
        multiscale_features,
        rolling_garch_forecasts,
    )

    return (
        fit_ridge_var,
        load_config,
        load_factors,
        make_ewma_predictor,
        make_har_predictor,
        make_predictor,
        mean_ci,
        multiscale_features,
        np,
        pd,
        pooled_loss,
        project_root,
        rolling_forecasts,
        rolling_garch_forecasts,
        target_positions,
        tau_grid
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Volatility forecasting

    ## Observable and information set

    The volatility target is the squared return

    \[
    q_t=r_t^2.
    \]

    Unlike signed returns, squared returns measure the magnitude of price movements and are therefore a simple proxy for daily volatility.

    For each factor, we construct trailing averages over three horizons:

    \[
    m_t^{(h)}
    =
    \frac{1}{h}
    \sum_{j=0}^{h-1}q_{t-j},
    \qquad
    h\in\{1,5,22\}.
    \]

    These represent daily, weekly, and approximately monthly volatility. With six factors and three horizons, the symmetric feature set contains \(6\times3=18\) variables.

    The asymmetric version adds

    \[
    q_t\mathbf{1}(r_t<0),
    \]

    which separates volatility associated with negative returns from total volatility. This adds another 18 features and gives 36 variables in total.

    At origin \(t\), these features use only information available through \(t\) and are used to forecast \(q_{t+1}\). The first 21 observations are discarded because a 22-day trailing average cannot be computed before a full history is available.
    """)
    return



@app.cell
def _(load_config, load_factors, project_root):
    cfg = load_config(project_root / "config.yaml")

    factors = load_factors(cfg["data"]["path"])
    returns_array = factors.to_numpy(dtype=float)

    bootstrap_settings = cfg["bootstrap"]

    return bootstrap_settings, cfg, factors, returns_array


@app.cell
def _(cfg, target_positions):
    def split_positions(factors):
        index = factors.index
        splits = cfg["splits"]

        # Development dates are used for model selection.
        dev_targets = target_positions(
            index,
            splits["dev_target_start"],
            splits["dev_target_end"],
        )

        # Holdout dates remain untouched until final evaluation.
        holdout_targets = target_positions(
            index,
            splits["test_target_start"],
            splits["test_target_end"],
        )

        return {
            "dev_origins": dev_targets - 1,
            "holdout_origins": holdout_targets - 1,
        }

    return (split_positions,)

@app.cell
def _(factors, split_positions):
    date_definitions = split_positions(factors)

    dev_origins = date_definitions["dev_origins"]
    holdout_origins = date_definitions["holdout_origins"]
    return dev_origins, holdout_origins


@app.cell
def _(
    make_ewma_predictor,
    make_har_predictor,
    make_predictor,
    np,
    rolling_forecasts,
):
    def run_forecasts(returns, origins, specifications, window=756):
        returns = np.asarray(returns, dtype=float)

        # the target is the squared return one day after each forecast origin
        targets = returns[np.asarray(origins) + 1] ** 2
        forecasts = {}

        for name, settings in specifications.items():

            if name == "Historical mean":
                # forecast the average squared return in the rolling window
                predictor = lambda w: (w ** 2).mean(axis=0, keepdims=True)
                input_returns = returns
    
            elif name == "Squared VAR":
                # fit a ridge VAR directly to squared returns
                predictor = make_predictor("ridge", settings)
                input_returns = returns ** 2
    
            elif name == "Symmetric HAR":
                # use daily, weekly, and monthly volatility features
                predictor = make_har_predictor(settings)
                input_returns = returns
    
            elif name == "Asymmetric HAR":
                # add separate features for negative-return volatility
                predictor = make_har_predictor(settings, asymmetric=True)
                input_returns = returns
    
            elif name == "EWMA":
                # use exponentially decaying weights on past squared returns
                predictor = make_ewma_predictor(settings)
                input_returns = returns
            else:
                raise ValueError(f"Unknown forecast family: {name}")
            forecasts[name], _ = rolling_forecasts(input_returns, origins, window, predictor)
        return forecasts, targets

    return (run_forecasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Development comparison: 2005–2014

    Model settings are selected using only the 2005–2014 development period. This prevents the final holdout period from influencing model choice.

    For each factor, the forecast error is normalized by a fixed variance estimated through 2004. The pooled daily loss is

    \[
    L_t=
    \frac{1}{6}
    \sum_{j=1}^{6}
    \frac{
    (q_{t+1,j}-\widehat q_{t+1,j})^2
    }{
    s_j^2
    }.
    \]

    Normalizing by \(s_j^2\) prevents the most volatile factor from dominating the comparison.

    Each model family is evaluated over its own tuning grid, and the setting with the lowest average development loss is selected. The historical mean is the benchmark.

    The block bootstrap preserves short-run dependence in the loss series by resampling consecutive 20-day blocks. A positive value of

    \[
    L_t^{\text{benchmark}}-L_t^{\text{model}}
    \]

    means that the model improves on the benchmark. Confidence intervals that exclude zero provide stronger evidence that the difference is not explained only by sampling variation.
    """)
    return


@app.cell
def _(cfg, np, tau_grid):
    forecast_window = cfg["window"]

    taus = tau_grid(cfg)

    ewma_lambdas = np.asarray(
        cfg["ewma_lambdas"],
        dtype=float,
    )

    return ewma_lambdas, forecast_window, taus


@app.cell
def _(cfg, factors, pd, returns_array):

    scale_end = pd.Timestamp(cfg["splits"]["scale_estimation_end"])

    train_mask = factors.index <= scale_end

    scales2 = (returns_array[train_mask] ** 2).var(axis=0, ddof=1)
    return (scales2,)


@app.cell
def _(
    dev_origins,
    ewma_lambdas,
    forecast_window,
    returns_array,
    run_forecasts,
    taus,
):
    # define the model families and their candidate tuning parameters
    development_specs = {
        "Historical mean": (None,),
        "Squared VAR": tuple(taus),
        "Symmetric HAR": tuple(taus),
        "Asymmetric HAR": tuple(taus),
        "EWMA": tuple(ewma_lambdas),
    }
    # evaluate every candidate on the 2005–2014 development period
    dev_forecasts, dev_targets = run_forecasts(
        returns_array, dev_origins, development_specs, forecast_window
    )
    return dev_forecasts, dev_targets, development_specs


@app.cell
def _(
    bootstrap_settings,
    dev_forecasts,
    dev_targets,
    development_specs,
    mean_ci,
    mo,
    np,
    pd,
    pooled_loss,
    scales2,
):
    # calculate normalized losses for every model and tuning parameter
    dev_losses = {
        name: pooled_loss((dev_targets - values[:, 0, :]) ** 2, scales2)

        if values.shape[1] == 1 
        else np.column_stack([
            pooled_loss((dev_targets - values[:, k, :]) ** 2, scales2)
            for k in range(values.shape[1])
        ])
        for name, values in dev_forecasts.items()
    }

    # select the parameter with the lowest average development loss
    selected = {
        name: int(np.argmin(loss.mean(axis=0))) if loss.ndim == 2 else 0
        for name, loss in dev_losses.items()
    }

    # keep only the loss series corresponding to the selected setting
    selected_losses = {
        name: loss if loss.ndim == 1 else loss[:, selected[name]]
        for name, loss in dev_losses.items()
    }

    # the historical-mean loss is the benchmark for all comparisons
    historical_loss = selected_losses["Historical mean"]

    # summarize the selected setting and its average normalized loss
    dev_summary = pd.DataFrame([
        {
            "model": name,
            "setting": (
                None 
                if name == "Historical mean" 
                else float(development_specs[name][selected[name]])
            ),
            "mean_normalized_loss": loss.mean(),
            "relative_to_historical": loss.mean() / historical_loss.mean(),
        }
        for name, loss in selected_losses.items()
    ])

    # compare each model's daily loss with the historical-mean loss
    dev_comparisons = []
    for _name, _loss in selected_losses.items():
        if _name == "Historical mean":
            continue
        ci = mean_ci(
            historical_loss - _loss, 
            block=20,
            n_boot=bootstrap_settings["n_boot"], 
            seed=bootstrap_settings["seed"],
        )
        dev_comparisons.append({
            "model_vs_historical": _name, 
            "mean_loss_improvement": ci["mean"],
            "bootstrap_se": ci["se"], 
            "ci_low": ci["lo"], 
            "ci_high": ci["hi"],
        })
    dev_summary = dev_summary.round(6)
    dev_uncertainty = pd.DataFrame(dev_comparisons).round(6)
    mo.vstack([
        mo.ui.table(dev_summary),
        mo.ui.table(dev_uncertainty),
    ])
    return dev_losses, selected, selected_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    On the development period, every model improves on the historical-mean benchmark. The benchmark loss is \(3.531\), compared with \(3.105\) for Squared VAR, \(2.757\) for symmetric HAR, \(2.768\) for asymmetric HAR, and \(2.648\) for EWMA.

    EWMA performs best, reducing normalized loss by about \(25.0\%\). The bootstrap intervals for all models remain above zero, so each improvement is supported by the development sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GARCH(1,1) benchmark

    GARCH provides a conventional volatility-model benchmark. For each factor, it models the conditional variance using its own recent squared returns and past conditional variance:

    \[
    \sigma_{t+1}^2
    =
    \omega+\alpha r_t^2+\beta\sigma_t^2.
    \]

    The conditional mean is assumed to be constant and innovations are treated as Gaussian.

    The model is refitted every 20 forecast origins and used to produce forecasts between refits. This reduces computation while still allowing the parameters to adapt over time.

    Each fit returns a status code. Non-converged estimates are recorded and cause the run to stop if necessary. This makes the comparison reproducible and prevents failed model fits from being mistaken for valid forecasts.
    """)
    return


@app.cell
def _(dev_origins, forecast_window, returns_array, rolling_garch_forecasts):
    # fit GARCH models on the development period.
    # the model is refitted every 20 forecast origins.
    garch_dev, garch_dev_status = rolling_garch_forecasts(
        returns_array, 
        dev_origins, 
        window=forecast_window, 
        refit_every=20
    )
    return garch_dev, garch_dev_status


@app.cell
def _(
    dev_targets,
    garch_dev,
    garch_dev_status,
    np,
    pd,
    pooled_loss,
    scales2,
    selected_losses,
):
    # compute the GARCH normalized forecast loss
    garch_dev_loss = pooled_loss((dev_targets - garch_dev) ** 2, scales2)


    # summarize performance and convergence information
    garch_summary = pd.DataFrame([{
        "model": "GARCH(1,1)", 
        "mean_normalized_loss": garch_dev_loss.mean(),
        "relative_to_historical": garch_dev_loss.mean() / selected_losses["Historical mean"].mean(),
        "successful_forecasts": int((garch_dev_status == 0).sum()),
        "status_codes": dict(zip(*np.unique(garch_dev_status, return_counts=True))),
    }]).round(6)
    garch_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    GARCH achieves a normalized loss of \(2.680\), around \(24.1\%\) below the historical mean. It is slightly worse than EWMA on the development period, whose loss is \(2.648\).

    All 15,102 GARCH forecasts succeeded, with status code zero throughout. This confirms that the benchmark was numerically stable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frozen holdout: 2015 onward

    After model settings are selected on 2005–2014, they are frozen and evaluated on the 2015 onward holdout period. No tuning or model selection is performed using holdout outcomes.

    At each holdout origin, the model may update its coefficients or volatility state using information that would actually have been available at that date, but the chosen hyperparameters remain unchanged.

    The out-of-sample \(R^2\) is calculated relative to the historical-mean benchmark:

    \[
    R^2_{\mathrm{OOS}}
    =
    1-
    \frac{
    \sum_t L_t^{\mathrm{model}}
    }{
    \sum_t L_t^{\mathrm{benchmark}}
    }.
    \]

    Positive values indicate an improvement over the benchmark, while negative values indicate worse forecasting performance.

    Negative squared-return forecasts are retained rather than clipped. Clipping would change the evaluated model after the forecast was produced, so the frequency of negative forecasts is reported separately as a validity diagnostic.

    The paired circular block bootstrap compares models using the same holdout dates. Positive loss improvement favours the first model in each comparison.
    """)
    return


@app.cell
def _(development_specs, selected):
    # freeze the settings selected on the development period
    # no holdout results are used to change these values
    frozen_specs = {
        name: (development_specs[name][selected[name]],)
        for name in development_specs
    }
    frozen_specs["Historical mean"] = (None,)
    return (frozen_specs,)


@app.cell
def _(
    forecast_window,
    frozen_specs,
    holdout_origins,
    returns_array,
    run_forecasts,
):
    # generate holdout forecasts using the frozen development settings
    holdout_forecasts, holdout_targets = run_forecasts(
        returns_array, 
        holdout_origins, 
        frozen_specs, 
        forecast_window
    )
    return holdout_forecasts, holdout_targets


@app.cell
def _(
    frozen_specs,
    holdout_forecasts,
    holdout_targets,
    pd,
    pooled_loss,
    scales2,
):
    # calculate holdout losses for each frozen model
    holdout_losses = {
        name: pooled_loss((holdout_targets - values[:, 0, :]) ** 2, scales2)
        for name, values in holdout_forecasts.items()
    }

    holdout_rows = []
    holdout_base = holdout_losses["Historical mean"]

    for _name, _loss in holdout_losses.items():
        forecast = holdout_forecasts[_name][:, 0, :]
        holdout_rows.append({
            "model": _name, 
            "frozen_setting": frozen_specs[_name][0],
            "mean_normalized_loss": _loss.mean(),
            "pooled_oos_r2": 1 - _loss.sum() / holdout_base.sum(),
            "negative_forecast_fraction": float((forecast < 0).mean()),
        })
    holdout_summary = pd.DataFrame(holdout_rows).round(6)
    holdout_summary
    return (holdout_losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Performance declines on the frozen 2015+ holdout, but all models still improve on the historical-mean benchmark. EWMA performs best with a loss of \(4.511\) and an out-of-sample \(R^2\) of \(9.39\%\).

    Gains are smaller for Squared VAR (\(3.57\%\)), symmetric HAR (\(8.40\%\)), and asymmetric HAR (\(8.19\%\)). The low negative-forecast fractions show that the selected models produce almost no invalid negative squared-return forecasts on the holdout.
    """)
    return


@app.cell
def _(
    forecast_window,
    holdout_origins,
    returns_array,
    rolling_garch_forecasts,
):
    # evaluate GARCH on the untouched holdout period
    garch_holdout, garch_holdout_status = rolling_garch_forecasts(
        returns_array, 
        holdout_origins, 
        window=forecast_window, 
        refit_every=20
    )
    return garch_holdout, garch_holdout_status


@app.cell
def _(
    garch_holdout,
    garch_holdout_status,
    holdout_losses,
    holdout_targets,
    np,
    pd,
    pooled_loss,
    scales2,
):
    # calculate the holdout GARCH loss
    garch_holdout_loss = pooled_loss((holdout_targets - garch_holdout) ** 2, scales2)

    garch_holdout_summary = pd.DataFrame([{
        "model": "GARCH(1,1)", 
        "mean_normalized_loss": garch_holdout_loss.mean(),
        "pooled_oos_r2": 1 - garch_holdout_loss.sum() / holdout_losses["Historical mean"].sum(),
        "negative_forecast_fraction": float((garch_holdout < 0).mean()),
        "successful_forecasts": int((garch_holdout_status == 0).sum()),
        "status_codes": dict(zip(*np.unique(garch_holdout_status, return_counts=True))),
    }]).round(6)
    garch_holdout_summary
    return (garch_holdout_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    GARCH achieves a holdout loss of \(4.522\), corresponding to an out-of-sample \(R^2\) of \(9.16\%\). It produces no negative forecasts and all 17,592 fits succeed.

    GARCH is therefore competitive with EWMA, although EWMA performs slightly better. The difference is small, so the main conclusion is that conventional volatility models and multiscale HAR models provide similar predictive performance on this holdout.
    """)
    return


@app.cell
def _(bootstrap_settings, garch_holdout_loss, holdout_losses, mean_ci, pd):
    # combine all holdout losses for pairwise comparisons
    holdout_comparison_losses = {**holdout_losses, "GARCH(1,1)": garch_holdout_loss}

    # define comparisons where the first model is evaluated against
    # the second model as its benchmark
    holdout_pairs = [
        ("Symmetric HAR", "Historical mean"),
        ("Symmetric HAR", "Squared VAR"),
        ("Symmetric HAR", "Asymmetric HAR"),
        ("EWMA", "Historical mean"),
        ("GARCH(1,1)", "Historical mean"),
    ]

    holdout_intervals = []
    for _model, _benchmark in holdout_pairs:
        _difference = holdout_comparison_losses[_benchmark] - holdout_comparison_losses[_model]
        for _block in bootstrap_settings["block_lengths"]:
            _ci = mean_ci(
                _difference, block=_block,
                n_boot=bootstrap_settings["n_boot"], 
                seed=bootstrap_settings["seed"],
            )
            holdout_intervals.append({
                "comparison": f"{_model} vs {_benchmark}", 
                "block_length": _block,
                "mean_loss_improvement": _ci["mean"], 
                "bootstrap_se": _ci["se"],
                "ci_low": _ci["lo"], 
                "ci_high": _ci["hi"],
            })
    holdout_uncertainty = pd.DataFrame(holdout_intervals).round(6)
    holdout_uncertainty
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The bootstrap comparisons provide evidence that symmetric HAR improves on the historical mean and on Squared VAR. All confidence intervals remain positive across the 5-, 20-, and 60-day block lengths.

    The difference between symmetric and asymmetric HAR is much smaller. Its confidence intervals include zero, so the data do not provide strong evidence that explicitly modelling negative-return asymmetry improves forecasts.

    Both EWMA and GARCH also significantly improve on the historical mean across all block lengths. These conclusions are conditional on the settings selected during development.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiscale Koopman diagnostic

    The multiscale features form an 18-dimensional observable state:

    \[
    \Psi_t=
    \left(
    m_{t,1}^{(1)},\ldots,m_{t,6}^{(1)},
    m_{t,1}^{(5)},\ldots,m_{t,6}^{(5)},
    m_{t,1}^{(22)},\ldots,m_{t,6}^{(22)}
    \right).
    \]

    A ridge-regularized linear transition is then fitted in this observable space:

    \[
    \Psi_{t+1}\approx B\Psi_t+c.
    \]

    This is a finite-dimensional approximation to Koopman evolution. Instead of modelling returns directly, we study how volatility-related observables evolve from one day to the next.

    The eigenvalues of \(B\) summarize the estimated dynamics. For an eigenvalue \(\lambda\):

    - \(|\lambda|<1\) indicates decay
    - \(|\lambda|\approx1\) indicates persistence
    - \(|\lambda|>1\) indicates growth in the lifted observable space

    The model is fitted only using data through 2004 and uses a fixed penalty chosen earlier.
    """)
    return


@app.cell
def _(cfg, factors, fit_ridge_var, multiscale_features, np, pd, returns_array):
    # use only the pre-test sample for the spectral diagnostic
    scale_end = pd.Timestamp(cfg["splits"]["scale_estimation_end"])
    pretest = returns_array[factors.index <= scale_end]

    # construct multiscale volatility observables
    train_features = multiscale_features(pretest)

    # remove rows without a complete 22-day history
    train_features = train_features[np.isfinite(train_features).all(axis=1)]

    # use the fixed ridge penalty specified in config.yaml
    tau_spectral = float(cfg["spectral_tau"])

    # fit the regularized transition operator in observable space
    koopman = fit_ridge_var(train_features, [tau_spectral])
    eigenvalues = np.linalg.eigvals(koopman["B"][0].T)

    koopman_summary = pd.DataFrame([{
        "observable_dimension": train_features.shape[1],
        "training_transitions": train_features.shape[0] - 1,
        "tau_spectral": tau_spectral,
        "largest_eigenvalue_modulus": float(np.abs(eigenvalues).max()),
    }]).round(6)

    koopman_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The Koopman diagnostic uses 18 daily, weekly, and monthly volatility observables and 10,426 training transitions. The largest eigenvalue modulus is \(0.797\), which is below one.

    This indicates stable but persistent dynamics in the lifted volatility state. Prturbations decay over time rather than growing exponentially. The corresponding dominant mode has an approximate half-life of three trading days if its modulus is interpreted as a geometric decay factor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Horizon ablation

    The ablation compares three feature sets:

    1. daily volatility only
    2. daily and weekly volatility
    3. daily, weekly, and monthly volatility

    All models use the same forecast window, targets, and ridge penalty. Therefore, differences in loss are attributed to the information contained in the additional horizons rather than to different tuning choices.

    The comparison tests whether volatility persistence is mainly short-term or whether longer-scale averages provide useful predictive information. A lower loss and higher out-of-sample \(R^2\) indicate that the added horizon contributes useful information beyond the shorter horizons.
    """)
    return


@app.cell
def _(
    dev_losses,
    dev_origins,
    dev_targets,
    forecast_window,
    make_har_predictor,
    pd,
    pooled_loss,
    returns_array,
    rolling_forecasts,
    scales2,
    selected,
    taus,
):
    # compare which volatility horizons are useful:
    # daily, daily plus weekly, or daily plus monthly
    ablation_specs = {
        "Daily": (1,), 
        "Daily + weekly": (1, 5), 
        "Daily + weekly + monthly": (1, 5, 22)
    }
    ablation_tau = float(taus[selected["Symmetric HAR"]])

    ablation_rows = []
    for label, horizons in ablation_specs.items():
        predictions, _ = rolling_forecasts(
            returns_array, 
            dev_origins, 
            forecast_window,
            make_har_predictor([ablation_tau], horizons=horizons),
        )
        daily = pooled_loss((dev_targets - predictions[:, 0, :]) ** 2, scales2)

        ablation_rows.append({
            "features": label, "tau": ablation_tau,
            "mean_normalized_loss": daily.mean(),
            "pooled_oos_r2_vs_historical": 1 - daily.sum() / dev_losses["Historical mean"].sum(),
        })
    ablation_table = pd.DataFrame(ablation_rows).round(6)
    ablation_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Adding longer volatility horizons improves development forecasts at every step. Daily features alone achieve an \(R^2\) of \(12.5\%\), adding weekly features increases this to \(18.8\%\), and including monthly features raises it further to \(21.9\%\).

    Because all three models use the same penalty and forecast targets, the improvement is attributable to the additional temporal scales. This supports the interpretation that volatility contains both short-term persistence and slower-moving components.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final conclusions

    This project finds a clear difference between forecasting returns and forecasting volatility.

    For signed factor returns, the linear ridge VAR performs slightly better than the historical-mean benchmark, but the gain is very small. Squares EDMD performs similarly, while the full quadratic dictionary does not provide a meaningful improvement. This suggests that nonlinear observables do not reveal much reliable one-day directional predictability.

    The results are stronger for squared returns. On the frozen holdout, EWMA achieves the best performance with an out-of-sample \(R^2\) of approximately \(9.4\%\), followed closely by GARCH at \(9.2\%\) and symmetric HAR at \(8.4\%\). The horizon ablation shows that daily, weekly, and monthly volatility features contribute complementary information.

    The Koopman analysis provides a spectral interpretation of this behaviour. The multiscale volatility operator has spectral radius approximately \(0.80\), indicating stable but persistent dynamics in the lifted observable space. The main contribution of the Koopman framework is therefore interpretability. It describes how volatility-related observables evolve over time, even though EDMD does not outperform conventional volatility models in this experiment.

    Overall, the evidence supports Koopman methods as a useful diagnostic framework for financial dynamics, but not as a demonstrated replacement for established volatility models.
    """)
    return


if __name__ == "__main__":
    app.run()
