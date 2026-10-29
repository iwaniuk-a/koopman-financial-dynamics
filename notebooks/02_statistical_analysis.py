import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Koopman Spectral Analysis of Financial Factor Dynamics
    ## 02 — Statistical diagnostics

    Before fitting linear, DMD, or EDMD models, we inspect the features that affect
    model choice: marginal tails, serial dependence, volatility clustering,
    cross-factor dependence, and stability across historical subperiods.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    import sys
    from pathlib import Path
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.data import load_factors

    return acorr_ljungbox, load_factors, mo, np, pd, plot_acf, plt, stats


@app.cell
def _(load_factors, pd):
    factors = load_factors("data/processed/factors_daily.csv")
    data_summary = pd.Series(
        {
            "start": str(factors.index.min().date()),
            "end": str(factors.index.max().date()),
            "observations": len(factors),
            "factors": factors.shape[1],
        },
        name="value",
    )
    data_summary
    return (factors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Marginal tails

    For a Gaussian random variable, skewness and excess kurtosis are both zero. The table
    compares those moments with the number of observations beyond
    \(\lvert z_t\rvert > 3\), whose Gaussian benchmark is
    \(2\Phi(-3) \approx 0.0027\).
    """)
    return


@app.cell
def _(factors, pd):
    distribution_stats = pd.DataFrame(
        {
            "mean": factors.mean(),
            "std": factors.std(),
            "skewness": factors.skew(),
            "excess_kurtosis": factors.kurt(),
            "min": factors.min(),
            "max": factors.max(),
        }
    )
    distribution_stats.round(4)
    return


@app.cell
def _(factors, pd, stats):
    z_scores = (factors - factors.mean()) / factors.std()
    empirical_tail_frequency = (z_scores.abs() > 3).mean()
    gaussian_tail_frequency = 2.0 * stats.norm.sf(3)

    tail_comparison = pd.DataFrame(
        {
            "empirical": empirical_tail_frequency,
            "gaussian": gaussian_tail_frequency,
            "empirical / gaussian": empirical_tail_frequency / gaussian_tail_frequency,
        }
    )
    tail_comparison.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The factors are clearly non-Gaussian. Three-standard-deviation events occur 4–7 times more often than under a normal distribution, and the high excess kurtosis confirms the presence of heavy tails.

    Skewness is mostly negative, especially for momentum, meaning large negative shocks are more common than equally large positive ones. These features suggest that Gaussian models may understate tail risk and support analysing volatility separately through squared returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Serial dependence

    The sample autocorrelation at lag \(k\) is
    \[
    \hat\rho(k)=
    \frac{\widehat{\operatorname{Cov}}(r_t,r_{t-k})}
         {\widehat{\operatorname{Var}}(r_t)}.
    \]
    We inspect a small set of lags = [1, 5, 10, 20].
    """)
    return


@app.cell
def _(factors, pd):
    acf_lags = [1, 5, 10, 20]
    return_acf = pd.DataFrame(
        {
            factor: [factors[factor].autocorr(lag=lag) for lag in acf_lags]
            for factor in factors.columns
        },
        index=acf_lags,
    )
    return_acf.index.name = "lag"
    return_acf.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Autocorrelation is positive at lag 1 for every factor, but it is generally small. Momentum, CMA, and RMW show the strongest short-term correlation, while market returns are close to uncorrelated.

    By lags 5–20, the correlations are mostly near zero, suggesting that linear dependence in daily returns fades quickly. This means any useful structure is likely to come from cross-factor relationships or volatility dynamics rather than long-memory prediction of returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Volatility clustering

    A stylized return model is \(r_t=\sigma_t\varepsilon_t\). Returns can have
    weak signed autocorrelation while for squared returns:
    \[
    \operatorname{Corr}(r_t^2,r_{t-k}^2)>0.
    \]
    Squared-return dependence is therefore a simple diagnostic for conditional
    volatility and for nonlinear observables used later by EDMD.
    """)
    return


@app.cell
def _(acorr_ljungbox, factors, pd):
    volatility_acf = pd.DataFrame(
        {
            "return_acf_1": factors.apply(lambda series: series.autocorr(lag=1)),
            "squared_return_acf_1": factors.apply(
                lambda series: (series**2).autocorr(lag=1)
            ),
            "squared_return_acf_5": factors.apply(
                lambda series: (series**2).autocorr(lag=5)
            ),
            "squared_return_acf_20": factors.apply(
                lambda series: (series**2).autocorr(lag=20)
            ),
        }
    )
    squared_return_lb_pvalue = pd.Series(
        {
            factor: acorr_ljungbox(
                factors[factor] ** 2,
                lags=[20],
                return_df=True,
            )["lb_pvalue"].iloc[-1]
            for factor in factors.columns
        },
        name="squared_return_lb_pvalue_lag20",
    )
    volatility_diagnostics = volatility_acf.assign(squared_return_lb_pvalue_lag20=squared_return_lb_pvalue)
    volatility_diagnostics.round(6)
    return


@app.cell
def _(factors, plot_acf, plt):
    factor_names = factors.columns.tolist()

    fig_vol, axes_vol = plt.subplots(
        nrows=len(factor_names),
        ncols=2,
        figsize=(12, 3 * len(factor_names)),
        constrained_layout=True,
    )

    for row, factor in enumerate(factor_names):
        plot_acf(factors[factor], lags=40, zero=False, ax=axes_vol[row, 0])
        plot_acf(factors[factor] ** 2, lags=40, zero=False, ax=axes_vol[row, 1])

        axes_vol[row, 0].set_title(f"{factor}: returns")
        axes_vol[row, 1].set_title(f"{factor}: squared returns")

    fig_vol
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The raw returns show little serial dependence, but their squared values are strongly autocorrelated. For example, lag-1 squared-return autocorrelation ranges from about 0.20 to 0.31 and remains positive even at longer lags.

    This is the signature of volatility clustering. Large moves tend to be followed by large moves, regardless of direction. The plots also show that this persistence is much clearer for squared returns than for raw returns, motivating the use of volatility models and nonlinear observables in the EDMD analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-factor structure

    The same-day correlation matrix is
    \[
    C_{ij}=\frac{\operatorname{Cov}(x_{t,i},x_{t,j})}{\sigma_i\sigma_j}.
    \]
    """)
    return


@app.cell
def _(factors):
    correlation_matrix = factors.corr()
    correlation_matrix.round(3)
    return (correlation_matrix,)


@app.cell
def _(correlation_matrix, factors, plt):
    fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
    image_corr = ax_corr.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax_corr.set_xticks(range(6), labels=factors.columns)
    ax_corr.set_yticks(range(6), labels=factors.columns)
    fig_corr.colorbar(image_corr, ax=ax_corr)
    ax_corr.set_title("Same-day factor correlations")
    fig_corr
    return


@app.cell
def _(correlation_matrix, np, pd):
    # PCA
    eigenvalues = np.linalg.eigvalsh(correlation_matrix.to_numpy())[::-1]
    explained_variance = eigenvalues / eigenvalues.sum()
    spectrum = pd.DataFrame(
        {
            "eigenvalue": eigenvalues,
            "explained_variance": explained_variance,
            "cumulative_variance": np.cumsum(explained_variance),
        }
    )
    spectrum.round(4)
    return (explained_variance,)


@app.cell
def _(explained_variance, plt):
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
    ax_spectrum.bar(range(1, 7), explained_variance)
    ax_spectrum.set_xlabel("Principal component")
    ax_spectrum.set_ylabel("Fraction of variance explained")
    ax_spectrum.set_title("Correlation-matrix spectrum")
    fig_spectrum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The factors are only moderately correlated with one another. The strongest relationship is the positive correlation between HML and CMA, while MKT and CMA have the strongest negative correlation. This suggests that the factors contain distinct sources of variation rather than being driven by one common signal.

    The correlation spectrum supports this. The first three principal components explain about 71% of the total variation. This indicates that the six-dimensional factor space can be represented reasonably well by a smaller number of principal components, while still retaining meaningful information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stability across subperiods

    If lag-one dependence changes across eras, a single global linear or Koopman
    operator can average across different regimes. The table below is a compact
    check. Later rolling-window models must distinguish genuine changes from
    estimation noise.
    """)
    return


@app.cell
def _(factors, pd):
    periods = {
        "1963–1989": ("1963-07-01", "1989-12-31"),
        "1990–2009": ("1990-01-01", "2009-12-31"),
        "2010–2026": ("2010-01-01", "2026-08-31"),
    }
    period_acf = pd.DataFrame(
        {
            name: factors.loc[start:end].apply(lambda series: series.autocorr(lag=1))
            for name, (start, end) in periods.items()
        }
    )
    period_acf.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Lag-one dependence is much stronger in the earliest period than in the later samples. Momentum, HML, and CMA show the clearest decline, while market returns become slightly negatively autocorrelated in 2010–2026.

    This suggests that a single global operator may average over different regimes. Rolling or subperiod models could therefore be useful, although the shorter recent sample means these changes should be interpreted cautiously.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Motivation for the linear benchmark

    These diagnostics motivate the linear benchmark used in the next notebook. We model the factor vector as

    \[
    x_{t+1}=Ax_t+\varepsilon_{t+1}.
    \]

    If the noise is uncorrelated with \(x_t\), then

    \[
    \Gamma_1 = A\Gamma_0,
    \]

    where \(\Gamma_1\) measures lag-one dependence and \(\Gamma_0\) measures covariance. This motivates estimating the transition matrix as

    \[
    \widehat A = \widehat\Gamma_1\widehat\Gamma_0^{-1}.
    \]

    The next notebook fits this model. Here, we only assess whether its assumptions appear reasonable.
    """)
    return


if __name__ == "__main__":
    app.run()
