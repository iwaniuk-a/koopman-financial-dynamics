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
    from src.evaluation import load_config
    from src.models import fit_dmd, fit_dmd_centered

    return (
        fit_dmd,
        fit_dmd_centered,
        load_config,
        load_factors,
        mo,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamic Mode Decomposition

    DMD fits a linear evolution operator to snapshot pairs. Here we check its
    least-squares interpretation, spectrum, rank sensitivity, and intercept
    handling on pre-2005 returns. Signed-return development selection belongs
    to notebook 03. Run from the repository root.
    """)
    return


@app.cell
def _(load_config, load_factors):
    cfg = load_config()
    train = load_factors(cfg["data"]["path"]).loc[:cfg["splits"]["scale_estimation_end"]]
    X, Y = train.to_numpy()[:-1].T, train.to_numpy()[1:].T
    return X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective and SVD solution

    With snapshots as columns, $X=[x_1,\ldots,x_m]$ and
    $Y=[x_2,\ldots,x_{m+1}]$, DMD solves
    \[
    A_* = \arg\min_A\|Y-AX\|_F^2=YX^\dagger.
    \]
    For $X\approx U_r\Sigma_rV_r^\top$,
    \[
    A_r=YV_r\Sigma_r^{-1}U_r^\top,\qquad
    \widetilde A_r=U_r^\top YV_r\Sigma_r^{-1}.
    \]
    This drops singular values which are below the cutoff.
    At full rank, the full and reduced operators are orthogonally similar.
    Truncation keeps high-variance directions, which need not be the most
    predictive ones.
    """)
    return


@app.cell
def _(X, Y, fit_dmd):
    dmd = fit_dmd(X, Y)
    A_dmd = dmd["A"]
    return A_dmd, dmd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DMD–VAR equivalence

    Let \(x_t\in\mathbb{R}^d\) be the factor-return vector at time \(t\). We construct
    \[
    X=[x_1,\ldots,x_m],
    \qquad
    Y=[x_2,\ldots,x_{m+1}],
    \]
    so that \(Y\) contains the one-step-ahead observations corresponding to \(X\).
    DMD estimates a matrix \(A\) satisfying
    \[
    Y\approx AX.
    \]
    This is exactly a least-squares linear transition model. The DMD solution is
    \[
    A_{\mathrm{DMD}}=YX^\dagger,
    \]
    where \(X^\dagger\) is the Moore–Penrose pseudoinverse.
    A row-oriented VAR instead writes the same problem as
    \[
    Y^\top\approx X^\top B,
    \]
    and solves
    \[
    \min_B\|Y^\top-X^\top B\|_F^2.
    \]
    Because \(B=A^\top\), both formulations estimate the same transition dynamics.
    """)
    return


@app.cell
def _(A_dmd, X, Y, dmd, np, pd):
    _pinv = Y @ np.linalg.pinv(X, rcond=dmd["rcond"]) # YX_pseudoinverse

    _var = np.linalg.lstsq(X.T, Y.T, rcond=dmd["rcond"])[0].T

    equivalence_table = pd.DataFrame({
        "comparison": ["SVD vs pseudoinverse", "DMD vs no-intercept VAR"],
        "max_abs_difference": [np.max(np.abs(A_dmd - _pinv)), np.max(np.abs(A_dmd - _var))],
        "agrees": [np.allclose(A_dmd, _pinv), np.allclose(A_dmd, _var)],
        "numerical_rank": dmd["numerical_rank"],
    })
    assert equivalence_table["agrees"].all()
    equivalence_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The DMD operator agrees with both the pseudoinverse solution and the no-intercept VAR, up to numerical precision. This confirms that the implementation is correct. In the full-rank linear case, DMD is simply another representation of least-squares VAR estimation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrum

    Suppose that \(v\) is an eigenvector of \(A\):
    \[
    Av=\lambda v,
    \qquad
    \lambda=re^{i\theta}.
    \]
    After \(t\) iterations,
    \[
    A^t v=\lambda^t v
          =r^t e^{it\theta}v.
    \]
    The modulus \(r=|\lambda|\) controls the amplitude:
    - \(r<1\): the mode decays
    - \(r=1\): the mode stays the same
    - \(r>1\): the mode grows

    The angle \(\theta\) controls oscillation. When \(\theta\neq 0\), the corresponding period is approximately
    \[
    T=\frac{2\pi}{|\theta|}
    \]
    time steps.
    """)
    return


@app.cell
def _(A_dmd, np, pd):
    eigenvalues = np.linalg.eigvals(A_dmd)
    spectrum_table = pd.DataFrame({
        "real": eigenvalues.real,
        "imaginary": eigenvalues.imag,
        "modulus": np.abs(eigenvalues),
        "angle": np.angle(eigenvalues),
    }).sort_values("modulus", ascending=False).round(6)
    spectrum_table
    return (eigenvalues,)


@app.cell
def _(eigenvalues, mo, np, plt):
    spectrum_figure, _ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    _theta = np.linspace(0, 2 * np.pi, 500)
    _ax.plot(np.cos(_theta), np.sin(_theta), "--", color="gray", label="Unit circle")
    _ax.scatter(eigenvalues.real, eigenvalues.imag, s=60, zorder=3)
    _ax.axhline(0, color="black", linewidth=0.6)
    _ax.axvline(0, color="black", linewidth=0.6)
    _ax.set(xlabel="Real part", ylabel="Imaginary part", title="Training DMD spectrum", aspect="equal")
    _ax.grid(alpha=0.3)
    _ax.legend()
    mo.output.append(spectrum_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    All eigenvalues lie well inside the unit circle, with spectral radius of about \(0.30\). The fitted dynamics are therefore stable and decay over time rather than growing exponentially.

    The complex-conjugate pairs indicate weak oscillatory components, with approximate periods of 11 and 28 trading days. However, their small moduli suggest that these oscillations decay quickly and should not be interpreted as cycles.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rank sensitivity

    DMD can be estimated using only the first \(r\) singular directions of the snapshot matrix \(X\). If
    \[
    X=U\Sigma V^\top,
    \]
    the rank-\(r\) approximation uses
    \[
    A_r=YV_r\Sigma_r^{-1}U_r^\top.
    \]
    A smaller rank removes weak or noisy directions, while a larger rank retains more of the original data.
    The training reconstruction error is
    \[
    \frac{\|Y-A_rX\|_F^2}{md},
    \]
    where \(d\) is the number of factors and \(m\) is the number of transitions. This error normally decreases as rank increases because a higher-rank model has more flexibility.
    The spectral radius is
    \[
    \rho(\widetilde A_r)=\max_j|\lambda_j|,
    \]
    where \(\widetilde A_r\) is the reduced operator acting on the retained subspace.
    """)
    return


@app.cell
def _(X, Y, dmd, fit_dmd, np, pd):
    _rows = []
    for _rank in range(1, dmd["numerical_rank"] + 1):
        _fit = fit_dmd(X, Y, rank=_rank, rcond=dmd["rcond"])
        _rows.append({
            "rank": _rank,
            "training_mse": np.linalg.norm(Y - _fit["A"] @ X, ord="fro") ** 2 / Y.size,
            "spectral_radius": np.abs(np.linalg.eigvals(_fit["A_reduced"])).max(),
        })
    rank_table = pd.DataFrame(_rows).round(7)
    rank_table
    return (rank_table,)


@app.cell
def _(mo, plt, rank_table):
    rank_figure, _axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    for _ax, _metric, _title in zip(
        _axes, ["training_mse", "spectral_radius"], ["Training MSE", "Reduced spectral radius"]
    ):
        _ax.plot(rank_table["rank"], rank_table[_metric], marker="o")
        _ax.set(xlabel="DMD rank", ylabel=_title)
    mo.output.append(rank_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Increasing the DMD rank reduces training error, but the improvement becomes very small after rank 2–3. At the same time, the spectral radius increases as weaker modes are added, although it remains below one for every rank.

    This suggests that a low-rank model may capture most of the useful structure while avoiding unnecessary modes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centering and the intercept

    Raw DMD estimates a model constrained to pass through the origin:
    \[
    x_{t+1}\approx Ax_t.
    \]
    That is a no-intercept VAR. If the factor returns have a nonzero sample mean, this restriction can force the matrix \(A\) to explain both the average level and the dynamic relationship.
    To allow for a constant level, we center the predictor and target snapshots separately:
    \[
    \bar{x}=\frac{1}{m}\sum_{t=1}^{m}x_t,
    \qquad
    \bar{y}=\frac{1}{m}\sum_{t=1}^{m}x_{t+1}.
    \]
    We then fit the centered relation
    \[
    Y-\bar y\mathbf{1}^\top
    \approx
    A_c\left(X-\bar x\mathbf{1}^\top\right),
    \]
    with
    \[
    A_c=
    (Y-\bar y\mathbf{1}^\top)
    (X-\bar x\mathbf{1}^\top)^\dagger.
    \]
    Transforming back to the original coordinates gives
    \[
    x_{t+1}\approx A_cx_t+c,
    \]
    where
    \[
    c=\bar y-A_c\bar x.
    \]
    The vector \(c\) is the intercept and it represents the average one-step shift that cannot be explained by the current factor state.
    """)
    return


@app.cell
def _(A_dmd, X, Y, dmd, fit_dmd_centered, np, pd):
    centered_dmd = fit_dmd_centered(X, Y, rcond=dmd["rcond"])

    _design = np.column_stack([np.ones(X.shape[1]), X.T])

    _coef = np.linalg.lstsq(_design, Y.T, rcond=dmd["rcond"])[0]

    _A, _c = centered_dmd["A"], centered_dmd["c"]

    centered_table = pd.DataFrame([
        {"model": _name,
         "training_mse": np.mean((Y - (_operator @ X + _intercept[:, None])) ** 2),
         "slope_difference_vs_intercept_var": np.max(np.abs(_operator - _coef[1:].T)),
         "intercept_difference_vs_intercept_var": np.max(np.abs(_intercept - _coef[0]))}
        for _name, _operator, _intercept in [
            ("Raw DMD", A_dmd, np.zeros(X.shape[0])),
            ("Centered DMD", _A, _c),
            ("Intercept VAR", _coef[1:].T, _coef[0]),
        ]
    ])
    assert np.allclose(_A, _coef[1:].T) and np.allclose(_c, _coef[0])
    centered_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Centering the predictor and target snapshots separates the average return level from the dynamic component. In the full-rank case, centered DMD is equivalent to a VAR with an intercept, while raw DMD corresponds to a no-intercept model.

    This distinction matters because forcing the dynamics through the origin can make the operator absorb the nonzero mean. The numerical equality with the intercept VAR confirms that the centered-DMD implementation is correct.
    """)
    return


if __name__ == "__main__":
    app.run()
