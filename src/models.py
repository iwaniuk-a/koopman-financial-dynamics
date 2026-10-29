"""One-step forecasters and DMD/EDMD operators.

Forecasters see only a training window ending at the forecast origin. Window
rows are observations, columns are factors; multi-parameter predictions have
shape (n_settings, n_factors). DMD instead takes snapshots as columns.
"""

import numpy as np


def _validate_window(window, min_rows=2):
    w = np.asarray(window, dtype=float)
    if w.ndim != 2 or w.shape[0] < min_rows or w.shape[1] == 0:
        raise ValueError(f"Expected a 2D window with at least {min_rows} rows and one factor")
    if not np.isfinite(w).all():
        raise ValueError("The window must contain only finite returns")
    return w


# Linear baselines

def predict_zero(window):
    """Return a zero next-day forecast, shape (n_factors,)."""
    return np.zeros(_validate_window(window).shape[1])


def predict_mean(window):
    """Forecast the mean of transition targets window[1:], shape (n_factors,)."""
    return _validate_window(window)[1:].mean(axis=0)


def fit_ridge_var(window, taus):
    """Fit VAR(1) in window-standardized coordinates with an unpenalized intercept.

    The objective is (1/n)||Yc - Xc B||_F^2 + tau ||B||_F^2, n = len(window)-1.
    Return scaling, transition means, last state, and B of shape (n_tau, d, d).
    """
    w = _validate_window(window)
    n = w.shape[0] - 1
    mu = w.mean(axis=0)
    sd = w.std(axis=0, ddof=1)
    z = (w - mu) / sd
    X, Y = z[:-1], z[1:]
    xbar, ybar = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - xbar, Y - ybar
    C = Xc.T @ Xc / n
    G = Xc.T @ Yc / n
    lam, V = np.linalg.eigh(C)
    VtG = V.T @ G
    taus = np.atleast_1d(np.asarray(taus, float))
    B = np.stack([V @ (VtG / (lam + t)[:, None]) for t in taus])
    return {"mu": mu, "sd": sd, "xbar": xbar, "ybar": ybar, "B": B, "z_last": z[-1]}


def predict_ridge_var(window, taus):
    """Forecast next-day returns for each tau, shape (n_tau, n_factors)."""
    f = fit_ridge_var(window, taus)
    zhat = f["ybar"] + (f["z_last"] - f["xbar"]) @ f["B"]
    return f["mu"] + f["sd"] * zhat


def make_predictor(kind, taus=None):
    """Build a zero, mean, OLS, or ridge window predictor with a settings axis."""
    if kind == "zero":
        return lambda w: predict_zero(w)[None, :]
    if kind == "mean":
        return lambda w: predict_mean(w)[None, :]
    if kind == "ols":
        return lambda w: predict_ridge_var(w, [0.0])
    if kind == "ridge":
        return lambda w: predict_ridge_var(w, taus)
    raise ValueError(kind)


# DMD: snapshots are columns, Y[:, k] follows X[:, k].

def default_rcond(shape):
    """Relative SVD cutoff shared by all DMD paths."""
    return max(shape) * np.finfo(float).eps


def fit_dmd(X, Y, rank=None, rcond=None):
    """Fit an uncentered operator to equal (d, m) snapshot matrices by SVD.

    Return the full/reduced operators, retained basis, singular values, and ranks.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim != 2 or X.shape != Y.shape:
        raise ValueError("X and Y must be 2D with equal shapes")
    if rcond is None:
        rcond = default_rcond(X.shape)

    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    num_rank = int(np.sum(s > rcond * s[0]))
    if rank is None:
        rank = num_rank
    if not 1 <= rank <= num_rank:
        raise ValueError(f"rank must be in [1, {num_rank}]")

    Ur, sr, Vr = U[:, :rank], s[:rank], Vh[:rank].T
    YV = Y @ Vr / sr
    return {"A": YV @ Ur.T, "A_reduced": Ur.T @ YV, "U": Ur, "s": s,
            "rank": rank, "numerical_rank": num_rank, "rcond": rcond}


def fit_dmd_centered(X, Y, rank=None, rcond=None):
    """Fit centered DMD and recover intercept c for the next state A @ x + c."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    xbar = X.mean(axis=1, keepdims=True)
    ybar = Y.mean(axis=1, keepdims=True)
    res = fit_dmd(X - xbar, Y - ybar, rank=rank, rcond=rcond)
    res["c"] = (ybar - res["A"] @ xbar).ravel()
    return res


# EDMD dictionaries and operators

DIM = {"linear": 6, "squared_only": 6, "squares": 12, "full": 27}


def dictionary(z, kind):
    """Lift six standardized returns: linear, squares, then cross-products i<j."""
    if kind == "linear":
        return z
    sq = z ** 2
    if kind == "squared_only":
        return sq
    if kind == "squares":
        return np.column_stack([z, sq])
    if kind == "full":
        cross = np.column_stack([z[:, i] * z[:, j] for i in range(6) for j in range(i + 1, 6)])
        return np.column_stack([z, sq, cross])
    raise ValueError(kind)


def fit_edmd(window, taus, kind):
    """Fit ridge transitions of standardized polynomial features within a window.

    Return scaling, transition means, last lifted state, and (n_tau, p, p) K.
    """
    w = _validate_window(window)
    mu, sd = w.mean(0), w.std(0, ddof=1)
    z = (w - mu) / sd
    F = dictionary(z, kind)
    m, s = F.mean(0), F.std(0, ddof=1)
    P = (F - m) / s
    X, Y = P[:-1], P[1:]
    n = len(X)
    xbar, ybar = X.mean(0), Y.mean(0)
    Xc, Yc = X - xbar, Y - ybar
    C, G = Xc.T @ Xc / n, Xc.T @ Yc / n
    lam, V = np.linalg.eigh(C)
    VtG = V.T @ G
    taus = np.atleast_1d(np.asarray(taus, float))
    K = np.stack([V @ (VtG / (lam + t)[:, None]) for t in taus])
    return {"mu": mu, "sd": sd, "m": m, "s": s, "xbar": xbar, "ybar": ybar,
            "K": K, "p_last": P[-1]}


def predict_edmd(window, taus, kind):
    """Forecast next-day returns from linear lifted coordinates, (n_tau, 6)."""
    f = fit_edmd(window, taus, kind)
    yhat = f["ybar"] + (f["p_last"] - f["xbar"]) @ f["K"]
    zhat = f["m"][:6] + f["s"][:6] * yhat[:, :6]
    return f["mu"] + f["sd"] * zhat


def predict_edmd_second_moment(window, taus, kind="full"):
    """Forecast next-day raw squared returns, (n_tau, 6), without clipping."""
    if kind not in ("squares", "full"):
        raise ValueError("Second-moment forecasting requires quadratic features")
    f = fit_edmd(window, taus, kind)
    y_hat = f["ybar"] + (f["p_last"] - f["xbar"]) @ f["K"]
    zhat = f["m"][:6] + f["s"][:6] * y_hat[:, :6]
    z2hat = f["m"][6:12] + f["s"][6:12] * y_hat[:, 6:12]
    mu, sd = f["mu"], f["sd"]
    return mu**2 + 2 * mu * sd * zhat + sd**2 * z2hat


def make_edmd_predictor(taus, kind):
    """Build a next-day signed-return EDMD window predictor."""
    return lambda w: predict_edmd(w, taus, kind)


def make_edmd_second_moment_predictor(taus, kind="full"):
    """Build a next-day squared-return EDMD window predictor."""
    return lambda w: predict_edmd_second_moment(w, taus, kind)


# Multiscale volatility

def multiscale_features(returns, horizons=(1, 5, 22), asymmetric=False):
    """Return trailing squared-return averages through each row, by horizon.

    Missing history is NaN. Optional negative-shock averages follow the symmetric
    blocks; output has d * len(horizons) columns, doubled when asymmetric.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("Expected a 2D returns array")
    squared = r**2
    negative_squared = squared * (r < 0)

    def rolling_averages(values, horizon):
        cumulative = np.vstack([
            np.zeros((1, values.shape[1])),
            np.cumsum(values, axis=0),
        ])
        averages = (cumulative[horizon:] - cumulative[:-horizon]) / horizon
        padded = np.full_like(values, np.nan)
        padded[horizon - 1:] = averages
        return padded

    features = [rolling_averages(squared, h) for h in horizons]
    if asymmetric:
        features += [rolling_averages(negative_squared, h) for h in horizons]
    return np.column_stack(features)


def predict_har_ridge(window, taus, asymmetric=False, horizons=(1, 5, 22)):
    """Fit multivariate HAR on trailing features to forecast next-day squares.

    Training pairs use features through s to predict window[s+1]**2. Return
    (n_tau, n_factors) forecasts from the final window row, without clipping.
    """
    w = _validate_window(window, min_rows=max(horizons) + 2)
    features = multiscale_features(w, horizons=horizons, asymmetric=asymmetric)
    start = max(horizons) - 1
    X = features[start:-1]
    Y = w[start + 1:] ** 2
    x_last = features[-1]
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=1)
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0, ddof=1)
    Xz = (X - x_mean) / x_std
    Yz = (Y - y_mean) / y_std
    xz_last = (x_last - x_mean) / x_std
    C = Xz.T @ Xz / len(Xz)
    G = Xz.T @ Yz / len(Xz)
    lam, V = np.linalg.eigh(C)
    VtG = V.T @ G
    taus = np.atleast_1d(np.asarray(taus, float))
    B = np.stack([V @ (VtG / (lam + tau)[:, None]) for tau in taus])
    forecasts_z = xz_last @ B
    return y_mean + y_std * forecasts_z


def make_har_predictor(taus, asymmetric=False, horizons=(1, 5, 22)):
    """Build a multivariate next-day squared-return HAR window predictor."""
    return lambda w: predict_har_ridge(w, taus, asymmetric=asymmetric, horizons=horizons)


def predict_univariate_har_ridge(window, taus, horizons=(1, 5, 22)):
    """Forecast next-day squares using each factor's own trailing HAR features.

    Return (n_tau, n_factors) forecasts without clipping; fit only in the window.
    """
    w = _validate_window(window, min_rows=max(horizons) + 2)
    features = multiscale_features(w, horizons=horizons)
    n_factors = w.shape[1]
    n_horizons = len(horizons)
    start = max(horizons) - 1
    X = features[start:-1]
    Y = w[start + 1:] ** 2
    x_last = features[-1]
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    forecasts = np.empty((len(taus), n_factors))
    for j in range(n_factors):
        columns = [h * n_factors + j for h in range(n_horizons)]
        Xj = X[:, columns]
        yj = Y[:, j]
        last_j = x_last[columns]
        x_mean = Xj.mean(axis=0)
        x_std = Xj.std(axis=0, ddof=1)
        y_mean = yj.mean()
        y_std = yj.std(ddof=1)
        Xz = (Xj - x_mean) / x_std
        yz = (yj - y_mean) / y_std
        last_z = (last_j - x_mean) / x_std
        C = Xz.T @ Xz / len(Xz)
        g = Xz.T @ yz / len(Xz)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        projected_g = eigenvectors.T @ g
        coefficients = np.stack([
            eigenvectors @ (projected_g / (eigenvalues + tau)) for tau in taus
        ])
        forecasts[:, j] = y_mean + y_std * (coefficients @ last_z)
    return forecasts


def make_univariate_har_predictor(taus, horizons=(1, 5, 22)):
    """Build a factor-by-factor next-day squared-return HAR window predictor."""
    return lambda window: predict_univariate_har_ridge(window, taus, horizons=horizons)


def predict_ewma(window, lambdas):
    """Filter squared returns through the last row for next-day second moments.

    Initialize with the first squared return; return (n_lambda, n_factors).
    """
    w = _validate_window(window)
    lambdas = np.atleast_1d(np.asarray(lambdas, dtype=float))
    if np.any((lambdas <= 0) | (lambdas >= 1)):
        raise ValueError("EWMA decay parameters must be between 0 and 1")
    squared = w ** 2
    forecasts = np.empty((len(lambdas), w.shape[1]))
    for j, decay in enumerate(lambdas):
        variance = squared[0].copy()
        for observation in squared[1:]:
            variance = decay * variance + (1 - decay) * observation
        forecasts[j] = variance
    return forecasts


def make_ewma_predictor(lambdas):
    """Build a next-day squared-return EWMA window predictor."""
    return lambda window: predict_ewma(window, lambdas)


# GARCH

def rolling_garch_forecasts(returns, origins, window=1000, refit_every=20):
    """Forecast next-day second moments for consecutive origins and each factor.

    Fit GARCH(1,1) to percent returns from origin-window+1 through origin, refit
    every refit_every origins, and filter intervening windows with fixed params.
    Return decimal squared-return forecasts and zero success flags, both shaped
    (n_origins, n_factors). Raise on nonconvergence; load arch only on use.
    """
    returns = np.asarray(returns, dtype=float)
    origins = np.asarray(origins, dtype=int)
    if returns.ndim != 2:
        raise ValueError("Returns must be a 2D array")
    if np.any(np.diff(origins) != 1):
        raise ValueError("Forecast origins must be consecutive")

    from arch import arch_model

    forecasts = np.empty((len(origins), returns.shape[1]))
    fit_status = np.empty((len(origins), returns.shape[1]), dtype=int)
    for j in range(returns.shape[1]):
        fitted_params = None
        for i, t in enumerate(origins):
            start = t - window + 1
            sample = 100 * returns[start:t + 1, j]
            model = arch_model(
                sample, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                dist="normal", rescale=False,
            )
            if i % refit_every == 0:
                fit = model.fit(disp="off", show_warning=False)
                if fit.convergence_flag != 0:
                    raise RuntimeError(f"GARCH convergence failure: factor={j}, origin={t}")
                fitted_params = fit.params
            forecast = model.forecast(fitted_params, horizon=1, reindex=False)
            variance = forecast.variance.iloc[-1, 0]
            conditional_mean = fitted_params["mu"]
            forecasts[i, j] = (variance + conditional_mean**2) / 10000
            fit_status[i, j] = 0
    return forecasts, fit_status
