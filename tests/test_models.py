import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import models as M


def _window(rows=120, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, 6)) * 0.01


def test_baseline_predictor_shapes():
    window = _window()
    taus = np.array([0.01, 0.1, 1.0])
    np.testing.assert_equal(M.predict_zero(window).shape, (6,))
    np.testing.assert_equal(M.predict_mean(window).shape, (6,))
    fitted = M.fit_ridge_var(window, taus)
    assert fitted["B"].shape == (3, 6, 6)
    np.testing.assert_equal(M.predict_ridge_var(window, taus).shape, (3, 6))
    for kind, expected in (("zero", 1), ("mean", 1), ("ols", 1), ("ridge", 3)):
        forecast = M.make_predictor(kind, taus)(window)
        assert forecast.shape == (expected, 6)


def test_dictionary_dimensions():
    z = _window()
    assert M.DIM["linear"] == 6
    assert M.DIM["squares"] == 12
    assert M.DIM["full"] == 27
    for kind in ("linear", "squares", "full"):
        assert M.dictionary(z, kind).shape == (len(z), M.DIM[kind])


def test_dictionary_feature_order():
    z = np.array([[1., -2., 3., -4., 5., -6.]])
    squares = [1., 4., 9., 16., 25., 36.]
    cross = [-2., 3., -4., 5., -6., -6., 8., -10., 12.,
             -12., 15., -18., -20., 24., -30.]
    np.testing.assert_array_equal(M.dictionary(z, "squared_only"), [squares])
    np.testing.assert_array_equal(M.dictionary(z, "squares"), [list(z[0]) + squares])
    np.testing.assert_array_equal(M.dictionary(z, "full"), [list(z[0]) + squares + cross])


def test_dmd_equivalences():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 300))
    A = rng.normal(size=(6, 6)) * 0.1
    c = rng.normal(size=(6, 1))
    Y = A @ X + c + rng.normal(size=(6, 300)) * 0.01
    expected = np.linalg.lstsq(X.T, Y.T, rcond=None)[0].T
    np.testing.assert_allclose(M.fit_dmd(X, Y)["A"], expected, atol=1e-10)
    result = M.fit_dmd_centered(X, Y)
    design = np.column_stack([np.ones(X.shape[1]), X.T])
    coef = np.linalg.lstsq(design, Y.T, rcond=None)[0]
    np.testing.assert_allclose(result["A"], coef[1:].T, atol=1e-10)
    np.testing.assert_allclose(result["c"], coef[0], atol=1e-10)
    assert M.default_rcond(X.shape) > 0


def test_dmd_cutoff_and_rank_validation():
    X = np.diag([1., 1e-9, 1e-12])
    result = M.fit_dmd(X, 2 * X, rcond=1e-10)
    assert result["numerical_rank"] == result["rank"] == 2
    np.testing.assert_array_equal(result["A"], np.diag([2., 2., 0.]))
    np.testing.assert_array_equal(result["A_reduced"], 2 * np.eye(2))
    for rank in (0, 3):
        with pytest.raises(ValueError, match="rank"):
            M.fit_dmd(X, 2 * X, rank=rank, rcond=1e-10)


def test_edmd_forecasts_are_finite():
    window = _window()
    taus = [0.1, 1.0]
    for forecast in (
        M.predict_edmd(window, taus, "full"),
        M.predict_edmd_second_moment(window, taus, "full"),
        M.make_edmd_predictor(taus, "squares")(window),
        M.make_edmd_second_moment_predictor(taus)(window),
    ):
        assert forecast.shape == (2, 6)
        assert np.isfinite(forecast).all()


def test_linear_edmd_equals_ridge_var():
    window = _window()
    np.testing.assert_allclose(
        M.predict_edmd(window, [0.01, 1.0], "linear"),
        M.predict_ridge_var(window, [0.01, 1.0]),
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["squares", "full"])
def test_edmd_second_moment_matches_direct_ridge(kind):
    # Fitting raw squared-return targets directly checks both the transition
    # sign and the inverse transformation's linear/quadratic feature slices.
    window = _window() + np.linspace(-0.02, 0.03, 6)
    z = (window - window.mean(0)) / window.std(0, ddof=1)
    features = np.column_stack([z, z**2])
    if kind == "full":
        cross = [z[:, i] * z[:, j] for i in range(6) for j in range(i + 1, 6)]
        features = np.column_stack([features, *cross])
    features = (features - features.mean(0)) / features.std(0, ddof=1)
    X = features[:-1]
    Xc = X - X.mean(0)
    targets = window[1:]**2
    target_mean = targets.mean(0)
    taus = [0.01, 1.0]
    expected = []
    for tau in taus:
        coef = np.linalg.solve(
            Xc.T @ Xc + len(X) * tau * np.eye(X.shape[1]),
            Xc.T @ (targets - target_mean),
        )
        expected.append(target_mean + (features[-1] - X.mean(0)) @ coef)
    np.testing.assert_allclose(
        M.predict_edmd_second_moment(window, taus, kind), expected, atol=1e-12,
    )


def test_mean_uses_transition_targets():
    window = np.array([[100., -100.], [2., 4.], [6., 8.]])
    np.testing.assert_array_equal(M.predict_mean(window), [4., 6.])


def test_volatility_feature_dimensions_and_predictors():
    window = _window()
    symmetric = M.multiscale_features(window)
    asymmetric = M.multiscale_features(window, asymmetric=True)
    assert symmetric.shape == (len(window), 18)
    assert asymmetric.shape == (len(window), 36)
    taus = [0.1, 1.0]
    assert M.predict_har_ridge(window, taus).shape == (2, 6)
    assert M.predict_har_ridge(window, taus, asymmetric=True).shape == (2, 6)
    assert M.predict_univariate_har_ridge(window, taus).shape == (2, 6)
    assert M.make_har_predictor(taus)(window).shape == (2, 6)
    assert M.make_univariate_har_predictor(taus)(window).shape == (2, 6)
    assert M.predict_ewma(window, [0.8, 0.95]).shape == (2, 6)
    assert M.make_ewma_predictor([0.8, 0.95])(window).shape == (2, 6)





def test_multiscale_features_use_only_past_and_current_returns():
    returns = np.array([[1., -2.], [-3., 4.], [5., -6.], [7., 8.]])
    features = M.multiscale_features(returns, horizons=(1, 2), asymmetric=True)
    assert np.isnan(features[0, [2, 3, 6, 7]]).all()
    np.testing.assert_array_equal(features[1], [9., 16., 5., 10., 9., 0., 4.5, 2.])
    np.testing.assert_allclose(
        features[:3],
        M.multiscale_features(returns[:3], horizons=(1, 2), asymmetric=True),
    )


def test_ewma_forecast_timing():
    # Start at 1, update with 9 then 25: 0.5 * (0.5 * 1 + 0.5 * 9) + 0.5 * 25.
    np.testing.assert_array_equal(M.predict_ewma([[1.], [3.], [5.]], [0.5]), [[15.]])


def test_signed_return_models_import_without_arch():
    code = """
import sys
class NoArch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'arch' or fullname.startswith('arch.'):
            raise ImportError('arch unavailable')
sys.meta_path.insert(0, NoArch())
from src.models import predict_ridge_var
forecast = predict_ridge_var([[1., 2.], [2., 3.], [4., 1.]], [0.1])
assert forecast.shape == (1, 2)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr

