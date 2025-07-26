import numpy as np
from src.generator import generate_coupled_pair, generate_multivariate
from src.crosscorr import lagged_xcorr, peak_lag, bootstrap_ci, permutation_test, xcorr_matrix

def test_zero_lag_returns_zero():
    x, y = generate_coupled_pair(1000, lag=0.0, noise_std=0.0, fs=100.0)
    res = lagged_xcorr(x, y, max_lag=1.0, fs=100.0)
    lag, _ = peak_lag(res)
    assert abs(lag) < 0.05

def test_known_lag_recovery():
    x, y = generate_coupled_pair(2000, lag=0.3, noise_std=0.05, fs=100.0)
    res = lagged_xcorr(x, y, max_lag=1.0, fs=100.0)
    lag, _ = peak_lag(res)
    assert abs(lag - 0.3) < 0.02

def test_negative_lag():
    x, y = generate_coupled_pair(2000, lag=0.2, noise_std=0.0, fs=100.0)
    # y is lagged relative to x by 0.2. So x is lagging y by -0.2
    res = lagged_xcorr(y, x, max_lag=1.0, fs=100.0)
    lag, _ = peak_lag(res)
    assert abs(lag - (-0.2)) < 0.02

def test_all_zero_input():
    x = np.zeros(500)
    y = np.zeros(500)
    res = lagged_xcorr(x, y, max_lag=1.0, fs=100.0)
    assert np.allclose(res['correlation'], 0.0)

def test_bootstrap_ci_contains_true_lag():
    x, y = generate_coupled_pair(2000, lag=0.25, noise_std=0.05, fs=100.0)
    ci = bootstrap_ci(x, y, max_lag=1.0, n_bootstrap=100, fs=100.0)
    assert ci['ci_lower'] <= 0.25 <= ci['ci_upper']

def test_permutation_test_significant():
    x, y = generate_coupled_pair(2000, lag=0.1, noise_std=0.01, fs=100.0)
    ptest = permutation_test(x, y, max_lag=1.0, n_permutations=100, fs=100.0)
    assert ptest['p_value'] < 0.05

def test_xcorr_matrix_shape():
    X = generate_multivariate(4, 1000, np.eye(4), np.zeros(4), fs=100.0)
    result = xcorr_matrix(X, max_lag=1.0, fs=100.0)
    assert result['lag_matrix'].shape == (4, 4)
    assert result['correlation_matrix'].shape == (4, 4)
