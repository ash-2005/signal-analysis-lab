import numpy as np
from src.lag_matrix import compute_lag_matrix, summarize_lag_distribution, lag_corrected_correlation
from src.crosscorr import xcorr_matrix
from src.generator import generate_multivariate, generate_coupled_pair

def test_diagonal_is_zero():
    X = generate_multivariate(3, 500, np.eye(3), [0.0, 0.1, 0.2], fs=100.0)
    L = compute_lag_matrix(X, max_lag=1.0, fs=100.0)
    assert np.allclose(np.diag(L), 0.0)

def test_antisymmetry():
    X = generate_multivariate(4, 500, np.eye(4), [0.0, 0.1, 0.2, -0.1], fs=100.0)
    L = compute_lag_matrix(X, max_lag=1.0, fs=100.0)
    assert np.allclose(L, -L.T)

def test_lag_corrected_correlation_improves():
    x, y = generate_coupled_pair(2000, lag=0.3, noise_std=0.01, fs=100.0)
    X = np.column_stack([x, y])
    L = compute_lag_matrix(X, max_lag=1.0, fs=100.0)
    
    C_uncorrected = np.corrcoef(X.T)
    C_corrected = lag_corrected_correlation(X, L, fs=100.0)
    
    assert C_corrected[0, 1] >= C_uncorrected[0, 1] - 0.01

def test_summarize_lag_n_pairs():
    # 4 channels -> 4C2 = 6 unique pairs
    X = generate_multivariate(4, 500, np.eye(4), np.zeros(4), fs=100.0)
    L = compute_lag_matrix(X, max_lag=1.0, fs=100.0)
    summary = summarize_lag_distribution(L)
    assert summary['n_pairs'] == 6
