import numpy as np
from src.generator import generate_coupled_pair, generate_multivariate, add_artifact

def test_generate_coupled_pair_shape():
    x, y = generate_coupled_pair(500, lag=0.1, fs=100.0)
    assert x.shape == (500,)
    assert y.shape == (500,)

def test_multivariate_shape():
    coupling = np.eye(3)
    lags = np.zeros(3)
    X = generate_multivariate(3, 1000, coupling, lags, fs=100.0)
    assert X.shape == (1000, 3)

def test_add_artifact_spike_changes_signal():
    x = np.zeros(200)
    x_art = add_artifact(x, 'spike', intensity=10.0)
    assert not np.allclose(x, x_art)
    assert np.max(np.abs(x_art)) == 10.0
