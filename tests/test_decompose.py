import numpy as np
from src.decompose import bandpass_filter, envelope, detrend_polynomial, zscore_normalize, decompose_pipeline
from src.generator import generate_coupled_pair

def test_bandpass_attenuates_out_of_band():
    fs = 1000.0
    t = np.arange(0, 5, 1/fs)
    x = np.sin(2 * np.pi * 50.0 * t)  # 50 Hz pure tone
    
    x_filtered = bandpass_filter(x, lowcut=1.0, highcut=10.0, fs=fs)
    assert np.std(x_filtered) < 0.01 * np.std(x)

def test_envelope_non_negative():
    x = np.random.normal(size=500)
    env = envelope(x, fs=100.0)
    assert np.all(env >= 0.0)

def test_detrend_removes_linear():
    t = np.linspace(0, 10, 500)
    x = 3.0 * t + 1.0
    detrended = detrend_polynomial(x, order=1)
    
    assert abs(np.mean(detrended)) < 1e-6
    assert np.std(detrended) < 0.01

def test_zscore_statistics():
    x = np.random.normal(loc=5.0, scale=2.0, size=500)
    x_norm = zscore_normalize(x)
    
    assert abs(np.mean(x_norm)) < 1e-10
    assert abs(np.std(x_norm) - 1.0) < 1e-10

def test_decompose_pipeline_output_keys():
    x, _ = generate_coupled_pair(500, lag=0.1, fs=100.0)
    result = decompose_pipeline(x, fs=100.0, lowcut=0.5, highcut=20.0)
    assert set(result.keys()) == {'detrended', 'filtered', 'normalised', 'envelope'}
