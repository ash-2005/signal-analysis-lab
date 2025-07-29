import numpy as np
from src.spectral import welch_psd, multitaper_psd, coherence, spectral_summary

def test_welch_peak_frequency():
    fs = 100.0
    t = np.linspace(0, 10, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5.0 * t)
    
    result = spectral_summary(x, fs=fs)
    assert abs(result['peak_freq'] - 5.0) < 0.5

def test_coherence_range():
    np.random.seed(42)
    x = np.random.normal(size=500)
    y = x + np.random.normal(scale=0.5, size=500)
    
    res = coherence(x, y, fs=50.0, method='welch')
    coh = res['coherence']
    assert np.all((coh >= 0.0) & (coh <= 1.0))

def test_multitaper_shape_matches_welch():
    np.random.seed(42)
    x = np.random.normal(size=1000)
    
    welch_res = welch_psd(x, fs=100.0)
    mt_res = multitaper_psd(x, fs=100.0)
    
    assert len(welch_res['freqs']) == len(mt_res['freqs'])

def test_spectral_summary_keys():
    np.random.seed(42)
    x = np.random.normal(size=500)
    
    summary = spectral_summary(x, fs=100.0)
    expected_keys = {'peak_freq', 'bandwidth', 'total_power', 'dominant_band'}
    assert expected_keys.issubset(summary.keys())
