import numpy as np
import scipy.signal
try:
    from scipy.fft import rfft, rfftfreq
except ImportError:
    from numpy.fft import rfft, rfftfreq

def welch_psd(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int = None,
) -> dict:
    """Estimate power spectral density using Welch's method.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment for FFT. Default 256.
        noverlap: Number of overlapping samples between segments.
            Default is nperseg // 2.

    Returns:
        result: dict with keys:
            'freqs': 1-D array of frequencies in Hz (one-sided).
            'psd': 1-D array of power spectral density values.
            'units': str, 'V^2/Hz' (placeholder — actual units depend on input).

    Example:
        >>> result = welch_psd(x, fs=100.0, nperseg=128)
        >>> len(result['freqs']) == len(result['psd'])
        True
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = scipy.signal.welch(x, fs=fs, nperseg=nperseg,
                                      noverlap=noverlap, window='hann')
    return {'freqs': freqs, 'psd': psd, 'units': 'V^2/Hz'}


def multitaper_psd(
    x: np.ndarray,
    fs: float = 1.0,
    n_tapers: int = 5,
) -> dict:
    """Estimate PSD using the multitaper method with Slepian (DPSS) windows.

    The multitaper method reduces spectral leakage by averaging across multiple
    orthogonal tapers, each providing an independent spectral estimate.

    Args:
        x: 1-D signal array. Length should be at least 4 * n_tapers.
        fs: Sampling frequency in Hz.
        n_tapers: Number of Slepian tapers to use. Default 5.

    Returns:
        result: dict with keys:
            'freqs': 1-D array of frequencies in Hz.
            'psd': 1-D array of averaged PSD values.
            'psd_per_taper': (n_tapers, n_freqs) array of individual taper PSDs.

    Example:
        >>> result = multitaper_psd(x, fs=100.0, n_tapers=5)
    """
    n = len(x)
    # Compute DPSS (discrete prolate spheroidal sequences)
    tapers, _ = scipy.signal.windows.dpss(n, NW=n_tapers/2, Kmax=n_tapers, return_ratios=True)
    
    # Apply each taper and compute FFT
    n_freqs = n // 2 + 1
    psd_per_taper = np.zeros((n_tapers, n_freqs))

    for k, taper in enumerate(tapers):
        x_tapered = x * taper
        X = rfft(x_tapered)
        psd_k = (np.abs(X) ** 2) / (fs * n)
        if n % 2 == 0:
            psd_k[1:-1] *= 2   # double for one-sided
        else:
            psd_k[1:] *= 2
        psd_per_taper[k] = psd_k

    psd = np.mean(psd_per_taper, axis=0)
    freqs = rfftfreq(n, d=1.0/fs)

    return {'freqs': freqs, 'psd': psd, 'psd_per_taper': psd_per_taper}


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    method: str = 'welch',
) -> dict:
    """Compute magnitude-squared coherence between two signals.

    Coherence measures the degree of linear relationship between x and y
    as a function of frequency, ranging from 0 (no relationship) to 1
    (perfect linear relationship at that frequency).

    Args:
        x: 1-D reference signal.
        y: 1-D second signal. Same length as x.
        fs: Sampling frequency in Hz.
        method: One of 'welch' or 'multitaper'. Default 'welch'.

    Returns:
        result: dict with keys:
            'freqs': 1-D array of frequencies in Hz.
            'coherence': 1-D array of coherence values in [0, 1].
            'method': str, the method used.

    Raises:
        ValueError: If method is not 'welch' or 'multitaper'.

    Example:
        >>> result = coherence(x, y, fs=100.0, method='welch')
        >>> all(0 <= c <= 1 for c in result['coherence'])
        True
    """
    if method == 'welch':
        freqs, coh = scipy.signal.coherence(x, y, fs=fs, nperseg=min(256, len(x)//4))
    elif method == 'multitaper':
        # compute cross-spectrum and auto-spectra using multitaper, then ratio
        Sxx = multitaper_psd(x, fs=fs)['psd']
        Syy = multitaper_psd(y, fs=fs)['psd']
        freqs = multitaper_psd(x, fs=fs)['freqs']
        # cross-spectrum (simplified — use Welch's cross-spectral estimate)
        _, Sxy = scipy.signal.csd(x, y, fs=fs, nperseg=min(256, len(x)//4))
        
        # interpolate Sxy to match frequencies if lengths are different
        if len(freqs) != len(Sxy):
            f_welch = scipy.signal.csd(x, y, fs=fs, nperseg=min(256, len(x)//4))[0]
            Sxy_mag = np.interp(freqs, f_welch, np.abs(Sxy))
        else:
            Sxy_mag = np.abs(Sxy)
            
        coh = (Sxy_mag**2) / (Sxx * Syy + 1e-20)
        coh = np.clip(coh, 0, 1)
    else:
        raise ValueError("Method must be 'welch' or 'multitaper'")

    return {'freqs': freqs, 'coherence': coh, 'method': method}


def spectral_summary(
    x: np.ndarray,
    fs: float = 1.0,
) -> dict:
    """Compute summary statistics of the power spectrum of a 1-D signal.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency in Hz.

    Returns:
        summary: dict with keys:
            'peak_freq': float, frequency in Hz at which PSD is maximum.
            'bandwidth': float, 3 dB bandwidth around the peak frequency in Hz.
            'total_power': float, total power (integral of PSD over all freqs).
            'dominant_band': str, one of 'low' (<1 Hz), 'mid' (1-10 Hz), 'high' (>10 Hz).

    Example:
        >>> summary = spectral_summary(x, fs=100.0)
        >>> 'peak_freq' in summary
        True
    """
    result = welch_psd(x, fs=fs)
    freqs = result['freqs']
    psd = result['psd']

    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    peak_power = psd[peak_idx]

    # 3 dB bandwidth: find frequencies where psd >= peak_power / 2
    half_power = peak_power / 2
    above = np.where(psd >= half_power)[0]
    bw = freqs[above[-1]] - freqs[above[0]] if len(above) > 1 else 0.0

    total_power = np.trapz(psd, freqs)

    if peak_freq < 1.0:
        band = 'low'
    elif peak_freq <= 10.0:
        band = 'mid'
    else:
        band = 'high'

    return {
        'peak_freq': peak_freq,
        'bandwidth': bw,
        'total_power': total_power,
        'dominant_band': band,
    }
