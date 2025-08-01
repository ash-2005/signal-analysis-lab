import numpy as np
import scipy.signal

def bandpass_filter(
    x: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a 1-D signal.

    Zero-phase filtering (via scipy.signal.filtfilt) is used to avoid
    introducing phase distortion, which would corrupt lag estimates.

    Args:
        x: 1-D input signal.
        lowcut: Lower cutoff frequency in Hz. Must be > 0.
        highcut: Upper cutoff frequency in Hz. Must be < fs/2.
        fs: Sampling frequency in Hz.
        order: Order of the Butterworth filter. Default 4.

    Returns:
        x_filtered: 1-D filtered signal, same length as x.

    Raises:
        ValueError: If lowcut >= highcut.
        ValueError: If highcut >= fs / 2.
        ValueError: If lowcut <= 0.
    """
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0")
    if highcut >= fs / 2:
        raise ValueError("highcut must be < fs/2")
    if lowcut >= highcut:
        raise ValueError("lowcut must be < highcut")

    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='band')
    x_filtered = scipy.signal.filtfilt(b, a, x)

    return x_filtered


def envelope(
    x: np.ndarray,
    fs: float = 1.0,
) -> np.ndarray:
    """Extract the amplitude envelope of a signal using the Hilbert transform.

    Args:
        x: 1-D signal array. Works best on bandpass-filtered input.
        fs: Sampling frequency in Hz

    Returns:
        env: 1-D array of non-negative envelope values, same length as x.
    """
    analytic = scipy.signal.hilbert(x)
    env = np.abs(analytic)
    return env


def detrend_polynomial(
    x: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Remove a polynomial trend from a 1-D signal.

    Args:
        x: 1-D signal array.
        order: Degree of the polynomial to fit and remove. Default 1 (linear).
            Use order=0 to remove only the mean.

    Returns:
        x_detrended: 1-D array with polynomial trend subtracted.
    """
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, order)
    trend = np.polyval(coeffs, t)
    return x - trend


def zscore_normalize(
    x: np.ndarray,
) -> np.ndarray:
    """Z-score normalise a 1-D signal to zero mean and unit standard deviation.

    Args:
        x: 1-D signal array.

    Returns:
        x_norm: 1-D normalised array. If std(x) == 0, returns array of zeros.
    """
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        return np.zeros_like(x)
    return (x - mu) / sigma


def decompose_pipeline(
    x: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    poly_order: int = 1,
) -> dict:
    """Run the full sequential decompose pipeline on a 1-D signal.

    Order of operations:
    1. Polynomial detrending
    2. Bandpass filtering
    3. Z-score normalisation
    4. Envelope extraction (applied to the normalised filtered signal)

    Args:
        x: 1-D input signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower bandpass cutoff in Hz.
        highcut: Upper bandpass cutoff in Hz.
        poly_order: Polynomial detrend order. Default 1.

    Returns:
        result: dict with keys:
            'detrended': signal after polynomial detrending.
            'filtered': signal after bandpass filter.
            'normalised': signal after z-score.
            'envelope': amplitude envelope of the normalised signal.
    """
    detrended = detrend_polynomial(x, poly_order)
    filtered = bandpass_filter(detrended, lowcut, highcut, fs)
    normalised = zscore_normalize(filtered)
    env = envelope(normalised, fs)

    return {
        'detrended': detrended,
        'filtered': filtered,
        'normalised': normalised,
        'envelope': env,
    }
