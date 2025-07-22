import numpy as np

def generate_coupled_pair(
    n_samples: int,
    lag: float,
    noise_std: float = 0.05,
    fs: float = 1.0,
    drift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two coupled oscillator signals with a known time lag.

    The reference signal x is a sum of two sinusoids. The lagged signal y
    is x shifted by `lag` seconds using linear interpolation, then corrupted
    with Gaussian noise. An optional linear drift is added to both channels.

    Args:
        n_samples: Number of time samples to generate.
        lag: Lag in seconds. Positive means y lags behind x. Can be fractional.
        noise_std: Standard deviation of additive Gaussian noise applied to y.
        fs: Sampling frequency in Hz. Used to convert lag (seconds) to samples.
        drift: Slope of linear drift added to both signals. Set to 0 for no drift.

    Returns:
        Tuple (x, y) where x is the reference signal and y is the lagged,
        noisy copy. Both are 1-D NumPy arrays of length n_samples.

    Raises:
        ValueError: If lag >= n_samples / fs (lag longer than signal duration).
        ValueError: If n_samples < 10.

    Example:
        >>> x, y = generate_coupled_pair(1000, lag=0.3, noise_std=0.1, fs=100.0)
        >>> x.shape, y.shape
        ((1000,), (1000,))
    """
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")
    if lag >= n_samples / fs:
        raise ValueError("lag must be less than signal duration (n_samples / fs)")

    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    x = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 2.3 * t)
    x += drift * t

    lag_samples = lag * fs
    
    if np.isclose(lag_samples, np.round(lag_samples)):
        shift = int(np.round(lag_samples))
        y_clean = np.roll(x, shift)
        if shift > 0:
            y_clean[:shift] = 0
        elif shift < 0:
            y_clean[shift:] = 0
    else:
        t_shifted = t + lag
        y_clean = np.interp(t, t_shifted, x, left=0, right=0)
        
    y = y_clean + np.random.normal(0, noise_std, n_samples)
    y += drift * t
    
    return x, y


def generate_multivariate(
    n_channels: int,
    n_samples: int,
    coupling_matrix: np.ndarray,
    lag_vector: np.ndarray,
    noise_std: float = 0.05,
    fs: float = 1.0,
) -> np.ndarray:
    """Generate an N-channel multivariate time series with prescribed coupling and lags.

    Channel 0 is the reference oscillator. Channel i receives a lagged, noise-
    corrupted copy of Channel 0, scaled by coupling_matrix[0, i]. The lag for
    Channel i is lag_vector[i].

    Args:
        n_channels: Number of output channels (columns).
        n_samples: Number of time samples per channel.
        coupling_matrix: (n_channels, n_channels) array. coupling_matrix[i, j]
            is the coupling weight from channel i to channel j. Only row 0 is
            currently used (all channels coupled to the reference).
        lag_vector: 1-D array of length n_channels. lag_vector[i] is the lag
            in seconds for channel i. lag_vector[0] must be 0.0.
        noise_std: Gaussian noise standard deviation applied per channel.
        fs: Sampling frequency in Hz.

    Returns:
        X: (n_samples, n_channels) NumPy array. Column i is channel i.

    Raises:
        ValueError: If coupling_matrix shape does not match (n_channels, n_channels).
        ValueError: If lag_vector length does not equal n_channels.
        ValueError: If lag_vector[0] != 0.0.

    Example:
        >>> lags = np.array([0.0, 0.2, 0.4])
        >>> coupling = np.eye(3)
        >>> X = generate_multivariate(3, 500, coupling, lags, fs=100.0)
        >>> X.shape
        (500, 3)
    """
    if coupling_matrix.shape != (n_channels, n_channels):
        raise ValueError("coupling_matrix shape must be (n_channels, n_channels)")
    if len(lag_vector) != n_channels:
        raise ValueError("lag_vector length must equal n_channels")
    if lag_vector[0] != 0.0:
        raise ValueError("lag_vector[0] must be 0.0")

    x_ref, _ = generate_coupled_pair(n_samples, lag=0.0, noise_std=0.0, fs=fs)
    X = np.zeros((n_samples, n_channels))
    X[:, 0] = x_ref

    for i in range(1, n_channels):
        coupling_weight = coupling_matrix[0, i]
        _, y = generate_coupled_pair(
            n_samples, lag=lag_vector[i], noise_std=noise_std, fs=fs
        )
        X[:, i] = coupling_weight * y

    return X


def add_artifact(
    signal: np.ndarray,
    artifact_type: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """Inject a synthetic artifact into a 1-D signal.

    Args:
        signal: 1-D input signal array. Not modified in place.
        artifact_type: One of 'spike', 'step', 'dropout'.
            'spike': Single-sample impulse at a random location.
            'step': Step discontinuity at the signal midpoint.
            'dropout': A 50-sample segment set to zero at a random location.
        intensity: Amplitude scaling of the artifact. For 'spike', intensity
            is the spike amplitude. For 'step', it is the step height.
            For 'dropout', intensity is ignored.

    Returns:
        signal_out: 1-D NumPy array with artifact injected. Same length as input.

    Raises:
        ValueError: If artifact_type is not one of the three allowed values.

    Example:
        >>> x = np.sin(np.linspace(0, 2*np.pi, 200))
        >>> x_art = add_artifact(x, 'spike', intensity=5.0)
    """
    out = np.copy(signal)
    n = len(signal)

    if artifact_type == 'spike':
        if n > 20:
            idx = np.random.randint(10, n - 10)
        else:
            idx = n // 2
        out[idx] += intensity

    elif artifact_type == 'step':
        out[n // 2:] += intensity

    elif artifact_type == 'dropout':
        if n > 60:
            start = np.random.randint(10, n - 60)
            out[start:start + 50] = 0.0
        else:
            out[:n // 2] = 0.0

    else:
        raise ValueError("artifact_type must be 'spike', 'step', or 'dropout'")

    return out
