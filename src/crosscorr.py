import numpy as np
import scipy.signal
from tqdm import tqdm

def lagged_xcorr(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: float,
    fs: float = 1.0,
) -> dict:
    """Compute the normalised cross-correlation between x and y across a lag range.

    Args:
        x: 1-D reference signal array.
        y: 1-D lagged signal array. Must be same length as x.
        max_lag: Maximum lag to consider, in seconds. The function computes
            correlation for lags in [-max_lag, +max_lag].
        fs: Sampling frequency in Hz. Used to convert lag samples to seconds.

    Returns:
        result: dict with keys:
            'lags': 1-D array of lag values in seconds.
            'correlation': 1-D array of normalised correlation coefficients.
            'max_lag_samples': int, the number of lag samples corresponding to max_lag.

    Raises:
        ValueError: If x and y have different lengths.
        ValueError: If max_lag * fs > len(x) / 2.

    Example:
        >>> result = lagged_xcorr(x, y, max_lag=1.0, fs=100.0)
        >>> result['lags'].shape == result['correlation'].shape
        True
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if max_lag * fs > len(x) / 2:
        raise ValueError("max_lag * fs must be <= len(x) / 2")

    max_lag_samples = int(np.round(max_lag * fs))

    # z-score both signals to remove amplitude scaling from correlation
    x_z = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_z = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # full correlation via FFT for efficiency
    full_corr = scipy.signal.correlate(x_z, y_z, mode='full', method='fft')
    # normalise by n (unbiased)
    full_corr /= len(x)

    # centre of full_corr corresponds to zero lag
    centre = len(x) - 1
    start_idx = centre - max_lag_samples
    end_idx = centre + max_lag_samples + 1
    
    corr_window = full_corr[start_idx:end_idx]
    lags = np.arange(-max_lag_samples, max_lag_samples + 1) / fs

    return {'lags': lags, 'correlation': corr_window, 'max_lag_samples': max_lag_samples}


def peak_lag(xcorr_result: dict) -> tuple[float, float]:
    """Extract the lag at maximum absolute correlation from a lagged_xcorr result.

    Args:
        xcorr_result: dict returned by lagged_xcorr, containing 'lags' and
            'correlation' arrays.

    Returns:
        Tuple (lag, correlation_at_peak) where lag is in seconds and
        correlation_at_peak is the normalised correlation value at that lag.

    Example:
        >>> result = lagged_xcorr(x, y, max_lag=1.0, fs=100.0)
        >>> lag, corr = peak_lag(result)
    """
    lags = xcorr_result['lags']
    corr = xcorr_result['correlation']
    idx = np.argmax(np.abs(corr))
    return lags[idx], corr[idx]


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: float,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    fs: float = 1.0,
) -> dict:
    """Estimate confidence interval on the peak lag using block bootstrap.

    Block bootstrap preserves autocorrelation structure within each signal.
    Block size is set to max(10, int(fs)) samples. Each bootstrap iteration
    resamples both signals with the same block indices (to preserve coupling).

    Args:
        x: 1-D reference signal.
        y: 1-D lagged signal. Same length as x.
        max_lag: Maximum lag in seconds, passed to lagged_xcorr.
        n_bootstrap: Number of bootstrap iterations. Default 500.
        confidence: Confidence level for the interval. Default 0.95.
        fs: Sampling frequency in Hz.

    Returns:
        result: dict with keys:
            'peak_lag': float, lag at peak correlation on original data.
            'ci_lower': float, lower bound of confidence interval (seconds).
            'ci_upper': float, upper bound of confidence interval (seconds).
            'bootstrap_lags': 1-D array of all n_bootstrap peak lag estimates.

    Example:
        >>> ci = bootstrap_ci(x, y, max_lag=1.0, n_bootstrap=500, fs=100.0)
        >>> ci['ci_lower'] <= ci['peak_lag'] <= ci['ci_upper']
        True
    """
    block_size = max(10, int(fs))
    n = len(x)
    n_blocks = n // block_size
    alpha = 1.0 - confidence

    original_result = lagged_xcorr(x, y, max_lag, fs)
    original_lag, _ = peak_lag(original_result)

    bootstrap_lags = []
    # Optionally suppress tqdm in tests by disabling or handling it wrapper
    for _ in range(n_bootstrap):
        # sample block indices with replacement
        chosen_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        idx_lists = [np.arange(b * block_size, (b + 1) * block_size) for b in chosen_blocks]
        idx = np.concatenate(idx_lists)
        idx = idx[:n]  # trim to n if needed
        
        x_boot = x[idx]
        y_boot = y[idx]
        
        res = lagged_xcorr(x_boot, y_boot, max_lag, fs)
        lag_b, _ = peak_lag(res)
        bootstrap_lags.append(lag_b)

    bootstrap_lags = np.array(bootstrap_lags)
    ci_lower = np.percentile(bootstrap_lags, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_lags, 100 * (1 - alpha / 2))

    return {
        'peak_lag': original_lag,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_lags': bootstrap_lags,
    }


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: float,
    n_permutations: int = 1000,
    fs: float = 1.0,
) -> dict:
    """Test whether the observed peak correlation is significant via permutation.

    The null distribution is built by randomly circular-shifting y and
    recomputing peak correlation. The p-value is the fraction of null-distribution
    peak correlations exceeding the observed peak.

    Args:
        x: 1-D reference signal.
        y: 1-D lagged signal.
        max_lag: Maximum lag in seconds.
        n_permutations: Number of permutation iterations. Default 1000.
        fs: Sampling frequency in Hz.

    Returns:
        result: dict with keys:
            'observed_peak': float, observed peak |correlation|.
            'p_value': float, two-tailed p-value.
            'null_distribution': 1-D array of null-distribution peak correlations.

    Example:
        >>> ptest = permutation_test(x, y, max_lag=1.0, n_permutations=1000, fs=100.0)
        >>> ptest['p_value'] < 0.05  # expect True for strongly coupled signals
        True
    """
    original_result = lagged_xcorr(x, y, max_lag, fs)
    _, observed_peak = peak_lag(original_result)
    observed_peak = abs(observed_peak)

    null_peaks = []
    n = len(x)
    for _ in range(n_permutations):
        shift = np.random.randint(n // 4, 3 * n // 4)
        y_shifted = np.roll(y, shift)   # circular shift preserves autocorrelation
        res = lagged_xcorr(x, y_shifted, max_lag, fs)
        _, null_peak = peak_lag(res)
        null_peaks.append(abs(null_peak))

    null_peaks = np.array(null_peaks)
    p_value = np.mean(null_peaks >= observed_peak)

    return {
        'observed_peak': observed_peak,
        'p_value': p_value,
        'null_distribution': null_peaks,
    }


def xcorr_matrix(
    multivariate_array: np.ndarray,
    max_lag: float,
    fs: float = 1.0,
) -> dict:
    """Compute pairwise peak lags and correlations for an N-channel array.

    Args:
        multivariate_array: (n_samples, n_channels) array. Each column is one channel.
        max_lag: Maximum lag in seconds.
        fs: Sampling frequency in Hz.

    Returns:
        result: dict with keys:
            'lag_matrix': (n_channels, n_channels) array of peak lags in seconds.
                lag_matrix[i, j] is the lag of channel j relative to channel i.
            'correlation_matrix': (n_channels, n_channels) array of peak correlations.
            'channel_pairs': list of (i, j) tuples for upper triangle.

    Example:
        >>> result = xcorr_matrix(X, max_lag=1.0, fs=100.0)
        >>> result['lag_matrix'].shape
        (3, 3)
    """
    n_samples, n_channels = multivariate_array.shape
    lag_mat = np.zeros((n_channels, n_channels))
    corr_mat = np.zeros((n_channels, n_channels))
    pairs = []

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            res = lagged_xcorr(multivariate_array[:, i], multivariate_array[:, j], max_lag, fs)
            lag, corr = peak_lag(res)
            lag_mat[i, j] = lag
            lag_mat[j, i] = -lag     # antisymmetric
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr    # symmetric
            pairs.append((i, j))

    # diagonal: zero lag, unit correlation
    np.fill_diagonal(lag_mat, 0.0)
    np.fill_diagonal(corr_mat, 1.0)

    return {'lag_matrix': lag_mat, 'correlation_matrix': corr_mat, 'channel_pairs': pairs}
