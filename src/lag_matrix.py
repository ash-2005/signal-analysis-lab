import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.crosscorr import xcorr_matrix

def compute_lag_matrix(
    X: np.ndarray,
    max_lag: float,
    fs: float = 1.0,
) -> np.ndarray:
    """Compute pairwise peak lags for all channel pairs in a multivariate array.

    Args:
        X: (n_samples, n_channels) array.
        max_lag: Maximum lag to consider in seconds.
        fs: Sampling frequency in Hz.

    Returns:
        lag_matrix: (n_channels, n_channels) array of peak lags in seconds.
    """
    result = xcorr_matrix(X, max_lag, fs)
    return result['lag_matrix']


def compute_correlation_matrix(
    X: np.ndarray,
) -> np.ndarray:
    """Compute the standard Pearson correlation matrix (zero-lag) for N channels.

    Args:
        X: (n_samples, n_channels) array.

    Returns:
        corr_matrix: (n_channels, n_channels) symmetric correlation matrix.
    """
    corr_matrix = np.corrcoef(X.T)
    return corr_matrix


def lag_corrected_correlation(
    X: np.ndarray,
    lag_matrix: np.ndarray,
    fs: float = 1.0,
) -> np.ndarray:
    """Compute correlation after correcting each channel pair for its estimated lag.

    Args:
        X: (n_samples, n_channels) array.
        lag_matrix: (n_channels, n_channels) lag matrix from compute_lag_matrix.
        fs: Sampling frequency in Hz.

    Returns:
        corr_corrected: (n_channels, n_channels) correlation matrix after lag correction.
    """
    n_samples, n_channels = X.shape
    corr_corrected = np.eye(n_channels)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            lag_samples = int(np.round(lag_matrix[i, j] * fs))
            y_shifted = np.roll(X[:, j], -lag_samples)
            
            c = np.corrcoef(X[:, i], y_shifted)[0, 1]
            corr_corrected[i, j] = c
            corr_corrected[j, i] = c

    return corr_corrected


def plot_lag_matrix(
    lag_matrix: np.ndarray,
    labels: list[str] = None,
    title: str = "Pairwise Lag Matrix",
) -> None:
    """Plot the lag matrix as a seaborn heatmap with diverging colormap.

    Args:
        lag_matrix: (n_channels, n_channels) array of lag values in seconds.
        labels: List of channel name strings. If None, uses ['ch0', 'ch1', ...].
        title: Title for the plot. Default 'Pairwise Lag Matrix'.
    """
    n = lag_matrix.shape[0]
    if labels is None:
        labels = [f'ch{i}' for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(4, n * 1.2)))
    sns.heatmap(lag_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def summarize_lag_distribution(
    lag_matrix: np.ndarray,
) -> dict:
    """Compute summary statistics of the off-diagonal lag values.

    Args:
        lag_matrix: (n_channels, n_channels) lag matrix.

    Returns:
        summary: dict with summary stats.
    """
    n = lag_matrix.shape[0]
    # extract upper triangle (unique pairs)
    upper_idx = np.triu_indices(n, k=1)
    upper_lags = lag_matrix[upper_idx]

    return {
        'mean_abs_lag': np.mean(np.abs(upper_lags)),
        'max_lag': np.max(np.abs(upper_lags)),
        'std_lag': np.std(upper_lags),
        'n_pairs': len(upper_lags),
    }
