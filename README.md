# signal-analysis-lab

> A self-contained Python library and CLI for multivariate time-series signal analysis.

---

## Motivation

Before I started working with neuroimaging data, I spent time trying to understand
what time-series signals actually are — not in the abstract, but at the level of
numbers and transformations. I built this library to answer a question I kept running
into: if two signals are correlated, but one arrives a few hundred milliseconds later
than the other, how much does that lag distort the apparent correlation between them?
The answer, it turns out, is *a lot* — and understanding that concretely, on synthetic
and real physiological signals, was what made the jump to fMRI preprocessing feel
natural rather than foreign. This library is the record of that learning process.
It is not fMRI-specific. It works on any multivariate time series you can load from a
CSV, NumPy array, or HDF5 file.

---

## Features

- **Synthetic signal generation** — coupled oscillator pairs with configurable lag,
  noise, sampling rate, and drift; multivariate N-channel generation with full
  coupling matrix control
- **Cross-correlation engine** — lag-dependent cross-correlation for regular and
  irregularly sampled series; block bootstrap confidence intervals (500 iterations);
  permutation significance testing; full N×N pairwise lag matrix
- **Spectral analysis** — Welch PSD, multitaper PSD, coherence estimation with
  confidence bands; spectral summary statistics per channel
- **Signal decomposition** — zero-phase Butterworth bandpass filter; Hilbert envelope
  extraction; polynomial detrending; z-score normalisation; sequential decompose
  pipeline
- **Command-line interface** — single entry point supporting xcorr, spectral,
  decompose, and lag_matrix modes; CSV, NumPy (.npz), and HDF5 input formats;
  automatic figure saving
- **Full test suite** — 22 pytest tests covering edge cases including zero lag,
  maximum lag, all-zero input, single channel, and irregular sampling
- **Two demonstration notebooks** — one on synthetic data, one on PhysioNet ECG data

---

## Installation

```bash
git clone https://github.com/ash-2005/signal-analysis-lab.git
cd signal-analysis-lab
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from src.generator import generate_coupled_pair
from src.crosscorr import lagged_xcorr, peak_lag
from src.decompose import bandpass_filter
import matplotlib.pyplot as plt

# Generate two coupled signals with a known 0.3s lag at 100 Hz
x, y = generate_coupled_pair(n_samples=1000, lag=0.3, noise_std=0.1, fs=100.0)

# Bandpass filter to 0.5–20 Hz
x_filt = bandpass_filter(x, lowcut=0.5, highcut=20.0, fs=100.0)
y_filt = bandpass_filter(y, lowcut=0.5, highcut=20.0, fs=100.0)

# Run cross-correlation
result = lagged_xcorr(x_filt, y_filt, max_lag=1.0, fs=100.0)

# Find peak lag
estimated_lag, peak_corr = peak_lag(result)
print(f"Estimated lag: {estimated_lag:.3f}s  |  Peak correlation: {peak_corr:.3f}")

# Plot
lags = result["lags"]
corr = result["correlation"]
plt.plot(lags, corr)
plt.axvline(estimated_lag, color="red", linestyle="--", label=f"lag={estimated_lag:.2f}s")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.title("Cross-Correlation")
plt.legend()
plt.tight_layout()
plt.savefig("xcorr_demo.png", dpi=150)
```

---

## CLI Usage

```bash
# Cross-correlation on a CSV file (two columns assumed)
python cli.py --input data.csv --mode xcorr --fs 100 --maxlag 2.0 --plot

# Spectral analysis with output saved to results/
python cli.py --input data.csv --mode spectral --fs 250 --output results/ --plot

# Full decompose pipeline
python cli.py --input data.csv --mode decompose --fs 100 --plot

# Pairwise lag matrix for multivariate data (N columns = N channels)
python cli.py --input multivariate.csv --mode lag_matrix --fs 100 --maxlag 1.0 --plot

# Using HDF5 input
python cli.py --input data.h5 --format hdf5 --mode xcorr --fs 100 --plot

# Suppress plots, save only numeric output
python cli.py --input data.csv --mode xcorr --fs 100 --output results/
```

**All flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | str | required | Path to input file |
| `--mode` | str | required | One of: xcorr, spectral, decompose, lag_matrix |
| `--output` | str | None | Directory to save output files |
| `--plot` | flag | False | Show and save matplotlib figures |
| `--fs` | float | 1.0 | Sampling frequency in Hz |
| `--maxlag` | float | 1.0 | Maximum lag to consider in seconds |
| `--format` | str | csv | Input format: csv, npz, hdf5 |

---

## Project Structure

```text
signal-analysis-lab/
├── src/
│   ├── __init__.py
│   ├── generator.py       # Synthetic signal generation
│   ├── crosscorr.py       # Cross-correlation and lag estimation
│   ├── spectral.py        # PSD and coherence analysis
│   ├── decompose.py       # Filtering, envelope, detrending
│   └── lag_matrix.py      # Pairwise lag matrix for multivariate data
├── tests/
│   ├── __init__.py
│   ├── test_generator.py
│   ├── test_crosscorr.py
│   ├── test_spectral.py
│   ├── test_decompose.py
│   └── test_lag_matrix.py
├── notebooks/
│   ├── demo_synthetic.ipynb
│   └── demo_physionet.ipynb
├── cli.py                 # Command-line entry point
├── requirements.txt
├── setup.py
├── .gitignore
├── CHANGELOG.md
└── README.md
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations, FFT |
| scipy | ≥1.10 | Signal processing, stats |
| pandas | ≥2.0 | CSV I/O, time-series frames |
| matplotlib | ≥3.7 | Plotting |
| seaborn | ≥0.12 | Heatmap visualisation |
| h5py | ≥3.8 | HDF5 file I/O |
| pytest | ≥7.3 | Unit testing |
| tqdm | ≥4.65 | Progress bars for bootstrap |

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

Contributions are welcome. Please open an issue describing the change before
submitting a pull request. All new functions must include a Google-style docstring,
at least two unit tests (one for the normal case, one for an edge case), and must
not break existing tests. Run `pytest tests/` before pushing.
