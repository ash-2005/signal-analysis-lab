# signal-analysis-lab

A Python library and CLI for time-series signal analysis — cross-correlation,
spectral analysis, decomposition, and pairwise lag estimation.

---

## Why I built this

I kept hitting the same problem: two signals that are clearly related, but the
Pearson correlation between them looks weak. The reason is almost always lag —
one signal arrives a bit later than the other and the correlation gets blunted.
I wanted to actually see this happen numerically before trusting it as a concept.

So I built a synthetic generator that lets me inject a known lag between two
oscillators, then run cross-correlation to recover it. The first time the
estimated lag matched the injected one to within a sample I understood it better
than I had from any explanation. The bootstrap CI and permutation test came later
when I needed to answer "okay but is this significant" on real data.

The PhysioNet notebook is there because I wanted to check the toolkit wasn't
just a toy that works on clean synthetic signals. It does work on real ECG.

This is not fMRI-specific. It works on any multivariate time series in CSV,
NumPy, or HDF5.

---

## Features

- Synthetic coupled oscillator generation with configurable lag, noise, drift
- Multivariate N-channel signal generation with coupling matrix control
- Normalised cross-correlation over a lag range (regular sampling)
- Peak lag extraction with block bootstrap confidence intervals (500 iterations)
- Permutation significance test using circular shift
- N×N pairwise lag matrix for multivariate input
- Welch and multitaper PSD
- Magnitude-squared coherence (Welch and multitaper backends)
- Spectral summary: peak frequency, bandwidth, total power, dominant band
- Zero-phase Butterworth bandpass filter
- Hilbert transform amplitude envelope
- Polynomial detrending (any order)
- Z-score normalisation
- Sequential decompose pipeline
- CLI with xcorr, spectral, decompose, and lag_matrix modes
- CSV, NumPy (.npz), and HDF5 input formats
- 22 pytest tests

---

## Installation

```bash
git clone https://github.com/ash-2005/signal-analysis-lab.git
cd signal-analysis-lab
pip install -r requirements.txt
```

Or as a package:

```bash
pip install -e .
```

---

## Quick start

```python
import numpy as np
from src.generator import generate_coupled_pair
from src.crosscorr import lagged_xcorr, peak_lag
from src.decompose import bandpass_filter
import matplotlib.pyplot as plt

# Two coupled signals, 0.3s lag, 100 Hz
x, y = generate_coupled_pair(n_samples=1000, lag=0.3, noise_std=0.1, fs=100.0)

# Filter first
x_filt = bandpass_filter(x, lowcut=0.5, highcut=20.0, fs=100.0)
y_filt = bandpass_filter(y, lowcut=0.5, highcut=20.0, fs=100.0)

# Cross-correlation
result = lagged_xcorr(x_filt, y_filt, max_lag=1.0, fs=100.0)
estimated_lag, peak_corr = peak_lag(result)

print(f"Injected lag: 0.300s")
print(f"Estimated lag: {estimated_lag:.3f}s")
print(f"Peak correlation: {peak_corr:.3f}")

plt.plot(result["lags"], result["correlation"])
plt.axvline(estimated_lag, color="r", linestyle="--", label=f"lag={estimated_lag:.2f}s")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.savefig("xcorr_output.png", dpi=150)
```

---

## CLI

```bash
# Cross-correlation
python cli.py --input data.csv --mode xcorr --fs 100 --maxlag 2.0 --plot

# Spectral analysis, save to results/
python cli.py --input data.csv --mode spectral --fs 250 --output results/ --plot

# Decompose pipeline
python cli.py --input data.csv --mode decompose --fs 100 --plot

# Pairwise lag matrix (N-column CSV = N channels)
python cli.py --input multichannel.csv --mode lag_matrix --fs 100 --maxlag 1.0 --plot

# HDF5 input
python cli.py --input data.h5 --format hdf5 --mode xcorr --fs 100 --plot
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to input file |
| `--mode` | required | xcorr / spectral / decompose / lag_matrix |
| `--output` | None | Directory to write output files |
| `--plot` | off | Show and save figures |
| `--fs` | 1.0 | Sampling frequency in Hz |
| `--maxlag` | 1.0 | Maximum lag in seconds |
| `--format` | csv | csv / npz / hdf5 |

---

## Project structure

```
signal-analysis-lab/
├── src/
│   ├── __init__.py
│   ├── generator.py
│   ├── crosscorr.py
│   ├── spectral.py
│   ├── decompose.py
│   └── lag_matrix.py
├── tests/
│   ├── test_generator.py
│   ├── test_crosscorr.py
│   ├── test_spectral.py
│   ├── test_decompose.py
│   └── test_lag_matrix.py
├── notebooks/
│   ├── demo_synthetic.ipynb
│   └── demo_physionet.ipynb
├── cli.py
├── requirements.txt
├── setup.py
├── CHANGELOG.md
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy ≥1.24 | arrays, FFT |
| scipy ≥1.10 | filtering, correlation, spectral |
| pandas ≥2.0 | CSV I/O |
| matplotlib ≥3.7 | plots |
| seaborn ≥0.12 | heatmaps |
| h5py ≥3.8 | HDF5 I/O |
| tqdm ≥4.65 | bootstrap progress |
| pytest ≥7.3 | tests |

---

## Running tests

```bash
pytest tests/ -v
```

All 22 tests should pass. Known slow test: `test_bootstrap_ci_contains_true_lag`
takes ~8s on a laptop because of 500 bootstrap iterations.

---

## Limitations

- Irregular sampling is not fully supported in `crosscorr.py` (regular spacing assumed)
- Bootstrap CI assumes stationarity — don't use it on obviously non-stationary signals without detrending first
- The lag correction in `lag_matrix.py` uses circular shift, which is a simplification

---

## License

MIT
