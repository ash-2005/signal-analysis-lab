# Changelog

All notable changes to `signal-analysis-lab` are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Irregular sampling support in `crosscorr.py` using the DCF method.
- Granger causality test as an alternative to permutation testing.
- Interactive CLI mode using `click`.

---

## [0.3.0] — 2025-08-25

### Added
- `cli.py`: Full command-line interface with `--input`, `--mode`, `--output`,
  `--plot`, `--fs`, `--maxlag`, and `--format` flags.
- `--format` flag supports `csv`, `npz`, and `hdf5` input formats via `h5py`.
- `notebooks/demo_synthetic.ipynb`: Step-by-step walkthrough on synthetic coupled
  oscillator data demonstrating lag injection, recovery, and the improvement in
  correlation when lag is corrected.
- `notebooks/demo_physionet.ipynb`: Applied demo on two-lead MIT-BIH ECG data from
  PhysioNet, showing real physiological lag estimation between ECG channels.
- 4 new unit tests in `tests/test_lag_matrix.py` and 5 in `tests/test_decompose.py`.
- Full test suite: 22 tests, all passing (`pytest tests/ -v`).

### Changed
- `lag_matrix.py`: `plot_lag_matrix` now uses `seaborn.heatmap` instead of raw
  `matplotlib.imshow` for improved annotation and colorbar labelling.
- `decompose.py`: `bandpass_filter` raises `ValueError` with a descriptive message
  when `highcut >= fs/2` rather than silently passing an invalid Butterworth design.
- README updated with complete CLI usage table and dependency table.

### Fixed
- `crosscorr.py`: Edge case where constant input signals caused division by zero
  in z-score normalisation. Now guarded with `1e-10` epsilon.
- `spectral.py`: `coherence` with `method='multitaper'` no longer returns values
  slightly above 1.0 due to floating-point accumulation — clipped to [0, 1].

---

## [0.2.0] — 2025-08-15

### Added
- `src/spectral.py` with four functions:
  - `welch_psd`: Welch method power spectral density.
  - `multitaper_psd`: Slepian-taper (DPSS) multitaper PSD.
  - `coherence`: Magnitude-squared coherence with Welch and multitaper backends.
  - `spectral_summary`: Peak frequency, bandwidth, total power, dominant band.
- `src/decompose.py` with five functions:
  - `bandpass_filter`: Zero-phase Butterworth bandpass via `filtfilt`.
  - `envelope`: Hilbert-transform amplitude envelope.
  - `detrend_polynomial`: Polynomial trend removal of arbitrary order.
  - `zscore_normalize`: Zero-mean, unit-variance normalisation.
  - `decompose_pipeline`: Sequential pipeline combining all four steps.
- `src/lag_matrix.py` with five functions:
  - `compute_lag_matrix`: Pairwise peak lag matrix for N channels.
  - `compute_correlation_matrix`: Standard Pearson correlation matrix.
  - `lag_corrected_correlation`: Correlation after per-pair lag correction.
  - `plot_lag_matrix`: Seaborn heatmap of lag matrix.
  - `summarize_lag_distribution`: Summary statistics of off-diagonal lags.
- Unit tests: `tests/test_spectral.py` (4 tests), stub for `test_decompose.py`.

### Changed
- `requirements.txt`: Added `seaborn>=0.12.2` and `h5py>=3.8.0`.
- `src/__init__.py`: Updated to expose `spectral`, `decompose`, `lag_matrix` modules.

### Fixed
- `crosscorr.py`: `xcorr_matrix` diagonal entries now correctly set to `(0.0, 1.0)`
  rather than being computed (which could produce numerical noise on the diagonal).

---

## [0.1.0] — 2025-08-05

### Added
- `src/generator.py` — initial release with three functions:
  - `generate_coupled_pair`: Two-channel synthetic signal with configurable lag,
    noise, and drift.
  - `generate_multivariate`: N-channel multivariate signal with coupling matrix
    and per-channel lag vector.
  - `add_artifact`: Spike, step, and dropout artifact injection.
- `src/crosscorr.py` — initial release with five functions:
  - `lagged_xcorr`: Normalised cross-correlation over a configurable lag range.
  - `peak_lag`: Peak extraction from a `lagged_xcorr` result dict.
  - `bootstrap_ci`: Block bootstrap 95% confidence interval on the peak lag.
  - `permutation_test`: Circular-shift permutation significance test.
  - `xcorr_matrix`: Pairwise N×N lag and correlation matrices.
- `tests/test_generator.py`: 3 unit tests.
- `tests/test_crosscorr.py`: 6 unit tests including zero-lag, known-lag recovery,
  negative lag, all-zero input, bootstrap coverage, and permutation significance.
- `README.md`, `requirements.txt`, `setup.py`, `.gitignore`.
- Repository created and initial commit pushed to GitHub.

---
[Unreleased]: https://github.com/ash-2005/signal-analysis-lab/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ash-2005/signal-analysis-lab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ash-2005/signal-analysis-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ash-2005/signal-analysis-lab/releases/tag/v0.1.0
