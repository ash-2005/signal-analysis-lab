# Changelog

---

## v0.3.0

- added CLI (`cli.py`) with xcorr, spectral, decompose, lag_matrix modes
- CSV, NPZ, HDF5 input formats all working
- `--plot` flag saves figures to output dir
- added `demo_physionet.ipynb` — runs on MIT-BIH ECG from PhysioNet, no login needed
- `plot_lag_matrix` switched to seaborn heatmap, the old imshow version had no annotations
- fixed: `coherence` with multitaper backend was occasionally returning values slightly above 1.0 due to float accumulation. clipped to [0,1].
- fixed: `bandpass_filter` now raises ValueError with a message if highcut >= fs/2 instead of silently passing bad params to butter()
- 22 tests passing

## v0.2.0

- added `spectral.py`: welch PSD, multitaper PSD, coherence, spectral summary
- added `decompose.py`: bandpass filter (zero-phase, filtfilt), hilbert envelope, polynomial detrend, zscore, pipeline wrapper
- added `lag_matrix.py`: pairwise lag matrix, correlation matrix, lag-corrected correlation, plot, summary stats
- added test files for spectral and lag_matrix
- added seaborn and h5py to requirements

## v0.1.0

- initial commit
- `generator.py`: `generate_coupled_pair`, `generate_multivariate`, `add_artifact`
- `crosscorr.py`: `lagged_xcorr`, `peak_lag`, `bootstrap_ci`, `permutation_test`, `xcorr_matrix`
- fixed edge case in `lagged_xcorr` where constant input caused division by zero in zscore step — added 1e-10 guard
- `xcorr_matrix` diagonal was computing self-correlation instead of setting 1.0 directly — fixed
- basic tests for generator and crosscorr
- README, requirements, setup.py
