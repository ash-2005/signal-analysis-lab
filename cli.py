import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    h5py = None

from src.crosscorr import lagged_xcorr, peak_lag
from src.spectral import welch_psd
from src.decompose import decompose_pipeline
from src.lag_matrix import compute_lag_matrix, plot_lag_matrix

def load_data(filepath, fmt):
    if fmt == 'csv':
        df = pd.read_csv(filepath)
        return df.values
    elif fmt == 'npz':
        data = np.load(filepath)
        keys = list(data.keys())
        return data[keys[0]]
    elif fmt == 'hdf5':
        if h5py is None:
            raise ImportError("h5py must be installed to read HDF5 files")
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            return f[keys[0]][:]
    else:
        raise ValueError(f"Unknown format: {fmt}")

def save_numeric_output(data, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    pd.DataFrame(data).to_csv(filepath, index=False)
    print(f"Saved {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Signal Analysis Lab CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--mode", type=str, required=True, choices=["xcorr", "spectral", "decompose", "lag_matrix"])
    parser.add_argument("--output", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--plot", action="store_true", help="Show and save matplotlib figures")
    parser.add_argument("--fs", type=float, default=1.0, help="Sampling frequency in Hz")
    parser.add_argument("--maxlag", type=float, default=1.0, help="Maximum lag to consider in seconds")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "npz", "hdf5"], help="Input format")
    
    args = parser.parse_args()
    
    try:
        data = load_data(args.input, args.format)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Loaded data with shape: {data.shape}")

    if args.mode == "xcorr":
        if data.shape[1] < 2:
            print("xcorr requires at least 2 columns")
            return
        x, y = data[:, 0], data[:, 1]
        res = lagged_xcorr(x, y, max_lag=args.maxlag, fs=args.fs)
        lag, peak_c = peak_lag(res)
        print(f"Peak lag: {lag:.3f}s, Correlation: {peak_c:.3f}")
        
        if args.output:
            save_numeric_output({"lag": res["lags"], "correlation": res["correlation"]}, args.output, "xcorr_result.csv")
            
        if args.plot:
            plt.plot(res['lags'], res['correlation'])
            plt.axvline(lag, color='red', linestyle='--')
            plt.title(f"Cross Correlation (Peak={lag:.3f}s)")
            if args.output:
                plt.savefig(os.path.join(args.output, "xcorr.png"))
            plt.show()

    elif args.mode == "spectral":
        x = data[:, 0]
        res = welch_psd(x, fs=args.fs)
        
        if args.output:
            save_numeric_output({"freqs": res["freqs"], "psd": res["psd"]}, args.output, "spectral_result.csv")
            
        if args.plot:
            plt.semilogy(res['freqs'], res['psd'])
            plt.title("Welch PSD")
            if args.output:
                plt.savefig(os.path.join(args.output, "spectral.png"))
            plt.show()

    elif args.mode == "decompose":
        x = data[:, 0]
        res = decompose_pipeline(x, fs=args.fs, lowcut=0.5, highcut=min(args.fs/2 - 1, 20.0))
        
        if args.output:
            save_numeric_output({"normalised": res["normalised"], "envelope": res["envelope"]}, args.output, "decompose_result.csv")
            
        if args.plot:
            plt.plot(res["normalised"], label="Normalised")
            plt.plot(res["envelope"], label="Envelope")
            plt.legend()
            plt.title("Decomposed Signal")
            if args.output:
                plt.savefig(os.path.join(args.output, "decompose.png"))
            plt.show()

    elif args.mode == "lag_matrix":
        lag_mat = compute_lag_matrix(data, max_lag=args.maxlag, fs=args.fs)
        print(f"Lag matrix built, shape: {lag_mat.shape}")
        
        if args.output:
            save_numeric_output(lag_mat, args.output, "lag_matrix.csv")
            
        if args.plot:
            plot_lag_matrix(lag_mat)
            if args.output:
                plt.savefig(os.path.join(args.output, "lag_matrix.png"))

if __name__ == "__main__":
    main()
