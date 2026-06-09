"""
Generic CWT Scalogram Generator for CSV Time-Series Data

This script generates Continuous Wavelet Transform (CWT) scalograms from any CSV file
containing time-series data. It auto-detects time columns, calculates appropriate
frequency ranges, and generates publication-quality scalogram images.

Usage:
    python vis/cwt_csv.py input.csv
    python vis/cwt_csv.py input.csv -o output_dir/ --tmin 0.001 --tmax 0.005
    python vis/cwt_csv.py input.csv --columns "avg(T [K])" "max(p [Pa])"
"""

import os
import sys
import re
import functools
import argparse
import pywt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt, ticker as mticker
from scipy.signal import savgol_filter

print = functools.partial(print, flush=True)
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_cwt_scales


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate CWT scalograms from CSV time-series data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - process all numeric columns
    python vis/cwt_csv.py data.csv

    # Specify output directory
    python vis/cwt_csv.py data.csv -o output/

    # Limit time range
    python vis/cwt_csv.py data.csv --tmin 0.001 --tmax 0.005

    # Process specific columns only
    python vis/cwt_csv.py data.csv --columns "avg(T [K])" "max(p [Pa])"

    # Custom wavelet and frequency range
    python vis/cwt_csv.py data.csv --wavelet cmor2.5-0.5 --fmin 100 --fmax 10000
        """
    )

    parser.add_argument('csv_file', type=str, help='Input CSV file path')
    parser.add_argument('-o', '--output', type=str, default='./cwt_output/',
                        help='Output directory (default: ./cwt_output/)')
    parser.add_argument('--time_col', type=str, default=None,
                        help='Name of time column (auto-detected if not specified)')
    parser.add_argument('--tmin', type=float, default=None,
                        help='Start time - trim data before this time')
    parser.add_argument('--tmax', type=float, default=None,
                        help='End time - trim data after this time')
    parser.add_argument('--columns', type=str, nargs='+', default=None,
                        help='Specific columns to process (default: all numeric)')
    parser.add_argument('--wavelet', type=str, default='cmor1.5-1.0',
                        help='Wavelet to use (default: cmor1.5-1.0)')
    parser.add_argument('--fmin', type=float, default=None,
                        help='Minimum frequency in Hz (default: auto from signal duration)')
    parser.add_argument('--fmax', type=float, default=None,
                        help='Maximum frequency in Hz (default: Nyquist frequency)')
    parser.add_argument('--num_scales', type=int, default=256,
                        help='Number of scales for CWT (default: 256)')
    parser.add_argument('--reset_time', action='store_true',
                        help='Reset time to start at zero after cropping')
    parser.add_argument('--savgol', type=int, default=None, metavar='WINDOW',
                        help='Apply Savitzky-Golay filter with specified window length (must be odd)')

    return parser.parse_args()


def detect_time_column(df):
    """
    Auto-detect the time column in a DataFrame.

    Searches for columns containing 'time', 't', or similar patterns.

    Args:
        df: pandas DataFrame

    Returns:
        str: Name of detected time column

    Raises:
        ValueError: If no time column can be detected
    """
    # Patterns to search for (case-insensitive)
    time_patterns = [
        r'^time$',           # Exact match "time"
        r'^t$',              # Exact match "t"
        r'^time[\s_\[\(]',   # "time" followed by space, underscore, bracket
        r'[\s_\]\)]time$',   # "time" at end after separator
        r'^t[\s_\[\(]',      # "t" followed by separator (e.g., "t [s]")
    ]

    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in time_patterns:
            if re.search(pattern, col_lower):
                return col

    # Fallback: check if first column looks like time data
    first_col = df.columns[0]
    if df[first_col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        # Check if it's monotonically increasing (characteristic of time)
        if df[first_col].is_monotonic_increasing:
            print(f"WARNING: Using first column '{first_col}' as time (monotonically increasing)")
            return first_col

    raise ValueError(
        "Could not auto-detect time column. Please specify with --time_col.\n"
        f"Available columns: {list(df.columns)}"
    )


def sanitize_filename(name):
    """
    Sanitize a column name for use as a filename.

    Replaces special characters with underscores.

    Args:
        name: Column name string

    Returns:
        str: Sanitized filename-safe string
    """
    # Replace common special characters
    sanitized = name
    replacements = {
        '(': '_', ')': '_',
        '[': '_', ']': '_',
        ' ': '_', '/': '_',
        '\\': '_', ':': '_',
        '<': '_', '>': '_',
        '"': '_', "'": '_',
        '|': '_', '?': '_',
        '*': '_', '%': 'pct',
    }

    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)

    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def get_numeric_columns(df, time_col):
    """
    Get list of numeric columns suitable for CWT analysis.

    Excludes time column and non-numeric columns.

    Args:
        df: pandas DataFrame
        time_col: Name of time column to exclude

    Returns:
        list: Column names suitable for CWT analysis
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove time column
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    return numeric_cols


def compute_cwt(signal, time_array, wavelet, num_scales, fmin, fmax):
    """
    Compute CWT for a signal with symmetric padding.

    Args:
        signal: 1D numpy array of signal values
        time_array: 1D numpy array of time values
        wavelet: Wavelet name string
        num_scales: Number of scales
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz

    Returns:
        tuple: (cwtmatr, freqs, t_mesh, f_mesh)
    """
    # Calculate sampling parameters
    dt = np.mean(np.diff(time_array))
    sampling_rate = 1 / dt
    sampling_period = dt

    # Get scales using existing utility
    scales, vmax = get_cwt_scales(
        wavelet,
        num=num_scales,
        fmin=fmin,
        fmax=fmax,
        sampling_rate=sampling_rate
    )

    # Apply symmetric padding (2x signal length on each side)
    pad_width = 2 * len(signal)
    s_pad = np.pad(signal, pad_width, mode='symmetric')

    # Compute CWT on padded signal
    cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)

    # Crop to original signal length (remove padding)
    cwtmatr = np.abs(cwtmatr[:, pad_width:pad_width + len(signal)])

    # Create mesh grids for plotting
    t_mesh, f_mesh = np.meshgrid(time_array, freqs)

    return cwtmatr, freqs, t_mesh, f_mesh, vmax


def compute_fft(signal, sampling_rate):
    """
    Compute FFT for a signal.

    Args:
        signal: 1D numpy array of signal values
        sampling_rate: Sampling rate in Hz

    Returns:
        tuple: (frequencies, magnitudes) - positive frequencies only
    """
    n = len(signal)

    # Remove DC component (mean)
    signal_centered = signal - np.mean(signal)

    # Compute FFT
    fft_result = np.fft.fft(signal_centered)
    fft_magnitude = np.abs(fft_result) / n  # Normalize

    # Get positive frequencies only
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    positive_mask = freqs > 0

    return freqs[positive_mask], fft_magnitude[positive_mask] * 2  # *2 for single-sided


def plot_fft(freqs, magnitude, column_name, output_path, fmin, fmax):
    """
    Generate and save FFT plot.

    Args:
        freqs: Frequency array (Hz)
        magnitude: FFT magnitude array
        column_name: Name of the signal column
        output_path: Path to save the figure
        fmin, fmax: Frequency range for axis limits
    """
    plt.rcParams.update({'font.size': 9})

    # Determine appropriate frequency units
    if fmax > 1000:
        freq_scale = 1000
        freq_unit = 'kHz'
    else:
        freq_scale = 1
        freq_unit = 'Hz'

    # Filter to frequency range of interest
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_plot = freqs[mask] / freq_scale
    mag_plot = magnitude[mask]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.suptitle(f'FFT Spectrum: {column_name}')

    ax.plot(freqs_plot, mag_plot, lw=0.75, color='tab:blue')
    ax.set_xlabel(f'Frequency [{freq_unit}]')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(fmin / freq_scale, fmax / freq_scale)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cwt(time_array, signal, cwtmatr, freqs, t_mesh, f_mesh,
             column_name, output_path, fmin, fmax):
    """
    Generate and save CWT scalogram plot.

    Args:
        time_array: Time values
        signal: Signal values
        cwtmatr: CWT coefficient matrix
        freqs: Frequency array
        t_mesh, f_mesh: Mesh grids for pcolormesh
        column_name: Name of the signal column
        output_path: Path to save the figure
        fmin, fmax: Frequency range for axis limits
    """
    # Scale text for readability at 1/2 A4 width (~105mm from 8" figure = 0.52x)
    # Target 8pt minimum -> need ~15pt in figure
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12})

    # Determine appropriate time and frequency units
    time_range = time_array[-1] - time_array[0]
    if time_range < 0.1:
        time_scale = 1000  # Use milliseconds
        time_unit = 'ms'
    else:
        time_scale = 1
        time_unit = 's'

    if fmax > 1000:
        freq_scale = 1000  # Use kHz
        freq_unit = 'kHz'
    else:
        freq_scale = 1
        freq_unit = 'Hz'

    # Normalize CWT magnitude to 0-1
    cwtmatr_norm = cwtmatr / cwtmatr.max()

    # Create figure with GridSpec for aligned axes
    # Layout: signal plot and CWT plot share same width, colorbar in separate column
    fig = plt.figure(figsize=(8, 6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[1, 0.03],
                          hspace=0.08, wspace=0.02)

    ax1 = fig.add_subplot(gs[0, 0])  # Signal plot (top row, main column)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # CWT plot (bottom row, main column)
    cax = fig.add_subplot(gs[1, 1])  # Colorbar (bottom row, narrow column)

    fig.suptitle(f'CWT Scalogram: {column_name}', fontsize=16)

    # Plot signal (no x-label since shared with bottom plot)
    ax1.plot(time_array * time_scale, signal, lw=0.75, color='tab:blue')
    ax1.set_xlim(time_array[0] * time_scale, time_array[-1] * time_scale)
    ax1.set_ylabel(column_name)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)  # Hide x-tick labels on top plot

    # Plot CWT scalogram with normalized colormap
    pcm = ax2.pcolormesh(t_mesh * time_scale, f_mesh / freq_scale,
                         cwtmatr_norm, cmap='viridis', shading='auto',
                         vmin=0, vmax=1)
    ax2.set_yscale('log', base=2)
    ax2.set_ylabel(f'Frequency [{freq_unit}]')
    ax2.set_xlabel(f'Time [{time_unit}]')
    ax2.grid(True, axis='x', alpha=0.3)  # Vertical gridlines to match signal plot

    # Set frequency axis limits and ticks (powers of 2 only)
    y_min = fmin / freq_scale
    y_max = fmax / freq_scale
    ax2.set_ylim(y_min, y_max)

    # Generate tick values (powers of 2 within range)
    tick_values = []
    power = int(np.floor(np.log2(y_min)))
    while 2**power <= y_max:
        if 2**power >= y_min:
            tick_values.append(2**power)
        power += 1

    ax2.set_yticks(tick_values)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    # Add colorbar in dedicated axis
    cbar = fig.colorbar(pcm, cax=cax, label='Magnitude (normalized)')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Validate input file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    # Load CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Detect or use specified time column
    if args.time_col:
        if args.time_col not in df.columns:
            print(f"ERROR: Specified time column '{args.time_col}' not found")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        time_col = args.time_col
    else:
        time_col = detect_time_column(df)
    print(f"  Time column: '{time_col}'")

    # Get time array
    time_array = df[time_col].values

    # Apply time range cropping FIRST (before calculating sampling rate)
    crop_mask = np.ones(len(time_array), dtype=bool)
    if args.tmin is not None:
        crop_mask &= (time_array >= args.tmin)
        print(f"  Applying tmin={args.tmin}")
    if args.tmax is not None:
        crop_mask &= (time_array <= args.tmax)
        print(f"  Applying tmax={args.tmax}")

    # Apply crop
    df_cropped = df[crop_mask].reset_index(drop=True)
    time_array = df_cropped[time_col].values

    if len(time_array) < 10:
        print(f"ERROR: Too few data points after cropping ({len(time_array)})")
        sys.exit(1)

    # Reset time to start at zero if requested
    if args.reset_time:
        time_offset = time_array[0]
        time_array = time_array - time_offset
        print(f"  Reset time to zero (offset: {time_offset:.6f} s)")

    print(f"  After cropping: {len(time_array)} samples")
    print(f"  Time range: {time_array[0]:.6f} to {time_array[-1]:.6f} s")

    # Calculate sampling rate from cropped data
    dt = np.mean(np.diff(time_array))
    sampling_rate = 1 / dt
    signal_duration = time_array[-1] - time_array[0]
    nyquist_freq = sampling_rate / 2

    print(f"  Sampling rate: {sampling_rate:.1f} Hz")
    print(f"  Signal duration: {signal_duration:.6f} s")
    print(f"  Nyquist frequency: {nyquist_freq:.1f} Hz")

    # Auto-detect or use specified frequency range
    fmin = args.fmin if args.fmin is not None else (1 / signal_duration)
    fmax = args.fmax if args.fmax is not None else nyquist_freq

    # Clamp fmax to Nyquist
    if fmax > nyquist_freq:
        print(f"  WARNING: fmax ({fmax:.1f} Hz) exceeds Nyquist ({nyquist_freq:.1f} Hz), clamping")
        fmax = nyquist_freq

    print(f"  Frequency range: {fmin:.1f} Hz to {fmax:.1f} Hz")
    print(f"  Wavelet: {args.wavelet}")
    if args.savgol:
        savgol_window = args.savgol if args.savgol % 2 == 1 else args.savgol + 1
        print(f"  Savitzky-Golay filter: window={savgol_window}, polyorder=2")

    # Determine columns to process
    if args.columns:
        # Validate specified columns exist
        missing = [c for c in args.columns if c not in df_cropped.columns]
        if missing:
            print(f"ERROR: Columns not found: {missing}")
            print(f"Available columns: {list(df_cropped.columns)}")
            sys.exit(1)
        columns_to_process = args.columns
    else:
        columns_to_process = get_numeric_columns(df_cropped, time_col)

    if not columns_to_process:
        print("ERROR: No numeric columns found to process")
        sys.exit(1)

    print(f"\nColumns to process ({len(columns_to_process)}):")
    for col in columns_to_process:
        print(f"  - {col}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Process each column
    print(f"\nGenerating CWT scalograms...")
    for i, col in enumerate(columns_to_process, 1):
        print(f"  [{i}/{len(columns_to_process)}] {col}")

        # Get signal data
        signal = df_cropped[col].values

        # Check for NaN values
        if np.any(np.isnan(signal)):
            nan_count = np.sum(np.isnan(signal))
            print(f"    WARNING: {nan_count} NaN values found, interpolating")
            signal = pd.Series(signal).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

        # Apply Savitzky-Golay filter if specified
        if args.savgol is not None:
            window = args.savgol
            # Ensure window is odd
            if window % 2 == 0:
                window += 1
            # Ensure window doesn't exceed signal length
            if window > len(signal):
                window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
            signal = savgol_filter(signal, window_length=window, polyorder=2)

        # Compute CWT
        cwtmatr, freqs, t_mesh, f_mesh, vmax = compute_cwt(
            signal, time_array, args.wavelet, args.num_scales, fmin, fmax
        )

        # Generate output filename
        safe_name = sanitize_filename(col)
        cwt_output_path = output_dir / f"{safe_name}_cwt.png"
        fft_output_path = output_dir / f"{safe_name}_fft.png"

        # Plot and save CWT
        plot_cwt(time_array, signal, cwtmatr, freqs, t_mesh, f_mesh,
                 col, cwt_output_path, fmin, fmax)

        # Compute and plot FFT
        fft_freqs, fft_magnitude = compute_fft(signal, sampling_rate)
        plot_fft(fft_freqs, fft_magnitude, col, fft_output_path, fmin, fmax)

        print(f"    Saved: {cwt_output_path.name}, {fft_output_path.name}")

    print(f"\nDone! Generated {len(columns_to_process)} CWT scalograms and FFT plots in {output_dir}")


if __name__ == '__main__':
    main()
