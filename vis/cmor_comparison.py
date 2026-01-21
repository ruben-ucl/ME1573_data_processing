#!/usr/bin/env python3
"""
Complex Morlet Wavelet Comparison Script

Generates test figures comparing different Complex Morlet (cmor) wavelets for CWT analysis.
Shows both the wavelet functions and their CWT results on photodiode data, focusing on
different bandwidth and center frequency parameters.

Usage:
    python vis/cmor_comparison.py

Author: Claude Code Assistant
"""

import os, sys, functools, h5py, pywt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, ticker as mticker

# Fix print buffering
print = functools.partial(print, flush=True)
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, get_cwt_scales

# Configuration
TARGET_TRACKID = '0110_04'  # Hardcoded test track
SAVE_OUTPUT = True
DEBUG = True

# Complex Morlet wavelets in 3x3 grid: bandwidth (rows) x center frequency (columns)
# Format: cmor<bandwidth>-<center_frequency>
BANDWIDTH_VALUES = [1.5, 2.5, 3.5]  # Low to high bandwidth (high freq res → high time res)
CENTER_FREQ_VALUES = [0.5, 1.0, 1.5]  # Low to high center frequency

def generate_cmor_grid():
    """Generate 3x3 grid of cmor wavelets."""
    wavelets = []
    for bw in BANDWIDTH_VALUES:
        row = []
        for cf in CENTER_FREQ_VALUES:
            row.append(f'cmor{bw}-{cf}')
        wavelets.append(row)
    return wavelets

CMOR_GRID = generate_cmor_grid()

def find_hdf5_file(trackid, folder):
    """Find the HDF5 file for a given trackid."""
    import glob
    pattern = os.path.join(folder, f"{trackid}*.hdf5")
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"No HDF5 file found for trackid {trackid} in {folder}")

def plot_wavelet_function(wavelet_name, ax):
    """Plot the wavelet function in time domain."""
    try:
        wavelet_obj = pywt.ContinuousWavelet(wavelet_name)
        [psi, x] = wavelet_obj.wavefun(level=10)
        
        # Handle complex wavelets - plot real and imaginary parts
        if np.iscomplexobj(psi):
            ax.plot(x, np.real(psi), 'b-', label='Real', linewidth=1.5)
            ax.plot(x, np.imag(psi), 'r--', label='Imag', linewidth=1.5)
            ax.legend(fontsize=8)
        else:
            ax.plot(x, psi, 'b-', linewidth=1.5)
        
        ax.set_xlim((-5, 5))
        ax.set_ylim((-1.2, 1.2))
        ax.set_title(f'{wavelet_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        
        return True
    except Exception as e:
        if DEBUG: print(f"Failed to plot wavelet {wavelet_name}: {e}")
        ax.text(0.5, 0.5, f'Error plotting\\n{wavelet_name}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{wavelet_name} (Error)', fontsize=10, color='red')
        return False

def compute_cwt_for_signal(signal, time, wavelet_name, sampling_period):
    """Compute CWT for given signal and wavelet with symmetric padding."""
    try:
        # Calculate sampling rate from period
        sampling_rate = 1.0 / sampling_period

        # Get scales for this wavelet with actual sampling rate
        # Note: We ignore the vmax from get_cwt_scales, will compute from actual data
        scales, _ = get_cwt_scales(wavelet_name, num=128, sampling_rate=sampling_rate)

        # Apply symmetric padding to minimize edge artifacts (best practice)
        signal_pad = np.pad(signal, len(signal), mode='symmetric')
        if DEBUG:
            print(f"  {wavelet_name}: Padded {len(signal)} → {len(signal_pad)} samples")

        # Compute CWT on padded signal
        cwtmatr, freqs = pywt.cwt(signal_pad, scales, wavelet_name, sampling_period)

        # Crop to original signal length (extract middle section, removing padding)
        cwtmatr = np.abs(cwtmatr[:, len(signal):2*len(signal)])

        # Automatically determine vmax from the actual scalogram data
        # Use 99th percentile to avoid outliers affecting the colormap
        vmax = np.percentile(cwtmatr, 99)

        if DEBUG:
            print(f"  {wavelet_name}: vmax = {vmax:.2f} (99th percentile, max = {cwtmatr.max():.2f})")

        return cwtmatr, freqs, vmax
    except Exception as e:
        if DEBUG: print(f"Failed to compute CWT for {wavelet_name}: {e}")
        return None, None, None

def plot_cwt_result(cwtmatr, freqs, time, vmax, ax, title):
    """Plot CWT result as a scalogram."""
    if cwtmatr is None:
        ax.text(0.5, 0.5, 'CWT computation failed', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title} (Error)', fontsize=10, color='red')
        return
    
    t_ax, f_ax = np.meshgrid(time*1000, freqs/1000)
    pcm = ax.pcolormesh(t_ax, f_ax, cwtmatr, cmap='jet', vmax=vmax)
    ax.set_yscale('log', base=2)
    ax.set_ylim(1, 50)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Freq [kHz]')
    ax.set_title(title, fontsize=10)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 50])
    
    return pcm

def create_cmor_comparison_figure():
    """Create the main cmor comparison figure in 3x3 grid layout."""
    folder = get_paths()['hdf5']

    # Find and load the target file
    try:
        filepath = find_hdf5_file(TARGET_TRACKID, folder)
        print(f"Loading data from: {filepath}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Load data from HDF5 file
    with h5py.File(filepath, 'r') as file:
        # Use AMPM photodiode data (same as cwt.py logic)
        group, time_key, series_key = ('AMPM', 'Time', 'Photodiode1Bits')

        try:
            t = np.array(file[f'{group}/{time_key}'])
            s = np.array(file[f'{group}/{series_key}'])
        except KeyError:
            print(f"Warning: AMPM data not found, trying KH data")
            group, time_key, series_key = ('KH', 'time', 'max_depth')
            t = np.array(file[f'{group}/{time_key}'])
            s = np.array(file[f'{group}/{series_key}'])

        # Calculate sampling parameters
        sampling_period = round(t[1]-t[0], 9)
        sampling_rate = round(1/sampling_period, 7)

        # Crop signals from 0 to 5 ms
        crop_start = 0.0  # seconds
        crop_end = 0.005  # 5 ms in seconds

        # Find indices for cropping
        start_idx = np.searchsorted(t, crop_start)
        end_idx = np.searchsorted(t, crop_end)

        # Crop the data
        t = t[start_idx:end_idx]
        s = s[start_idx:end_idx]

        # Apply logarithm to smooth peaks at laser onset
        s = np.log(s + 1)  # Add 1 to avoid log(0)

        if DEBUG:
            print(f"Original data: {len(t) + start_idx + (len(s) - end_idx)} samples")
            print(f"Cropped data: {len(t)} samples (0-5 ms)")
            print(f"Sampling period: {sampling_period*1000:.3f} ms")
            print(f"Sampling rate: {sampling_rate:.0f} Hz")
            print(f"Time range: {t[0]*1000:.1f} to {t[-1]*1000:.1f} ms")
            print(f"Signal range: {s.min():.2f} to {s.max():.2f}")

    # Create 3x3 grid figure
    fig = plt.figure(figsize=(18, 14), dpi=150)
    fig.suptitle(f'Complex Morlet Wavelet Parameter Grid - {TARGET_TRACKID}\n'
                 f'Bandwidth (rows) × Center Frequency (columns)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create 3x3 grid with space for colorbars
    # Reduced top margin to 0.89 to make room for CF labels above plots
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05],
                         hspace=0.4, wspace=0.3,
                         left=0.08, right=0.95, top=0.89, bottom=0.08)

    # Iterate through grid: rows = bandwidth, cols = center frequency
    for row_idx, bw in enumerate(BANDWIDTH_VALUES):
        for col_idx, cf in enumerate(CENTER_FREQ_VALUES):
            wavelet_name = CMOR_GRID[row_idx][col_idx]

            # Create subplot for this wavelet's CWT
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if DEBUG:
                print(f"\nProcessing [{row_idx}, {col_idx}]: {wavelet_name}")

            # Compute and plot CWT
            cwtmatr, freqs, vmax = compute_cwt_for_signal(s, t, wavelet_name, sampling_period)
            pcm = plot_cwt_result(cwtmatr, freqs, t, vmax, ax, wavelet_name)

            # Add row labels (bandwidth) on the left
            if col_idx == 0:
                ax.set_ylabel(f'Freq [kHz]', fontsize=9)
                # Add bold bandwidth label with extra spacing
                ax.text(-0.35, 0.5, f'BW={bw}', transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       ha='center', va='center')
            else:
                ax.set_ylabel('Freq [kHz]', fontsize=9)

            # Add column labels (center frequency) on top
            if row_idx == 0:
                # Normal wavelet name at regular title position
                ax.set_title(f'{wavelet_name}', fontsize=9)
                # Bold CF label positioned above with spacing
                ax.text(0.5, 1.18, f'CF={cf}', transform=ax.transAxes,
                       fontsize=11, fontweight='bold',
                       ha='center', va='bottom')
            else:
                ax.set_title(wavelet_name, fontsize=9)

            # Only show x-label on bottom row
            if row_idx < len(BANDWIDTH_VALUES) - 1:
                ax.set_xlabel('')

    # Add a single colorbar on the right for the entire figure
    cbar_ax = fig.add_subplot(gs[:, 3])
    # Use the last pcm for colorbar (they should all have similar ranges with auto vmax)
    if pcm is not None:
        fig.colorbar(pcm, cax=cbar_ax, label='CWT Coefficient Magnitude')

    return fig

def main():
    """Main execution function."""
    print("=" * 60)
    print("Complex Morlet Wavelet Parameter Grid Comparison")
    print("=" * 60)

    print(f"Target trackid: {TARGET_TRACKID}")
    print(f"Grid layout: 3x3 (Bandwidth × Center Frequency)")
    print(f"Bandwidth values: {BANDWIDTH_VALUES}")
    print(f"Center frequency values: {CENTER_FREQ_VALUES}")
    print(f"Total wavelets: {len(BANDWIDTH_VALUES) * len(CENTER_FREQ_VALUES)}")
    
    # Generate the comparison figure
    try:
        fig = create_cmor_comparison_figure()
        
        if fig is None:
            print("Failed to create comparison figure")
            return
        
        # Save the figure with new filename
        if SAVE_OUTPUT:
            folder = get_paths()['hdf5']
            output_dir = Path(folder, 'CWT')
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f'cmor_comparison_{TARGET_TRACKID}.png'
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"\\n✅ Saved cmor comparison figure to: {output_path}")
        else:
            plt.show()
        
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating comparison figure: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()