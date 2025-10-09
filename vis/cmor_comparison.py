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

# Complex Morlet wavelets to test - organized by characteristics
CMOR_WAVELETS_TO_TEST = {
    'High Frequency Resolution (Low Bandwidth)': ['cmor1.0-1.0', 'cmor1.5-1.0'],
    'Balanced Resolution (Medium Bandwidth)': ['cmor2.0-1.0', 'cmor2.5-1.0'], 
    'High Time Resolution (High Bandwidth)': ['cmor3.0-1.0', 'cmor4.0-1.0'],
    'Different Center Frequencies': ['cmor2.0-0.5', 'cmor2.0-2.0']
}

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
    """Compute CWT for given signal and wavelet using dynamic scale calculation."""
    try:
        # Use the new dynamic scale calculation from tools.py
        scales, vmax = get_cwt_scales(wavelet_name, 128)  # Fewer scales for faster computation
        
        # Compute CWT
        cwtmatr, freqs = pywt.cwt(signal, scales, wavelet_name, sampling_period)
        cwtmatr = np.abs(cwtmatr)
        
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
    """Create the main cmor comparison figure."""
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
        
        if DEBUG:
            print(f"Original data: {len(t) + start_idx + (len(s) - end_idx)} samples")
            print(f"Cropped data: {len(t)} samples (0-5 ms)")
            print(f"Sampling period: {sampling_period*1000:.3f} ms")
            print(f"Sampling rate: {sampling_rate:.0f} Hz")
            print(f"Time range: {t[0]*1000:.1f} to {t[-1]*1000:.1f} ms")
            print(f"Signal range: {s.min():.2f} to {s.max():.2f}")
    
    # Collect all cmor wavelets to test
    all_wavelets = []
    for category, wavelets in CMOR_WAVELETS_TO_TEST.items():
        all_wavelets.extend(wavelets)
    
    n_wavelets = len(all_wavelets)
    n_cols = 3  # Wavelet function, CWT result, colorbar
    n_rows = n_wavelets
    
    # Create the figure with more space between subplots
    fig = plt.figure(figsize=(15, 3.5*n_wavelets), dpi=150)
    fig.suptitle(f'Complex Morlet Wavelet Comparison - {TARGET_TRACKID} ({series_key})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create subplots with increased spacing
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=[1, 2, 0.1], 
                         hspace=0.6, wspace=0.3)
    
    for i, wavelet_name in enumerate(all_wavelets):
        # Create subplots for this wavelet
        ax_wavelet = fig.add_subplot(gs[i, 0])
        ax_cwt = fig.add_subplot(gs[i, 1])
        ax_cbar = fig.add_subplot(gs[i, 2])
        
        # Plot wavelet function
        success = plot_wavelet_function(wavelet_name, ax_wavelet)
        
        if success:
            # Compute and plot CWT using dynamic scale calculation
            cwtmatr, freqs, vmax = compute_cwt_for_signal(s, t, wavelet_name, sampling_period)
            pcm = plot_cwt_result(cwtmatr, freqs, t, vmax, ax_cwt, f'CWT: {wavelet_name}')
            
            # Add colorbar
            if pcm is not None:
                fig.colorbar(pcm, cax=ax_cbar, label='Intensity')
            else:
                ax_cbar.axis('off')
        else:
            ax_cwt.axis('off')
            ax_cbar.axis('off')
    
    # Add signal plot at the top with more space for title
    ax_signal = fig.add_axes([0.1, 0.92, 0.8, 0.04])
    ax_signal.plot(t*1000, s, 'k-', linewidth=0.8)
    ax_signal.set_xlim(t[0]*1000, t[-1]*1000)
    ax_signal.set_ylabel('PD Signal', fontsize=8)
    ax_signal.tick_params(labelsize=8)
    ax_signal.set_title(f'Raw Signal: {series_key}', fontsize=10)
    
    # Adjust layout with more space for title
    plt.subplots_adjust(top=0.90)
    return fig

def main():
    """Main execution function."""
    print("=" * 60)
    print("Complex Morlet Wavelet Comparison Script")
    print("=" * 60)
    
    print(f"Target trackid: {TARGET_TRACKID}")
    print(f"Wavelets to test: {sum(len(w) for w in CMOR_WAVELETS_TO_TEST.values())}")
    
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