import os, sys, functools, h5py, pywt, glob, argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import pyplot as plt, ticker as mticker
from scipy.signal import find_peaks, savgol_filter

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, printProgressBar, get_cwt_scales

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate CWT visualizations with optional peak labeling')
parser.add_argument('--peak-label-mode', action='store_true',
                   help='Enable peak labeling mode (reads from config file and processes all configured trackids)')
parser.add_argument('--trackid', type=str, default=None,
                   help='Single trackid to analyze in peak labeling mode (overrides config file)')
parser.add_argument('--timestamps', type=float, nargs='+', default=None,
                   help='Timestamps (in seconds) to mark peaks (overrides config file)')
parser.add_argument('--peak-number', type=int, default=None,
                   help='Number of peaks to mark at each timestamp (overrides config file)')
parser.add_argument('--peak-prominence', type=float, default=None,
                   help='Minimum prominence for peak detection (overrides config file)')
args = parser.parse_args()

folder = get_paths()['hdf5']

# Hardcoded path to config file
PEAK_LABEL_CONFIG_PATH = Path(folder) / 'cwt_peak_label_config.json'

def load_peak_label_config():
    """Load peak labeling configuration from JSON file."""
    if not PEAK_LABEL_CONFIG_PATH.exists():
        print(f"WARNING: Config file not found: {PEAK_LABEL_CONFIG_PATH}")
        print("Using default parameters")
        return {
            'trackids': {},
            'default_parameters': {
                'timestamps': [0.0015, 0.002625, 0.003575],
                'peak_number': 2,
                'peak_prominence': 0.01,
                'smoothing_window': 5,
                'smoothing_polyorder': 2
            }
        }

    with open(PEAK_LABEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config

def get_trackid_config(config, trackid):
    """Get configuration for a specific trackid, falling back to defaults."""
    trackid_config = config.get('trackids', {}).get(trackid, {})
    defaults = config.get('default_parameters', {})

    # Get timestamps in ms and convert to seconds
    timestamps_ms = trackid_config.get('timestamps_ms', defaults.get('timestamps_ms', [1.5, 2.625, 3.575]))
    timestamps_s = [t / 1000.0 for t in timestamps_ms]  # Convert ms to seconds

    # Merge trackid-specific config with defaults
    return {
        'timestamps': timestamps_s,
        'timestamps_ms': timestamps_ms,  # Keep original for display
        'peak_number': trackid_config.get('peak_number', defaults.get('peak_number', 2)),
        'peak_prominence': trackid_config.get('peak_prominence', defaults.get('peak_prominence', 0.01)),
        'time_averaging_window': trackid_config.get('time_averaging_window', defaults.get('time_averaging_window', 3)),
        'smoothing_window': trackid_config.get('smoothing_window', defaults.get('smoothing_window', 5)),
        'smoothing_polyorder': trackid_config.get('smoothing_polyorder', defaults.get('smoothing_polyorder', 2))
    }

group, time, series = ('AMPM', 'Time', 'Photodiode1Bits')
# group, time, series = ('KH', 'time', 'max_depth')
running_mean_window = None

mode = 'save' # 'preview' or 'save'
show_wavelet = True
debug = not args.peak_label_mode  # Disable debug output in peak label mode
series_name = f'{group}_{series}'

# Load peak labeling configuration
peak_label_enabled = args.peak_label_mode
trackids_to_process = []

if peak_label_enabled:
    config = load_peak_label_config()
    print(f"Loaded config from: {PEAK_LABEL_CONFIG_PATH}\n")

    # If CLI trackid specified, use only that trackid
    if args.trackid:
        trackids_to_process = [args.trackid]
        print(f"Peak labeling mode enabled for single trackid (CLI): {args.trackid}")
    else:
        # Otherwise, use all trackids from config
        trackids_to_process = list(config.get('trackids', {}).keys())
        if len(trackids_to_process) == 0:
            print("ERROR: No trackids found in config file")
            sys.exit(1)
        print(f"Peak labeling mode enabled for {len(trackids_to_process)} trackids from config:")
        for tid in trackids_to_process:
            print(f"  - {tid}")

    print()  # Blank line

def plot_wavelet(wavelet):
    [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(level=10)
    fig, ax = plt.subplots(1, 1, figsize=(3.15, 3.15), dpi = 300)
    ax.plot(x, psi)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-1, 1))
    if mode == 'save':
        output_folder = Path(folder, 'CWT', series_name, wavelet)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(Path(output_folder, f'{wavelet}.png'))
    else:
        plt.show()
        plt.close()

files = sorted(glob.glob(f'{folder}/*.hdf5'))

# Filter files in peak labeling mode
if peak_label_enabled:
    filtered_files = []
    for trackid in trackids_to_process:
        matching_files = [f for f in files if Path(f).name[:7] == trackid]
        if len(matching_files) == 0:
            print(f"WARNING: Trackid {trackid} not found in {folder}")
        else:
            filtered_files.extend(matching_files)

    if len(filtered_files) == 0:
        print("ERROR: No HDF5 files found for any configured trackid")
        sys.exit(1)

    files = filtered_files
    print(f"Processing {len(files)} file(s) in peak labeling mode\n")

for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]

    # Get trackid-specific config if in peak labeling mode
    if peak_label_enabled:
        trackid_cfg = get_trackid_config(config, trackid)

        # CLI arguments override config file values
        peak_label_config = {
            'enabled': True,
            'timestamps': args.timestamps if args.timestamps is not None else trackid_cfg['timestamps'],
            'peak_number': args.peak_number if args.peak_number is not None else trackid_cfg['peak_number'],
            'prominence': args.peak_prominence if args.peak_prominence is not None else trackid_cfg['peak_prominence'],
            'time_averaging_window': trackid_cfg['time_averaging_window'],
            'smoothing_window': trackid_cfg['smoothing_window'],
            'smoothing_polyorder': trackid_cfg['smoothing_polyorder']
        }

        print(f"{'='*80}")
        print(f"Processing trackid: {trackid}")
        print(f"Timestamps: {peak_label_config['timestamps']}")
        print(f"Peak number: {peak_label_config['peak_number']}")
        print(f"Prominence: {peak_label_config['prominence']}")
        print(f"{'='*80}\n")
    else:
        # Normal mode (not peak labeling)
        peak_label_config = {'enabled': False}
        print(trackid)
    
    with h5py.File(filepath, 'r') as file:
        t = np.array(file[f'{group}/{time}'])[510:-510]
        s = np.array(file[f'{group}/{series}'])[510:-510]
        s = np.log(s + 1)
        # t = np.array(file[f'{group}/{time}'])
        # s = np.array(file[f'{group}/{series}'])
        if running_mean_window != None:
            s = np.convolve(s, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        xray_im = np.array(file['bs-f40'])[-1]
        
        # Frequency range calculation
        sampling_period = round(t[1]-t[0], 9)
        print(sampling_period)
        sampling_duration = round(t[-1]-t[0], 9)
        sampling_rate = round(1/sampling_period, 7)
        print(sampling_rate)
        
        nyquist_freq = sampling_rate / 2
        min_freq = 1 / sampling_duration  # Lowest resolvable frequency
        max_freq = nyquist_freq
        
        wavelet = "cmor2.5-0.5"
        # wavelet = "fbsp4-0.6-1.0"
        # wavelet = 'cmor10.0-0.3'
        scales, vmax = get_cwt_scales(wavelet, num=256, sampling_rate=sampling_rate)
        if debug: print('scales:\n', scales)
        
        if debug:
            print(f'min: {pywt.scale2frequency(wavelet, scales[-1])*sampling_rate} ',
                f'max: {pywt.scale2frequency(wavelet, scales[0])*sampling_rate}')
        
        if show_wavelet:
            plot_wavelet(wavelet)
            show_wavelet = False

        # Apply symmetric padding to minimize edge artifacts (best practice)
        # Pad with 2x signal length on each side (5x total)
        pad_width = 2 * len(s)
        s_pad = np.pad(s, pad_width, mode='symmetric')
        if debug: print(f'Padded signal length: {len(s_pad)} (5x original: {len(s)})')

        # Perform CWT on padded signal
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)

        # Normalise to preserve amplitude proportionality
        # cwtmatr /= np.sqrt(scales[:, None])

        # Crop to original signal length (extract middle section, removing padding)
        cwtmatr = np.abs(cwtmatr[:, pad_width:pad_width+len(s)])

        # Prepare common variables for plotting
        t_ax, f_ax = np.meshgrid(t*1000, freqs/1000)
        cwt_cmap = 'jet'
        plt.rcParams.update({'font.size': 9})
        kw = {'height_ratios':[1, 1, 1], "width_ratios":[95, 5]}

        # Skip normal CWT plot in peak labeling mode
        if not peak_label_config['enabled']:
            fig, ((ax1, ax1b), (ax2, ax2b), (ax3, ax3b)) = plt.subplots(3, 2,
                figsize = [6.3, 7],
                dpi = 300,
                gridspec_kw = kw)
            fig.suptitle(f'{trackid} - {series}')

            ax1.plot(t*1000, s, lw=0.75)
            ax1.set_xlim(t[0]*1000, t[-1]*1000)
            ax1.set_ylabel('Intensity')
            ax1.set_xlabel('Time [ms]')

            pcm = ax2.pcolormesh(t_ax, f_ax, cwtmatr, cmap=cwt_cmap)#, vmax=vmax)
            ax2.set_yscale('log', base=2)
            ax2.set_ylim(1, 50)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax2.set_xlabel('Time [ms]')
            ax2.set_ylabel('Frequency [kHz]')
            ax2.set_yticks([1, 2, 4, 8, 16, 32, 50])
            fig.colorbar(pcm, cax=ax2b, label='Intensity')

            ax3.imshow(xray_im[150:450], cmap='gray')  # xray_im[150:450] for full frame 40 kHz
            scalebar = ScaleBar(4.3,
                "µm",
                length_fraction = 0.15,
                width_fraction = 0.02,
                frameon = False,
                color = 'w',
                location = 'lower right')
            ax3.add_artist(scalebar)

            for ax in [ax1b, ax3, ax3b]:
                ax.axis('off')
            plt.tight_layout()

            if mode == 'save':
                output_folder = Path(folder, 'CWT', series_name, wavelet, cwt_cmap)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                plt.savefig(Path(output_folder, f'{trackid}_{series}_CWT_{wavelet}.png'))
            else:
                plt.show()
            plt.close()

        # Generate peak-labeled version if configured
        if peak_label_config['enabled']:
            print(f"\nGenerating peak-labeled CWT for {trackid}...")

            # Create new figure with same structure
            fig, ((ax1, ax1b), (ax2, ax2b), (ax3, ax3b)) = plt.subplots(3, 2,
                figsize = [6.3, 7],
                dpi = 300,
                gridspec_kw = kw)
            fig.suptitle(f'{trackid} - {series} (Labeled Peaks)')

            # Plot signal
            ax1.plot(t*1000, s, lw=0.75)
            ax1.set_xlim(t[0]*1000, t[-1]*1000)
            ax1.set_ylabel('Intensity')
            ax1.set_xlabel('Time [ms]')

            # Plot CWT
            pcm = ax2.pcolormesh(t_ax, f_ax, cwtmatr, cmap=cwt_cmap)
            ax2.set_yscale('log', base=2)
            ax2.set_ylim(1, 50)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax2.set_xlabel('Time [ms]')
            ax2.set_ylabel('Frequency [kHz]')
            ax2.set_yticks([1, 2, 4, 8, 16, 32, 50])
            fig.colorbar(pcm, cax=ax2b, label='Intensity [a.u.]', fraction=0.01)

            # Draw vertical dashed lines at each timestamp
            for timestamp in peak_label_config['timestamps']:
                time_ms = timestamp * 1000  # Convert to milliseconds
                ax2.axvline(x=time_ms, color='black', linestyle='--', linewidth=0.75, alpha=1)

            # Find and mark peaks at specified timestamps using scipy peak detection
            print(f"\nDEBUG: Time range: [{t[0]:.4f}, {t[-1]:.4f}] seconds ({t[0]*1000:.2f} - {t[-1]*1000:.2f} ms)")
            print(f"DEBUG: Peak detection at {len(peak_label_config['timestamps'])} timestamps")

            # Store peak detection data for diagnostic plot
            peak_detection_data = []

            for timestamp in peak_label_config['timestamps']:
                # Find closest time index
                time_idx = np.argmin(np.abs(t - timestamp))
                actual_time = t[time_idx]
                print(f"\n  Timestamp {timestamp}s (actual: {actual_time:.3f}s, index: {time_idx})")

                # Time averaging: average vectors over window centered at timestamp
                # For window=3: average indices [time_idx-1, time_idx, time_idx+1]
                avg_window = peak_label_config['time_averaging_window']
                half_window = avg_window // 2

                # Calculate valid index range with bounds checking
                start_idx = max(0, time_idx - half_window)
                end_idx = min(cwtmatr.shape[1], time_idx + half_window + 1)

                print(f"    Time averaging window: {avg_window} (indices {start_idx} to {end_idx-1})")

                # Average intensity vectors over time window
                intensity_vector_raw = np.mean(cwtmatr[:, start_idx:end_idx], axis=1)

                # Apply light smoothing (Savitzky-Golay filter from config)
                intensity_vector = savgol_filter(intensity_vector_raw,
                                                window_length=peak_label_config['smoothing_window'],
                                                polyorder=peak_label_config['smoothing_polyorder'])

                print(f"    Intensity vector shape: {intensity_vector.shape}")
                print(f"    Intensity range (smoothed): [{intensity_vector.min():.2f}, {intensity_vector.max():.2f}]")

                # Find peaks using scipy.signal.find_peaks with prominence threshold
                peaks, properties = find_peaks(intensity_vector,
                                              prominence=peak_label_config['prominence'])
                print(f"    Found {len(peaks)} peaks with prominence >= {peak_label_config['prominence']}")

                # Sort peaks by intensity (height) and select top N
                top_peaks = np.array([])
                if len(peaks) > 0:
                    peak_intensities = intensity_vector[peaks]
                    peak_freqs = freqs[peaks] / 1000  # Convert to kHz

                    # Show all peaks found
                    for i, (pk_idx, pk_int, pk_freq) in enumerate(zip(peaks, peak_intensities, peak_freqs)):
                        print(f"      Peak {i+1}: freq={pk_freq:.2f} kHz, intensity={pk_int:.2f}")

                    sorted_indices = np.argsort(peak_intensities)[::-1]  # Descending order
                    top_peaks = peaks[sorted_indices[:peak_label_config['peak_number']]]

                    print(f"    Marking top {len(top_peaks)} peaks:")

                    # Mark top peaks with black 'x'
                    for peak_idx in top_peaks:
                        freq_val = freqs[peak_idx] / 1000  # Convert to kHz
                        time_val = t[time_idx] * 1000  # Convert to ms
                        intensity_val = intensity_vector[peak_idx]
                        print(f"      -> freq={freq_val:.2f} kHz, time={time_val:.2f} ms, intensity={intensity_val:.2f}")
                        ax2.plot(time_val, freq_val, 'kx', markersize=5, markeredgewidth=1)
                else:
                    print(f"    WARNING: No peaks found! Try lowering --peak-prominence")

                # Store data for diagnostic plot
                peak_detection_data.append({
                    'timestamp': timestamp,
                    'actual_time': actual_time,
                    'time_idx': time_idx,
                    'intensity_vector_raw': intensity_vector_raw.copy(),
                    'intensity_vector': intensity_vector.copy(),
                    'freqs': freqs.copy(),
                    'all_peaks': peaks.copy(),
                    'top_peaks': top_peaks.copy(),
                    'properties': properties
                })

            # Plot X-ray image
            ax3.imshow(xray_im[150:450], cmap='gray')
            scalebar = ScaleBar(4.3,
                "µm",
                length_fraction = 0.15,
                width_fraction = 0.02,
                frameon = False,
                color = 'w',
                location = 'lower right')
            ax3.add_artist(scalebar)

            for ax in [ax1b, ax3, ax3b]:
                ax.axis('off')
            plt.tight_layout()

            # Save to labeled_peaks subfolder
            labeled_output_folder = Path(folder, 'CWT', series_name, wavelet, cwt_cmap, 'labelled_peaks')
            if not os.path.exists(labeled_output_folder):
                os.makedirs(labeled_output_folder)
            plt.savefig(Path(labeled_output_folder, f'{trackid}_{series}_CWT_{wavelet}_labeled.png'))
            plt.close()

            print(f"✓ Saved labeled peak plot to: {labeled_output_folder}")

            # Generate peak detection diagnostic plot
            print(f"\nGenerating peak detection diagnostic plot...")
            n_timestamps = len(peak_detection_data)
            fig_diag, axes = plt.subplots(n_timestamps, 1, figsize=(8, 3*n_timestamps), dpi=150, sharex=True)
            if n_timestamps == 1:
                axes = [axes]  # Make it iterable

            for idx, (ax, data) in enumerate(zip(axes, peak_detection_data)):
                freqs_khz = data['freqs'] / 1000  # Convert to kHz
                intensity_raw = data['intensity_vector_raw']
                intensity = data['intensity_vector']
                all_peaks = data['all_peaks']
                top_peaks = data['top_peaks']
                timestamp = data['timestamp']
                actual_time = data['actual_time']

                # Plot raw intensity vs frequency (light grey)
                ax.plot(freqs_khz, intensity_raw, color='lightgrey', linewidth=1,
                       label='Raw CWT Intensity', alpha=0.7)

                # Plot smoothed intensity vs frequency
                ax.plot(freqs_khz, intensity, 'b-', linewidth=1.5, label='Smoothed CWT Intensity')

                # Mark all detected peaks
                if len(all_peaks) > 0:
                    ax.plot(freqs_khz[all_peaks], intensity[all_peaks],
                           'go', markersize=6, label=f'All peaks (n={len(all_peaks)})')

                # Mark top peaks
                if len(top_peaks) > 0:
                    ax.plot(freqs_khz[top_peaks], intensity[top_peaks],
                           'rx', markersize=10, markeredgewidth=2,
                           label=f'Top {len(top_peaks)} peaks')

                    # Annotate top peaks with frequency
                    for peak_idx in top_peaks:
                        freq_val = freqs_khz[peak_idx]
                        intensity_val = intensity[peak_idx]
                        ax.annotate(f'{freq_val:.1f} kHz',
                                   xy=(freq_val, intensity_val),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color='red')

                ax.set_xlabel('Frequency [kHz]')
                ax.set_ylabel('CWT Intensity')
                ax.set_title(f'Timestamp {timestamp:.4f}s (actual: {actual_time:.4f}s, {actual_time*1000:.2f} ms)')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3, which='both')
                ax.set_xscale('log', base=2)
                ax.set_xlim(1, 50)  # Match CWT plot frequency range
                ax.set_xticks([1, 2, 4, 8, 16, 32, 50])
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

            fig_diag.suptitle(f'{trackid} - Peak Detection Diagnostics\nProminence threshold: {peak_label_config["prominence"]}',
                            fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Save diagnostic plot
            diag_filename = f'{trackid}_{series}_peak_detection_diagnostics.png'
            plt.savefig(Path(labeled_output_folder, diag_filename))
            plt.close()

            print(f"✓ Saved peak detection diagnostic plot: {diag_filename}")

        printProgressBar(i+1, len(files), prefix='Progress', suffix=trackid)


    