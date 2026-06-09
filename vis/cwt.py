import os, sys, functools, h5py, pywt, glob, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import pyplot as plt, ticker as mticker
from scipy.signal import find_peaks, savgol_filter

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, printProgressBar, get_cwt_scales

# ─── Command Line Arguments ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Generate CWT visualizations with optional peak labeling')
parser.add_argument('--peak_label_mode', action='store_true',
                   help='Enable peak labeling mode (reads from config file and processes all configured trackids)')
parser.add_argument('--trackid', type=str, default=None,
                   help='Single trackid to analyze in peak labeling mode (overrides config file)')
parser.add_argument('--timestamps', type=float, nargs='+', default=None,
                   help='Timestamps (in seconds) to mark peaks (overrides config file)')
parser.add_argument('--peak_number', type=int, default=None,
                   help='Number of peaks to mark at each timestamp (overrides config file)')
parser.add_argument('--peak_prominence', type=float, default=None,
                   help='Minimum prominence for peak detection (overrides config file)')
parser.add_argument('--auto_vmax', action='store_true',
                   help='Auto-scale colormap to full range per image instead of using preset vmax')
args = parser.parse_args()

# ─── Global Settings ───────────────────────────────────────────────────────────

folder = get_paths()['hdf5']

# group, time, series = ('AMPM', 'Time', 'Photodiode1Bits')
group, time, series = ('KH', 'time', 'area')
mad_threshold = None   # MAD outlier removal before filtering, e.g. 4.0; None to disable
running_mean_window = None
savgol_window = None   # set to an odd integer to enable, e.g. 5
savgol_polyorder = 3   # polynomial order for savgol filter

# AMPM signals have 500 samples additional margin on each side, trim these away
trim_start = 510 if group == 'AMPM' else 10
trim_end = 510  if group == 'AMPM' else 10

mode = 'save'  # 'preview' or 'save'
show_wavelet = True
debug = False
series_name = f'{group}_{series}'
cwt_cmap = 'inferno'
plt.rcParams.update({'font.size': 9})

wavelet = "cmor1.5-1.0"
# wavelet = "fbsp4-0.6-1.0"
# wavelet = 'cmor10.0-0.3'
cwt_scales_num = 256
pad_factor = 2   # padding = pad_factor * len(s) on each side
freq_min = 1000  # Hz, or 'auto' to use 1/signal_duration
freq_max = 'auto'  # Hz, or 'auto' to use nyquist frequency

# ─── Peak Labeling Module ──────────────────────────────────────────────────────
# Toggle via --peak_label_mode CLI flag. All peak labeling logic lives here.

def load_peak_label_config():
    return {
        'default_parameters': {
            'timestamps_ms': [1.00, 2.00, 3.00],
            'peak_number': 1,
            'peak_prominence': 0.0003,
            'time_averaging_window': 3,
            'smoothing_window': 21,
            'smoothing_polyorder': 2
        },
        'trackids': {
            '0323_05': {
                'timestamps_ms': [0.3, 0.5, 0.625, 1.525, 2.6, 3.575, 4.0],
                'peak_number': 2,
                'peak_prominence': 10,
                'time_averaging_window': 3,
                'smoothing_window': 21,
                'smoothing_polyorder': 2
            }
        }
    }

def get_trackid_config(config, trackid):
    """Get configuration for a specific trackid, falling back to default_parameters."""
    trackid_config = config.get('trackids', {}).get(trackid, {})
    defaults = config['default_parameters']
    timestamps_ms = trackid_config.get('timestamps_ms', defaults['timestamps_ms'])
    return {
        'timestamps': [t / 1000.0 for t in timestamps_ms],
        'timestamps_ms': timestamps_ms,
        'peak_number': trackid_config.get('peak_number', defaults['peak_number']),
        'peak_prominence': trackid_config.get('peak_prominence', defaults['peak_prominence']),
        'time_averaging_window': trackid_config.get('time_averaging_window', defaults['time_averaging_window']),
        'smoothing_window': trackid_config.get('smoothing_window', defaults['smoothing_window']),
        'smoothing_polyorder': trackid_config.get('smoothing_polyorder', defaults['smoothing_polyorder'])
    }

def detect_peaks_at_timestamps(cwtmatr, freqs, t, peak_cfg):
    """
    Detect CWT peaks at each configured timestamp.
    Returns a list of result dicts, one per timestamp.
    """
    
    print(f"\nDEBUG: Time range: [{t[0]:.4f}, {t[-1]:.4f}] seconds ({t[0]*1000:.2f} - {t[-1]*1000:.2f} ms)")
    print(f"DEBUG: Peak detection at {len(peak_cfg['timestamps'])} timestamps")

    results = []
    for timestamp in peak_cfg['timestamps']:
        time_idx = np.argmin(np.abs(t - timestamp))
        actual_time = t[time_idx]
        print(f"\n  Timestamp {timestamp}s (actual: {actual_time:.3f}s, index: {time_idx})")

        avg_window = peak_cfg['time_averaging_window']
        half_window = avg_window // 2
        start_idx = max(0, time_idx - half_window)
        end_idx = min(cwtmatr.shape[1], time_idx + half_window + 1)
        print(f"    Time averaging window: {avg_window} (indices {start_idx} to {end_idx-1})")

        intensity_raw = np.mean(cwtmatr[:, start_idx:end_idx], axis=1)
        intensity = savgol_filter(intensity_raw,
                                  window_length=peak_cfg['smoothing_window'],
                                  polyorder=peak_cfg['smoothing_polyorder'])

        print(f"    Intensity vector shape: {intensity.shape}")
        print(f"    Intensity range (smoothed): [{intensity.min():.2f}, {intensity.max():.2f}]")

        peaks, properties = find_peaks(intensity, prominence=peak_cfg['prominence'])
        print(f"    Found {len(peaks)} peaks with prominence >= {peak_cfg['prominence']}")

        top_peaks = np.array([])
        if len(peaks) > 0:
            peak_intensities = intensity[peaks]
            peak_freqs = freqs[peaks] / 1000
            for i, (pk_idx, pk_int, pk_freq) in enumerate(zip(peaks, peak_intensities, peak_freqs)):
                print(f"      Peak {i+1}: freq={pk_freq:.2f} kHz, intensity={pk_int:.2f}")
            sorted_indices = np.argsort(peak_intensities)[::-1]
            top_peaks = peaks[sorted_indices[:peak_cfg['peak_number']]]
            print(f"    Marking top {len(top_peaks)} peaks:")
            for peak_idx in top_peaks:
                freq_val = freqs[peak_idx] / 1000
                time_val = t[time_idx] * 1000
                intensity_val = intensity[peak_idx]
                print(f"      -> freq={freq_val:.2f} kHz, time={time_val:.2f} ms, intensity={intensity_val:.2f}")
        else:
            print(f"    WARNING: No peaks found! Try lowering --peak_prominence")

        results.append({
            'timestamp': timestamp,
            'actual_time': actual_time,
            'time_idx': time_idx,
            'intensity_vector_raw': intensity_raw.copy(),
            'intensity_vector': intensity.copy(),
            'freqs': freqs.copy(),
            'all_peaks': peaks.copy(),
            'top_peaks': top_peaks.copy(),
            'properties': properties
        })
    return results

def plot_peak_diagnostics(peak_data, trackid, peak_cfg, output_folder, fmin_hz, fmax_hz):
    """Generate and save the peak detection diagnostic plot."""
    print(f"\nGenerating peak detection diagnostic plot...")
    n = len(peak_data)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), dpi=150, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, peak_data):
        freqs_khz = data['freqs'] / 1000
        intensity_raw = data['intensity_vector_raw']
        intensity = data['intensity_vector']
        all_peaks = data['all_peaks']
        top_peaks = data['top_peaks']

        ax.plot(freqs_khz, intensity_raw, color='lightgrey', linewidth=1,
                label='Raw CWT Intensity', alpha=0.7)
        ax.plot(freqs_khz, intensity, 'b-', linewidth=1.5, label='Smoothed CWT Intensity')

        if len(all_peaks) > 0:
            ax.plot(freqs_khz[all_peaks], intensity[all_peaks],
                    'go', markersize=6, label=f'All peaks (n={len(all_peaks)})')
        if len(top_peaks) > 0:
            ax.plot(freqs_khz[top_peaks], intensity[top_peaks],
                    'rx', markersize=10, markeredgewidth=2,
                    label=f'Top {len(top_peaks)} peaks')
            for peak_idx in top_peaks:
                freq_val = freqs_khz[peak_idx]
                ax.annotate(f'{freq_val:.1f} kHz',
                            xy=(freq_val, intensity[peak_idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='red')

        ax.set_ylabel('CWT Intensity')
        ax.set_title(f"Timestamp {data['timestamp']:.4f}s "
                     f"(actual: {data['actual_time']:.4f}s, {data['actual_time']*1000:.2f} ms)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xscale('log', base=2)
        ax.set_xlim(fmin_hz / 1000, fmax_hz / 1000)
        ax.set_xticks(_freq_yticks_khz(fmin_hz, fmax_hz))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel('Frequency [kHz]')

    fig.suptitle(f'{trackid} - Peak Detection Diagnostics\nProminence threshold: {peak_cfg["prominence"]}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    diag_filename = f'{trackid}_{series}_peak_detection_diagnostics.png'
    plt.savefig(Path(output_folder, diag_filename))
    plt.close()
    print(f"✓ Saved peak detection diagnostic plot: {diag_filename}")

# ─── Utility Functions ─────────────────────────────────────────────────────────

def mad_interpolate(s, threshold):
    """
    Detect outliers using the Median Absolute Deviation and replace them
    with linear interpolation from neighbouring clean samples.

    Outlier condition: |s - median(s)| > threshold * MAD
    where MAD = median(|s - median(s)|)
    """
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return s  # Flat signal — no outliers possible
    outliers = np.abs(s - med) > threshold * mad
    if not np.any(outliers):
        return s
    clean = s.copy()
    x_all = np.arange(len(s))
    x_good = x_all[~outliers]
    clean[outliers] = np.interp(x_all[outliers], x_good, s[x_good])
    return clean

def _freq_yticks_khz(fmin_hz, fmax_hz):
    """Power-of-2 tick positions in kHz spanning fmin_hz–fmax_hz, including endpoints."""
    lo, hi = fmin_hz / 1000, fmax_hz / 1000
    p2 = [2**i for i in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)
          if lo <= 2**i <= hi]
    return sorted(set([round(lo, 4)] + p2 + [round(hi, 4)]))

def plot_wavelet(wavelet):
    [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(level=10)
    fig, ax = plt.subplots(1, 1, figsize=(3.15, 3.15), dpi=300)
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

def build_cwt_figure(t, s, t_ax, f_ax, cwtmatr, xray_im, trackid, fmin_hz, fmax_hz, vmax=None, title_suffix=''):
    """Create the standard 3-panel CWT figure. Returns (fig, ax2) for further annotation."""
    kw = {'height_ratios': [1, 1, 1], 'width_ratios': [95, 5]}
    fig, ((ax1, ax1b), (ax2, ax2b), (ax3, ax3b)) = plt.subplots(
        3, 2, figsize=[6.3, 7], dpi=300, gridspec_kw=kw)
    fig.suptitle(f'{trackid} - {series}{title_suffix}')

    ax1.plot(t*1000, s, lw=0.75)
    ax1.set_xlim(t[0]*1000, t[-1]*1000)
    ax1.set_ylabel('Intensity')
    ax1.tick_params(labelbottom=False)

    pcm = ax2.pcolormesh(t_ax, f_ax, cwtmatr, cmap=cwt_cmap, vmax=vmax)
    ax2.set_yscale('log', base=2)
    ax2.set_ylim(fmin_hz / 1000, fmax_hz / 1000)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Frequency [kHz]')
    ax2.set_yticks(_freq_yticks_khz(fmin_hz, fmax_hz))
    cbar_label = 'Intensity [a.u.]' if title_suffix else 'Intensity'
    fig.colorbar(pcm, cax=ax2b, label=cbar_label)

    ax3.imshow(xray_im[150:450], cmap='gray')
    scalebar = ScaleBar(4.3, "µm", length_fraction=0.15, width_fraction=0.02,
                        frameon=False, color='w', location='lower right')
    ax3.add_artist(scalebar)

    for ax in [ax1b, ax3, ax3b]:
        ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)

    return fig, ax2

# ─── Main ──────────────────────────────────────────────────────────────────────

peak_label_enabled = args.peak_label_mode
trackids_to_process = []

if peak_label_enabled:
    config = load_peak_label_config()
    print()

    if args.trackid:
        trackids_to_process = [args.trackid]
        print(f"Peak labeling mode enabled for single trackid (CLI): {args.trackid}")
    else:
        trackids_to_process = list(config.get('trackids', {}).keys())
        if len(trackids_to_process) == 0:
            print("ERROR: No trackids found in config file")
            sys.exit(1)
        print(f"Peak labeling mode enabled for {len(trackids_to_process)} trackids from config:")
        for tid in trackids_to_process:
            print(f"  - {tid}")

    print()

files = sorted(glob.glob(f'{folder}/*.hdf5'))

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

skipped = []

for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]

    if peak_label_enabled:
        trackid_cfg = get_trackid_config(config, trackid)
        peak_cfg = {
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
        print(f"Timestamps: {peak_cfg['timestamps']}")
        print(f"Peak number: {peak_cfg['peak_number']}")
        print(f"Prominence: {peak_cfg['prominence']}")
        print(f"{'='*80}\n")
    else:
        peak_cfg = {'enabled': False}
        if debug: print(trackid)

    with h5py.File(filepath, 'r') as file:
        try:
            t = np.array(file[f'{group}/{time}'])[trim_start:-trim_end if trim_end else None]
            s = np.array(file[f'{group}/{series}'])[trim_start:-trim_end if trim_end else None]
        except KeyError:
            skipped.append(trackid)
            printProgressBar(i+1, len(files), prefix=trackid, suffix=' [skipped]')
            continue
        # s = np.log(s + 1)
        # t = np.array(file[f'{group}/{time}'])
        # s = np.array(file[f'{group}/{series}'])
        if mad_threshold is not None:
            s = mad_interpolate(s, mad_threshold)
        if running_mean_window != None:
            s = np.convolve(s, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        if savgol_window != None:
            s = savgol_filter(s, window_length=savgol_window, polyorder=savgol_polyorder)
        xray_im = np.array(file['bs-f40'])[-1]

        # Frequency range calculation
        sampling_period = round(t[1]-t[0], 9)
        if debug: print(sampling_period)
        sampling_duration = round(t[-1]-t[0], 9)
        sampling_rate = round(1/sampling_period, 7)
        if debug: print(sampling_rate)

        nyquist_freq = sampling_rate / 2
        min_freq = 1 / sampling_duration  # Lowest resolvable frequency
        max_freq = nyquist_freq

        fmin = min_freq if freq_min == 'auto' else freq_min
        fmax = max_freq if freq_max == 'auto' else freq_max
        if debug: print(f'Frequency range: {fmin/1000:.4g} - {fmax/1000:.4g} kHz')

        scales, vmax = get_cwt_scales(wavelet, num=cwt_scales_num, sampling_rate=sampling_rate,
                                      fmin=fmin, fmax=fmax)
        if debug: print('scales:\n', scales)

        if debug:
            print(f'min: {pywt.scale2frequency(wavelet, scales[-1])*sampling_rate} ',
                f'max: {pywt.scale2frequency(wavelet, scales[0])*sampling_rate}')

        if show_wavelet:
            plot_wavelet(wavelet)
            show_wavelet = False

        # Apply symmetric padding to minimize edge artifacts (best practice)
        pad_width = pad_factor * len(s)
        s_pad = np.pad(s, pad_width, mode='symmetric')
        if debug: print(f'Padded signal length: {len(s_pad)} (5x original: {len(s)})')

        # Perform CWT on padded signal
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)
        freqs = freqs.real  # pywt returns complex dtype for freqs when using complex wavelets

        # Normalise to preserve amplitude proportionality
        # cwtmatr /= np.sqrt(scales[:, None])

        # Crop to original signal length (extract middle section, removing padding)
        cwtmatr = np.abs(cwtmatr[:, pad_width:pad_width+len(s)])

        if args.auto_vmax:
            vmax = np.percentile(cwtmatr, 99)

        # Prepare common variables for plotting
        t_ax, f_ax = np.meshgrid(t*1000, freqs/1000)

        # Normal CWT plot
        if not peak_cfg['enabled']:
            fig, _ = build_cwt_figure(t, s, t_ax, f_ax, cwtmatr, xray_im, trackid, fmin, fmax, vmax=vmax)

            if mode == 'save':
                output_folder = Path(folder, 'CWT', series_name, wavelet, cwt_cmap)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = str(output_folder / f'{trackid}_{series}_CWT_{wavelet}.png')
                for attempt in range(3):
                    try:
                        plt.savefig(output_path)
                        break
                    except OSError as e:
                        if attempt < 2:
                            import time; time.sleep(1.0)
                        else:
                            skipped.append(f'{trackid} (save error)')
                            printProgressBar(i+1, len(files), prefix=trackid, suffix=' [save failed]')
                            print(f'\nWarning: could not save {output_path}: {e}')
            else:
                plt.show()
            plt.close()

        # Peak-labeled CWT plot
        if peak_cfg['enabled']:
            print(f"\nGenerating peak-labeled CWT for {trackid}...")
            fig, ax2 = build_cwt_figure(t, s, t_ax, f_ax, cwtmatr, xray_im, trackid, fmin, fmax,
                                        vmax=vmax, title_suffix=' (Labeled Peaks)')

            for timestamp in peak_cfg['timestamps']:
                ax2.axvline(x=timestamp*1000, color='k', linestyle='--', linewidth=0.75, alpha=1)
            peak_data = detect_peaks_at_timestamps(cwtmatr, freqs, t, peak_cfg)

            for data in peak_data:
                for peak_idx in data['top_peaks']:
                    ax2.plot(data['actual_time']*1000, freqs[peak_idx]/1000,
                             'kx', markersize=5, markeredgewidth=1)

            labeled_output_folder = Path(folder, 'CWT', series_name, wavelet, cwt_cmap, 'labelled_peaks')
            if not os.path.exists(labeled_output_folder):
                os.makedirs(labeled_output_folder)
            plt.savefig(Path(labeled_output_folder, f'{trackid}_{series}_CWT_{wavelet}_labeled.png'))
            plt.close()
            print(f"✓ Saved labeled peak plot to: {labeled_output_folder}")

            plot_peak_diagnostics(peak_data, trackid, peak_cfg, labeled_output_folder, fmin, fmax)

    printProgressBar(i+1, len(files), prefix=trackid, suffix='         ')

if skipped:
    print(f"\nSkipped {len(skipped)} file(s) with missing '{group}/{series}' dataset:")
    for tid in skipped:
        print(f"  - {tid}")
