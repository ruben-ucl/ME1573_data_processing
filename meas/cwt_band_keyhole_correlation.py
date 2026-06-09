"""
CWT Band Power vs Keyhole Depth Correlation
============================================
For each HDF5 file, computes CWT band power statistics from the AMPM photodiode signal
over short time windows, then extracts matching keyhole geometry statistics (depth,
length, area, FKW angle) from the KH datasets in the same file.

Outputs:
  {hdf5_folder}/cwt_band_kh_correlation.csv       - flat CSV, one row per (trackid, window)
  {hdf5_folder}/cwt_band_kh_correlation_scatter.png - summary scatter plots

Usage:
    conda activate ml
    python meas/cwt_band_keyhole_correlation.py
"""

import os
import sys
import json
import functools
import glob
import h5py
import pywt
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, kurtosis as scipy_kurtosis
from scipy.signal import find_peaks, coherence
from matplotlib import pyplot as plt

print = functools.partial(print, flush=True)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import get_paths, get_cwt_scales, printProgressBar, get_excluded_trackids, get_logbook, mad_interpolate

# ── Configuration ───────────────────────────────────────────────────────────

wavelet        = "cmor1.5-1.0"
cwt_scales_num = 256
pad_factor     = 2   # symmetric padding = pad_factor * len(s) on each side

# Frequency bands of interest (Hz)
# CWT is computed over the union range [min fmin, max fmax]
freq_bands = [
    {'name': '1-2kHz',   'fmin':  1000, 'fmax':  2000},
    {'name': '2-4kHz',   'fmin':  2000, 'fmax':  4000},
    {'name': '4-8kHz',   'fmin':  4000, 'fmax':  8000},
    {'name': '8-16kHz',  'fmin':  8000, 'fmax': 16000},
    {'name': '16-32kHz', 'fmin': 16000, 'fmax': 32000},
    {'name': '32-50kHz', 'fmin': 32000, 'fmax': 50000},
]

# Time windowing (1 ms windows, 0.2 ms step — matches ML windowing)
window_size_s = 0.001
window_step_s = 0.0002

# Skip edge windows to avoid start/end artefacts
skip_first_n_windows = 1
skip_last_n_windows  = 1

# AMPM signal settings
ampm_group  = 'AMPM'
ampm_time   = 'Time'
ampm_series = 'Photodiode1Bits'
trim_start  = 510   # strip 500-sample margins (plus a few extra)
trim_end    = 510

# KH datasets to extract from KH/ group in HDF5
kh_datasets = ['max_depth', 'max_length', 'area', 'depth_at_max_length', 'fkw_angle']

# Detector resolution floor for KH geometric measurements.
# Values at or below this limit are censored (below detection), not true zeros.
# Substituted with this value for log/power fits. Units must match HDF5 KH data.
kh_detection_limit = 4.3e-3   # 4 µm expressed in mm (adjust if KH data stored in µm → use 4.0)

# Hardcoded track IDs for per-track CWT similarity figures.
# SCALOGRAM_TRACKIDS = ['0323_03']
SCALOGRAM_TRACKIDS = ['1112_01']

# Signal cleaning parameters (applied inside per-track similarity plots)
MAD_THRESHOLD    = 4.0   # KH only: replace outliers > MAD_THRESHOLD × MAD

# KH timing offsets relative to PD (seconds). Positive = KH is delayed; subtract
# from kh_time to bring it into alignment with the PD time axis.
# Values sourced from MANUAL_LAG_CORRECTIONS in vis/timeseries_compare.py
KH_LAG_CORRECTIONS = {
    '1112_01': 0.00162,
    '1112_02': 0.00162,
    '1112_03': 0.00162,
    '1112_04': 0.00168,
    '1112_05': 0.00168,
    '1112_06': 0.00168,
}

# ── Derived band frequency range for CWT computation ────────────────────────
cwt_fmin = min(b['fmin'] for b in freq_bands)
cwt_fmax = max(b['fmax'] for b in freq_bands)


# ── Signal preparation helpers ───────────────────────────────────────────────

def _prepare_kh(arr, kh_time):
    """Interpolate NaN and MAD-clean a KH array."""
    a = arr.copy().astype(float)
    mask = ~np.isfinite(a)
    if mask.any():
        x = np.arange(len(a))
        a[mask] = np.interp(x[mask], x[~mask], a[~mask])
    a = mad_interpolate(a, MAD_THRESHOLD)
    return a


def _prepare_pd(s, sampling_rate):
    """Cast PD signal to float (smoothing disabled)."""
    return s.astype(float)


# ── Helper functions ─────────────────────────────────────────────────────────

def _band_stats(band_data, total_energy=None):
    """Return dict of statistics for a 2-D array of CWT amplitudes (log-compressed power)."""
    flat = band_data.ravel()
    nan_base = dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                    median=np.nan, energy=np.nan, energy_ratio=np.nan, entropy=np.nan)
    if flat.size == 0:
        return nan_base
    S = np.log1p(flat ** 2)
    band_energy = float(S.sum())
    result = dict(
        mean=float(np.mean(S)),
        std=float(np.std(S)),
        min=float(np.min(S)),
        max=float(np.max(S)),
        median=float(np.median(S)),
        energy=band_energy,
        energy_ratio=band_energy / total_energy if (total_energy and total_energy > 0) else np.nan,
    )
    # Shannon entropy of normalised band power
    p = S / (S.sum() + 1e-12)
    result['entropy'] = float(-np.sum(p * np.log(p + 1e-12)))
    return result


def _cwt_global_features(cwt_window, freqs):
    """Return dict of 11 global CWT features from a full-frequency CWT slice.

    Parameters
    ----------
    cwt_window : ndarray, shape [n_scales, n_time]  (raw amplitudes, not power)
    freqs      : ndarray, shape [n_scales]
    """
    nan_dict = {k: np.nan for k in (
        'cwt_entropy', 'cwt_kurtosis', 'cwt_spectral_centroid', 'cwt_spectral_spread',
        'cwt_dominant_freq', 'cwt_activity_ratio', 'cwt_temporal_variance',
        'cwt_peak_count', 'cwt_ridge_mean_freq', 'cwt_ridge_freq_std', 'cwt_ridge_smoothness',
    )}
    if cwt_window.size == 0 or cwt_window.shape[1] < 2:
        return nan_dict

    S = np.log1p(cwt_window ** 2)   # [n_scales, n_time], log-compressed power

    # Marginals
    M_f = S.sum(axis=1)   # [n_scales]  — power per frequency
    M_t = S.sum(axis=0)   # [n_time]    — power per time step

    total = S.sum()
    if total <= 0:
        return nan_dict

    # Shannon entropy over full scalogram
    p = S.ravel() / (total + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # Kurtosis (impulsiveness)
    kurt = float(scipy_kurtosis(S.ravel()))

    # Spectral centroid & spread
    freq_sum = float(M_f.sum())
    if freq_sum > 0:
        centroid = float(np.sum(freqs * M_f) / freq_sum)
        spread   = float(np.sqrt(np.sum(M_f * (freqs - centroid) ** 2) / freq_sum))
    else:
        centroid, spread = np.nan, np.nan

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(M_f)])

    # Temporal activity ratio: fraction of time steps above mean energy
    mean_mt = float(M_t.mean())
    activity_ratio = float(np.mean(M_t > mean_mt))

    # Temporal variance (std of M_t)
    temporal_var = float(np.std(M_t))

    # Peak count in temporal marginal
    peaks, _ = find_peaks(M_t, height=mean_mt)
    peak_count = int(len(peaks))

    # Ridge: dominant frequency at each time step
    ridge_idx = np.argmax(S, axis=0)   # [n_time]
    ridge_freqs = freqs[ridge_idx]
    ridge_mean = float(np.mean(ridge_freqs))
    ridge_std  = float(np.std(ridge_freqs))
    ridge_smoothness = float(1.0 / (1.0 + np.std(np.diff(ridge_freqs))))

    return {
        'cwt_entropy':          entropy,
        'cwt_kurtosis':         kurt,
        'cwt_spectral_centroid': centroid,
        'cwt_spectral_spread':  spread,
        'cwt_dominant_freq':    dom_freq,
        'cwt_activity_ratio':   activity_ratio,
        'cwt_temporal_variance': temporal_var,
        'cwt_peak_count':       float(peak_count),
        'cwt_ridge_mean_freq':  ridge_mean,
        'cwt_ridge_freq_std':   ridge_std,
        'cwt_ridge_smoothness': ridge_smoothness,
    }


def _kh_stats(kh_values):
    """Return dict of mean/std/min/max/n for a 1-D array of KH values."""
    vals = kh_values[np.isfinite(kh_values)]
    if vals.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, median=np.nan, n=0)
    return dict(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        median=float(np.median(vals)),
        n=int(vals.size),
    )


# ── Main processing loop ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CWT band–keyhole correlation analysis.')
    parser.add_argument('--figs_only', action='store_true',
                        help='Skip batch processing; regenerate single-track figures only.')
    args = parser.parse_args()

    pipeline_track = '0323_04'

    paths = get_paths()
    folder = paths['hdf5']

    out_dir = Path(folder) / 'CWT' / 'band_kh_correlation'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figs_only:
        print("--figs_only: skipping batch processing, generating single-track figures.")
        _plot_cwt_scalogram_comparison(out_dir, folder)
        _plot_spectral_coherence(out_dir, folder)
        return

    files  = sorted(glob.glob(str(folder / '*.hdf5')))

    if not files:
        print(f"ERROR: No HDF5 files found in {folder}")
        sys.exit(1)

    print(f"Found {len(files)} HDF5 files in {folder}")
    print(f"CWT: wavelet={wavelet}, scales={cwt_scales_num}, "
          f"freq range={cwt_fmin/1000:.0f}-{cwt_fmax/1000:.0f} kHz, "
          f"pad_factor={pad_factor}")
    print(f"Windowing: {window_size_s*1000:.1f} ms windows, "
          f"{window_step_s*1000:.2f} ms step\n")

    all_records = []
    skipped     = []   # missing AMPM signal
    no_kh       = []   # no KH measurements (silently skipped)
    excluded    = get_excluded_trackids()

    for file_idx, filepath in enumerate(files):
        trackid = Path(filepath).name[:7]

        if trackid in excluded:
            printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix=' [excluded]')
            continue

        with h5py.File(filepath, 'r') as hf:

            # ── Load AMPM signal ─────────────────────────────────────────
            try:
                t_raw = np.array(hf[f'{ampm_group}/{ampm_time}'])
                s_raw = np.array(hf[f'{ampm_group}/{ampm_series}'])
            except KeyError:
                skipped.append(trackid)
                printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix=' [skipped]')
                continue

            t = t_raw[trim_start: len(t_raw) - trim_end if trim_end else None]
            s = s_raw[trim_start: len(s_raw) - trim_end if trim_end else None]

            if len(t) < 2:
                skipped.append(trackid)
                printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix=' [skipped]')
                continue

            # ── Load KH datasets (silently skip if absent) ────────────────
            if 'KH/time' not in hf:
                no_kh.append(trackid)
                printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix=' [no KH]  ')
                continue
            kh_time = np.array(hf['KH/time'])
            if kh_time.size == 0:
                no_kh.append(trackid)
                printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix=' [no KH]  ')
                continue

            kh_data = {}
            for ds_name in kh_datasets:
                try:
                    kh_data[ds_name] = np.array(hf[f'KH/{ds_name}'], dtype=float)
                except KeyError:
                    kh_data[ds_name] = None  # will produce NaN columns

            # ── CWT ──────────────────────────────────────────────────────
            sampling_period = float(round(t[1] - t[0], 9))
            sampling_rate   = round(1.0 / sampling_period, 7)

            scales, _ = get_cwt_scales(
                wavelet, num=cwt_scales_num,
                sampling_rate=sampling_rate,
                fmin=cwt_fmin, fmax=cwt_fmax,
            )

            pad_width = pad_factor * len(s)
            s_pad     = np.pad(s.astype(float), pad_width, mode='symmetric')

            cwtmatr_pad, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)
            freqs = freqs.real  # complex wavelet returns complex freqs dtype

            # Crop padding; keep raw absolute amplitudes — no normalisation
            cwtmatr = np.abs(cwtmatr_pad[:, pad_width: pad_width + len(s)])
            # cwtmatr shape: [n_scales, n_time]
            # freqs[0] = highest freq (smallest scale), freqs[-1] = lowest freq

            # Pre-compute boolean masks for each band (rows of cwtmatr)
            band_masks = {
                b['name']: (freqs >= b['fmin']) & (freqs <= b['fmax'])
                for b in freq_bands
            }

            # ── KH CWT (computed once per track, sliced per window) ──────
            kh_cwt_cache   = {}   # ds_name -> (kh_cwtm, kh_band_masks)
            kh_freqs_kh    = None
            kh_valid_bands = []
            if kh_time.size >= 4:
                kh_sampling_period = float(round(kh_time[1] - kh_time[0], 9))
                kh_sampling_rate   = round(1.0 / kh_sampling_period, 7)
                kh_nyquist         = kh_sampling_rate / 2.0
                kh_valid_bands     = [b for b in freq_bands if b['fmax'] <= kh_nyquist]

                if kh_valid_bands:
                    kh_cwt_fmin = min(b['fmin'] for b in kh_valid_bands)
                    kh_cwt_fmax = max(b['fmax'] for b in kh_valid_bands)
                    kh_scales, _ = get_cwt_scales(
                        wavelet, num=cwt_scales_num,
                        sampling_rate=kh_sampling_rate,
                        fmin=kh_cwt_fmin, fmax=kh_cwt_fmax,
                    )

                    for ds_name in kh_datasets:
                        arr = kh_data[ds_name]
                        if arr is None or arr.size < 4:
                            continue
                        median_val = np.nanmedian(arr)
                        if not np.isfinite(median_val):
                            median_val = 0.0
                        arr_clean = np.where(np.isfinite(arr), arr, median_val)
                        kh_pad_w  = pad_factor * len(arr_clean)
                        kh_s_pad  = np.pad(arr_clean.astype(float), kh_pad_w, mode='symmetric')
                        kh_cwtm_pad, kh_freqs_out = pywt.cwt(
                            kh_s_pad, kh_scales, wavelet, kh_sampling_period)
                        kh_freqs_out = kh_freqs_out.real
                        kh_cwtm = np.abs(
                            kh_cwtm_pad[:, kh_pad_w: kh_pad_w + len(arr_clean)])
                        kh_band_masks_ds = {
                            b['name']: (kh_freqs_out >= b['fmin']) & (kh_freqs_out <= b['fmax'])
                            for b in kh_valid_bands
                        }
                        kh_cwt_cache[ds_name] = (kh_cwtm, kh_band_masks_ds)
                        if kh_freqs_kh is None:
                            kh_freqs_kh = kh_freqs_out

            # ── Windowing ────────────────────────────────────────────────
            t0 = t[0]
            t_end_track = t[-1]
            window_starts = np.arange(t0, t_end_track - window_size_s, window_step_s)

            if skip_last_n_windows:
                window_starts = window_starts[skip_first_n_windows: -skip_last_n_windows]
            else:
                window_starts = window_starts[skip_first_n_windows:]

            for t_start in window_starts:
                t_win_end = t_start + window_size_s

                # AMPM indices
                i0 = int(np.searchsorted(t, t_start))
                i1 = int(np.searchsorted(t, t_win_end))
                if i1 <= i0:
                    continue

                record = {
                    'trackid':   trackid,
                    't_start_s': float(t_start),
                    't_end_s':   float(t_win_end),
                }

                # Global CWT features (computed over full scalogram window)
                cwt_window = cwtmatr[:, i0:i1]
                S_total    = np.log1p(cwt_window ** 2)
                total_energy = float(S_total.sum())
                record.update(_cwt_global_features(cwt_window, freqs))

                # CWT band stats
                for b in freq_bands:
                    mask = band_masks[b['name']]
                    band_slice = cwtmatr[mask, i0:i1]
                    stats = _band_stats(band_slice, total_energy=total_energy)
                    for stat_key, val in stats.items():
                        record[f"cwt_{b['name']}_{stat_key}"] = val

                # KH stats
                kh_mask = (kh_time >= t_start) & (kh_time < t_win_end)
                for ds_name in kh_datasets:
                    arr = kh_data[ds_name]
                    if arr is None:
                        stats = dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, median=np.nan, n=0)
                    else:
                        stats = _kh_stats(arr[kh_mask])
                    for stat_key, val in stats.items():
                        record[f"kh_{ds_name}_{stat_key}"] = val

                # KH CWT band stats + global features (depth only)
                if kh_valid_bands and kh_cwt_cache:
                    j0 = int(np.searchsorted(kh_time, t_start))
                    j1 = int(np.searchsorted(kh_time, t_win_end))
                    if j1 > j0:
                        for ds_name in kh_datasets:
                            if ds_name not in kh_cwt_cache:
                                continue
                            kh_cwtm, kh_band_masks_ds = kh_cwt_cache[ds_name]
                            kh_win = kh_cwtm[:, j0:j1]
                            kh_total_energy = float(np.log1p(kh_win ** 2).sum())
                            for b in kh_valid_bands:
                                mask  = kh_band_masks_ds[b['name']]
                                stats = _band_stats(kh_win[mask], total_energy=kh_total_energy)
                                for stat_key in ('energy', 'energy_ratio', 'entropy'):
                                    record[f"kh_cwt_{ds_name}_{b['name']}_{stat_key}"] = stats[stat_key]
                            if ds_name == 'max_depth' and kh_freqs_kh is not None:
                                gf = _cwt_global_features(kh_win, kh_freqs_kh)
                                for feat_name, val in gf.items():
                                    record[f"kh_cwt_depth_{feat_name}"] = val

                all_records.append(record)

        printProgressBar(file_idx + 1, len(files), prefix=trackid, suffix='         ')

    if not all_records:
        print("\nNo records generated — check HDF5 file contents.")
        _write_log(out_dir, folder, files, all_records, skipped, no_kh, excluded)
        return

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)

    try:
        logbook    = get_logbook()
        regime_map = logbook.dropna(subset=['trackid']).set_index('trackid')['Melting regime'].to_dict()
        df['melting_regime'] = df['trackid'].map(regime_map)
    except Exception as exc:
        print(f"Could not load regime labels: {exc}")
        df['melting_regime'] = np.nan

    output_csv = out_dir / 'cwt_band_kh_correlation.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nSaved {len(df)} rows to: {output_csv}")

    # ── Write log ─────────────────────────────────────────────────────────────
    _write_log(out_dir, folder, files, all_records, skipped, no_kh, excluded)

    # ── R² correlation table ──────────────────────────────────────────────────
    r2_table = _compute_r2_table(df)
    r2_csv = out_dir / 'cwt_band_kh_r2_table.csv'
    r2_table.to_csv(r2_csv, index=False, encoding='utf-8')
    print(f"Saved R² table:     {r2_csv}  ({len(r2_table)} pairs)")
    print("\nTop 10 correlations by R²:")
    print(r2_table.head(10).to_string(index=False))

    # ── Summary scatter plot ──────────────────────────────────────────────────
    _plot_summary(df, out_dir)

    # ── Top-N R² scatter plot (points only; curve-fit plots are separate) ────
    _plot_top_correlations(df, r2_table, out_dir, top_n=20)

    # ── Signed Pearson-r heatmap (all features × all KH targets) ─────────────
    _plot_correlation_heatmap(df, r2_table, out_dir)

    # ── Per-track CWT similarity figures ─────────────────────────────────────
    _plot_cwt_scalogram_comparison(out_dir, folder)
    _plot_band_energy_timeseries(df, out_dir)
    _plot_spectral_coherence(out_dir, folder)

    # ── Global CWT features by melting regime ────────────────────────────────
    _plot_global_features_by_regime(df, out_dir)

    # ── Pipeline example figure ───────────────────────────────────────────────
    try:
        _plot_pipeline_example(df, r2_table, out_dir, folder, trackid=pipeline_track)
    except Exception as exc:
        print(f"Pipeline example figure failed: {exc}")


def _write_log(out_dir, hdf5_folder, files, records, skipped, no_kh, excluded):
    """Write a JSON log file capturing all run metadata and per-track outcomes."""
    processed_trackids = sorted({r['trackid'] for r in records})
    log = {
        'run_timestamp':     datetime.now().isoformat(timespec='seconds'),
        'hdf5_folder':       str(hdf5_folder),
        'output_folder':     str(out_dir),
        'cwt_settings': {
            'wavelet':          wavelet,
            'scales_num':       cwt_scales_num,
            'pad_factor':       pad_factor,
            'freq_bands_hz':    freq_bands,
            'cwt_fmin_hz':      cwt_fmin,
            'cwt_fmax_hz':      cwt_fmax,
        },
        'signal_settings': {
            'group':            ampm_group,
            'time_dataset':     ampm_time,
            'series_dataset':   ampm_series,
            'trim_start':       trim_start,
            'trim_end':         trim_end,
        },
        'windowing': {
            'window_size_s':        window_size_s,
            'window_step_s':        window_step_s,
            'skip_first_n_windows': skip_first_n_windows,
            'skip_last_n_windows':  skip_last_n_windows,
        },
        'kh_datasets':       kh_datasets,
        'summary': {
            'total_hdf5_files':  len(files),
            'excluded':          len(excluded),
            'no_kh':             len(no_kh),
            'skipped_no_signal': len(skipped),
            'processed':         len(processed_trackids),
            'total_windows':     len(records),
        },
        'tracks': {
            'processed':         processed_trackids,
            'excluded':          sorted(excluded),
            'no_kh':             sorted(no_kh),
            'skipped_no_signal': sorted(skipped),
        },
    }

    log_path = out_dir / 'cwt_band_kh_correlation_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print(f"Saved log:          {log_path}")


def _plot_summary(df, folder):
    """One subplot per KH dataset; one series/colour per frequency band."""
    band_names = [b['name'] for b in freq_bands]
    n_kh = len(kh_datasets)

    fig, axes = plt.subplots(1, n_kh, figsize=(4.5 * n_kh, 4.5), dpi=150)
    if n_kh == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax_idx, ds_name in enumerate(kh_datasets):
        ax = axes[ax_idx]
        kh_col = f'kh_{ds_name}_mean'
        if kh_col not in df.columns:
            ax.set_title(ds_name)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        for b_idx, band_name in enumerate(band_names):
            cwt_col = f'cwt_{band_name}_mean'
            if cwt_col not in df.columns:
                continue

            sub = df[[cwt_col, kh_col]].dropna()
            if sub.empty:
                continue

            x = sub[kh_col].values
            y = sub[cwt_col].values

            # Pearson r
            if len(x) >= 2:
                r, _ = pearsonr(x, y)
                label = f"{band_name} (r={r:.2f})"
            else:
                label = band_name

            ax.scatter(x, y, s=2, alpha=0.3, color=colors[b_idx % len(colors)],
                       label=label, rasterized=True)

        ax.set_xlabel(ds_name)
        ax.set_ylabel('CWT band mean amplitude')
        ax.set_title(ds_name)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, markerscale=4)

    fig.suptitle('CWT Band Power vs Keyhole Geometry', fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = Path(folder) / 'cwt_band_kh_correlation_scatter.png'  # folder is already out_dir
    plt.savefig(out_path)
    plt.close()
    print(f"Saved scatter plot: {out_path}")


def _compute_r2_table(df):
    """Compute R² for every (cwt_*, kh_*) column pair.

    Returns a DataFrame sorted by R² descending with columns:
    cwt_col, kh_col, r2, r, n
    """
    cwt_cols = [c for c in df.columns if c.startswith('cwt_')]
    # Exclude _n (window count, not a measurement) from KH side
    kh_cols  = [c for c in df.columns if c.startswith('kh_') and not c.endswith('_n')]

    rows = []
    for kh_col in kh_cols:
        for cwt_col in cwt_cols:
            sub = df[[cwt_col, kh_col]].dropna()
            n = len(sub)
            if n < 3:
                continue
            r, _ = pearsonr(sub[cwt_col].values, sub[kh_col].values)
            rows.append({'cwt_col': cwt_col, 'kh_col': kh_col,
                         'r2': r ** 2, 'r': r, 'n': n})

    return pd.DataFrame(rows).sort_values('r2', ascending=False).reset_index(drop=True)


def _fit_curve(x, y, fit_type):
    """Fit y = f(x) with the requested curve type.

    Returns (x_line, y_line, fit_r2, fit_equation_str) or None if the fit
    cannot be applied to this data (e.g. non-positive x for log/power fits).

    fit_type options:
        'linear'  — y = mx + b
        'log'     — y = a·ln(x) + b   (requires x > 0)
        'power'   — y = a·x^b          (requires x > 0 and y > 0)
        'poly2'   — y = ax² + bx + c
    """
    x_line = np.linspace(x.min(), x.max(), 300)

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if fit_type == 'linear':
        m, b = np.polyfit(x, y, 1)
        y_line = m * x_line + b
        return x_line, y_line, _r2(y, m * x + b), f'y = {m:.3g}x + {b:.3g}'

    if fit_type == 'log':
        # Below-detection values are censored, not true zeros — substitute detection limit
        xm = np.maximum(x, kh_detection_limit)
        a, b = np.polyfit(np.log(xm), y, 1)
        x_plot = np.maximum(x_line, kh_detection_limit)
        y_line = a * np.log(x_plot) + b
        return x_plot, y_line, _r2(y, a * np.log(xm) + b), f'y = {a:.3g}·ln(x) + {b:.3g}'

    if fit_type == 'power':
        # Substitute detection limit for x; mask y <= 0 (genuinely silent CWT windows)
        xm = np.maximum(x, kh_detection_limit)
        valid = y > 0
        if valid.sum() < 3:
            return None
        xv, yv = xm[valid], y[valid]
        log_a, b = np.polyfit(np.log(xv), np.log(yv), 1)
        a = np.exp(log_a)
        x_plot = np.maximum(x_line, kh_detection_limit)
        y_line = a * x_plot ** b
        # R² in log-log space — consistent with where the fit minimises residuals.
        # Computing R² in linear space after a log-space fit can go arbitrarily
        # negative when back-transformation amplifies outliers.
        r2_loglog = _r2(np.log(yv), log_a + b * np.log(xv))
        return x_plot, y_line, r2_loglog, f'y = {a:.3g}·x^{b:.3g}  (R² in log)'

    if fit_type == 'poly2':
        c2, c1, c0 = np.polyfit(x, y, 2)
        y_line = c2 * x_line ** 2 + c1 * x_line + c0
        return x_line, y_line, _r2(y, c2 * x ** 2 + c1 * x + c0), \
               f'y = {c2:.3g}x² + {c1:.3g}x + {c0:.3g}'

    raise ValueError(f"Unknown fit_type: {fit_type!r}")


# Human-readable names and file-safe slugs for each fit type
_FIT_META = {
    'linear': ('Linear',    'linear'),
    'log':    ('Log',       'log'),
    'power':  ('Power law', 'power'),
    'poly2':  ('Quadratic', 'poly2'),
}


def _plot_top_correlations(df, r2_table, out_dir, top_n=5):
    """Scatter plots for the top_n (cwt, kh) pairs ranked by Pearson R²."""
    # Filter to original KH geometry means only (exclude kh_cwt_* derived columns)
    kh_geom_cols = {f'kh_{ds}_mean' for ds in kh_datasets}
    table = r2_table[
        r2_table['kh_col'].isin(kh_geom_cols) &
        r2_table['cwt_col'].str.startswith('cwt_')
    ].head(top_n)

    if table.empty:
        print("No valid pairs for top-R² scatter — skipping.")
        return

    top = []
    for _, row in table.iterrows():
        sub = df[[row['cwt_col'], row['kh_col']]].dropna()
        if len(sub) < 3:
            continue
        top.append(dict(
            cwt_col=row['cwt_col'], kh_col=row['kh_col'],
            pearson_r=row['r'], pearson_r2=row['r2'], n=row['n'],
            x=sub[row['kh_col']].values, y=sub[row['cwt_col']].values,
        ))

    if not top:
        return

    fig, axes = plt.subplots(1, len(top), figsize=(4.5 * len(top), 4.5), dpi=150)
    if len(top) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, entry in zip(axes, top):
        ax.scatter(entry['x'], entry['y'], s=2, alpha=0.3, color=colors[0], rasterized=True)
        ax.set_xlabel(entry['kh_col'], fontsize=8)
        ax.set_ylabel(entry['cwt_col'], fontsize=8)
        ax.set_title(
            f"Pearson r={entry['pearson_r']:+.3f}  R²={entry['pearson_r2']:.3f}\n"
            f"n={entry['n']}",
            fontsize=7,
        )

    fig.suptitle(
        f'Top {len(top)} CWT–KH Correlations (ranked by Pearson R²)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    out_path = out_dir / 'cwt_band_kh_top_r2_scatter.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved top-R² scatter: {out_path}")


def _plot_correlation_heatmap(df, r2_table, out_dir):
    """Figure A — signed Pearson-r heatmap: all CWT features × all KH targets (mean aggregate)."""
    band_names = [b['name'] for b in freq_bands]
    kh_cols = [f'kh_{ds}_mean' for ds in kh_datasets if f'kh_{ds}_mean' in df.columns]
    kh_labels = [c.replace('kh_', '').replace('_mean', '') for c in kh_cols]

    global_features = [
        'cwt_entropy', 'cwt_kurtosis', 'cwt_spectral_centroid', 'cwt_spectral_spread',
        'cwt_dominant_freq', 'cwt_activity_ratio', 'cwt_temporal_variance',
        'cwt_peak_count', 'cwt_ridge_mean_freq', 'cwt_ridge_freq_std', 'cwt_ridge_smoothness',
    ]
    band_energy_ratio = [f'cwt_{bn}_energy_ratio' for bn in reversed(band_names)]
    band_entropy      = [f'cwt_{bn}_entropy'       for bn in reversed(band_names)]
    band_mean         = [f'cwt_{bn}_mean'           for bn in reversed(band_names)]
    band_energy       = [f'cwt_{bn}_energy'         for bn in reversed(band_names)]

    row_groups = [
        ('Global features',    global_features),
        ('Band energy ratio',  band_energy_ratio),
        ('Band entropy',       band_entropy),
        ('Band mean',          band_mean),
        ('Band energy',        band_energy),
    ]

    # Filter to columns that actually exist in df
    row_groups = [(grp_label, [c for c in cols if c in df.columns])
                  for grp_label, cols in row_groups]
    row_groups = [(lbl, cols) for lbl, cols in row_groups if cols]

    all_row_cols = [c for _, cols in row_groups for c in cols]
    if not all_row_cols or not kh_cols:
        print("Skipping heatmap — no feature or KH columns found.")
        return

    # Build r-matrix
    r_matrix = np.full((len(all_row_cols), len(kh_cols)), np.nan)
    for ci, kh_col in enumerate(kh_cols):
        for ri, cwt_col in enumerate(all_row_cols):
            sub = df[[cwt_col, kh_col]].dropna()
            if len(sub) >= 3:
                r, _ = pearsonr(sub[cwt_col].values, sub[kh_col].values)
                r_matrix[ri, ci] = r

    row_labels = [c.replace('cwt_', '') for c in all_row_cols]

    fig_h = max(6, len(all_row_cols) * 0.35 + 1.5)
    fig_w = max(5, len(kh_cols) * 1.4 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    im = ax.imshow(r_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')

    ax.set_xticks(range(len(kh_labels)))
    ax.set_xticklabels(kh_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticklabels(kh_labels, rotation=30, ha='left', fontsize=8)

    # Annotate cells
    for ri in range(len(all_row_cols)):
        for ci in range(len(kh_cols)):
            val = r_matrix[ri, ci]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(ci, ri, f'{val:+.2f}', ha='center', va='center',
                        fontsize=6, color=text_color)

    # Horizontal dividers between row groups
    boundary = 0
    for _, cols in row_groups[:-1]:
        boundary += len(cols)
        ax.axhline(boundary - 0.5, color='black', linewidth=1.2)

    # Group labels on the left
    boundary = 0
    for grp_label, cols in row_groups:
        mid = boundary + len(cols) / 2 - 0.5
        ax.annotate(grp_label, xy=(-0.01, 1.0 - (mid + 0.5) / len(all_row_cols)),
                    xycoords='axes fraction', ha='right', va='center',
                    fontsize=7, fontstyle='italic', rotation=0,
                    annotation_clip=False)
        boundary += len(cols)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Pearson r', fontsize=8)

    ax.set_title('CWT Feature × KH Target Correlation (Pearson r)', fontsize=10, fontweight='bold', pad=12)
    plt.tight_layout()

    out_path = out_dir / 'cwt_kh_correlation_heatmap.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap:      {out_path}")


def _plot_pipeline_example(df, r2_table, out_dir, hdf5_folder, trackid='0323_04'):
    """Pedagogical multi-panel figure showing the full CWT feature-extraction pipeline.

    Layout (2 rows × 3 cols):
      Row 0: b (signal) | a (radiograph) | c (KH depth)
      Row 1: f (bars)   | e (CWT)        | g (scatter)
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker

    band_names = [b['name'] for b in freq_bands]

    # --- 1. Find + open HDF5 file ---
    matches = glob.glob(str(Path(hdf5_folder) / f'{trackid}*.hdf5'))
    if not matches:
        print(f"Pipeline example: no HDF5 file found for trackid={trackid!r}")
        return

    filepath = matches[0]

    with h5py.File(filepath, 'r') as hf:
        # --- 2. Load signal + KH data ---
        try:
            t_raw = np.array(hf[f'{ampm_group}/{ampm_time}'])
            s_raw = np.array(hf[f'{ampm_group}/{ampm_series}'])
        except KeyError:
            print(f"Pipeline example: AMPM signal not found in {filepath}")
            return

        t = t_raw[trim_start: len(t_raw) - trim_end if trim_end else None]
        s = s_raw[trim_start: len(s_raw) - trim_end if trim_end else None]

        if len(t) < 2:
            print("Pipeline example: signal too short after trimming")
            return

        kh_time      = np.array(hf['KH/time'])      if 'KH/time'      in hf else np.array([])
        kh_max_depth = np.array(hf['KH/max_depth'], dtype=float) if 'KH/max_depth' in hf else None

        if 'bs-f40_lagrangian' in hf:
            n_frames = hf['bs-f40_lagrangian'].shape[0]
            frame = hf['bs-f40_lagrangian'][n_frames // 2]
        else:
            frame = None

    # --- 3. Compute full CWT ---
    sampling_period = float(round(t[1] - t[0], 9))
    sampling_rate   = round(1.0 / sampling_period, 7)

    scales, _ = get_cwt_scales(
        wavelet, num=cwt_scales_num,
        sampling_rate=sampling_rate,
        fmin=cwt_fmin, fmax=cwt_fmax,
    )

    pad_width = pad_factor * len(s)
    s_pad     = np.pad(s.astype(float), pad_width, mode='symmetric')

    cwtmatr_pad, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)
    freqs    = freqs.real
    cwtmatr  = np.abs(cwtmatr_pad[:, pad_width: pad_width + len(s)])

    band_masks = {
        b['name']: (freqs >= b['fmin']) & (freqs <= b['fmax'])
        for b in freq_bands
    }

    # --- 4. Select middle window ---
    window_starts = np.arange(t[0], t[-1] - window_size_s, window_step_s)
    window_starts = window_starts[skip_first_n_windows: -skip_last_n_windows if skip_last_n_windows else None]

    if len(window_starts) == 0:
        print("Pipeline example: no valid windows")
        return

    t_win = window_starts[len(window_starts) // 2]
    i0    = int(np.searchsorted(t, t_win))
    i1    = int(np.searchsorted(t, t_win + window_size_s))

    # --- 5. Extract window features ---
    cwt_window   = cwtmatr[:, i0:i1]
    total_energy = float(np.log1p(cwt_window ** 2).sum())

    band_means = []
    for bn in band_names:
        mask       = band_masks[bn]
        band_slice = cwtmatr[mask, i0:i1]
        stats      = _band_stats(band_slice, total_energy=total_energy)
        band_means.append(stats['mean'])

    # --- 6. Row in df + scatter pair ---
    track_df = df[df['trackid'] == trackid]

    # x-axis is always kh_max_depth_mean; pick the best cwt feature for that target
    kh_col  = 'kh_max_depth_mean'
    cwt_col = 'cwt_4-8kHz_energy_ratio'   # default fallback
    if r2_table is not None and not r2_table.empty:
        depth_rows = r2_table[
            (r2_table['kh_col'] == kh_col) &
            r2_table['cwt_col'].str.startswith('cwt_')
        ]
        if not depth_rows.empty:
            cwt_col = depth_rows.iloc[0]['cwt_col']

    x_win_val, y_win_val = np.nan, np.nan
    if not track_df.empty and cwt_col in df.columns and kh_col in df.columns:
        nearest_idx = (track_df['t_start_s'] - t_win).abs().idxmin()
        x_win_val   = df.loc[nearest_idx, kh_col]
        y_win_val   = df.loc[nearest_idx, cwt_col]

    # --- 7. Build figure: 3 rows × 3 cols ---
    # Row 0: b (signal),      a (radiograph), c (KH depth)
    # Row 1: e (CWT),         f (bars),       g (scatter)
    # Row 2: cbar (colorbar)  —               —
    fig = plt.figure(figsize=(15, 9.5))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            height_ratios=[1.0, 1.0, 0.10],
                            hspace=0.42, wspace=0.32)

    ax_b    = fig.add_subplot(gs[0, 0])
    ax_a    = fig.add_subplot(gs[0, 1])
    ax_c    = fig.add_subplot(gs[0, 2])
    ax_e    = fig.add_subplot(gs[1, 0])
    ax_f    = fig.add_subplot(gs[1, 1])
    ax_g    = fig.add_subplot(gs[1, 2])
    ax_cbar = fig.add_subplot(gs[2, 0])

    # --- Panel a: Radiograph (top centre) ---
    if frame is not None:
        ax_a.imshow(frame, cmap='gray', aspect='auto')
    else:
        ax_a.text(0.5, 0.5, 'No radiograph data', transform=ax_a.transAxes, ha='center', fontsize=9)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    # --- Panel b: Full signal (top left) ---
    ax_b.plot(t, s, lw=0.4, color='k', rasterized=True)
    ax_b.axvspan(t_win, t_win + window_size_s, alpha=0.25, color='C0')
    ax_b.set_xlabel('Time (s)', fontsize=9)
    ax_b.set_ylabel('Signal (bits)', fontsize=9)
    ax_b.tick_params(labelsize=8)

    # --- Panel c: KH max_depth (top right) ---
    if kh_max_depth is not None and kh_time.size > 0:
        ax_c.scatter(kh_time, kh_max_depth, s=1, color='k', rasterized=True)
        ax_c.axvspan(t_win, t_win + window_size_s, alpha=0.25, color='C0')
    else:
        ax_c.text(0.5, 0.5, 'No KH data', transform=ax_c.transAxes, ha='center', fontsize=9)
    ax_c.set_xlabel('Time (s)', fontsize=9)
    ax_c.set_ylabel(r'Depth ($\mu$m)', fontsize=9)
    ax_c.tick_params(labelsize=8)

    # --- Panel e: CWT scalogram (bottom left) ---
    scalogram = np.log1p(cwtmatr[:, i0:i1] ** 2)
    t_e       = t[i0:i1] * 1000 - t[i0] * 1000   # relative ms
    freqs_khz = freqs / 1000

    pc = ax_e.pcolormesh(t_e, freqs_khz, scalogram, cmap='jet', shading='auto')
    ax_e.set_yscale('log', base=2)
    cwt_yticks = [1, 2, 4, 8, 16, 32, 50]
    ax_e.set_yticks(cwt_yticks)
    ax_e.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_e.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_e.set_xlabel('Time (ms)', fontsize=9)
    ax_e.set_ylabel('Frequency (kHz)', fontsize=9)
    ax_e.tick_params(labelsize=8)

    for b in freq_bands:
        ax_e.axhline(b['fmin'] / 1000, color='white', lw=0.8, ls='--')
    ax_e.axhline(freq_bands[-1]['fmax'] / 1000, color='white', lw=0.8, ls='--')

    cbar_e = fig.colorbar(pc, cax=ax_cbar, orientation='horizontal')
    cbar_e.set_label('log power', fontsize=8)
    cbar_e.ax.tick_params(labelsize=7)

    # --- Panel f: Band mean bars (bottom centre) ---
    for b_idx, (bn, bm) in enumerate(zip(band_names, band_means)):
        ax_f.barh(b_idx, bm if np.isfinite(bm) else 0, color='C0', height=0.7)
    ax_f.set_yticks(range(len(band_names)))
    ax_f.set_yticklabels(band_names, fontsize=8)
    ax_f.set_xlabel('Mean log power', fontsize=9)
    ax_f.tick_params(labelsize=8)

    # --- Panel g: Scatter (bottom right) ---
    if cwt_col in df.columns and kh_col in df.columns:
        sub = df[[kh_col, cwt_col]].dropna()
        if len(sub) >= 3:
            x_all = sub[kh_col].values
            y_all = sub[cwt_col].values

            ax_g.scatter(x_all, y_all, s=2, alpha=0.3, color='silver', rasterized=True)

            if np.isfinite(x_win_val) and np.isfinite(y_win_val):
                ax_g.scatter([x_win_val], [y_win_val], s=60, marker='x',
                             color='C0', linewidths=1.5, zorder=5, label='This window')
                ax_g.legend(fontsize=8)

            x_label = kh_col.replace('kh_', '').replace('_mean', '').replace('_', ' ')
            y_label = cwt_col.replace('cwt_', '').replace('_', ' ')
            ax_g.set_xlabel(x_label, fontsize=9)
            ax_g.set_ylabel(y_label, fontsize=9)
        else:
            ax_g.text(0.5, 0.5, 'Insufficient data', transform=ax_g.transAxes, ha='center')
    else:
        ax_g.text(0.5, 0.5, f'Columns not found:\n{cwt_col}\n{kh_col}',
                  transform=ax_g.transAxes, ha='center', fontsize=8)
    ax_g.tick_params(labelsize=8)

    out_path = out_dir / f'cwt_pipeline_example_{trackid}.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved pipeline example: {out_path}")


def _plot_cwt_scalogram_comparison(out_dir, hdf5_folder):
    """PD and KH max_depth CWT scalograms stacked for each SCALOGRAM_TRACKIDS entry.

    Both panels are cropped to the same time and frequency range for direct
    visual comparison.  PD signal and KH depth are each overlaid as a white
    line on their respective panel with a secondary y-axis showing values.
    Vertical colourbar on the far right avoids overlapping the secondary axes.
    Sized for half A4 page width: tick labels 8 pt, axis labels/titles 9 pt.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker

    HALF_A4_W = 4.134   # inches (105 mm)
    TICK_FS   = 8
    LABEL_FS  = 9
    TITLE_FS  = 9

    band_boundaries_khz = sorted({b['fmin'] / 1000 for b in freq_bands} |
                                  {b['fmax'] / 1000 for b in freq_bands})

    for trackid in SCALOGRAM_TRACKIDS:
        matches = glob.glob(str(Path(hdf5_folder) / f'{trackid}*.hdf5'))
        if not matches:
            print(f"Scalogram comparison: no HDF5 file for trackid={trackid!r}")
            continue

        with h5py.File(matches[0], 'r') as hf:
            try:
                t_raw = np.array(hf[f'{ampm_group}/{ampm_time}'])
                s_raw = np.array(hf[f'{ampm_group}/{ampm_series}'])
            except KeyError:
                print(f"Scalogram comparison: AMPM signal not found for {trackid}")
                continue

            t = t_raw[trim_start: len(t_raw) - trim_end if trim_end else None]
            s = s_raw[trim_start: len(s_raw) - trim_end if trim_end else None]
            if len(t) < 2:
                continue

            kh_time  = np.array(hf['KH/time'])      if 'KH/time'      in hf else np.array([])
            kh_depth = (np.array(hf['KH/max_depth'], dtype=float)
                        if 'KH/max_depth' in hf else None)

        # ── PD CWT ──────────────────────────────────────────────────────
        sampling_period = float(round(t[1] - t[0], 9))
        sampling_rate   = round(1.0 / sampling_period, 7)

        s_clean = _prepare_pd(s, sampling_rate)

        scales, _ = get_cwt_scales(
            wavelet, num=cwt_scales_num,
            sampling_rate=sampling_rate,
            fmin=cwt_fmin, fmax=cwt_fmax,
        )
        pad_width = pad_factor * len(s_clean)
        s_pad = np.pad(s_clean, pad_width, mode='symmetric')
        cwtmatr_pad, freqs_pd = pywt.cwt(s_pad, scales, wavelet, sampling_period)
        freqs_pd = freqs_pd.real
        cwtmatr  = np.abs(cwtmatr_pad[:, pad_width: pad_width + len(s_clean)])
        S_pd = np.log1p(cwtmatr ** 2)
        freqs_pd_khz = freqs_pd / 1000
        time_s = t - t[0]   # relative seconds

        # ── KH CWT ──────────────────────────────────────────────────────
        S_kh         = None
        freqs_kh_khz = None
        depth_clean  = None
        kh_time_rel  = None

        if kh_depth is not None and kh_time.size >= 4:
            kh_sp = float(round(kh_time[1] - kh_time[0], 9))
            kh_sr = round(1.0 / kh_sp, 7)
            kh_nyquist = kh_sr / 2.0
            kh_valid = [b for b in freq_bands if b['fmax'] <= kh_nyquist]

            if kh_valid:
                depth_clean = _prepare_kh(kh_depth, kh_time)
                kh_fmin = min(b['fmin'] for b in kh_valid)
                kh_fmax = max(b['fmax'] for b in kh_valid)
                kh_scales, _ = get_cwt_scales(
                    wavelet, num=cwt_scales_num,
                    sampling_rate=kh_sr,
                    fmin=kh_fmin, fmax=kh_fmax,
                )
                kh_pad_w = pad_factor * len(depth_clean)
                kh_s_pad = np.pad(depth_clean, kh_pad_w, mode='symmetric')
                kh_cwtm_pad, freqs_kh = pywt.cwt(kh_s_pad, kh_scales, wavelet, kh_sp)
                freqs_kh = freqs_kh.real
                kh_cwtm = np.abs(kh_cwtm_pad[:, kh_pad_w: kh_pad_w + len(depth_clean)])
                S_kh = np.log1p(kh_cwtm ** 2)
                freqs_kh_khz = freqs_kh / 1000
                # Apply lag correction: positive lag means KH is delayed → subtract
                kh_lag = KH_LAG_CORRECTIONS.get(trackid, 0.0)
                kh_time_rel = kh_time - t[0] - kh_lag

        # ── Crop to common frequency range; PD sets the time axis ────────
        if S_kh is not None:
            f_lo = max(freqs_pd_khz.min(), freqs_kh_khz.min())
            f_hi = min(freqs_pd_khz.max(), freqs_kh_khz.max())
            if f_lo >= f_hi:
                S_kh = None   # no frequency overlap — fall back to PD-only

        if S_kh is not None:
            pd_fm = (freqs_pd_khz >= f_lo) & (freqs_pd_khz <= f_hi)
            kh_fm = (freqs_kh_khz >= f_lo) & (freqs_kh_khz <= f_hi)
            kh_tm = (kh_time_rel >= time_s[0]) & (kh_time_rel <= time_s[-1])

            time_kh_plot  = kh_time_rel[kh_tm]
            freqs_kh_plot = freqs_kh_khz[kh_fm]
            S_kh_plot     = S_kh[kh_fm][:, kh_tm]
            depth_plot    = depth_clean[kh_tm]

            if S_kh_plot.size == 0:
                print(f"  {trackid}: KH slice empty after lag/freq crop "
                      f"(kh_tm={kh_tm.sum()}, kh_fm={kh_fm.sum()}) — PD only")
                S_kh = None

        if S_kh is not None:
            # Both panels: crop PD frequency to match KH range
            time_pd_plot  = time_s
            freqs_pd_plot = freqs_pd_khz[pd_fm]
            S_pd_plot     = S_pd[pd_fm]
            s_plot        = s_clean
        else:
            # PD-only fallback: show full frequency range
            time_pd_plot  = time_s
            freqs_pd_plot = freqs_pd_khz
            S_pd_plot     = S_pd
            s_plot        = s_clean

        # ── Figure layout (half A4 width, 9 pt labels) ───────────────────
        n_rows   = 1 if S_kh is None else 2
        fig_h    = 2.5 * n_rows + 0.7

        fig = plt.figure(figsize=(HALF_A4_W, fig_h), dpi=300)
        gs  = gridspec.GridSpec(n_rows, 1, hspace=0.15)

        # — PD panel —
        ax_pd = fig.add_subplot(gs[0, 0])
        vmax_pd = float(np.percentile(S_pd_plot, 99))
        ax_pd.pcolormesh(time_pd_plot, freqs_pd_plot, S_pd_plot,
                         cmap='jet', shading='auto', vmin=0, vmax=vmax_pd)
        ax_pd.set_yscale('log', base=2)
        ax_pd.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.4g}'))
        for bnd in band_boundaries_khz:
            if freqs_pd_plot.size and freqs_pd_plot.min() <= bnd <= freqs_pd_plot.max():
                ax_pd.axhline(bnd, color='white', lw=0.5, ls='--', alpha=0.6)
        ax_pd.set_ylabel('Frequency (kHz)', fontsize=LABEL_FS)
        ax_pd.set_title('PD signal', fontsize=TITLE_FS)
        ax_pd.tick_params(labelbottom=(n_rows == 1), labelsize=TICK_FS)

        ax_pd2 = ax_pd.twinx()
        ax_pd2.patch.set_visible(False)
        ax_pd2.plot(time_pd_plot, s_plot, color='white', lw=0.7, alpha=0.85)
        ax_pd2.set_ylabel('PD (bits)', fontsize=TICK_FS)
        ax_pd2.tick_params(labelsize=TICK_FS)

        axes_data = [ax_pd]

        # — KH panel —
        if S_kh is not None:
            ax_kh = fig.add_subplot(gs[1, 0], sharex=ax_pd)
            axes_data.append(ax_kh)
            vmax_kh = float(np.percentile(S_kh_plot, 99))
            ax_kh.pcolormesh(time_kh_plot, freqs_kh_plot, S_kh_plot,
                             cmap='jet', shading='auto', vmin=0, vmax=vmax_kh)
            ax_kh.set_yscale('log', base=2)
            ax_kh.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.4g}'))
            for bnd in band_boundaries_khz:
                if freqs_kh_plot.size and freqs_kh_plot.min() <= bnd <= freqs_kh_plot.max():
                    ax_kh.axhline(bnd, color='white', lw=0.5, ls='--', alpha=0.6)
            ax_kh.set_ylabel('Frequency (kHz)', fontsize=LABEL_FS)
            ax_kh.set_title('KH max depth', fontsize=TITLE_FS)
            ax_kh.set_xlabel('Time (s)', fontsize=LABEL_FS)
            ax_kh.tick_params(labelsize=TICK_FS)

            ax_kh2 = ax_kh.twinx()
            ax_kh2.patch.set_visible(False)
            ax_kh2.plot(time_kh_plot, depth_plot, color='white', lw=0.8, alpha=0.85)
            ax_kh2.set_ylabel('Depth (mm)', fontsize=TICK_FS)
            ax_kh2.tick_params(labelsize=TICK_FS)
        else:
            ax_pd.set_xlabel('Time (s)', fontsize=LABEL_FS)
            ax_pd.tick_params(labelbottom=True, labelsize=TICK_FS)

        # — Vertical colourbar on far right, clear of secondary y-axis labels —
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_data, orientation='vertical',
                            fraction=0.05, pad=0.22)
        cbar.set_label('log power (norm.)', fontsize=TICK_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)

        fig.suptitle(f'PD vs KH CWT — {trackid}', fontsize=TITLE_FS + 1, fontweight='bold')

        out_path = out_dir / f'cwt_scalogram_comparison_{trackid}.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved scalogram comparison: {out_path}")


def _plot_band_energy_timeseries(df, out_dir):
    """Stacked CWT band energy time series with KH depth overlay, one figure per track.

    Uses pre-computed df (no HDF5 needed).
    """
    band_names = [b['name'] for b in freq_bands]
    n_bands = len(band_names)
    colors = plt.cm.tab10.colors

    for trackid in SCALOGRAM_TRACKIDS:
        track_df = df[df['trackid'] == trackid].sort_values('t_start_s')
        if track_df.empty:
            print(f"Band energy timeseries: no data for trackid={trackid!r}")
            continue

        t = track_df['t_start_s'].values

        fig, axes = plt.subplots(
            n_bands + 1, 1,
            figsize=(12, 1.8 * (n_bands + 1)),
            sharex=True, dpi=150,
        )

        # High → low frequency (top → bottom)
        for i, band_name in enumerate(reversed(band_names)):
            ax = axes[i]
            col = f'cwt_{band_name}_energy'
            if col in track_df.columns:
                vals = track_df[col].values
                ax.fill_between(t, vals, alpha=0.5,
                                color=colors[i % len(colors)])
                ax.set_ylabel(band_name, fontsize=8, rotation=0,
                              labelpad=55, va='center')
            else:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', fontsize=8)
            ax.tick_params(labelbottom=False, labelsize=7)

        # Bottom subplot: KH depth
        ax_kh = axes[n_bands]
        kh_col = 'kh_max_depth_mean'
        if kh_col in track_df.columns:
            ax_kh.plot(t, track_df[kh_col].values, color='black', lw=1.0)
        ax_kh.set_ylabel('KH depth\n(mean)', fontsize=8)
        ax_kh.set_xlabel('Time (s)', fontsize=9)
        ax_kh.tick_params(labelsize=7)

        fig.suptitle(f'CWT Band Energy vs KH depth — {trackid}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()

        out_path = out_dir / f'cwt_band_energy_timeseries_{trackid}.png'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved band energy timeseries: {out_path}")


def _plot_spectral_coherence(out_dir, hdf5_folder):
    """Magnitude-squared coherence between raw PD signal and KH max_depth, per track.

    All tracks in SCALOGRAM_TRACKIDS are stacked in one figure.
    """
    import matplotlib.ticker as ticker

    band_boundaries_khz = sorted({b['fmin'] / 1000 for b in freq_bands} |
                                  {b['fmax'] / 1000 for b in freq_bands})

    n_tracks = len(SCALOGRAM_TRACKIDS)
    if n_tracks == 0:
        return

    fig, axes = plt.subplots(n_tracks, 1,
                              figsize=(10, 3.5 * n_tracks), dpi=150,
                              squeeze=False)

    for row, trackid in enumerate(SCALOGRAM_TRACKIDS):
        ax = axes[row, 0]

        matches = glob.glob(str(Path(hdf5_folder) / f'{trackid}*.hdf5'))
        if not matches:
            ax.text(0.5, 0.5, f'No HDF5 for {trackid}',
                    transform=ax.transAxes, ha='center')
            continue

        with h5py.File(matches[0], 'r') as hf:
            try:
                t_raw = np.array(hf[f'{ampm_group}/{ampm_time}'])
                s_raw = np.array(hf[f'{ampm_group}/{ampm_series}'])
            except KeyError:
                ax.text(0.5, 0.5, 'No AMPM signal', transform=ax.transAxes, ha='center')
                continue

            t = t_raw[trim_start: len(t_raw) - trim_end if trim_end else None]
            s = s_raw[trim_start: len(s_raw) - trim_end if trim_end else None]
            if len(t) < 2:
                continue

            kh_time  = np.array(hf['KH/time'])      if 'KH/time'      in hf else np.array([])
            kh_depth = (np.array(hf['KH/max_depth'], dtype=float)
                        if 'KH/max_depth' in hf else None)

        if kh_depth is None or kh_time.size < 4:
            ax.text(0.5, 0.5, 'No KH depth data', transform=ax.transAxes, ha='center')
            continue

        sampling_period = float(round(t[1] - t[0], 9))
        sampling_rate   = round(1.0 / sampling_period, 7)

        s_clean     = _prepare_pd(s, sampling_rate)
        depth_clean = _prepare_kh(kh_depth, kh_time)

        # Interpolate KH depth onto PD time axis
        depth_interp = np.interp(t, kh_time, depth_clean)

        # Compute coherence
        nperseg = int(sampling_rate * 0.01)  # 10 ms segments
        f_coh, Cxy = coherence(s_clean, depth_interp, fs=sampling_rate, nperseg=nperseg)
        f_khz = f_coh / 1000

        # Plot up to cwt_fmax
        mask = f_khz <= cwt_fmax / 1000
        ax.fill_between(f_khz[mask], Cxy[mask], alpha=0.6, color='steelblue')
        ax.plot(f_khz[mask], Cxy[mask], color='steelblue', lw=0.8)

        for bnd in band_boundaries_khz:
            ax.axvline(bnd, color='gray', lw=0.7, ls='--', alpha=0.7)

        ax.set_ylim(0, 1)
        ax.set_ylabel('Coherence', fontsize=9)
        ax.set_title(f'{trackid}', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1, 0].set_xlabel('Frequency (kHz)', fontsize=9)
    fig.suptitle('PD–KH Spectral Coherence (mag-squared)', fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_path = out_dir / 'cwt_spectral_coherence.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved spectral coherence: {out_path}")


def _plot_global_features_by_regime(df, out_dir):
    """Box plots comparing PD vs KH-depth global CWT features grouped by melting regime."""
    from matplotlib.patches import Patch

    if 'melting_regime' not in df.columns:
        print("Skipping regime box plots — no melting_regime column.")
        return

    global_feat_keys = [
        'cwt_entropy', 'cwt_kurtosis', 'cwt_spectral_centroid', 'cwt_spectral_spread',
        'cwt_dominant_freq', 'cwt_activity_ratio', 'cwt_temporal_variance',
        'cwt_peak_count', 'cwt_ridge_mean_freq', 'cwt_ridge_freq_std', 'cwt_ridge_smoothness',
    ]

    feat_names = [k for k in global_feat_keys if k in df.columns]
    if not feat_names:
        print("Skipping regime box plots — no global CWT features found.")
        return

    has_kh = any(f'kh_cwt_depth_{k}' in df.columns for k in feat_names)

    regimes = sorted(df['melting_regime'].dropna().unique())
    if not regimes:
        print("Skipping regime box plots — no regime labels found.")
        return

    ncols = 3
    nrows = int(np.ceil(len(feat_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), dpi=150)
    axes_flat = np.array(axes).ravel()

    colors = ['C0', 'C1']
    n_regimes = len(regimes)
    offset    = 0.2
    pd_pos    = [i + 1 - offset for i in range(n_regimes)]
    kh_pos    = [i + 1 + offset for i in range(n_regimes)]

    def _norm_log(col_vals, group_vals):
        """Min-max normalise group_vals using the global range of col_vals, then log1p."""
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size == 0:
            return group_vals
        vmin, vmax = finite.min(), finite.max()
        rng = vmax - vmin
        if rng == 0:
            return np.zeros_like(group_vals)
        normed = (group_vals - vmin) / rng
        return np.log1p(np.clip(normed, 0, None))

    for feat_idx, feat_key in enumerate(feat_names):
        ax     = axes_flat[feat_idx]
        pd_col = feat_key
        kh_col = f'kh_cwt_depth_{feat_key}'

        # Global column values used to establish the normalisation range per channel
        pd_all = df[pd_col].dropna().values if pd_col in df.columns else np.array([])
        kh_all = df[kh_col].dropna().values if has_kh and kh_col in df.columns else np.array([])

        pd_groups = []
        kh_groups = []
        for regime in regimes:
            mask = df['melting_regime'] == regime
            g_pd = df.loc[mask, pd_col].dropna().values if pd_col in df.columns else np.array([])
            g_kh = (df.loc[mask, kh_col].dropna().values
                    if has_kh and kh_col in df.columns else np.array([]))
            pd_groups.append(_norm_log(pd_all, g_pd))
            kh_groups.append(_norm_log(kh_all, g_kh))

        def _safe(g):
            return g if len(g) > 0 else np.array([np.nan])

        bp_pd = ax.boxplot([_safe(g) for g in pd_groups],
                           positions=pd_pos, widths=0.35,
                           patch_artist=True, manage_ticks=False)
        for patch in bp_pd['boxes']:
            patch.set_facecolor(colors[0])
            patch.set_alpha(0.6)

        if has_kh:
            bp_kh = ax.boxplot([_safe(g) for g in kh_groups],
                               positions=kh_pos, widths=0.35,
                               patch_artist=True, manage_ticks=False)
            for patch in bp_kh['boxes']:
                patch.set_facecolor(colors[1])
                patch.set_alpha(0.6)

        ax.set_xticks(list(range(1, n_regimes + 1)))
        ax.set_xticklabels([str(r) for r in regimes], fontsize=7, rotation=15, ha='right')
        ax.set_xlim(0.5, n_regimes + 0.5)
        ax.set_ylabel('log(1 + normalised value)', fontsize=8)
        ax.set_title(feat_key.replace('cwt_', ''), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

        if feat_idx == 0 and has_kh:
            legend_elements = [Patch(facecolor=colors[0], alpha=0.6, label='PD'),
                               Patch(facecolor=colors[1], alpha=0.6, label='KH depth')]
            ax.legend(handles=legend_elements, fontsize=7, loc='best')

    for ax_idx in range(len(feat_names), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle('Global CWT Features by Melting Regime (PD vs KH depth)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_path = out_dir / 'cwt_global_features_by_regime.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved regime box plots: {out_path}")


if __name__ == '__main__':
    main()
