# -*- coding: utf-8 -*-
"""
Pure CWT computation and image-save utilities shared between dataset_labeller
and generate_scalograms.  No PyQt5 / GUI dependency.
"""

import sys
import numpy as np
import pywt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import get_cwt_scales


# ---------------------------------------------------------------------------
# CWT computation
# ---------------------------------------------------------------------------

def cwt_full_signal(s, n_points, scales, wavelet, sampling_period):
    """Symmetric-padded CWT on the full signal.

    Returns (cwtmatr, freqs) cropped to the original signal length.
    """
    s_pad = np.pad(s, len(s), mode='symmetric')
    cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=sampling_period)
    cwtmatr = np.abs(cwtmatr[:, n_points:2 * n_points])
    return cwtmatr, freqs


def cwt_per_window(s, t, wStart_ms, wEnd_ms, scales, wavelet, sampling_period):
    """Smart-padded CWT on a windowed region.

    Uses actual signal as padding where available, synthetic reflection only at edges.
    Returns (cwtmatr, freqs) already trimmed to the window region.
    """
    t_ms = t * 1000
    window_start_idx = np.argmin(np.abs(t_ms - wStart_ms))
    window_end_idx   = np.argmin(np.abs(t_ms - wEnd_ms))
    window_length    = window_end_idx - window_start_idx
    pad_length       = window_length

    # Left padding
    if window_start_idx >= pad_length:
        left_pad = s[window_start_idx - pad_length : window_start_idx]
    else:
        available_left   = window_start_idx
        left_signal      = s[0:window_start_idx]
        synthetic_needed = pad_length - available_left
        synthetic_left   = np.pad(left_signal, (synthetic_needed, 0), mode='symmetric')[:synthetic_needed]
        left_pad         = np.concatenate([synthetic_left, left_signal])

    window = s[window_start_idx:window_end_idx]

    # Right padding
    if window_end_idx + pad_length <= len(s):
        right_pad = s[window_end_idx : window_end_idx + pad_length]
    else:
        available_right  = len(s) - window_end_idx
        right_signal     = s[window_end_idx : len(s)]
        synthetic_needed = pad_length - available_right
        synthetic_right  = np.pad(right_signal, (0, synthetic_needed), mode='symmetric')[-synthetic_needed:]
        right_pad        = np.concatenate([right_signal, synthetic_right])

    s_pad = np.concatenate([left_pad, window, right_pad])
    cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=sampling_period)
    cwtmatr = np.abs(cwtmatr[:, pad_length : pad_length + window_length])
    return cwtmatr, freqs


def apply_coi_masking(cwtmatr, freqs, sampling_rate):
    """Zero out COI edge regions in the CWT coefficient matrix."""
    cwtmatr_masked = cwtmatr.copy()
    for i, freq in enumerate(freqs):
        scale     = sampling_rate / freq
        coi_width = int(np.ceil(np.sqrt(2) * scale))
        cwtmatr_masked[i, :coi_width]  = 0
        cwtmatr_masked[i, -coi_width:] = 0
    return cwtmatr_masked


def compute_cwt(s, t, wavelet, n_freqs, cwt_mode, wStart_ms=None, wEnd_ms=None):
    """High-level CWT wrapper that mirrors Controller.cwt().

    Returns a cwt_spec dict: {t, freqs, cwtmatr, vmax, n_samples_window}.
    wStart_ms / wEnd_ms are only required when cwt_mode == 'per-window'.
    """
    sampling_period = round(float(t[1] - t[0]), 9)
    sampling_rate   = int(round(1.0 / sampling_period))
    n_points        = len(t)

    scales, vmax = get_cwt_scales(wavelet, n_freqs, sampling_rate=sampling_rate)

    if cwt_mode == 'per-window':
        if wStart_ms is None or wEnd_ms is None:
            raise ValueError("wStart_ms and wEnd_ms are required for per-window mode")
        cwtmatr, freqs = cwt_per_window(s, t, wStart_ms, wEnd_ms, scales, wavelet, sampling_period)
    else:
        cwtmatr, freqs = cwt_full_signal(s, n_points, scales, wavelet, sampling_period)

    n_samples_window = int((wEnd_ms - wStart_ms) * sampling_rate / 1000) if wStart_ms is not None else 0

    return {
        't':                t,
        'freqs':            freqs,
        'cwtmatr':          cwtmatr,
        'vmax':             vmax,
        'n_samples_window': n_samples_window,
        'sampling_rate':    sampling_rate,
    }


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_cwt_image(cwt_spec, wStart_ms, wEnd_ms, cwt_mode, cmap,
                   coi_masking, output_path, override_vmax=None):
    """Save a CWT window as a PNG using PIL (fast, no matplotlib overhead).

    Mirrors Controller._save_cwt_from_cached().
    output_path: full Path to the destination .png file.
    """
    import matplotlib as mpl
    from PIL import Image

    t    = cwt_spec['t']
    t_ms = t * 1000

    if cwt_mode == 'per-window':
        cwt_windowed = cwt_spec['cwtmatr']
    else:
        i0 = np.argmin(np.abs(t_ms - wStart_ms))
        i1 = np.argmin(np.abs(t_ms - wEnd_ms))
        cwt_windowed = cwt_spec['cwtmatr'][:, i0:i1]

    if coi_masking:
        sampling_rate = cwt_spec.get('sampling_rate', int(round(1.0 / (t[1] - t[0]))))
        cwt_windowed = apply_coi_masking(cwt_windowed, cwt_spec['freqs'], sampling_rate)

    vmax           = override_vmax if override_vmax is not None else cwt_spec['vmax']
    cwt_normalized = np.clip(cwt_windowed / vmax, 0, 1)

    cmap_func  = mpl.colormaps.get_cmap(cmap)
    cwt_colored = cmap_func(cwt_normalized)                       # RGBA float
    cwt_rgb     = (cwt_colored[:, :, :3] * 255).astype(np.uint8) # RGB uint8

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cwt_rgb, mode='RGB').save(output_path, optimize=True)
