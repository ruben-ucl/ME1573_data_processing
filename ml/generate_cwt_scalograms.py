# -*- coding: utf-8 -*-
"""
Generate windowed CWT scalogram images from segmented AMPM HDF5 files.

Identical processing to dataset_labeller's auto-save pipeline but headless
(no PyQt5, no labels).  Edit the constants below, then run:

    conda activate ml
    python ml/generate_scalograms.py
"""

import sys, functools, glob, h5py, time
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.cwt_utils import compute_cwt, save_cwt_image

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------
HDF5_DIR    = Path(r"F:\lab_validation\hdf5")        # directory of 1200_XX.hdf5 files
OUT_DIR     = Path(r"F:\lab_validation\scalograms")  # image output root
PD_CHANNEL  = 'Photodiode1Bits'                      # dataset inside AMPM/ group
WAVELET     = 'cmor1.5-1.0'
N_FREQS     = 256
WINDOW_MS   = 1.0     # window length in ms
OFFSET_MS   = 0.2     # sliding step between window starts in ms
CWT_MODE    = 'per-window'  # 'full' (compute once per file) or 'per-window'
CMAP        = 'grey'
COI_MASKING = False
TRIM        = 0       # samples to drop from each end (0 = no trim; dataset_labeller uses 500)
# ---------------------------------------------------------------------------


def output_folder(base, channel, wavelet, window_ms, freq_min, freq_max, n_freqs, cmap, cwt_mode):
    mode_dir = 'full_signal' if cwt_mode == 'full' else 'per_window'
    folder = base / channel / wavelet.replace('.', '_') / f'{window_ms}_ms' \
             / f'{freq_min}-{freq_max}_Hz_{n_freqs}_steps' / cmap / mode_dir
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def process_file(filepath, out_base):
    trackid = Path(filepath).stem

    with h5py.File(filepath, 'r') as hf:
        t = np.array(hf['AMPM/Time'])
        s = np.array(hf[f'AMPM/{PD_CHANNEL}'])

    if TRIM:
        t = t[TRIM:-TRIM]
        s = s[TRIM:-TRIM]

    t_ms = t * 1000
    t_max_ms = t_ms[-1]

    cwt_time = 0.0

    # Full-signal mode: compute CWT once then slide windows over it
    # Per-window mode: compute_cwt is called per-window inside the loop
    if CWT_MODE == 'full':
        t0 = time.perf_counter()
        cwt_spec = compute_cwt(s, t, WAVELET, N_FREQS, CWT_MODE)
        cwt_time += time.perf_counter() - t0
        freq_min = int(round(cwt_spec['freqs'][-1]))
        freq_max = int(round(cwt_spec['freqs'][0]))
        out_dir  = output_folder(out_base, PD_CHANNEL, WAVELET, WINDOW_MS,
                                 freq_min, freq_max, N_FREQS, CMAP, CWT_MODE)

    n_windows  = 0
    wIndex     = 0
    wStart     = round(OFFSET_MS * (wIndex + 1), 3)
    wEnd       = round(wStart + WINDOW_MS, 3)

    while wEnd <= t_max_ms:
        if CWT_MODE == 'per-window':
            t0 = time.perf_counter()
            cwt_spec = compute_cwt(s, t, WAVELET, N_FREQS, CWT_MODE, wStart, wEnd)
            cwt_time += time.perf_counter() - t0
            if wIndex == 0:
                freq_min = int(round(cwt_spec['freqs'][-1]))
                freq_max = int(round(cwt_spec['freqs'][0]))
                out_dir  = output_folder(out_base, PD_CHANNEL, WAVELET, WINDOW_MS,
                                         freq_min, freq_max, N_FREQS, CMAP, CWT_MODE)

        out_path = out_dir / f'{trackid}_{round(wStart, 1)}-{round(wEnd, 1)}ms.png'
        save_cwt_image(cwt_spec, wStart, wEnd, CWT_MODE, CMAP, COI_MASKING, out_path)
        n_windows += 1

        wIndex += 1
        wStart  = round(OFFSET_MS * (wIndex + 1), 3)
        wEnd    = round(wStart + WINDOW_MS, 3)

    return n_windows, cwt_time


def main():
    files = sorted(glob.glob(str(HDF5_DIR / '*.hdf5')))
    if not files:
        print(f'No HDF5 files found in {HDF5_DIR}')
        return

    print(f'Generating scalograms for {len(files)} files -> {OUT_DIR}')
    total = 0
    total_cwt_time = 0.0
    for i, fp in enumerate(files, 1):
        n, cwt_time = process_file(fp, OUT_DIR)
        print(f'  [{i}/{len(files)}] {Path(fp).name}: {n} windows')
        total += n
        total_cwt_time += cwt_time

    n_cwt_calls = total if CWT_MODE == 'per-window' else len(files)
    per_call_ms = (total_cwt_time / n_cwt_calls * 1000) if n_cwt_calls else 0
    print(f'Done. {total} images saved.')
    print(f'CWT compute: {total_cwt_time:.2f}s total | {per_call_ms:.1f}ms per call '
          f'({n_cwt_calls} calls, mode={CWT_MODE})')


if __name__ == '__main__':
    main()
