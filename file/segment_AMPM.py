# -*- coding: utf-8 -*-
"""
Segment an AMPM .dat file by Modulate signal and save each active segment
to a separate HDF5 file under the group AMPM/.

Edit DAT_PATH, OUT_DIR, and TRACK_BASE before running.
"""

import sys
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from file.read_AMPM import readAMPMdat

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
DAT_PATH   = Path(r"F:\Lab validation\AMPM_1_L4_100K_2025-11-10_16-56-34.dat")   # path to source .dat file
OUT_DIR    = Path(r"F:\Lab validation\hdf5")  # HDF5 output root
TRACK_BASE = "1200"   # output files: 1200_01.hdf5, 1200_02.hdf5, ...
# ---------------------------------------------------------------------------


def find_segments(beam_dump):
    """Return list of (start, end) index pairs where BeamDumpDiodeBits is active.
    Values < 20 are treated as zero (noise floor).
    """
    mod_bin = (beam_dump >= 20).astype(int)
    diff = np.diff(mod_bin, prepend=0)
    rising  = np.where(diff ==  1)[0]
    falling = np.where(diff == -1)[0]

    segments = []
    for r in rising:
        after = falling[falling > r]
        end = after[0] if len(after) else len(mod_bin) - 1
        if end > r:
            segments.append((r, end))
    return segments


def write_segment(df, start, end, out_path):
    seg = df.iloc[start:end].copy()
    seg['Time'] = seg['Time'] - seg['Time'].iloc[0]
    with h5py.File(out_path, 'w') as hf:
        grp = hf.require_group('AMPM')
        for col in seg.columns:
            grp.create_dataset(col, data=seg[col].values)


def main():
    if not DAT_PATH.exists():
        print(f"DAT_PATH not found: {DAT_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {DAT_PATH.name} ...")
    df = readAMPMdat(str(DAT_PATH))
    print(f"  {len(df)} samples, {df.shape[1]} channels")

    segments = find_segments(df['BeamDumpDiodeBits'].values)
    print(f"  {len(segments)} segments found")

    for i, (start, end) in enumerate(segments, 1):
        out_path = OUT_DIR / f"{TRACK_BASE}_{i:02d}.hdf5"
        write_segment(df, start, end, out_path)
        duration = df['Time'].iloc[end] - df['Time'].iloc[start]
        print(f"  [{i:02d}] samples {start}–{end}  ({duration*1e3:.1f} ms)  -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
