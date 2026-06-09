"""
FKW Angle → HDF5 Migration
===========================
Reads per-trackid FKW angle CSVs from get_paths()['FKW_meas'] and writes
KH/fkw_angle into each matching HDF5 file, aligned to the existing KH/time axis.

Time alignment:
  FKW CSV time is in milliseconds, starting at 0.0 (= laser onset).
  KH/time is in seconds; KH/time[0] ≈ -0.05 ms (2 frames before laser onset).
  Mapping: kh_idx = np.searchsorted(kh_time, t_fkw_ms / 1000)

Usage:
    conda activate ml
    python file/fkw_angle_to_hdf5.py [--overwrite]
"""

import functools
import glob
import os
import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import get_paths, printProgressBar


def main():
    parser = argparse.ArgumentParser(description='Migrate FKW angle CSVs into HDF5 KH/fkw_angle.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing KH/fkw_angle datasets (default: skip if present)')
    args = parser.parse_args()

    mode = 'overwrite' if args.overwrite else 'append'

    if mode == 'overwrite':
        confirm = input("Are you sure you want to overwrite existing KH/fkw_angle datasets? (y/n): ")
        if confirm.strip().lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    paths = get_paths()
    hdf5_folder = paths['hdf5']
    fkw_folder  = paths['FKW_meas']

    hdf5_files = sorted(glob.glob(str(hdf5_folder / '*.hdf5')))
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {hdf5_folder}")
        sys.exit(1)

    print(f"HDF5 folder : {hdf5_folder}  ({len(hdf5_files)} files)")
    print(f"FKW folder  : {fkw_folder}")
    print(f"Mode        : {mode}\n")

    n_written   = 0
    n_skipped   = 0   # no KH/time
    n_no_csv    = 0   # no matching FKW CSV
    n_existing  = 0   # already has KH/fkw_angle (append mode)

    for i, filepath in enumerate(hdf5_files):
        trackid = Path(filepath).name[:7]
        printProgressBar(i + 1, len(hdf5_files), prefix=trackid, suffix='         ')

        # Check matching FKW CSV
        csv_path = fkw_folder / f'{trackid}_fkw_angle_measurements.csv'
        if not csv_path.exists():
            n_no_csv += 1
            continue

        with h5py.File(filepath, 'r+') as hf:
            # Skip files without KH/time
            if 'KH/time' not in hf:
                n_skipped += 1
                continue

            kh_time = np.array(hf['KH/time'])
            if kh_time.size == 0:
                n_skipped += 1
                continue

            # Handle existing dataset
            if 'KH/fkw_angle' in hf:
                if mode == 'append':
                    n_existing += 1
                    continue
                else:  # overwrite
                    del hf['KH/fkw_angle']

            # Load FKW CSV
            fkw_df = pd.read_csv(csv_path, encoding='utf-8')

            # Build NaN array aligned to KH/time
            fkw_arr = np.full(len(kh_time), np.nan)
            for _, row in fkw_df.iterrows():
                t_s = row['time'] / 1000.0  # ms → s
                idx = int(np.searchsorted(kh_time, t_s))
                if 0 <= idx < len(kh_time):
                    fkw_arr[idx] = row['fkw_angle']

            hf.create_dataset('KH/fkw_angle', data=fkw_arr)
            n_written += 1

    print(f"\n{'='*50}")
    print(f"Written   : {n_written}")
    print(f"No FKW CSV: {n_no_csv}")
    print(f"No KH/time: {n_skipped}")
    if mode == 'append':
        print(f"Already had KH/fkw_angle (skipped): {n_existing}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
