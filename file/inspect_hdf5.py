import h5py, glob, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v2.0'

'''
CHANGELOG
    v0.1 - simple loop that prints dataset names and gives option to delete them
    v0.2 - prints dataset attributes in a table
    v1.0 - multilevel tables for displaying up to two levels of data subgroups
         - improved control loop for automatically deleting a dataset from all files
         - added loop break with 'x' input
    v1.1 - added purge_kh mode
    v2.0 - unified three-mode system: one_by_one | from_list | from_all
           any dataset or group can be targeted via TARGET
'''

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE   = 'one_by_one'   # 'one_by_one' | 'from_list' | 'from_all'
TARGET = 'keyhole_bin'           # dataset or group to delete (used in from_list / from_all)

# Trackids to operate on — only used when MODE == 'from_list'
TRACKIDS = ['0301_01', '0301_03', '0301_05', '0302_01',
            '0304_04', '0306_02', '0307_01', '0307_06',
            '0504_01', '0504_02', '0504_06', '0506_04',
            '0506_05', '0507_02', '0507_03', '0507_05',
            '0507_06', '0516_05'
            ]

# ─────────────────────────────────────────────────────────────────────────────

hdf5_dir = Path(get_paths()['hdf5'])

col_w = [50, 25, 15, 10]
total_w = np.sum(col_w) + 3
col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
tab_rule = '-' * total_w

# ── Shared helpers ────────────────────────────────────────────────────────────

def print_contents(f):
    print(f'\n{f.filename}  —  datasets:\n{tab_rule}')
    props = {}
    for i in f.keys():
        item = f[i]
        try:
            props[i] = [str(item.shape), str(item.dtype), str(round(item.nbytes / 1e9, 6))]
        except AttributeError:
            props[i + '/'] = ['', 'Group', '']
            for j in f[i].keys():
                sub = f[i][j]
                try:
                    props[f'    {j}'] = [str(sub.shape), str(sub.dtype), str(round(sub.nbytes / 1e9, 6))]
                except AttributeError:
                    props[f'    {j}/'] = ['', 'Sub-group', '']
                    for k in f[i][j].keys():
                        subsub = f[i][j][k]
                        try:
                            props[f'        {k}'] = [str(subsub.shape), str(subsub.dtype), str(round(subsub.nbytes / 1e9, 6))]
                        except AttributeError:
                            pass
    print(col_format.format('Name', 'Shape', 'Datatype', 'Gigabytes'))
    print(tab_rule)
    for name, (shape, dtype, nb) in props.items():
        print(col_format.format(name, shape, dtype, nb))

def delete_target(f, target):
    if target in f:
        del f[target]
        return True
    return False

# ── one_by_one mode ───────────────────────────────────────────────────────────

def run_one_by_one():
    for path in sorted(hdf5_dir.glob('*.h*5')):
        with h5py.File(path, 'a') as f:
            while True:
                print_contents(f)
                cmd = input("\nDataset/group to delete, 'c' to continue, 'x' to exit: ").strip()
                if cmd == 'x':
                    print('Done.')
                    return
                elif cmd == 'c':
                    break
                elif cmd in f:
                    del f[cmd]
                    print(f"Deleted '{cmd}'.")
                else:
                    print(f"'{cmd}' not found.")
    print('Done.')

# ── from_list / from_all shared batch logic ───────────────────────────────────

def run_batch(paths, target):
    if not paths:
        print('No files to process.')
        return

    print(f"Will delete '{target}' from {len(paths)} file(s):")
    for p in paths:
        print(f'  {p.stem}')
    confirm = input('\nProceed? (y/n): ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        return

    deleted, skipped = [], []
    for path in paths:
        if not path.exists():
            print(f'  {path.stem}: file not found, skipping')
            skipped.append(path.stem)
            continue
        with h5py.File(path, 'a') as f:
            if delete_target(f, target):
                print(f'  {path.stem}: deleted')
                deleted.append(path.stem)
            else:
                print(f"  {path.stem}: '{target}' not found, skipping")
                skipped.append(path.stem)

    print(f'\nDeleted: {len(deleted)}  |  Skipped: {len(skipped)}')
    print('Done.')

# ─────────────────────────────────────────────────────────────────────────────

def main():
    if MODE == 'one_by_one':
        run_one_by_one()
    elif MODE == 'from_list':
        paths = [hdf5_dir / f'{tid}.hdf5' for tid in TRACKIDS]
        run_batch(paths, TARGET)
    elif MODE == 'from_all':
        paths = sorted(hdf5_dir.glob('*.h*5'))
        run_batch(paths, TARGET)
    else:
        print(f"Unknown MODE '{MODE}'. Choose: one_by_one | from_list | from_all")

main()
