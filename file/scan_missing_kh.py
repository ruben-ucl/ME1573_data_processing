import sys, h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import get_paths

def main():
    hdf5_dir = Path(get_paths()['hdf5'])
    files = sorted(hdf5_dir.glob('*.hdf5'))
    print(f'Scanning {len(files)} HDF5 files in {hdf5_dir}\n')

    missing = []
    for path in files:
        with h5py.File(path, 'r') as f:
            if 'KH' not in f:
                missing.append(path.stem)

    if missing:
        print(f'Missing KH group ({len(missing)}/{len(files)}):')
        for tid in missing:
            print(f'  {tid}')
    else:
        print('All files contain KH group.')

if __name__ == '__main__':
    main()
