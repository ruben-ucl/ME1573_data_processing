import os, sys, functools, h5py, glob, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, get_logbook, get_logbook_data, printProgressBar
folder = get_paths()['hdf5']

parser = argparse.ArgumentParser()
parser.add_argument('--pd', type=int, choices=[1, 2], default=2,
                    help='Photodiode number to summarise (default: 2)')
args = parser.parse_args()

group, series = ('AMPM', f'Photodiode{args.pd}Bits')

# Initialse dictionary to store results:
results = {'trackid': [],
    'n': [],
    'min': [],
    'q25': [],
    'median': [],
    'q75': [],
    'max': [],
    'iqr': [],
    'mean': [],
    'std': [],
    'se': [],
    'skewness': [],
    }

# Iterate through HDF5 files in folder
files = sorted(glob.glob(f'{folder}/*.hdf5'))
n_files = len(files)

for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]
    printProgressBar(i, n_files-1, suffix=f'Working on {trackid}')
    
    with h5py.File(filepath, 'r') as file:
        s = np.array(file[f'{group}/{series}'])[500:-500]   # discard the 500 frame margin before and after laser scan
    
    q25, median, q75 = np.percentile(s, [25, 50, 75])
    n = len(s)
    std = np.std(s)

    results['trackid'].append(trackid)
    results['n'].append(n)
    results['min'].append(np.min(s))
    results['q25'].append(q25)
    results['median'].append(median)
    results['q75'].append(q75)
    results['max'].append(np.max(s))
    results['iqr'].append(q75 - q25)
    results['mean'].append(np.mean(s))
    results['std'].append(std)
    results['se'].append(std / np.sqrt(n))
    results['skewness'].append(float(stats.skew(s)))
    
results = pd.DataFrame(results) # Convert to dataframe

results.to_csv(Path(folder, f'{group}_{series}_summary.csv'))