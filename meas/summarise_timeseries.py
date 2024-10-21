import os, sys, functools, h5py, glob
import pandas as pd
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, get_logbook, get_logbook_data, printProgressBar
folder = get_paths()['hdf5']
logbook = get_logbook

group, series = ('AMPM', 'Photodiode1Bits')

# Initialse dictionary to store results:
results = {'trackid': [],
    'min': [],
    'max': [],
    'mean': [],
    'std': []
    }

# Iterate through HDF5 files in folder
files = sorted(glob.glob(f'{folder}/*.hdf5'))
n_files = len(files)

for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]
    printProgressBar(i, n_files-1, suffix=f'Working on {trackid}')
    
    with h5py.File(filepath, 'r') as file:
        s = np.array(file[f'{group}/{series}'])[500:-500]   # discard the 500 frame margin before and after laser scan
    
    results['trackid'].append(trackid)
    results['mean'].append(np.mean(s))
    results['min'].append(np.min(s))
    results['max'].append(np.max(s))
    results['std'].append(np.std(s))
    
results = pd.DataFrame(results) # Convert to dataframe

results.to_csv(Path(folder, f'{group}_{series}_summary.csv'))