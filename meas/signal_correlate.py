import h5py, functools, scipy, glob, sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tools import get_paths, define_collumn_labels

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

path_dict = get_paths()
label_dict = define_collumn_labels()
folder = path_dict['hdf5']

group1, time1, series1, colour1 = ('AMPM', 'Time', 'Photodiode1Bits', 'r')
group2, time2, series2, colour2 = ('KH', 'time', 'max_depth', 'b')

for filepath in sorted(glob.glob(f'{folder}/*.hdf5')):
    trackid = Path(filepath).name[:7]
    print(trackid)
    
    with h5py.File(filepath, 'r') as file:
        t1 = np.array(file[f'{group1}/{time1}']) * 1000
        t2 = np.array(file[f'{group2}/{time2}']) * 1000
        s1 = np.array(file[f'{group1}/{series1}'])
        s2 = np.array(file[f'{group2}/{series2}'])
    
    # Plot data
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(6.3, 3.15), dpi=300, tight_layout=True)
    fig.suptitle(f'{trackid}')
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(-0.5, max(t2)+0.5)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel(series1, c=colour1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(label_dict[series2][1], c=colour2)
    
    ax1.plot(t1, s1, c=colour1, lw=0.5)
    ax2.plot(t2, s2, c=colour2, lw=0.5)
    plt.show()
    plt.close()
    