import h5py, glob, functools, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import 

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_AMPM_channel_names

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

hdf5_path = get_paths()['hdf5']

channels = get_AMPM_channel_names()

mode = 'append' # 'append' or 'overwrite'

for filepath in sorted(glob.glob(f'{hdf5_path}\*.hdf5')):
    trackid = Path(filepath).name[:7]
    print(trackid)
    
    if len(glob.glob(f'{hdf5_path}/AMPM_plots/{trackid}_AMPM.png')) == 1 and mode != 'overwrite':
        continue
    
    # Read AMPM data to dataframe
    with h5py.File(filepath, 'r+') as file:
        df = pd.DataFrame()
        try:
            for key in file['AMPM'].keys():
                df[key] = file[f'AMPM/{key}']
        except KeyError:
            print('No AMPM data in file')
            continue
        
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(6.3, 9.45), dpi=600, tight_layout=True)
    fig.suptitle(f'{trackid} AMPM')
    ax1 = fig.add_subplot(311)
    ax1b = ax1.twinx()
    ax2 = fig.add_subplot(312, sharex = ax1)
    ax3 = fig.add_subplot(313, sharex = ax1)
    ax3b = ax3.twinx()
    ax3.set_xlabel('Time [ms]')
    
    for ax in fig.axes:
        if ax != ax3: plt.setp(ax.get_xticklabels(), visible=False)
    
    t = df['Time'] * 1000

    # Subplot 1 primary axis (galvo cartesian demand and actual)
    ax1.plot(t, df[channels[1]], 'b--', label=channels[1], lw=0.5)
    ax1.plot(t, df[channels[7]], 'b-', label=channels[7], lw=0.5)
    
    ax1.plot(t, df[channels[3]], 'r--', label=channels[3], lw=0.5)
    ax1.plot(t, df[channels[9]], 'r-', label=channels[9], lw=0.5)
    
    ax1.set_ylabel('Galvo cartesian [mm]')
    ax1.legend(fontsize='xx-small', loc='upper left')

    # Subplot 1 secondary axis (focus demand and actual)
    ax1b.plot(t, df[channels[5]], 'k--', label=channels[5], lw=0.5)
    ax1b.plot(t, df[channels[11]], 'k-', label=channels[11], lw=0.5)
    
    ax1b.set_ylabel('Focus [mm]')
    ax1b.legend(fontsize='xx-small', loc='upper right')
    
    # Subplot 2 primary axis (laser control and feedback)
    ax2.plot(t, df[channels[12]], 'r--', label=channels[12], lw=0.5)
    ax2.plot(t, df[channels[22]], 'b--', label=channels[22], lw=0.5)
    ax2.plot(t, df[channels[13]], 'k-', label=channels[13], lw=0.5)
    
    ax2.set_ylabel('Laser control/beam dump signal [bits]')
    ax2.legend(fontsize='xx-small', loc='upper right')
    
    # Subplot 3 primary axis (photodiodes)
    ax3.plot(t, df[channels[15]], 'r-', label=channels[15], lw=0.5)
    ax3b.plot(t, df[channels[17]], 'b-', label=channels[17], lw=0.5)
    
    ax3.set_ylabel('Photodiode 1 signal [bits]')
    ax3b.set_ylabel('Photodiode 2 signal [bits]')
    ax3.legend(fontsize='xx-small', loc='upper left')
    ax3b.legend(fontsize='xx-small', loc='upper right')
    
    # Save output figure
    plt.savefig(f'{hdf5_path}/AMPM_plots/{trackid}_AMPM.png')
    plt.close()