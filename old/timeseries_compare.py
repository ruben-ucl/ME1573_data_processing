import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob, functools, os, sys
from pathlib import Path
from scipy.signal import medfilt, find_peaks

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, interpolate_low_quality_data

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

##################################################################################################

path1 = get_paths()['KH_meas']
label1 = 'max_depth'

path2 = get_paths()['FKW_meas']
label2 = 'fkw_angle'

med_filt_window = None # None or odd-valued int
running_mean_window = 3 # None or odd-valued int
interpolate_isolated = True # bool

labels = {'area': ['Keyhole area', ' [μm\u00b2]'],
          'max_depth': ['Keyhole depth', ' [μm]'],
          'max_length': ['Keyhole length', '\n[μm]'],
          'AR': ['Keyhole aspect ratio', ',\nd/l'],
          'Ra': ['Roughness, Ra', ' [μm]'],
          'fkw_angle': ['Keyhole front wall angle', ' [°]']
          }

##################################################################################################

def main():
    files = sorted(glob.glob(str(Path(path1, '*keyhole_measurements.csv'))))
    all_peaks = []
    for f1 in files:
        f1name = Path(f1).name
        print('\nReading %s' % f1name)
        trackid = f1name[:7]
        f2 = glob.glob(str(Path(path2, f'{trackid}_fkw_angle_measurements_raw.csv')))[0]
        
        df1 = pd.read_csv(f1, index_col=0)
        df2 = pd.read_csv(f2, index_col=0)
        t1 = df1['time'][:-running_mean_window+1]
        t2 = df2['time'][:-running_mean_window+1]
        var1 = get_data(label1, df1)
        var2 = get_data(label2, df2)
        
        
        if interpolate_isolated:
            var1, _ = interpolate_low_quality_data(var1, None)
            var2, _ = interpolate_low_quality_data(var2, None)
        if med_filt_window != None:
            var1 = medfilt(var1, med_filt_window)
            var2 = medfilt(var2, med_filt_window)
        if running_mean_window != None:
            var1 = np.convolve(var1, np.ones(running_mean_window)/running_mean_window, mode='valid')
            var2 = np.convolve(var2, np.ones(running_mean_window)/running_mean_window, mode='valid')
        
        
        plt.rcParams.update({'font.size': 8})
        # fig = plt.figure(figsize=(6.3, 3.15), dpi=300, tight_layout=True)
        fig = plt.figure(figsize=(4, 5), dpi=300, tight_layout=True)
        fig.suptitle(f'{trackid}: {labels[label1][0]}, {labels[label2][0]}\n')
        
        ax1 = fig.add_subplot(111)
        ax1.plot(t1, var1, '-', c='tab:blue', lw=1, marker='o', markersize=3.5, label=labels[label1][0])
        # ax1.plot(t1, var1, '-', c='tab:blue', lw=0.8, markersize=2, label=labels[label1][0])
        ax1.set_xlabel('Time [ms]', fontsize=10)
        ax1.set_xlim(0.08, 0.2)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel(labels[label1][0]+labels[label1][1], c='tab:blue', fontsize=10)
        ax1.legend(loc='lower left', bbox_to_anchor=(0.01, 0.1), framealpha=0)
        
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 80)
        ax2.plot(t2, var2, '--', c= 'tab:red', lw=1, marker='x', markersize=4, label=labels[label2][0])
        # ax2.plot(t2, var2, '--', c= 'tab:red', lw=0.8, markersize=3, label=labels[label2][0])
        ax2.set_ylabel(labels[label2][0]+labels[label2][1], c='tab:red', fontsize=10)
        ax2.legend(loc='lower left', bbox_to_anchor=(0.01, 0.02), framealpha=0)
        
        plt.show()
        # plt.savefig(str(Path(path1, 'plots_smoothed', f'{trackid}_KH_{label1}_and_{label2}')))
        plt.close()

def get_data(label, df):
    var = np.nan_to_num(np.divide(df['max_depth'], df['max_length'])) if label == 'AR' else df[label]
    var *= 4.3**2 if label == 'area' else 1 # convert area from px to um^2
    return var

if __name__ == '__main__':
    main()