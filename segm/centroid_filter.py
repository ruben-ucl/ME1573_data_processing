import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob, functools, sys, os
from pathlib import Path
from scipy.signal import medfilt, find_peaks

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

##################################################################################################

path1 = get_paths()['KH_meas']
label1 = 'centroid'

save_mode = 'save'

# Cutoff margins above and below dataset median values
x_high = 5
x_low = 6

y_high = 7

##################################################################################################

def main():
    files = sorted(glob.glob(str(Path(path1, '1*nofilt.csv'))))
    all_peaks = []
    for f1 in files:
        f1name = Path(f1).name
        print('\nReading %s' % f1name)
        trackid = f1name[:7]
        
        df1 = pd.read_csv(f1, index_col=0, keep_default_na=False)
        t1 = df1['time'].copy().tolist()
        sr1 = 504 # kHz
        var1 = get_data(label1, df1)
        cx = []
        cy = []
        for e in var1.copy():
            if e == '':
                cx.append(np.nan)
                cy.append(np.nan)
            else:
                coords = e[1:-1].split(', ')
                cx.append(float(coords[1]))
                cy.append(float(coords[0]))
            
        cx_median = np.median([x for x in cx if str(x) != 'nan'])
        cy_median = np.median([y for y in cy if str(y) != 'nan'])
        print(cx_median)
        print(cy_median)
        
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(6.3, 3.15), dpi=300, tight_layout=True)
        fig.suptitle(f'{trackid} KH centroid')
        
        ax1 = fig.add_subplot(111)
        
        ax1.scatter(t1, cx, c='tab:blue', marker='o', s=1, label='x')
        ax1.plot([t1[0], t1[-1]], [cx_median+x_high, cx_median+x_high], '--', c='tab:blue', linewidth=0.6)
        ax1.plot([t1[0], t1[-1]], [cx_median-x_low, cx_median-x_low], '--', c='tab:blue', linewidth=0.6)
        
        ax1.scatter(t1, cy, c='tab:red', marker='o', s=1, label='y')
        ax1.plot([t1[0], t1[-1]], [cy_median+y_high, cy_median+y_high], '--', c='tab:red', linewidth=0.6)
        ax1.plot([t1[0], t1[-1]], [3    , 3], '--', c='tab:red', linewidth=0.6)
        
        ax1.set_xlabel('Time [ms]')
        # ax1.set_xlim(0.08, 0.2)
        ax1.set_ylabel('Centroid [px]')
        ax1.legend(framealpha=0)
        
        # plt.show()
        # plt.savefig(str(Path(path1, f'{trackid}_KH_{label1}_and_{label2}')))
        plt.close()
        
        df_out = df1.copy()
        df_out['centroid-x'] = cx
        df_out['centroid-y'] = cy
        df_out['valid'] = np.ones_like(cx)
        
        for label in ['area',
                      'max_depth',
                      'max_length',
                      'apperture_length',
                      'depth_at_max_length',
                      'valid']:
            df_out[label][np.logical_not(
                              np.logical_and(
                                  np.logical_and(df_out['centroid-x'] > cx_median-x_low, df_out['centroid-x'] < cx_median+x_high),
                                  np.logical_and(df_out['centroid-y'] > 3, df_out['centroid-y'] < cy_median+y_high)
                          ))] = 0
        
        # for i, row in df_out.iterrows():
            # xi = row['centroid-x']
            # yi = row['centroid-y']
            # if np.logical_and(np.logical_and(xi > cx_median-6, xi < cx_median+4),
                              # np.logical_and(yi > 3, yi < cy_median+7)
                              # ):
                # df_out['area'][i] = 0
                # df_out['max_depth'][i] = 0
                # df_out['max_length'][i] = 0
                # df_out['depth_at_max_length'][i] = 0
                # df_out['apperture_length'][i] = 0
                # df_out['outlier'][i] = 1
        
        plt.plot(df_out['time'], df_out['max_depth'])
        
        print(df_out)
        # plt.show()
        plt.close()
        if save_mode == 'save':
            pd.DataFrame(df_out).to_csv(Path(path1, f'{trackid}_keyhole_measurements_v2.csv'))

def get_data(label, df):
    var = np.nan_to_num(np.divide(df['max_depth'], df['max_length'])) if label == 'AR' else df[label]
    var *= 4.3**2 if label == 'area' else 1 # convert area from px to um^2
    return var

if __name__ == '__main__':
    main()