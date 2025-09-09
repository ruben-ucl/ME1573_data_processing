import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, functools, os, sys, h5py
from pathlib import Path
from scipy.signal import medfilt, find_peaks, savgol_filter
from scipy.stats import describe

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# data_path = get_paths()['hdf5']
data_path = r'E:/sim_segmented_300W_800mm_s/SLM_Al10SiMg_1st_layer_4mm_350W_800mms'
# group = 'AMPM/'
group = 'meas/'
# time_label = 'Time'
time_label = 'time'
# data_label = 'Photodiode1Bits'
data_label = 'sum(Q) (stats)'
general_filename = 'absorption.hdf5'

labels = {'area': ['Keyhole area', ' [μm\u00b2]'],
          'max_depth': ['Keyhole depth', '\n[μm]'],
          'max_length': ['Keyhole length', '\n[μm]'],
          'AR': ['Keyhole aspect ratio', ',\nd/l'],
          'fkw_angle': ['Keyhole front wall angle', '\n[°]'],
          'Photodiode1Bits': ['Photodiode 1 intensity', '\n'],
          'KH_depth_um': ['Keyhole depth', '\n[μm]'],
          'EnergyAbsorbed_W': ['Laser absorption', '\n[W]'],
          'sum(Q) (stats)': ['Laser absorption', '\n[W]']
          }

plot_x_fraction = 0.25 # Set fraction of frequency range to plot on FFT inset
plot_y_fraction = 1.05 # Set fraction of fft magnitude to plot on FFT inset
med_filt_window = None # None or odd-valued int
running_mean_window = None # None or odd-valued int
savgol_window = 5 # None or int
plot_sum = False
label_peaks = False
mode = 'save'

def main():
    files = sorted(glob.glob(str(Path(data_path, general_filename))))
    print(f'Found {len(files)} files')
    fft_dict = {}
    for f in files:
        fname = Path(f).name
        print('\nReading %s' % fname) 
        trackid = fname[:7]
        
        # df = pd.read_csv(f, index_col=0)
        with h5py.File(f, 'r') as file:
            try:
                # t = np.array(file[group+time_label])[10:-10]#[500:-490]
                # t = np.array(file[group+time_label])[510:-510]
                # t = np.array(file[group+time_label])
                t = np.array(file[group+time_label])[116:-1]
                # x = np.array(file[group+data_label])[10:-10]#[500:-490]
                # x = np.array(file[group+data_label])[510:-510]
                # x = np.array(file[group+data_label])
                x = np.array(file[group+data_label])[116:-1]
                print(describe(x))
                delta_t = (t[-1]-t[0])/len(t) # s
                sr = 0.001/delta_t # kHz
                
            except KeyError:
                print(f'Dataset \'{data_label}\' not found - skipping file')
                continue
        
        if med_filt_window != None:
            x = medfilt(x, med_filt_window)
        if running_mean_window != None:
            x = np.convolve(x, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        if savgol_window != None:
            x = savgol_filter(x, savgol_window, 1)
        
        N = len(t)
        X = np.fft.fft(x-np.mean(x))/1000
        freq = np.fft.fftfreq(N, 1/sr)
        if 'freq' not in fft_dict.keys(): fft_dict['freq'] = freq
        fft_dict[trackid] = np.abs(X)
        
        print('Creating figure')
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(6.3, 1.8), dpi=300, tight_layout=True)
        fig.suptitle(f'{trackid} KH flickering FFT - {labels[data_label][0]}')
        
        ax1 = fig.add_subplot(121)
        ax1.plot(t*1000, x, lw=0.8)
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel(labels[data_label][0]+labels[data_label][1])
        
        ax2 = fig.add_subplot(122)
        ax2.stem(freq, np.abs(X), markerfmt=' ', basefmt=' ')
        ax2.set_xlabel('Frequency [kHz]')
        ax2.set_ylabel('FFT Amplitude')
        x_max = sr/2
        # x_max = 20
        ax2.set_xlim(0, x_max)
        ax2.set_ylim(0, None)
        # ax2.set_xticks([0, 5, 10, 15, 20])
        
        if True:
            ax2_sub = ax2.inset_axes([0.4,0.45,0.55,0.45])
            ax2_sub.stem(freq, np.abs(X), markerfmt=' ', basefmt=' ')
            x2_max =x_max*plot_x_fraction
            y2_max = np.max(np.abs(X))*plot_y_fraction
            ax2_sub.set_xlim(0, x2_max)
            ax2_sub.set_ylim(0, y2_max)
            # ax2_sub.set_xticks([0, 2, 4])
        
        if mode == 'save':
            output_folder = f'{data_path}/FFT_{data_label}'
            print('Figure saved to ', output_folder)
            if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            plt.savefig(str(Path(output_folder, f'{trackid}_{data_label}_fft.png')))
            plt.close()
        else:
            plt.show()
    
    if plot_sum == True:
        try:
            fft_sum = np.zeros_like(fft_dict['freq'])
            for k in fft_dict.keys():
                if k != 'freq':
                    fft_sum = fft_sum + fft_dict[k]
            peaks, peak_props = find_peaks(fft_sum, prominence=37)
            
            fig = plt.figure(figsize=(3.15, 1.8), dpi=300, tight_layout=True)
            fig.suptitle(f'504 kHz FFT sum: {data_label}')
            ax1 = fig.add_subplot(111)
            ax1.stem(fft_dict['freq'], fft_sum, markerfmt=' ', basefmt=' ')
            if label_peaks == True:
                ax1.scatter(fft_dict['freq'][peaks], fft_sum[peaks], marker='x', c='k', s=10, linewidths=0.7, zorder=10)
                for p in peaks[:len(peaks)//2+1]:
                    ax1.text(fft_dict['freq'][p]+1, fft_sum[p]+160, round(fft_dict['freq'][p], 1), fontsize='xx-small')
            ax1.set_xlabel('Frequency [kHz]')
            ax1.set_ylabel('FFT Amplitude')
            ax1.set_xlim(0, None)
            ax1.set_ylim(0, None)
            if True:
                ax1_sub = ax1.inset_axes([0.4,0.45,0.55,0.45])
                ax1_sub.stem(freq, np.abs(X), markerfmt=' ', basefmt=' ')
                x_max = x_max/2*plot_x_fraction
                y_max = np.max(fft_sum)*plot_y_fraction
                ax1_sub.set_xlim(0, x_max)
                ax1_sub.set_ylim(0, None)
                # ax1_sub.set_xticks([0, 10, 20, 30, 40, 50])
            # plt.show()
            plt.savefig(str(Path(data_path, f'{data_label}_fft_sum.png')))
        except ValueError:
            print('Series lengths do not match - cannot calulate FFT sum')
    

if __name__ == '__main__':
    main()