import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, functools
from pathlib import Path
from scipy.signal import medfilt, find_peaks

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

data_path = r'J:\ESRF ME1573 LTP 6 Al data HDF5\ffc\keyhole_measurements_lagrangian'
data_label = 'area'
general_filename = '1*v2.csv'

labels = {'area': ['Keyhole area', ' [μm\u00b2]'],
          'max_depth': ['Keyhole depth', '\n[μm]'],
          'max_length': ['Keyhole length', '\n[μm]'],
          'AR': ['Keyhole aspect ratio', ',\nd/l'],
          'fkw_angle': ['Keyhole front wall angle', '\n[°]']
          }

plot_x_fraction = 0.2 # Set fraction of frequency range to plot on FFT inset
plot_y_fraction = 0.5 # Set fraction of fft magnitude to plot on FFT inset
med_filt_window = None # None or odd-valued int
running_mean_window = 5 # None or odd-valued int
plot_sum = True
label_peaks = False
sr = 504 # kHz

def main():
    files = sorted(glob.glob(str(Path(data_path, general_filename))))
    print(f'Found {len(files)} files')
    fft_dict = {}
    for f in files:
        fname = Path(f).name
        print('\nReading %s' % fname) 
        trackid = fname[:7]
        
        df = pd.read_csv(f, index_col=0)
        t = df['time']
        x = np.nan_to_num(np.divide(df['max_depth'], df['max_length'])) if data_label == 'AR' else df[data_label]
        x *= 4.3**2 if data_label == 'area' else 1 # convert area from px to um^2
        
        if med_filt_window != None:
            x = medfilt(x, med_filt_window)
        if running_mean_window != None:
            x = np.convolve(x, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        
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
        ax1.plot(t, x, lw=0.8)
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel(labels[data_label][0]+labels[data_label][1])
        
        ax2 = fig.add_subplot(122)
        ax2.stem(freq, np.abs(X), markerfmt=' ', basefmt=' ')
        ax2.set_xlabel('Frequency [kHz]')
        ax2.set_ylabel('FFT Amplitude')
        x_max = sr/2*plot_x_fraction
        y_max = np.abs(np.max(X))
        ax2.set_xlim(0, None)
        ax2.set_ylim(0, None)
        
        if True:
            ax2_sub = ax2.inset_axes([0.4,0.45,0.55,0.45])
            ax2_sub.stem(freq, np.abs(X), markerfmt=' ', basefmt=' ')
            x_max = sr/2*plot_x_fraction
            y_max = np.max(np.abs(X))*plot_y_fraction
            ax2_sub.set_xlim(0, x_max)
            ax2_sub.set_ylim(0, y_max)
            # ax2_sub.set_xticks([0, 10, 20, 30, 40])
        
        # plt.show()
        plt.savefig(str(Path(data_path, f'{trackid}_KH_{data_label}_fft.png')))
        plt.close()
    
    
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
                x_max = sr/2*plot_x_fraction
                y_max = np.max(fft_sum)*plot_y_fraction
                ax1_sub.set_xlim(0, x_max)
                ax1_sub.set_ylim(0, None)
                # ax1_sub.set_xticks([0, 10, 20, 30, 40, 50])
            # plt.show()
            plt.savefig(str(Path(data_path, f'KH_{data_label}_fft_sum.png')))
        except ValueError:
            print('Series lengths do not match - cannot calulate FFT sum')
    

if __name__ == '__main__':
    main()