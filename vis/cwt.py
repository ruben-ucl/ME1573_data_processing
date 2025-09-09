import os, sys, functools, h5py, pywt, glob
import pandas as pd
import numpy as np
import ssqueezepy as ssq
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import pyplot as plt, ticker as mticker

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, printProgressBar, get_cwt_scales
folder = get_paths()['hdf5']

# group, time, series = ('AMPM', 'Time', 'Photodiode1Bits')
group, time, series = ('KH', 'time', 'max_depth')
running_mean_window = None

mode = 'preview' # 'preview' or 'save'
show_wavelet = True
debug = True

def plot_wavelet(wavelet):    
    [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(level=10)
    fig, ax = plt.subplots(1, 1, figsize=(3.15, 3.15), dpi = 300)
    ax.plot(x, psi)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-1, 1))
    if mode == 'save':
        output_folder = Path(folder, 'CWT', wavelet)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(Path(output_folder, f'{wavelet}.png'))
    else:
        plt.show()
        plt.close()

files = sorted(glob.glob(f'{folder}/*.hdf5'))
for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]
    print(trackid)
    
    with h5py.File(filepath, 'r') as file:
        # t = np.array(file[f'{group}/{time}'])[510:-510]
        # s = np.array(file[f'{group}/{series}'])[510:-510]
        t = np.array(file[f'{group}/{time}'])
        s = np.array(file[f'{group}/{series}'])
        if running_mean_window != None:
            s = np.convolve(s, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        xray_im = np.array(file['bs-f40'])[-1]
        
        # Frequency range calculation
        sampling_period = round(t[1]-t[0], 9)
        print(sampling_period)
        sampling_duration = round(t[-1]-t[0], 9)
        sampling_rate = round(1/sampling_period, 7)
        print(sampling_rate)
        
        nyquist_freq = sampling_rate / 2
        min_freq = 1 / sampling_duration  # Lowest resolvable frequency
        max_freq = nyquist_freq
        
        # wavelet = "cmor1.5-1.0"
        wavelet = "fbsp4-0.6-1.0" 
        # wavelet = 'cmor10.0-0.3'
        scales, vmax = get_cwt_scales(wavelet, 256)
        if debug: print('scales:\n', scales)
        
        if debug:
            print(f'min: {pywt.scale2frequency(wavelet, scales[-1])*sampling_rate} ',
                f'max: {pywt.scale2frequency(wavelet, scales[0])*sampling_rate}')
        
        if show_wavelet:
            plot_wavelet(wavelet)
            show_wavelet = False
        
        cwtmatr, freqs = pywt.cwt(s, scales, wavelet, sampling_period)
        cwtmatr = np.abs(cwtmatr[:-1, :-1])
        
        plt.rcParams.update({'font.size': 9})
        kw = {'height_ratios':[1, 1, 1], "width_ratios":[95, 5]}
        fig, ((ax1, ax1b), (ax2, ax2b), (ax3, ax3b)) = plt.subplots(3, 2,
            figsize = [6.3, 7],
            dpi = 300,
            gridspec_kw = kw)
        fig.suptitle(f'{trackid} - {series}')
        
        ax1.plot(t*1000, s, lw=0.75)
        ax1.set_xlim(t[0]*1000, t[-1]*1000)
        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Time [ms]')
        
        t_ax, f_ax = np.meshgrid(t*1000, freqs/1000)
        cwt_cmap = 'jet'
        pcm = ax2.pcolormesh(t_ax, f_ax, cwtmatr, cmap=cwt_cmap, vmax=vmax)
        ax2.set_yscale('log', base=2)
        ax2.set_ylim(1, 50)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Frequency [kHz]')
        ax2.set_yticks([1, 2, 4, 8, 16, 32, 50])
        fig.colorbar(pcm, cax=ax2b, label='Intensity')
        
        ax3.imshow(xray_im[150:450], cmap='gray')  # xray_im[150:450] for full frame 40 kHz
        scalebar = ScaleBar(4.3,
            "µm",
            length_fraction = 0.15,
            width_fraction = 0.02,
            frameon = False,
            color = 'w',
            location = 'lower right')
        ax3.add_artist(scalebar)
        
        for ax in [ax1b, ax3, ax3b]:
            ax.axis('off')
        plt.tight_layout()
        
        if mode == 'save':
            output_folder = Path(folder, 'CWT', wavelet, cwt_cmap)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(Path(output_folder, f'{trackid}_{series}_CWT_{wavelet}.png'))
        else:
            plt.show()
        plt.close()
        
        printProgressBar(i+1, len(files), prefix='Progress', suffix=trackid)


    