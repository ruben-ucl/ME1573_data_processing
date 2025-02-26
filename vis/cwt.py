import os, sys, functools, h5py, pywt, glob
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import pyplot as plt, ticker as mticker

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, printProgressBar
folder = get_paths()['hdf5']

group, time, series, colour = ('AMPM', 'Time', 'Photodiode1Bits', 'r')
running_mean_window = 4

mode = 'preview' # 'preview' or 'save'

def plot_wavelet():    
    [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(level=10)
    plt.plot(x, psi)
    plt.show()

files = sorted(glob.glob(f'{folder}/0110_01.hdf5'))
for i, filepath in enumerate(files):
    trackid = Path(filepath).name[:7]
    
    with h5py.File(filepath, 'r') as file:
        t = np.array(file[f'{group}/{time}'])[450:-450]
        s = np.array(file[f'{group}/{series}'])[450:-450]
        if running_mean_window != None:
            s = np.convolve(s, np.ones(running_mean_window)/running_mean_window, mode='valid')
            t = t[:-running_mean_window+1]
        xray_im = np.array(file['bs-f40'])[-1]
        
        scales = np.logspace(2, 7, num=256, base=2, endpoint=True)
        wavelet = "cmor1.5-1.0"
        sampling_period = round(t[1]-t[0], 9)
        sampling_rate = round(1/sampling_period, 7)
        f = pywt.scale2frequency(wavelet, 1)/sampling_period
        
        plot_wavelet()
        
        cwtmatr, freqs = pywt.cwt(s, scales, wavelet, sampling_period=sampling_period)
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
        pcm = ax2.pcolormesh(t_ax, f_ax, cwtmatr, cmap='jet')
        ax2.set_yscale('log', base=2)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Frequency [kHz]')
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
            if not os.path.exists(f'{folder}/CWT_jet'):
                os.makedirs(f'{folder}/CWT_jet')
            plt.savefig(f'{folder}/CWT_jet/{trackid}_{series}_CWT.png')
        else:
            plt.show()
        plt.close()
        
        printProgressBar(i+1, len(files), prefix='Progress', suffix=trackid)


    