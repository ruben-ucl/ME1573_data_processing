import pandas as pd
import numpy as np
from pathlib import Path
from skimage import filters, exposure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, functools

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    # print(f'Reading data from {filepath}\n')

# def get_logbook(logbook_path = Path('J:\Logbook_Al_ID19_combined_RLG.xlsx')):
def get_logbook(logbook_path = Path('E:\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\Logbook_Al_ID19_combined_RLG.xlsx')):
    print(f'Trying to read logbook: {logbook_path.name}')
    try:
        logbook = pd.read_excel(logbook_path,
                                sheet_name='Logbook',
                                usecols='C, D, E, F, I, J, M, O, P, Q, R, S, T, U, AM, AP, AS, AT, AU, AV, AW, AX, AY, AZ, BA',
                                converters={'Substrate No.': str, 'Sample position': str}
                                )
        # logging.info('Logbook data aquired from %s' % logbook_path)
        print('Logbook read successfully')
    
        return logbook
    
    except Exception as e:
        print('Error: Failed to read logbook')
        print(e)
        # logging.info('Failed to read logbook - unable to continue')
        # logging.debug(str(e))
        raise
        
def get_logbook_data(logbook, trackid, layer_n=1):  # Get scan speed and framerate from logbookprint('Reading scan speed and framerate from logbook')
    substrate_n = trackid[1:4]
    track_n = trackid[-1]
    track_row = logbook.loc[(logbook['Substrate No.'] == substrate_n) &
                            (logbook['Sample position'] == track_n) &
                            (logbook['Layer'] == layer_n)
                            ]
    # print(track_row)
    track_data = {}
    track_data['peak_power'] = int(track_row['Power [W]'])
    track_data['avg_power'] = int(track_row['Avg. power [W]'])
    track_data['pt_dist'] = int(track_row['Point distance [um]'])
    track_data['exp_time'] = int(track_row['Exposure time [us]'])
    track_data['pt_jump_delay'] = int(track_row['Point jump delay [us]'])
    track_data['scan_speed'] = int(track_row['Scan speed [mm/s]'])
    track_data['LED'] = int(track_row['LED [J/m]'])
    track_data['framerate'] = int(track_row['Frame rate (kHz)'] * 1000)
    track_data['laser_onset_frame'] = int(track_row['Laser onset frame #'])
    track_data['keyhole_regime'] = track_row['Melting regime'].values[0]
    
    return track_data
    
    
def get_start_end_frames(trackid, logbook, margin=50, start_frame_offset=0):
    track_data = get_logbook_data(logbook, trackid)
    framerate = track_data['framerate']
    scan_speed = track_data['scan_speed']
    start_frame = track_data['laser_onset_frame']
    
    n_frames = round(framerate * 4 / scan_speed) # based on track length of 4 mm
    f1 = start_frame - margin - start_frame_offset
    f2 = start_frame + n_frames + margin - start_frame_offset 
    
    return f1, f2
    
def get_substrate_mask(trackid, shape, substrate_surface_measurements_fpath):   # Generates a 2d mask of the substrate
    substrate_mask = np.zeros(shape, dtype=bool)
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    print(f'Substrate edge equation retrieved: y = {round(m, 3)}x + {round(c)}')
    n_rows, n_cols = substrate_mask.shape
    print('Calculating mask dimensions')
    for x in range(n_cols):
        surface_height = int(m * x + c)
        substrate_mask[surface_height:, x] = True
    
    return substrate_mask

def get_substrate_surface_coords(shape, substrate_surface_measurements_fpath, trackid):
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    xx = range(shape[2])
    yy = [round(m * x + c) for x in xx]
    
    return xx, yy
    
def median_filt(dset, kernel):
    if dset.dtype == np.uint8:
        median_filter = filters.rank.median_filt # faster method for 8-bit integers
    else:
        median_filter = filters.median
    
    tic = time.perf_counter()
    try:
        if dset.ndim == 3:
            output_dset = np.zeros_like(dset)
            for i, frame in enumerate(dset):
                print(f'Median filtering frame {i}', end='\r')
                output_dset[i] = median_filter(frame, kernel)
        elif dset.ndim == 2:
            output_dset = median_filter(dset, kernel)
        else:
            raise UnexpectedShape
        
        toc = time.perf_counter()
        print(f'Median filter duration: {toc-tic:0.4f} seconds')
        
        return output_dset
    except UnexpectedShape:
        print(f'Expected dataset of 2 or 3 dimensions, but received dataset with {dset.ndim} dimension(s)')

def view_histogram(a, title=None, show_std=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    if title != None:
        fig.suptitle(title)
    if a.ndim == 3:
        # Create panel of 4 frames from stack to display
        l = len(a)
        r1 = np.concatenate((a[0], a[l//3]), axis=1)
        r2 = np.concatenate((a[2*l//3], a[-1]), axis=1)
        im = np.concatenate((r1, r2))
    else:
        im = a
    ax1.imshow(im, cmap='gray')
    im_aspect = im.shape[0] / im.shape[1]
    ax2.set_box_aspect(im_aspect)
    ax2.hist(a.ravel(), bins=255, density=True, zorder=0)
    ax2.annotate(f'max: {np.max(a)}\nmin: {np.min(a)}\ndtype: {a.dtype}', xy=(0.01, 0.8), xycoords='axes fraction')
    
    if show_std:
        a_mean = np.mean(a)
        a_sigma = np.std(a)
        a_std_intervals = [a_mean + i * a_sigma for i in range(-3, 4)]
        marker_h = ax2.get_ylim()[1] / 30
        ax2.scatter(a_std_intervals, [marker_h for i in range(7)], c='k', marker='v')
        ax2.scatter((np.min(a), np.max(a)), (marker_h, marker_h), c='r', marker='v')
        txt_h = ax2.get_ylim()[1] / 15
        ax2.annotate('min', (np.min(a), txt_h), ha='center', color='r')
        ax2.annotate('max', (np.max(a), txt_h), ha='center', color='r')
        sigma_text = ['-3\u03C3', '-2\u03C3', '-\u03C3', '\u00B5', '\u03C3', '2\u03C3', '3\u03C3']
        for i, x in enumerate(a_std_intervals):
            ax2.annotate(sigma_text[i], (x, txt_h), ha='center')
    
    plt.show()
    
def compare_histograms(im_dict, fig_title=None):
    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(len(im_dict), 2, figsize=(5, len(im_dict)*2), dpi=300, tight_layout=True, sharex='col')
    # fig, axs = plt.subplots(len(im_dict), 2, figsize=(4, 4), dpi=300, tight_layout=True, sharex='col')
    if fig_title != None:
        fig.suptitle(fig_title)
    for i, key in enumerate(im_dict):
        im = im_dict[key]
        axs[i, 0].imshow(im, cmap='gray')
        axs[i, 0].title.set_text(key)
        axs[i, 0].axis('off')
        im_aspect = im.shape[0] / im.shape[1]
        axs[i, 1].set_box_aspect(im_aspect)
        axs[i, 1].set_xlim(0, 255)
        axs[i, 1].hist(im.ravel(), bins=255, density=True)
        axs[i, 1].annotate(f'max: {np.max(im)}\nmin: {np.min(im)}\ndtype: {im.dtype}', xy=(0.05, 0.6), xycoords='axes fraction')
    axs[len(im_dict)-1, 1].set_xlabel('greyscale value [0, 255]')
    
    plt.show()    

def hist_eq(dset):
    tic = time.perf_counter()
    # output_dset = np.zeros_like(dset, dtype=np.uint8)
    # for i, im in enumerate(dset):
        # output_dset[i] = (exposure.equalize_hist(im) * 255).astype(np.uint8)
    output_dset = (exposure.equalize_hist(dset) * 255).astype(np.uint8)
    toc = time.perf_counter()
    print(f'Histogram equalisation duration: {toc-tic:0.4f} seconds')
    
    return output_dset
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
class UnexpectedShape(Exception):
    "Raised when input data is not the expected shape"
    pass