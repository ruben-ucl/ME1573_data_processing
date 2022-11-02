# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:14:27 2022

@author: Ruben

---------------------------------------------------

CHANGES:
    v0.1 - Initial version. Applies flat field correction and rotates images.
    v0.2 - Deactivated rotation feature, added background subtraction and 8-bit conversion stages.
    v0.3 - Stripped down to just flat field correction functionality to work with tiff_to_hdf5 v1.3, saves output as new hdf5 file
    v0.4 - Added logging
    v0.4.1 - Added logging of warnings to catch zero division RuntimeWarnings from numpy
    v0.5 - Added dynamic range adjustment before compresing to 8-bit
    v1.0 - Operates on entire datasets as numpy arrays instead of going frame by frame 
           Reads start and end frames from database to define trimming of videos
           Applies more aggressive histogram stretching before compression to 8-bit to maximise grey-level resolution
    
INTENDED CHANGES:
    - 
    
"""
__version__ = '0.5.1'

import h5py, os, glob, logging, warnings, functools
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import *
from skimage import exposure

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes
np.seterr(divide='warn')    # Log zero division errors

compare_histograms = True
flip_images = False

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
output_fpath = Path(filepath, 'ffc')

def create_log(folder):  # Create log file for storing error details
    init_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(folder):
            os.makedirs(folder)
    log_file_path = Path(folder, 'flat_field_correction_%s.log' % init_time)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s    %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    logging.captureWarnings(capture=True)  # Capture warnings with the logging module and write to the the same log
    warnings.formatwarning(message='%(asctime)s    %(message)s', category=Warning, filename=log_file_path, lineno=-1)
    print('Logging to: %s' % str(log_file_path))
    
def duplicate_folder(original_dir, output_dir):
    name = Path(original_dir).name
    output_path = Path(output_dir, name)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return output_path

def flat_field_correction(images, flats):
    raw = images[-100] / 4096
    # Calculate averaged flat field
    avg_flat = np.clip(np.mean(flats, axis=0), 1, None)   # Set 0 value pixels to 1 to avoid zero division error in ffc
    print('Performing flat field correction')
    images = np.divide(images, avg_flat)
    ffc = images[-100]
    print('Scaling pixel intensities')
    lp, up = np.percentile(images, (1, 99))
    images = exposure.rescale_intensity(images, in_range=(lp, up))
    if flip_images == True:
        print('Flipping images')
        images = np.flip(images, axis=(1, 2))
        ri_flip = images[-100]
    print('Converting to 8-bit')
    images = (images * 255).astype(np.uint8)
    if compare_histograms == True:
        compare_histograms({'raw': raw,
                            'avg_flat': avg_flat / 4096,
                            'ff_corr': ffc,
                            'ff_corr_stretched': images[-100] / 255
                            })  
    return images

def compare_histograms(images, save_to_csv=False, generate_plots=True):
    n_vals = 4096
    histograms = pd.DataFrame()
    histograms['value'] = [i for i in np.arange(0, 1, 1/n_vals)] # First column contains integers 0-255 for 8-bit greyscale values
    for key in images:
        hist, _ = np.histogram(images[key], bins = n_vals, range = [0, 1])
        histograms[key] = hist
    if save_to_csv == True:        
        histograms.to_csv(Path(filepath, f'{input_dset_name}_histograms.csv'), index=False)    
    if generate_plots == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi = 200, figsize = (9, 3))
        x = histograms['value']
        for col in histograms.columns:
            if col != 'value':
                ax1.plot(x, histograms[col], label = col)
        ax1.set_xlabel('normalised pixel value')
        ax1.set_ylabel('count')
        ax1.legend()
        ax2.imshow(images['ff_corr_stretched'], cmap='gray')
        fig.set_tight_layout(True)
        plt.show()

def main():
    create_log(output_fpath)
    logbook = get_logbook()
    files = glob.glob(str(Path(filepath, '*.hdf5')))
    for file in sorted(files):     # For each substrate, iterate through track folders
        try:
            fname = Path(file).name
            trackid = fname[:5] + '0' + fname[-6]
            logging.info(fname)
            output_file = Path(output_fpath, trackid)
            if os.path.exists(output_file):
                print(fname + 'file already exists in output folder')
                logging.info('File of the same name exists in output directory - skipping file')
                raise FileExistsError()
            print('\nReading %s' % fname)
            with h5py.File(file, 'r') as fi:
                _, end_frame = get_start_end_frames(trackid, logbook, margin=50) # Define end frame as 50 frames after laser off
                flats = np.array(fi['flats'])[:200, :, :]   # Take first 200 flat field images
                images = np.array(fi['raw'])[:end_frame, :, :]  # Discard frames after end_frame
                # elem_size = fi['xray_images'].attrs.get('element_size_um')   # Get element size to pass on to new file
            print('Running flat field correction')
            ff_corrected_images = flat_field_correction(images, flats)
            with h5py.File(output_file, 'x') as fo:
                fo['ff_corrected'] = ff_corrected_images
                # fo['ff_corrected'].attrs.create('element_size_um', 4.3)
            print('Output file saved')
            logging.info('Complete')
            # input('Done. Press any key to continue to next file.')
        except FileExistsError as e:
            print('Skipping file')
            logging.info(str(e))

if __name__ == "__main__":
	main()	
