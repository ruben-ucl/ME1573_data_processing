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
    v1.0 - Operates on entire datasets as numpy arrays where possible instead of going frame by frame 
           Reads start and end frames from database to define trimming of videos
           Applies histogram stretching between 0.1 and 99.9 percentiles before compression to 8-bit
           Option to generate figure showing histogram at each stage of processing
    
INTENDED CHANGES:
    - 
    
"""
__version__ = '1.0'

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

show_histograms = False
flip_images = False
trim = True # Trim video to laser scan accoding to start_frame number in logbook
margin = 50 # Number of frames kept before and after laser scan

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

def flat_field_correction(images, flats):
    raw = images[-100] / 4096
    # Calculate averaged flat field
    avg_flat = np.clip(np.median(flats, axis=0), 1, None)   # Set 0 value pixels to 1 to avoid zero division error in ffc
    images = np.divide(images, avg_flat)
    ffc = images[-100]
    print('Scaling pixel intensities')
    
    a, b = np.percentile(images, (0.1, 99.9))
    images = np.clip((images - a) / (b - a), 0, 1)
    images = (images * 255).astype(np.uint8)
    
    if flip_images:
        print('Flipping images')
        images = np.flip(images, axis=(1, 2))
        ri_flip = images[-100]

    if show_histograms:
        compare_histograms({'raw': raw,
                            'avg_flat': avg_flat / 4096,
                            'ff_corr': ffc,
                            'ff_corr_stretched': images[-100] / 255
                            })  
    return images

def main():
    create_log(output_fpath)
    logbook = get_logbook()
    files = glob.glob(str(Path(filepath, '*.hdf5')))
    for file in sorted(files):     # For each substrate, iterate through track folders
        try:
            fname = Path(file).name
            trackid = fname[:5] + '0' + fname[-6]
            logging.info(fname)
            output_file = Path(output_fpath, f'{trackid}.hdf5')
            if os.path.exists(output_file):
                print(f'{trackid}.hdf5 already exists in output folder')
                logging.info('File of the same name exists in output directory - skipping file')
                raise FileExistsError()
            print('\nReading %s' % fname)
            with h5py.File(file, 'r') as fi:
                if trim == True:
                    start_frame, end_frame = get_start_end_frames(trackid, logbook, margin=margin) # Define end frame as 50 frames after laser off
                    end_frame = start_frame + 870 # For 500 kHz, 800 mm/s scan
                    print(f'Trimming to frames {start_frame}-{end_frame}')
                else:
                    start_frame = 0
                    end_frame = -1
                flats = np.array(fi['flats'])[:200]   # Take first 200 flat field images
                images = np.array(fi['raw'])[start_frame:end_frame]  # Discard frames after end_frame
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
