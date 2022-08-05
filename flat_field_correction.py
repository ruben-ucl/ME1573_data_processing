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
    
INTENDED CHANGES:
    - Solve divide by zero errors - only affecting first file?
    - Finish implementing dynamic range adjustment
    
"""
__version__ = '0.4.1'

import h5py, os, glob, pathlib, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt

np.seterr(divide='warn')

# Read data folder path from .txt file
def get_filepath():
    with open('data_path.txt', encoding='utf8') as f:
        filepath = fr'{f.read()}'
        print(f'Reading from {filepath}\n')
        return filepath

def create_log(folder):  # Create log file for storing error details
    init_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(folder):
            os.makedirs(folder)
    log_file_path = pathlib.PurePath(folder, 'flat_field_correction_%s.log' % init_time)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s    %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    logging.captureWarnings(capture=True)  # Capture warnings with the logging module and write to the the same log
    warnings.formatwarning(message='%(asctime)s    %(message)s', category=Warning, filename=log_file_path, lineno=-1)
    print('\nLogging to: %s' % str(log_file_path))

# def wd_query():
    # try:
        # wd_query_response = input('\nIs the current working directory the root folder for the input data? (y/n)\n')
        # if wd_query_response == 'y':
            # data_root = os.getcwd()
        # elif wd_query_response == 'n':
            # data_root = input('\nPlease input the global path of the input data root folder.\n')
        # else:
            # raise ValueError('Invalid input, try again.')
    # except ValueError:
        # wd_query()
    # return data_root

def output_query(data_root):
    output_query_response = input('\nNow enter the name of the output root folder.\n')
    output_root = pathlib.PurePath(data_root, '..', output_query_response)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    return output_root
    
def duplicate_folder(original_dir, output_dir):
    name = pathlib.PurePath(original_dir).name
    output_path = pathlib.PurePath(output_dir, name)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return output_path

def flat_field_correction(images, flats):
    # Calculate averaged flat field
    avg_flat = np.clip(np.mean(flats, axis=0), 1, None)   # Set 0 value pixels to 1 to avoid zero division error in ffc
    avg_flat_mean = avg_flat.mean()
    # Subtract the averaged flat field from each image, increase contrast and convert to 8bit
    ff_corrected_images = np.zeros(images.shape, dtype=np.uint8)
    for i, im in enumerate(images):
        im_ff_corr_norm = np.divide(im, avg_flat) * avg_flat_mean / 4096
        lim_a = 0.25    # Calibrate these limits from un-stretched flat field corrected image histograms
        lim_b = 0.7
        # im_ff_corr = (im_ff_corr_norm - lim_a) / (lim_a - lim_b) * 255   # Stretch histogram to limits defined above to increase dynamic range
        im_ff_corr_norm_stretch = (im_ff_corr_norm - lim_a) / (lim_b - lim_a)
        ff_corrected_images[i, :, :] = np.round(np.clip(im_ff_corr_norm_stretch * 255, 0, 255)).astype(np.uint8)
        # compare_histograms({'raw': im/4096,
                            # 'avg_flat': avg_flat/4096,
                            # 'ff_corr': im_ff_corr_norm,
                            # 'ff_corr_stretched': im_ff_corr_norm_stretch
                            # })
    return ff_corrected_images

# def rotation_correction(image, angle):
    # image_centre = tuple(np.array(image.shape) / 2)
    # rot_mat = cv2.getRotationMatrix2D(image_centre[::-1], angle, 1.0)
    # rotated_image = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
    # return rotated_image

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
        fig, ax = plt.subplots(figsize = [9, 6])
        x = histograms['value']
        for col in histograms.columns:
            if col != 'value':
                ax.plot(x, histograms[col], label = col)
        ax.set_xlabel('pixel value')
        ax.set_ylabel('count')
        ax.legend()
        fig.set_tight_layout(True)
        plt.show()

def main():
    data_path = get_filepath()
    output_path = output_query(data_path)
    create_log(output_path)
    for file in glob.glob(f'{data_path}/*.hdf5'):     # For each substrate, iterate through track folders
        try:
            trackid = pathlib.PurePath(file).name
            logging.info(trackid)
            output_file = pathlib.PurePath(output_path, trackid)
            if os.path.exists(output_file):
                print(trackid + 'file already exists in output folder')
                logging.info('File of the same name exists in output directory - skipping file')
                raise FileExistsError()
            print('\nReading %s' % trackid)
            with h5py.File(file, 'r') as fi:
                flats = np.array(fi['flats'])
                images = np.array(fi['raw'])
                # elem_size = fi['xray_images'].attrs.get('element_size_um')   # Get element size to pass on to new file
            print('Processing data')
            ff_corrected_images = flat_field_correction(images, flats)
            with h5py.File(output_file, 'x') as fo:
                fo['ff_corrected'] = ff_corrected_images
                fo['ff_corrected'].attrs.create('element_size_um', 4.3)
            print('Output file saved')
            logging.info('Complete')
            # input('Done. Press any key to continue to next file.')
        except FileExistsError as e:
            print('Skipping file')
            logging.info(str(e))

if __name__ == "__main__":
	main()	
