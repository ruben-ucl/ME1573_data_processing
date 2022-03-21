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
    
INTENDED CHANGES:
    - Solve divide by zero errors - only affecting first file?
    - 
    
"""
__version__ = '0.4.1'

import h5py, os, glob, pathlib, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime as dt

np.seterr(divide='warn')


def create_log(folder):  # Create log file for storing error details
    init_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(folder):
            os.makedirs(folder)
    log_file_path = pathlib.PurePath(folder, 'flat_field_correction_%s.log' % init_time)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s    %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    logging.captureWarnings(capture=True)  # Capture warnings with the logging module and write to the the same log
    warnings.formatwarning(message='%(asctime)s    %(message)s', category=Warning, filename=log_file_path, lineno=-1)
    print('\nLogging to: %s' % str(log_file_path))

def wd_query():
    try:
        wd_query_response = input('\nIs the current working directory the root folder for the input data? (y/n)\n')
        if wd_query_response == 'y':
            data_root = os.getcwd()
        elif wd_query_response == 'n':
            data_root = input('\nPlease input the global path of the input data root folder.\n')
        else:
            raise ValueError('Invalid input, try again.')
    except ValueError:
        wd_query()
    return data_root

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
    avg_flat = np.zeros(flats[0].shape, dtype=np.uint32)       # Initialise array for storing averaged flat field
    for im in flats:
        avg_flat = np.add(avg_flat, im)
    avg_flat = (avg_flat / len(flats)).astype(np.uint8)     # Is rounding down here causing divide by zero errors on line 73?
    # Subtract the averaged flat field from each image
    ff_corrected_images = np.zeros(images.shape, dtype=np.uint8)
    avg_flat_mean = avg_flat.mean()
    for i, im in enumerate(images):
        im_ff_corr = (im / avg_flat) * avg_flat_mean
        ff_corrected_images[i, :, :] = np.round(np.clip(im_ff_corr, 0, 255))
    return ff_corrected_images

# def rotation_correction(image, angle):
    # image_centre = tuple(np.array(image.shape) / 2)
    # rot_mat = cv2.getRotationMatrix2D(image_centre[::-1], angle, 1.0)
    # rotated_image = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
    # return rotated_image

data_root = wd_query()
output_root = output_query(data_root)
create_log(output_root)

for material_folder in glob.glob('%s\*\\' % data_root):     # Iterate through material folders (top level of folder structure in data root folder)
    material_output_folder = duplicate_folder(material_folder, output_root)
    print('\nWriting to %s' % material_output_folder)
    logging.info('Writing to %s' % material_output_folder)
            
    for file in glob.glob('%s\*.hdf5' % material_folder):     # For each substrate, iterate through track folders
        try:
            trackid = pathlib.PurePath(file).name
            logging.info(trackid)
            output_file = pathlib.PurePath(material_output_folder, trackid)
            if os.path.exists(output_file):
                print(trackid + 'file already exists in output folder')
                logging.info('File of the same name exists in output directory - skipping file')
                raise FileExistsError()
            print('\nReading %s' % trackid)
            with h5py.File(file, 'r') as f:
                flats = np.array(f['xray_flats'])
                images = np.array(f['xray_images'])
                elem_size = f['xray_images'].attrs.get('element_size_um')   # Get element size to pass on to new file
            print('Processing data')
            ff_corrected_images = flat_field_correction(images, flats)
            with h5py.File(output_file, 'x') as output_file:
                output_file['ff_corrected'] = ff_corrected_images
                output_file['ff_corrected'].attrs.create('element_size_um', elem_size)
            print('Output file saved')
            logging.info('Complete')
            # input('Done. Press any key to continue to next file.')
        except FileExistsError as e:
            print('Skipping file')
            logging.info(str(e))

	
