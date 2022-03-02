# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:14:27 2022

@author: Ruben

---------------------------------------------------

CHANGES:
    v0.1 - Initial version. Applies flat field correction and rotates images.
    v0.2 - Deactivated rotation feature, added background subtraction and 8-bit conversion stages.
    
INTENDED CHANGES:
    - Make work with tiff_to_hdf5 v1.0+ which already takes care of flipping the image, trimming and 8-bit conversion
    
"""
__version__ = '0.2'

import h5py, os, glob, pathlib, cv2
from skimage.io import imshow
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def wd_query():
    try:
        wd_query_response = input('Is the current working directory the root folder for the input data? (y/n)\n')
        if wd_query_response == 'y':
            data_root = os.getcwd()
        elif wd_query_response == 'n':
            data_root = input('Please input the global path of the beamtime data root folder.\n')
        else:
            raise ValueError('Invalid input, try again.')
    except ValueError:
        wd_query()
    return data_root

def output_query(data_root):
    output_query_response = input('Now enter the name of the output root folder.\n')
    output_root = pathlib.PurePath(data_root, '..', output_query_response)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    return output_root

def overlays_query():
    r1 = input('Would you like to perform background subtraction using the first frame of each stack? (y/n)\n')
    r2 = input('Would you like to convert images to 8-bit greyscale? (y/n)\n')
    return [True if r=='y' else None for r in [r1, r2]]
    
def duplicate_folder(original_dir, output_dir):
    name = pathlib.PurePath(original_dir).name
    output_path = pathlib.PurePath(output_dir, name)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return output_path

def flat_field_correction(images, flats):
    # Calculate averaged flat field
    avg_flat = np.zeros(flats[0].shape)       # Initialise array for storing averaged flat field
    for im in flats:
        avg_flat = np.add(avg_flat, im)
    avg_flat = avg_flat / len(flats)
    # Subtract the averaged flat field from each image
    ff_corrected_images = np.zeros(images.shape)
    for i, im in enumerate(images):
        im_corr = np.flip((im / avg_flat) * avg_flat.mean(), axis=(0, 1))
        if bg_sub_query_response == True:
            if i == 0:
                im_corr_0 = im_corr
            else:
                im_corr = im_corr / im_corr_0
        ff_corrected_images[i, :, :] = im_corr
    if convert_query_response == True:
        print('Performing 8-bit conversion')
        ff_corrected_images = (np.clip(ff_corrected_images / 4095 * 255, 0, 255)).astype('uint8')
        print(ff_corrected_images.dtype)
        print(ff_corrected_images.shape)
        print(ff_corrected_images)
    return ff_corrected_images

def rotation_correction(image, angle):
    image_centre = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_centre[::-1], angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

data_root = wd_query()
output_root = output_query(data_root)
bg_sub_query_response, convert_query_response = overlays_query()

for material_folder in glob.glob('%s\*\\' % data_root):     # Iterate through material folders (top level of folder structure in data root folder)
    material_output_folder = duplicate_folder(material_folder, output_root)
    print('Writing to %s' % material_output_folder)
            
    for file in glob.glob('%s\*.hdf5' % material_folder):     # For each substrate, iterate through track folders
        try:
            trackid = pathlib.PurePath(file).name
            output_file = pathlib.PurePath(material_output_folder, trackid)
            if os.path.exists(output_file):
                print(trackid + 'file already exists in output folder')
                raise FileExistsError()
            print('Reading %s' % trackid)
            with h5py.File(file, 'r') as f:
                flats = np.array(f['xray_flats'])
                images = np.array(f['xray_images'])
            print('Processing %s' % trackid)
            ff_corrected_images = flat_field_correction(images, flats)
            print('Converting %s to 8-bit' % output_file)
            with h5py.File(output_file, 'x') as output_file:
                output_file['ff_corrected_images'] = ff_corrected_images
            print('Output file saved')
            # input('Done. Press any key to continue to next file.')
        except FileExistsError as e:
            print('%s - Skipping file' % str(e))

	
