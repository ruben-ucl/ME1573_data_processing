# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:32:09 2022

@author: Ruben

---------------------------------------------------

CHANGES:
    v0.1 - Initial version. Makes a copy of the original folder structure and creates hdf5 files for each individual track containing datasets for xray images and flat field images.
    v0.2 - Added file skipping if file already exists so that process can be interrupted and restarted easily
    v0.3 - Added pixel value conversion to 8-bit greyscale to save space and speed up subsequent processing
    v0.4 - Write element size into dataset attributes so that it can be read by ImageJ automatically by ImageJ
         - Trim first 800 frames and final frames (number calculated from frame rate and scan speed with track lenght of 4 mm)
    v1.0 - Fixed bugs, testing on full Al dataset
    v1.1 - Working on workstation MXIF27, added some extra print statements to make it easier to keep track of progress
    
INTENDED CHANGES:
    - Switch to chunked storage to allow lossless compression
    - Add more error handling
    
"""

__author__ = 'Rubén Lambert-Garcia'
__version__ = '1.1'

username = 'lbn38569' # Set to PC username so that correct Dropbox directory can be located

import h5py
import numpy as np
import pandas as pd
import os, glob, pathlib
from skimage.io import imread

# File structure: '\Beamtime root folder\Material\Substrate\Track\[Datasets]',
# E.g '\ESRF ME1573\AlSi10Mg\0101\1\flats'

# Dict containing starting frame number for each track position on substrate
start_frame = {'01': 850,
               '02': 890,
               '03': 930,
               '04': 970,
               '05': 1010,
               '06': 1050
               }

# Query user to confirm working directory. Working directory should be the root folder of the beamtime data, e.g. '\ESRF ME1573\'. 
def wd_query():
    try:
        wd_query_response = input('Is the current working directory the root folder for the original beamtime data? (y/n)\n')
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

def duplicate_folder_tree(data_root, output_root, logbook):
    for material_folder in sorted(glob.glob('%s/*/' % data_root)):     # Iterate through material folders (top level of folder structure in data root folder)
        material_output_folder = duplicate_folder(material_folder, output_root)
        print('\nMaterial: %s' % str(pathlib.PurePath(material_folder).name))
        
        for substrate_folder in sorted(glob.glob('%s/*/' % material_folder)):      # For each material, iterate through contained substrate folders
            substrate_number = pathlib.PurePath(substrate_folder).name
            print('\nSubstrate: %s' % str(pathlib.PurePath(substrate_folder).name))
                
            for track_folder in sorted(glob.glob('%s/*/' % substrate_folder)):     # For each substrate, iterate through track folders
                track_number = pathlib.PurePath(track_folder).name
                print('Track: %s' % str(track_number))
                try:
                    make_hdf5(material_output_folder, track_folder, substrate_number, track_number, logbook)
                except ValueError as e:
                    print('Error: could not create file - %s - Check logbook for bad value in row %s_%s' % (str(e), str(substrate_number), str(track_number)))
                    
def duplicate_folder(original_dir, output_dir):
    name = pathlib.PurePath(original_dir).name
    output_path = pathlib.PurePath(output_dir, name)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return output_path

def get_logbook():
    logbook_path = pathlib.PurePath('C:/Users/%s/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19', '220121-Logbook_EndofBeamtime.xlsx' % username)
    logbook = pd.read_excel(logbook_path,
                            usecols='C,D,P,AJ',
                            converters={'Substrate No.': str, 'Sample position': str}
                            )
    return logbook

def get_logbook_data(logbook, trackid):  # Get scan speed and framerate from logbook
    substrate_no = trackid[1:4]
    track_no = trackid[-1]
    track_row = logbook.loc[(logbook['Substrate No.'] == substrate_no) & (logbook['Sample position'] == track_no)]
    scan_speed = int(track_row['scan speed [mm/s]'])
    framerate = int(track_row['Frame rate (kHz)'] * 1000)
    return framerate, scan_speed
    
def make_hdf5(substrate_output_folder, track_folder, substrate_number, track_number, logbook):
    trackid = '%s_%s' % (substrate_number, track_number)
    input_subfolders = sorted(glob.glob('%s/*/' % track_folder))   # Sorted list of subfolders containing tiff stacks. When sorted, image folder is at index zero and flats are at index 1 due to the file naming convention.
    output_filepath = pathlib.PurePath(substrate_output_folder, '%s.hdf5' % trackid)
    framerate, scan_speed = get_logbook_data(logbook, trackid)
    n_frames = round(framerate * 4 / scan_speed) # based on track length of 4 mm
    n_frames += 100 # Margin to allow for variation in track start time due to different scan speeds
    first_frame = start_frame[str(track_number)]
    try:
        with h5py.File(output_filepath, 'x') as output_file:
            create_dataset(output_file, 'xray_images', input_subfolders, n_frames, first_frame, index = 0, element_size = [0, 4.3, 4.3])
            create_dataset(output_file, 'xray_flats', input_subfolders, n_frames, first_frame, index = 1, element_size = [0, 4.3, 4.3])
    except OSError as e:
        print('Error: %s - Skipping file' % str(e))
        
def create_dataset(file, dset_name, dset_folders, n_frames, first_frame, index, element_size):
    dset_source = dset_folders[index]
    dset_images = sorted(glob.glob('%s/*.tif' % dset_source))
    dset_im0 = imread(dset_images[0])
    dset_shape = tuple([n_frames] + list(dset_im0.shape))
    dset = file.require_dataset(dset_name, shape=dset_shape, dtype=np.uint8)
    dset.attrs.create('element_size_um', element_size)
    print('%s - Creating %s' % (file, dset))
    for i, tiff_file in enumerate(dset_images[first_frame:first_frame+n_frames]):
        im = np.flip(imread(tiff_file), axis=(0, 1)) # Read TIFF image and rotate 180 degrees (by flipping along both axes)
        im_8bit = np.round(im / 4095 * 255).astype(np.uint8) # Convert to 8-bit greyscale 
        dset[i, :, :] = im_8bit
        
def main():
    data_root = wd_query()
    output_root = output_query(data_root)
    print('Fetching logbook data')
    logbook = get_logbook()
    duplicate_folder_tree(data_root, output_root, logbook)

if __name__ == "__main__":
	main()