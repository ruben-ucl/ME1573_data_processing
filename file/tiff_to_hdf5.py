# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:32:09 2022

@author: Ruben

---------------------------------------------------

CHANGES:
    v0.1 - Initial version. Makes a copy of the original folder structure and creates hdf5 files for each individual
           track containing datasets for xray images and flat field images.
    v0.2 - Added file skipping if file already exists so that process can be interrupted and restarted easily
    v0.3 - Added pixel value conversion to 8-bit greyscale to save space and speed up subsequent processing
    v0.4 - Write element size into dataset attributes so that it can be read by ImageJ automatically by ImageJ
         - Trim empty frames from the start and end of each dataset (start time defined in dictionary, number of
           frames calculated from frame rate and scan speed with track lenght of 4 mm)
    v1.0 - Fixed bugs, testing on full Al dataset
    v1.1 - Working on workstation MXIF27, added some extra print statements to make it easier to keep track of progress
    v1.2 - Added logging for full error messages to clean up console output and facillitate debugging
    v1.3 - Now copies all flat field frames
    v1.3.1 - Updated all start frames with intention that all videos start 50 frames before laser onset
             Only saves first 100 flat field frames to reduce file size
    v1.3.2 - Added ability to output 16-bit int images
    
INTENDED CHANGES:
    - Switch to chunked storage to allow lossless compression
    - Add more error handling
    
"""

__author__ = 'Rubén Lambert-Garcia'
__version__ = '1.3.2'

username = 'lbn38569' # Set to PC username so that correct Dropbox directory can be located

import os, glob, pathlib, h5py, logging
import numpy as np
import pandas as pd
from skimage.io import imread
from datetime import datetime as dt

# Frame number at which laser interaction begins for each track position on substrate
start_frames = {'01': 867,
                '02': 910,
                '03': 851,
                '04': 1006,
                '05': 947,
                '06': 1102
                }

output_dtype = '16_bit' # Set to '8_bit' or '16_bit'

# File structure: '\Beamtime root folder\Material\Substrate\Track\[Datasets]',
# E.g '\ESRF ME1573\AlSi10Mg\0101\01\flats'

# Query user to confirm working directory. Working directory should be the root folder of the beamtime data, e.g. '\ESRF ME1573\'. 
def wd_query():
    try:
        wd_query_response = input('\nIs the current working directory the root folder for the original beamtime data? (y/n)\n')
        if wd_query_response == 'y':
            data_root = os.getcwd()
        elif wd_query_response == 'n':
            data_root = input('\nPlease input the global path of the beamtime data root folder.\n')
        else:
            raise ValueError('Invalid input, try again.')
    except ValueError:
        wd_query()
    logging.debug('Working directory: %s' % str(data_root))
    return data_root

def output_query(data_root):
    output_query_response = input('\nNow enter the name of the output root folder.\n')
    output_root = pathlib.PurePath(data_root, '..', output_query_response)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    logging.debug('Destination directory: %s' % str(output_root))
    return output_root
    
def create_log(folder):  # Create log file for storing error details
    init_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(folder):
            os.makedirs(folder)
    log_file_path = pathlib.PurePath(folder, 'tiff_to_hdf5_%s.log' % init_time)
    logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG, force=True)
    print('Logging to: %s' % str(log_file_path))

def timestamp():
    return dt.today().strftime('%Y/%m/%d %H:%M:%S')

def duplicate_folder_tree(data_root, output_root, logbook):
    for material_folder in sorted(glob.glob('%s/*/' % data_root)):     # Iterate through material folders (top level of folder structure in data root folder)
        material_output_folder = duplicate_folder(material_folder, output_root)
        print('\nMaterial: %s' % str(pathlib.PurePath(material_folder).name))
        
        for substrate_folder in sorted(glob.glob('%s/*/' % material_folder)):      # For each material, iterate through contained substrate folders
            substrate_number = pathlib.PurePath(substrate_folder).name
            print('\nSubstrate: %s' % str(pathlib.PurePath(substrate_folder).name))
                
            for track_folder in sorted(glob.glob('%s/*/' % substrate_folder)):     # For each substrate, iterate through track folders
                track_number = pathlib.PurePath(track_folder).name
                trackid = '%s_%s' % (str(substrate_number), str(track_number))
                print('Track: %s' % trackid[-2:])
                logging.info('\n%s    Track ID: %s' % (timestamp(), trackid))
                try:
                    make_hdf5(material_output_folder, track_folder, trackid, logbook)
                except (ValueError, TypeError) as e:
                    print('Error: Possible bad value in logbook: could not create output for %s - skipping file' % trackid)
                    logging.info('Failed: Check logbook for bad values')
                    logging.debug(str(e))
                
def duplicate_folder(original_dir, output_dir):
    name = pathlib.PurePath(original_dir).name
    output_path = pathlib.PurePath(output_dir, name)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return output_path

def get_logbook():
    try:
        logbook_path = pathlib.PurePath('C:/Users/%s/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19' % username, '220121-Logbook_EndofBeamtime.xlsx')
        logbook = pd.read_excel(logbook_path,
                                usecols='C,D,P,AJ',
                                converters={'Substrate No.': str, 'Sample position': str}
                                )
        logging.info('Logbook data aquired from %s' % logbook_path)
        return logbook
    except Exception as e:
        print('Failed to read logbook - terminating process')
        logging.info('Failed to read logbook - unable to continue')
        logging.debug(str(e))
        raise

def get_logbook_data(logbook, trackid):  # Get scan speed and framerate from logbook
    substrate_no = trackid[1:4]
    track_no = trackid[-1]
    track_row = logbook.loc[(logbook['Substrate No.'] == substrate_no) & (logbook['Sample position'] == track_no)]
    scan_speed = int(track_row['scan speed [mm/s]'])
    framerate = int(track_row['Frame rate (kHz)'] * 1000)
    return framerate, scan_speed
    
def make_hdf5(substrate_output_folder, track_folder, trackid, logbook):
    input_subfolders = sorted(glob.glob('%s/*/' % track_folder))   # Sorted list of subfolders containing tiff stacks. When sorted, image folder is at index zero and flats are at index 1 due to the file naming convention.
    output_filepath = pathlib.PurePath(substrate_output_folder, '%s.hdf5' % trackid)
    framerate, scan_speed = get_logbook_data(logbook, trackid)
    n_frames = round(framerate * 4 / scan_speed) # based on track length of 4 mm
    n_frames += 100     # Margin for ~50 frames before and after scan
    first_frame = start_frames[trackid[-2:]] - 50   # Begin video 50 frames before laser onset
    try:
        with h5py.File(output_filepath, 'x') as output_file:
            create_dataset(output_file, 'xray_images', input_subfolders, index = 0, element_size = [0, 4.3, 4.3], n_frames=n_frames, first_frame=first_frame)
            create_dataset(output_file, 'xray_flats', input_subfolders, index = 1, element_size = [0, 4.3, 4.3], n_frames=200)
            logging.info('File complete')
    except OSError as e:
        print('Error: file already exists - skipping file')
        logging.info('Failed - file exists')
        logging.debug(str(e))
        
def create_dataset(file, dset_name, dset_folders, index, element_size, n_frames=0, first_frame=0):
    dset_source = dset_folders[index]
    dset_images = sorted(glob.glob('%s/*.tif' % dset_source))
    dset_im0 = imread(dset_images[0])
    if n_frames == 0:
        n_frames = len(dset_images)
    dset_shape = tuple([n_frames] + list(dset_im0.shape))
    dset = file.require_dataset(dset_name, shape=dset_shape, dtype=np.uint8)
    dset.attrs.create('element_size_um', element_size)
    print('Writing %s %s' % (dset_name, dset_shape))
    logging.debug('%s - Writing %s %s' % (file, dset_name, dset_shape))
    for i, tiff_file in enumerate(dset_images[first_frame:first_frame+n_frames]):
        im = np.flip(imread(tiff_file), axis=(0, 1)) # Read TIFF image and rotate 180 degrees (by flipping along both axes)
        if output_dtype == '8_bit':
            im_converted = np.round(im / 4095 * 255).astype(np.uint8) # Convert to 8-bit greyscale
        elif output_dtype == '16_bit':
            im_converted = np.round(im / 4095 * 65535).astype(np.uint16) # Convert to 16-bit greyscale
        dset[i, :, :] = im_converted
        
def main():
    data_root = wd_query()
    output_root = output_query(data_root)
    create_log(output_root)  # Set folder name for log files that will be created in the same directory as this script
    print('\nProcess started at %s - creating new log' % timestamp())
    print('Fetching experiment logbook data')
    logbook = get_logbook()
    duplicate_folder_tree(data_root, output_root, logbook)

if __name__ == "__main__":
	main()