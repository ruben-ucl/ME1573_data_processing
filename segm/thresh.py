import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bs-p10-s37_lagrangian'
calculate_thresh_from = 'bs-p10-s37'
apply_substrate_mask = False
thresh_algo = 'tri'
thresh_offset = 85  # Default 35
output_dset_name = f'{input_dset_name}_bin'
view_hists = False
save_output = True

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def calc_thresh(dset, trackid):
    if apply_substrate_mask == True:
        a_masked, vals = mask_substrate(dset, trackid)
    else:
        a_masked = dset
        vals = np.array(dset)
    print('Calculating threshold')
    if thresh_algo == 'tri':
        thresh = triangle(vals, thresh_offset)
    if thresh_algo == 'yen':
        thresh = yen(vals, thresh_offset)
    return thresh

def triangle(vals, offset):
    thresh = filters.threshold_triangle(vals) + offset  
    print(f'Threshold calulated by Triangle method with offset of {offset}: {thresh}')
    return thresh

def yen(vals, offset):
    thresh = filters.threshold_yen(vals) + offset
    print(f'Threshold calulated by Yen method with offset of {offset}: {thresh}')
    return thresh

def mask_substrate(dset, trackid):  # Function to mask the images so that only the substrate region is used for thresholding. Sets region above to zero.
    print('Masking ROI')
    substrate_mask = get_substrate_mask(trackid, dset[0].shape, substrate_surface_measurements_fpath)
    print('Creating 3d mask from 2d slice')
    substrate_mask_3d = np.stack([substrate_mask for i in range(len(dset))])
    print('Setting pixel values above substrate surface to zero')
    stack_masked = np.array(dset)
    stack_masked[np.invert(substrate_mask_3d)] = 0
    print('Extracting substrate grey values for threshold calculation')
    val_inds = np.nonzero(stack_masked)
    vals = stack_masked[val_inds]
    print('Masking operation complete')
    return stack_masked, vals

def main():
    for f in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
        fname = Path(f).name
        print('Reading %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        with h5py.File(f, 'r+') as file:
            if output_dset_name not in file:
                # Calculate threshold values
                thresh_val = calc_thresh(file[calculate_thresh_from], trackid)
                # Apply threshold to dataset
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                output_dset = (dset > thresh_val).astype(np.uint8) * 255                
                if view_hists == True:
                    view_histogram(output_dset, trackid)
                if save_output == True:
                    file[output_dset_name] = output_dset
                print('Done\n')
            else:
                print(f'Dataset \'{output_dset_name}\' already exists - skipping file\n')

if __name__ == "__main__":
	main()