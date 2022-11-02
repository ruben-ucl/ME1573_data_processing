import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'ffc_bg_sub_prev_10_frames_hist_eq_med_filt_r3'

output_dset_name = 'bgs_prev_10/keyhole_binary_tri'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def threshold(dset, trackid):
    a = np.array(dset)
    a_masked, vals = mask_substrate(dset, trackid)
    print('Calculating threshold')
    thresh_offset = 0   # Defaults for moving window bg sub: 5 frames: Yen -5, 10 frames: Yen -25?
    thresh = triangle(vals, thresh_offset)
    # thresh = yen(vals, thresh_offset)
    output_dset = np.zeros_like(dset)
    mask = a_masked > thresh
    output_dset[mask] = 255
    return output_dset

def triangle(vals, offset):
    thresh = filters.threshold_triangle(vals) + offset
    print(f'Threshold calulated by Triangle method with offset of {offset}: {thresh}')
    return thresh
    
def yen(vals, offset):
    thresh = filters.threshold_yen(vals) + offset
    print(f'Threshold calulated by Yen method with offset of {offset}: {thresh}')

def mask_substrate(dset, trackid):  # Function to mask the images so that only the substrate region is used for thresholding. Sets region above to zero.
    print('Masking ROI')
    substrate_mask = get_substrate_mask(dset[0].shape, substrate_surface_measurements_fpath, trackid)
    
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
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(f).name
        print('Reading %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                output_dset = threshold(dset, trackid)
                
                file[output_dset_name] = output_dset
            print('Done\n')
        except OSError as e:
            # print('Error: output dataset with the same name already exists - skipping file\n')
            print(e, '\n')
            
if __name__ == "__main__":
	main()