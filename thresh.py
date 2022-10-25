import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'ffc_bg_sub_prev_5_frames_hist_eq_med_filt_r3'

output_dset_name = 'keyhole_binary_yen'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def threshold(dset, trackid):
    a = np.array(dset)
    a_masked, vals = mask_substrate(dset, trackid)
    # a_masked[a_masked == 0] = 255
    # thresh = filters.threshold_minimum(a)     # Does not work the same as ImageJ, different implementation of histogram smoothing
    print('Calculating threshold')
    thresh_offset = -5
    thresh = filters.threshold_yen(vals) + thresh_offset
    # thresh = filters.threshold_otsu(a)
    print(f'Threshold calulated by Yen method with offset of {thresh_offset}: {thresh}')
    output_dset = np.zeros_like(dset)
    mask = a_masked > thresh
    output_dset[mask] = 255
    return output_dset

def mask_substrate(dset, trackid):  # Function to mask the images so that only the substrate region is used for thresholding. Sets region above to zero.
    print('Masking ROI')
    substrate_mask = np.ones_like(dset[0], dtype=bool)
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    print(f'Substrate edge equation retrieved: y = {m}x + {c}')
    n_rows, n_cols = substrate_mask.shape
    print('Calculating mask dimensions')
    for x in range(n_cols):
        surface_height = round(m * x + c)
        substrate_mask[surface_height:, x] = False
    
    print('Creating 3d mask from 2d slice')
    substrate_mask_3d = np.stack([substrate_mask for i in range(len(dset))])
    print('Setting pixel values above substrate surface to zero')
    stack_masked = np.array(dset)
    stack_masked[substrate_mask_3d] = 0
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