import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bg_sub_prev_5_frames_hist_eq_med_filt_r3'

output_dset_name = 'keyhole_binary'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

def threshold(dset):
    a = np.array(dset)
    # thresh = filters.threshold_minimum(a)     # Does not work the same as ImageJ, different implementation of histogram smoothing
    thresh = filters.threshold_yen(a)
    # thresh = filters.threshold_otsu(a)
    print(f'Threshold calulated by Yen method: {thresh}')
    output_dset = np.zeros_like(dset)
    mask = dset > thresh
    output_dset[mask] = 255
    return output_dset

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                output_dset = threshold(dset)
                
                file[output_dset_name] = output_dset
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()