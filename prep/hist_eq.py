import h5py, glob, functools, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import exposure

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'ffc_bg_sub_prev_5_frames'
op_name = 'hist_eq'

output_dset_name = f'{input_dset_name}_{op_name}'

filepath = get_paths()['hdf5']

def hist_eq(dset):
    output_dset = (exposure.equalize_hist(np.array(dset)) * 255).astype(np.uint8)
    # print(f'min: {np.amin(output_dset)}, max: {np.amax(output_dset)}, mean: {np.mean(output_dset)}')
    return output_dset

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print(f'Peforming {op_name} on dataset')
                
                output_dset = hist_eq(dset)
                
                file[output_dset_name] = output_dset
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()