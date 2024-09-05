import h5py, glob, cv2, functools
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bg_sub_prev_5_frames'
op_name = 'placeholder'

output_dset_name = f'{input_dset_name}_{op_name}'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

"""Input function to execute on dataset here"""
def placeholder_func_name(dset):
    output_dset = None
    return output_dset

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print(f'Peforming {op_name} on dataset')
                
                """Update function name here"""
                output_dset = placeholder_func_name(dset)
                
                file[output_dset_name] = output_dset
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()