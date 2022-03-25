import h5py, glob
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.2'

'''
CHANGELOG
    v0.1 - Carries out background subtraction on specified hdf5 file datasets in either 'previous frame' or 'first n frames' mode
           and apppends output to the original file as a new dataset
    v0.2 - Added moving window background subtraction mode 'prev_n_frames'
           
INTENDED CHANGES
    - Add more background subtraction options
    
'''
# Input informaton
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'ff_corrected'

# Output information
mode = 'prev_n_frames' # Set to 'first_n_frames' or 'prev_frame'
n = 10

def main(mode, n):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                if mode == 'first_n_frames':
                    output_dset = first_n_frames(dset, n)
                    output_dset_name = 'bg_sub_first_%s_frames' % str(n)
                if mode == 'prev_n_frames':
                    output_dset = prev_n_frames(dset, n)
                    output_dset_name = 'bg_sub_prev_%s_frames' % str(n)
                file[output_dset_name] = output_dset
                transfer_attr(file[input_dset_name], file[output_dset_name], 'element_size_um')
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')

def prev_n_frames(dset, n):
    output_dset = np.zeros_like(dset[n:])
    dset_mean = np.mean(dset)
    for i, frame in enumerate(dset):
                if i < n:
                    continue
                else:
                    prev_n_frames_avg = np.mean(dset[i-n:i, :, :], axis=0)
                    output_dset[i-n, :, :] = np.clip(frame / prev_n_frames_avg * dset_mean, 0, 255)
    return output_dset
    
def first_n_frames(dset, n):
    output_dset = np.zeros_like(dset)
    dset_mean = np.mean(dset)
    first_n_frames_mean = np.mean(dset[:n, :, :], axis=0)
    for i, frame in enumerate(dset):
        output_dset[i, :, :] = np.clip(frame/first_n_frames_mean * dset_mean, 0, 255)
    return output_dset
    
def transfer_attr(dset_1, dset_2, attr):    # Copy attribute from dset_1 to dset_2
    data = dset_1.attrs.get(attr)
    dset_2.attrs.create(attr, data)

main(mode, n)
    