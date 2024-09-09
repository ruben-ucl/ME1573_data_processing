import h5py, glob, os, sys
import numpy as np
from pathlib import Path
from skimage import filters
from skimage import exposure
from skimage.morphology import disk, ball
import functools

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook, median_filt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.3'

'''
CHANGELOG
    v0.1 - Carries out background subtraction on specified hdf5 file datasets in either 'previous frame' or 'first n frames' mode
           and apppends output to the original file as a new dataset
    v0.2 - Added moving window background subtraction mode 'prev_n_frames'
    v0.3 - Rewritten algorithms 
           
INTENDED CHANGES
    - Add more background subtraction options
    
'''
print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Input information
filepath = get_paths()['hdf5']
input_dset_name = 'ff_corrected'

# Output information
mode = 'prev_n_frames_skip_m' # Set to 'first_n_frames', 'prev_n_frames' or 'prev_n_frames_skip_m'
n = 10
m = 37

def main(mode, n):
    logbook = get_logbook()
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(f).name
        trackid = fname[:5] + '0' + fname[-6]
        print('Reading %s' % fname)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                if mode == 'first_n_frames':
                    output_dset_name = f'bs-f{n}'
                    check_for_dset(file, output_dset_name)
                    output_dset = first_n_frames(dset, n)
                elif mode == 'prev_n_frames':
                    output_dset_name = f'bs-p{n}'
                    check_for_dset(file, output_dset_name)
                    output_dset = prev_n_frames(dset, n)
                elif mode == 'prev_n_frames_skip_m':
                    output_dset_name = f'bs-p{n}-s{m}'
                    check_for_dset(file, output_dset_name)
                    output_dset = prev_n_frames(dset, n, m)
                file[output_dset_name] = output_dset
                # transfer_attr(file[input_dset_name], file[output_dset_name], 'element_size_um')
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')

def first_n_frames(dset, n):
    first_n_frames_avg = np.clip(np.median(dset[:n], axis=0), 1, None) # Set 0 value pixels to 1 to avoid zero division errors
    bg_sub = dset / first_n_frames_avg
    # view_histogram(bg_sub, show_std=True, title='bg_sub')
    # bg_sub_filt = median_filt(bg_sub, kernel=disk(radius=3))
    bg_sub_filt = bg_sub
    # view_histogram(bg_sub_filt, show_std=True, title='bg_sub_filt')
    output_dset_8bit = rescale_to_8bit(bg_sub_filt)
    # view_histogram(output_dset_8bit, show_std=True, title='output_dset_8bit')
    return output_dset_8bit

def prev_n_frames(dset, n, m=0):
    output_dset = np.zeros(dset.shape, dtype=np.float64)
    for i, frame in enumerate(dset):
                print(f'Working on frame {i+1}', end='\r')
                if (i < n+m):
                    prev_n_frames_avg = np.clip(np.median(dset[:n], axis=0), 1, None)
                else:
                    prev_n_frames_avg = np.clip(np.median(dset[i-m-n:i-m], axis=0), 1, None)
                bg_sub = np.clip(frame / prev_n_frames_avg, 1, None)
                output_dset[i] = bg_sub
    bg_sub_filt = median_filt(output_dset, kernel=disk(radius=3))
    # bg_sub_filt = output_dset
    output_dset_8bit = rescale_to_8bit(bg_sub_filt)
    # view_histogram(output_dset_8bit[-100])
    
    return output_dset_8bit

def rescale_to_8bit(dset):
    # clip values outside +- 3 sigma (99.7 % confidence interval) and rescale image to 8bit
    # For lower limit values <0, clip to 0
    print('Stretching stack histogram and rescaling values to 8-bit')
    bg_sub_mean = np.mean(dset)
    bg_sub_std = np.std(dset)
    a = np.clip(bg_sub_mean - 6 * bg_sub_std, 0, None)
    b = bg_sub_mean + 6 * bg_sub_std                
    output_dset_norm = np.clip((dset - a) / (b - a), 0, 1)
    # view_histogram(output_dset_norm, show_std=True, title='Normalised')
    
    output_dset_8bit = (output_dset_norm * 255).astype(np.uint8)
    # view_histogram(output_dset_8bit, show_std=True, title='8bit')
    
    return output_dset_8bit
    
def transfer_attr(dset_1, dset_2, attr):    # Copy attribute from dset_1 to dset_2
    data = dset_1.attrs.get(attr)
    dset_2.attrs.create(attr, data)
    
def check_for_dset(file, dset_name):
    if dset_name in file.keys():    # Check if dataset with output name exists already, and skip file if so
        raise OSError

if __name__ == '__main__':
    main(mode, n)
    