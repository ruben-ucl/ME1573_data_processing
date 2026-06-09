import h5py, glob, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters
from skimage import exposure
from skimage.morphology import disk, ball
from cv2 import inpaint, INPAINT_TELEA
import functools
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook, median_filt, view_histogram

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.4'

'''
CHANGELOG
    v0.1 - Carries out background subtraction on specified hdf5 file datasets in either 'previous frame' or 'first n frames' mode
           and apppends output to the original file as a new dataset
    v0.2 - Added moving window background subtraction mode 'prev_n_frames'
    v0.3 - Rewritten algorithms 
    v0.4 - Improved and cleaned up normalisation method, and added saturated pixel interpolation function
           
INTENDED CHANGES
    - Add more background subtraction options
    
'''
print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Input information
filepath = get_paths()['hdf5']
input_dset_name = 'ff_corrected'

# Output information
mode = 'prev_n_frames' # Set to 'first_n_frames', 'prev_n_frames' or 'prev_n_frames_skip_m'
n = 1
m = 37

debug = False

def main(mode, n):
    logbook = get_logbook()
    for f in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
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
            print('\nDone\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')

def first_n_frames(dset, n):
    first_n_frames_avg = np.clip(np.median(dset[:n], axis=0), 1, None) # Set 0 value pixels to 1 to avoid zero division errors
    bg_sub = dset / first_n_frames_avg
    # bg_sub_filt = median_filt(bg_sub, kernel=disk(radius=3))
    bg_sub_filt = bg_sub
    output_dset_8bit = rescale_to_8bit(bg_sub_filt)
    output_dset_8bit_dsat = interpolate_saturated_values(output_dset_8bit)
    return output_dset_8bit

def prev_n_frames(dset, n, m=0):
    output_dset = np.zeros(dset.shape, dtype=np.float64)
    for i, frame in enumerate(dset):
                print(f'Working on frame {i+1}', end='\r')
                if (i < n+m):
                    prev_n_frames_avg = np.clip(np.mean(dset[:n], axis=0), 1, None)
                else:
                    prev_n_frames_avg = np.clip(np.mean(dset[i-m-n:i-m], axis=0), 1, None)
                bg_sub = np.clip(np.divide(frame, prev_n_frames_avg), 0, 2)
                output_dset[i] = bg_sub

    if debug:
        print('\nBackground subtraction results')
        print('------------------------------')
        print(f'min: {output_dset.min()}')
        print(f'max: {output_dset.max()}')
        print(f'avg: {output_dset.mean()}')
        print(f'std: {output_dset.std()}')
        print('------------------------------\n')
        plt.hist(output_dset.flatten())
        plt.show()
        plt.close()

    # Apply filter
    # bg_sub_filt = median_filt(output_dset, kernel=disk(radius=3))
    bg_sub_filt = output_dset
    
    # Rescale to 8bit
    output_dset_8bit = rescale_to_8bit(bg_sub_filt)
    
    # Interpolate saturated pixels
    output_dset_8bit_dsat = interpolate_saturated_values(output_dset_8bit)
    
    if debug:
        print('\n8-bit conversion results')
        print('------------------------------')
        print(f'min: {output_dset_8bit_dsat.min()}')
        print(f'max: {output_dset_8bit_dsat.max()}')
        print(f'avg: {output_dset_8bit_dsat.mean()}')
        print(f'std: {output_dset_8bit_dsat.std()}')
        print('------------------------------\n')
        plt.hist(output_dset_8bit_dsat.flatten())
        plt.show()
        plt.close()
    
    return output_dset_8bit_dsat

def rescale_to_8bit(dset):
    if debug:
        print('Stretching stack histogram and rescaling values to 8-bit')
    
    output_dset_norm = normalise(dset)
    output_dset_8bit = (output_dset_norm * 255).astype(np.uint8)
    
    return output_dset_8bit
    
def transfer_attr(dset_1, dset_2, attr):    # Copy attribute from dset_1 to dset_2
    data = dset_1.attrs.get(attr)
    dset_2.attrs.create(attr, data)
    
def check_for_dset(file, dset_name):
    if dset_name in file.keys():    # Check if dataset with output name exists already, and skip file if so
        raise OSError

def normalise(dset, target_mean=1, min_q=0.1, max_q=99.9):
    mn, mx = np.percentile(dset, min_q), np.percentile(dset, max_q)
    normed = (dset - mn) / (mx - mn)
    
    return normed
    
def interpolate_saturated_values(dset, sat_low=0, sat_high=255):
    if debug:
        print('interpolating saturated pixels')
    
    T, H, W = dset.shape
    result = dset.astype(float).reshape(T, -1)

    result[(result == sat_low) | (result == sat_high)] = np.nan

    # Temporal interpolation
    df = pd.DataFrame(result)
    df = df.interpolate(method='linear', axis=0, limit_direction='both')
    result = df.values.reshape(T, H, W)
    
    if debug:
        print('temporal interpolation complete')

    # Spatial fallback with OpenCV inpainting
    for t in range(T):
        nan_mask = np.isnan(result[t])
        if nan_mask.any():
            frame = np.nan_to_num(result[t], nan=0).astype(np.float32)
            mask = nan_mask.astype(np.uint8)
            result[t] = inpaint(frame, mask, 3, INPAINT_TELEA)
    
    if debug:
        print('spatial interpolation complete')
    
    return np.clip(np.round(result), 0, 255).astype(np.uint8)

if __name__ == '__main__':
    main(mode, n)
    