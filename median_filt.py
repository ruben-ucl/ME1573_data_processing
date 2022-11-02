import h5py, glob, functools
import numpy as np
from pathlib import Path
from skimage import filters
from skimage.morphology import disk
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'ffc_bg_sub_prev_10_frames_hist_eq'
op_name = 'med_filt'
filt_rad = 3

output_dset_name = f'{input_dset_name}_{op_name}_r{filt_rad}'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

username = 'MSM35_Admin' # Set to PC username so that correct Dropbox directory can be located
logbook_fpath = Path(f'C:/Users/{username}/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19/LTP 2 June 2022', '20220622_ME1573_Logbook_Final.xlsx')
logbook = get_logbook(logbook_fpath)

def median_filt(dset, kernel, f1=0, f2=-1):
    output_dset = np.zeros_like(dset)
    frame_inds = range(len(dset))[f1:f2]
    for i in frame_inds:
        print(f'Working on frame {i}', end='\r')
        frame = dset[i, :, :]
        output_dset[i, :, :] = filters.rank.median(frame, kernel)
    return output_dset

def main():
    kernel = disk(radius=filt_rad)
    print(f'Filter kernel disk r{filt_rad}:')
    print(kernel)
    print()
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(f).name
        print('\nReading %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print(f'Peforming {op_name} on dataset')
                
                f1, f2 = get_start_end_frames(trackid, logbook)
                print(f'Working on frames {f1} to {f2}')
                output_dset = median_filt(dset, kernel, f1, f2)
                
                file[output_dset_name] = output_dset
            print('Done                 \n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()