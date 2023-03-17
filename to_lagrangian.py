import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from my_funcs import *

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

mode = 'save'
input_dset_name = 'bs-p5-s5'
output_dset_name = f'{input_dset_name}_lagrangian_meltpool'  # Set fov_w = 220 for keyhole, fov_w = 350 for meltpool
logbook = get_logbook()

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def main():
    files = glob.glob(str(Path(filepath, '*.hdf5')))
    for f in files:
        fname = Path(f).name
        trackid = fname[:-5]
        print('\nReading %s: %s' % (fname, input_dset_name)) 
        try:
            with h5py.File(f, 'r+') as file:
                dset = file[input_dset_name]
                print('Input: shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                track_data = get_logbook_data(logbook, trackid)
                framerate = track_data['framerate']
                scan_speed = track_data['scan_speed']
                
                _, substrate_surface_coords = get_substrate_surface_coords(dset.shape, substrate_surface_measurements_fpath, trackid)
                dset_lagrangian = to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords)
                
                print('Output: shape: %s, dtype: %s'% (dset_lagrangian.shape, dset_lagrangian.dtype))
                if mode == 'save':
                    file[output_dset_name] = dset_lagrangian
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            print(e)
            
def to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords,
                  px_size=4.3, start_frame=48, track_l = 4, fov_w=350, fov_h=130, shift_h=30, laser_start_pos=90, v_correction=0.97
                  ):
    output_shape = (len(dset) - start_frame, fov_h, fov_w)
    output_dset = np.zeros(output_shape, dtype=np.uint8)
    for i, frame in enumerate(dset[start_frame:]):
        laser_end_pos = laser_start_pos + track_l * 1000 / px_size
        scan_speed_px = 1000 * scan_speed / (px_size * framerate)
        laser_pos = int(np.clip(laser_start_pos + i * scan_speed_px * v_correction, None, laser_end_pos))
        # print(f'laser_pos: {laser_pos}')
        row_min = int(substrate_surface_coords[laser_pos])
        row_max = row_min + fov_h
        col_max = laser_pos + shift_h
        col_min = col_max - fov_w
        # print(f'rows: {row_min} - {row_max}\ncols: {col_min} - {col_max}')
        pad = fov_w - laser_start_pos - shift_h
        frame_padded = np.pad(frame, pad, mode='constant')
        frame_cropped = frame_padded[row_min+pad:row_max+pad, col_min+pad:col_max+pad]
        
        preview_int = 50
        if (mode == 'preview') & (i in [n * preview_int for n in range(len(dset)//preview_int)]):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(frame_padded, cmap='gray')
            ax1.plot([i + pad for i in [col_min, col_max, col_max, col_min, col_min]],
                     [i + pad for i in [row_min, row_min, row_max, row_max, row_min]],
                     color = 'b',
                     lw = 1
                     )
            ax2.imshow(frame_cropped, cmap='gray')
            plt.show()

        elif mode == 'save':
            output_dset[i] = frame_cropped
    
    return output_dset
            
if __name__ == "__main__":
	main()