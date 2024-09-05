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

mode = 'save'   # 'preview' or 'save'
input_dset_name = 'bs-p10-s37'
output_dset_name = f'{input_dset_name}_lagrangian'  # Set fov_h = 220 for keyhole, fov_h = 350 for meltpool
# output_dset_name = 'ffc_lagrangian'  # Set fov_h = 220 for keyhole, fov_h = 350 for meltpool
logbook = get_logbook()

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def main():
    files = glob.glob(str(Path(filepath, '*.h*5')))
    for f in sorted(files):
        fname = Path(f).name
        trackid = fname[:7]
        print('\nReading %s: %s' % (fname, input_dset_name)) 
        try:
            with h5py.File(f, 'r+') as file:
                # if output_dset_name in file.keys():    # Check if dataset with output name exists already, and skip file if so
                    # raise OSError
                dset = file[input_dset_name]
                print('Input: shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                track_data = get_logbook_data(logbook, trackid)
                framerate = track_data['framerate']
                scan_speed = track_data['scan_speed']
                
                _, substrate_surface_coords = get_substrate_surface_coords(dset.shape, substrate_surface_measurements_fpath, trackid)
                # dset_lagrangian = to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords)    # Default for keyhole
                # dset_lagrangian = to_lagrangian(dset,
                                                # scan_speed,
                                                # framerate,
                                                # substrate_surface_coords,
                                                # fov_h = 350,
                                                # shift_v = 100,
                                                # fov_v = 230
                                                # start_frame=118,
                                                # laser_start_pos=30
                                                # )    # For meltpool  
                dset_lagrangian = to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords, track_l=256*0.0043, fov_v=50, fov_h=60, shift_v=0, shift_h=10, laser_start_pos=0, start_frame=50) # For 504 kHz
                # dset_lagrangian = to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords, fov_v=250, shift_v=250, shift_h=100) # For spatter
                
                print('Output: shape: %s, dtype: %s'% (dset_lagrangian.shape, dset_lagrangian.dtype))
                if mode == 'save':
                    file[output_dset_name] = dset_lagrangian
            print('Done')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            print(e)
            
def to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords,
                  px_size=4.3, start_frame=48, track_l = 4, fov_h=220, fov_v=130, shift_v=0, shift_h=30, laser_start_pos=90, v_correction=0.97
                  ):
    # Create zero-filled output dataset to store the lagrangian keyhole video
    output_shape = (len(dset) - start_frame, fov_v, fov_h)
    output_dset = np.zeros(output_shape, dtype=np.uint8)
    
    # Iterate through frames, cropping each so that the keyhole is in the same position relative to the new frame edges
    for i, frame in enumerate(dset[start_frame:]):
        laser_end_pos = laser_start_pos + track_l * 1000 / px_size
        scan_speed_px = 1000 * scan_speed / (px_size * framerate)
        laser_pos = int(np.clip(laser_start_pos + i * scan_speed_px * v_correction, None, laser_end_pos))
        # print(f'laser_pos: {laser_pos}')
        row_min = int(substrate_surface_coords[np.clip(laser_pos, None, len(substrate_surface_coords)-1)]) - shift_v
        row_max = row_min + fov_v
        col_max = laser_pos + shift_h
        col_min = col_max - fov_h
        # print(f'rows: {row_min} - {row_max}\ncols: {col_min} - {col_max}')
        # pad_start = fov_h - laser_start_pos - shift_h
        pad = fov_h
        frame_padded = np.pad(frame, pad, mode='constant')
        frame_cropped = frame_padded[row_min+pad:row_max+pad, col_min+pad:col_max+pad]
        
        preview_int = 50
        if (mode == 'preview') & (i in [n * preview_int for n in range(len(dset)//preview_int)]):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # Plot with frame padding
            # ax1.imshow(frame_padded, cmap='gray')
            # ax1.plot([i + pad for i in [col_min, col_max, col_max, col_min, col_min]],
                     # [i + pad for i in [row_min, row_min, row_max, row_max, row_min]],
                     # color = 'b',
                     # lw = 1
                     # )
            # Plot without frame padding
            ax1.imshow(frame, cmap='gray')
            ax1.plot([col_min, col_max, col_max, col_min, col_min],
                     [row_min, row_min, row_max, row_max, row_min],
                     color = 'b',
                     lw = 1
                     )
            ax2.imshow(frame_cropped, cmap='gray')
            for spine in ax2.spines.values():
                spine.set_edgecolor('b')
            for ax in [ax1, ax2]:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            plt.show()

        elif mode == 'save':
            output_dset[i] = frame_cropped
    
    return output_dset
            
if __name__ == "__main__":
	main()