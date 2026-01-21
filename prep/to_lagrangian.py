import h5py, glob, functools, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook, get_logbook_data, get_substrate_surface_coords

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

# Re-implement print to fix issue where print statements do not show in console until after script execution completes
print = functools.partial(print, flush=True)



# Main inputs
mode = 'save'   # 'preview' or 'save'
input_dset_name = 'bs-f40'
output_dset_name = f'{input_dset_name}_lagrangian_long'  # Set fov_h = 220 for keyhole, fov_h = 350 for meltpool
# output_dset_name = 'ffc_lagrangian'  # Set fov_h = 220 for keyhole, fov_h = 350 for meltpool

# Read logbook for accessing scan speeds and laser onset frame numbers
logbook = get_logbook()

# Read folder path to hdf5 files
# filepath = get_paths()['hdf5']
filepath = r'F:/ESRF ME1573 LTP 6 Al data HDF5/ffc'

# Read .csv file containing substrate linear fits
# File should have headers: 'trackid', 'm', 'c'
# 'trackid' is of the format 0123_01
# 'm' is the gradient of the substrate surface (usually found to be approx. -0.01)
# 'c' is y-intercept at x = 0 (in pixels)
substrate_surface_measurements_fpath = Path(filepath,
    'substrate_surface_measurements',
    'substrate_surface_locations.csv')

# Main batch processing loop
def main():
    # Get list of hdf5 files
    files = glob.glob(str(Path(filepath, '*06.h*5')))
    
    # Iterate through files
    for f in sorted(files):
        # Extract file name and track ID
        fname = Path(f).name
        trackid = fname[:7]
        print('\nReading %s: %s' % (fname, input_dset_name))
        
        # Read hdf5 file
        try:
            with h5py.File(f, 'r+') as file:
                # Check if dataset with output name exists already, and skip file if so
                if output_dset_name in file.keys():
                    raise OSError
                
                # Load specified dataset
                dset = file[input_dset_name]
                print('Input: shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                # Get scan information from logbook
                track_data = get_logbook_data(logbook, trackid)
                framerate = track_data['framerate']
                scan_speed = track_data['scan_speed']
                
                # Generate array of substrate surface heights from previously generated linear fits
                # (using get_substrate_surface.py or manually in ImageJ)
                _, substrate_surface_coords = get_substrate_surface_coords(dset.shape,
                    substrate_surface_measurements_fpath,
                    trackid)
                
                # Run Lagrangian cropping operation
                # For 40 kHz video
                # dset_lagrangian = to_lagrangian(dset,
                    # scan_speed,
                    # framerate,
                    # substrate_surface_coords,
                    # fov_h = 100,
                    # shift_v = 100,
                    # fov_v = 230)

                # For 504 kHz
                dset_lagrangian = to_lagrangian(dset,
                    scan_speed,
                    framerate,
                    substrate_surface_coords,
                    track_l=256*0.0043,
                    fov_v=45,
                    fov_h=120,
                    shift_v=3,
                    shift_h=25,
                    laser_start_pos=0,
                    start_frame=88)
                
                print('Output: shape: %s, dtype: %s'% (dset_lagrangian.shape, dset_lagrangian.dtype))
                
                # If in 'save' mode, save the new dataset into the existing hdf5 file
                if mode == 'save':
                    file[output_dset_name] = dset_lagrangian
            print('Done')
            
        # Skip the file if a dataset with the designated output name already exists
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            print(e)

# Lagrangian cropping function    
def to_lagrangian(dset, scan_speed, framerate, substrate_surface_coords,
                  px_size=4.3, start_frame=48, track_l = 4, fov_h=220, fov_v=130,
                  shift_v=0, shift_h=30, laser_start_pos=90, speed_correction=0.97):
    """
    Args:
       dset: Input dataset containing the uncropped images (numpy array)
       scan_speed: Laser scan speed (mm/s)
       framerate: Frame rate of the radiography dataset (kHz)
       substrate_surface_coords: Coordinates of the substrate surface (numpy array)
       px_size: Pixel size in um/pixel (default: 4.3)
       start_frame: Frame number to start processing from (default: 48)
       track_l: Track length in mm (default: 4)
       fov_h: Horizontal field of view (length) in pixels (default: 220)
       fov_v: Vertical field of view (height) in pixels (default: 130)
       shift_v: Vertical field of view shift in pixels (default: 0, corresponding to crop at substrate level)
       shift_h: Horizontal shift parameter (default: 30, corresponding to 30 pixels ahead of the laser)
       laser_start_pos: Starting x-position of the laser in pixels (default: 30)
       speed_correction: Scan speed correction factor (default: 0.97)
       
    Returns:
       Cropped image dataset in Lagrangian reference frame with respect to the melt pool (numpy array)
    """
    
    # Create zero-filled output dataset to store the lagrangian keyhole video
    output_shape = (len(dset) - start_frame, fov_v, fov_h)
    output_dset = np.zeros(output_shape, dtype=np.uint8)
    
    # Calculate laser end position in pixels
    laser_end_pos = laser_start_pos + track_l * 1000 / px_size
        
    # Calculate scan speed in pixels
    scan_speed_px = 1000 * scan_speed / (px_size * framerate)
    print('Scan speed: ', scan_speed, ' mm/s')
    
    # Iterate through frames, cropping each so that the keyhole is in the same position relative to the new frame edges
    for i, frame in enumerate(dset[start_frame:]):
        # Calculate laser position in frame i
        laser_pos = int(np.clip(laser_start_pos + i * scan_speed_px * speed_correction, None, laser_end_pos))

        # Calculate field of view bounds for cropping operation
        row_min = int(substrate_surface_coords[np.clip(laser_pos, None, len(substrate_surface_coords)-1)]) - shift_v
        row_max = row_min + fov_v
        col_max = laser_pos + shift_h
        col_min = col_max - fov_h

        # Add zero padding in horizontal direction for when the field of view exceeds the bounds of the original image
        pad = fov_h
        frame_padded = np.pad(frame, pad, mode='constant')
        frame_cropped = frame_padded[row_min+pad:row_max+pad, col_min+pad:col_max+pad]
        
        # If in 'preview' mode, plot every 50th frame to check the cropping result
        preview_int = 50
        if (mode == 'preview') & (i in [n * preview_int for n in range(len(dset)//preview_int)]):
            # create figure with two subplots, ax1 for original image and ax2 for cropped image
            fig, (ax1, ax2) = plt.subplots(1, 2)
            
            # Display original image and plot cropped field of view boundaries in blue on top
            ax1.imshow(frame, cmap='gray')
            ax1.plot([col_min, col_max, col_max, col_min, col_min],
                     [row_min, row_min, row_max, row_max, row_min],
                     color = 'b',
                     lw = 1)
                     
            # Display cropped image, and outline it in blue
            ax2.imshow(frame_cropped, cmap='gray')
            for spine in ax2.spines.values():
                spine.set_edgecolor('b')
            for ax in [ax1, ax2]:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            plt.show()
        
        # If in 'save' mode, save the cropped frame i into the output dataset array
        elif mode == 'save':
            output_dset[i] = frame_cropped
    
    # Return cropped dataset
    return output_dset
            
if __name__ == "__main__":
	main()