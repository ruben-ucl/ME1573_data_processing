import h5py, glob, cv2, os, functools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from my_funcs import *

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v1.0'

'''
CHANGELOG
    v0.1 - Measures keyhole binary images to extract geometry metrics
         - Outputs individual .csv files with full measurements for each track
         - Outputs summary .csv file containing summary stats for all tracks
    v1.0 - Switched output units to um instead of px
         - Works on full frame and Lagrangian cropped datasets
    
INTENDED CHANGES
    - 
'''

"""Controls"""
input_dset_name = 'keyhole_bin_nofilt'
save_mode = 'save' # Set to 'preview' or 'save'
frame_mode = 'cropped' # Set to 'full_frame' or 'cropped'
plot_keyholes = False
ignore_first_n_frames = 70
ignore_last_n_frames = 100

um_per_pix = 4.3
capture_framerate = 504000 # fps

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

def main():
    logbook = get_logbook()
    # Initialise dictionary for storing summary statistics
    keyhole_data_summary = {'trackid': [],
                            'n_frames': [],
                            'n_samples': [],
                            'area_mean': [],
                            'area_sd': [],
                            'max_depth_mean': [],
                            'max_depth_sd': [],
                            'max_length_mean': [],
                            'max_length_sd': [],
                            'depth_at_max_length_mean': [],
                            'depth_at_max_length_sd': [],
                            'apperture_length_mean': [],
                            'apperture_length_sd': []
                            }
    # Iterate through files measuring keyholes frame by frame
    for file in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
        fname = Path(file).name
        print('\nReading file %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        with h5py.File(file, 'a') as f:
            try:
                dset = f[input_dset_name]
            except KeyError:
                continue
            n_frames = len(dset)
            keyhole_data = {'time': [],
                            'centroid': [],
                            'area': [],
                            'max_depth': [],
                            'max_length': [],
                            'depth_at_max_length': [],
                            'apperture_length': [],
                            }
            # f1, f2 = get_start_end_frames(trackid, logbook, margin=3, start_frame_offset=5)
            f1 = 0
            # frame_inds = range(f1, f2)
            # len_frame_inds = len(frame_inds)
            len_frame_inds = len(dset[ignore_first_n_frames:-ignore_last_n_frames])
            for i, im in enumerate(dset[ignore_first_n_frames:-ignore_last_n_frames]):
                # Get timestamp of frame
                time = i * 1/capture_framerate * 1000 # ms
                keyhole_data['time'].append(time)
                
                # Get keyhole region properties
                try:
                    props = measure.regionprops(im)
                    keyhole_data['centroid'].append(props[0].centroid)  # Assume keyhole is at index 1, 0 is background
                    keyhole_data['area'].append(props[0].area)
                    max_depth, max_length, depth_at_max_length, apperture_length = get_keyhole_measurements(im, trackid, i+1)
                    keyhole_data['max_depth'].append(max_depth)
                    keyhole_data['max_length'].append(max_length)
                    keyhole_data['depth_at_max_length'].append(depth_at_max_length)
                    keyhole_data['apperture_length'].append(apperture_length)
                except IndexError:
                    keyhole_data['centroid'].append(None)
                    keyhole_data['area'].append(0)
                    keyhole_data['max_depth'].append(0)
                    keyhole_data['max_length'].append(0)
                    keyhole_data['depth_at_max_length'].append(0)
                    keyhole_data['apperture_length'].append(0)
                
                if plot_keyholes == False: 
                    printProgressBar(i-f1+1, len_frame_inds, prefix='Measuring keyholes', suffix='Complete', length=50)
            
        keyhole_data = pd.DataFrame(keyhole_data)   # Convert to pandas dataframe
        keyhole_data_summary = generate_summary_stats(keyhole_data, keyhole_data_summary, trackid)
        if save_mode == 'save':
            output_folder = Path(filepath, 'keyhole_measurements_lagrangian')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            keyhole_data.to_csv(Path(output_folder, f'{trackid}_keyhole_measurements_nofilt.csv'))
        else:
            print(keyhole_data[100:110])
    if save_mode == 'save':
        pd.DataFrame(keyhole_data_summary).to_csv(Path(output_folder, 'keyhole_measurements_summary_nofilt.csv'))

def generate_summary_stats(keyhole_data, keyhole_data_summary, trackid):
    print('Calculating summary stats')
    keyhole_data_summary['trackid'].append(trackid)
    keyhole_data_summary['n_frames'].append(len(keyhole_data))
    for col_name in ['area', 'max_depth', 'max_length', 'depth_at_max_length', 'apperture_length']:
        data = keyhole_data[col_name].to_numpy()
        if col_name == 'depth_at_max_length':
            data_nonzero = data
        else:
            data_nonzero = data[np.nonzero(data)]
        if col_name == 'area':
            keyhole_data_summary['n_samples'].append(len(data_nonzero))
        mean = data_nonzero.mean()
        keyhole_data_summary[f'{col_name}_mean'].append(mean)
        sd = data_nonzero.std()
        keyhole_data_summary[f'{col_name}_sd'].append(sd)
    return keyhole_data_summary
            
def get_keyhole_measurements(im, trackid, frame_n):
    # Initialise variables
    max_length_px = 0
    max_depth_px = 0
    apperture_length_px = 0
    depth_at_max_length_px = 0
    lengths_px = []
    depths_px = []
    im_height, im_length = im.shape
    _, min_col, max_row, max_col = measure.regionprops(im)[0]['bbox']
    
    if frame_mode == 'full_frame':
        substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
        substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
        slope = substrate_surface_df.at[trackid, 'm']
        intercept = substrate_surface_df.at[trackid, 'c']
        substrate_height = round(slope * min_col + intercept).astype(np.uint16)
        min_row = substrate_height
    else:
        min_row = 0
    
    # Iterate through image rows to find max_length, depth_at_max_length and apperture_length 
    for r in range(min_row, max_row):
        row = im[r]
        try:
            length_px = pd.value_counts(row).at[255]   # Count white pixels in row to get keyhole length at that row
            if plot_keyholes == True:
                lengths_px.append(length_px)
            if r == min_row:
                apperture_length_px = length_px
            if length_px >= max_length_px:
                max_length_px = length_px
                depth_at_max_length_px = r - min_row
        except KeyError:
            if plot_keyholes == True:
                lengths_px.append(0)
            
    # Iterate through columns to find max_depth
    for c in range(min_col, max_col):
        col = im.T[c]
        try:
            depth_px = np.where(col == 255)[0][-1] - min_row   # Find distance between substrate top of last (lowest) occurance of 255 in column
            if plot_keyholes == True:
                depths_px.append(depth_px)
            if depth_px > max_depth_px:
                max_depth_px = depth_px
        except KeyError:
            if plot_keyholes == True:
                depths_px.append(0)
    
    # Convert from pixels to micrometers
    output_px = [max_depth_px, max_length_px, depth_at_max_length_px, apperture_length_px]
    output_um = [x * um_per_pix for x in output_px]
    
    if plot_keyholes == True:
        print(f'Plotting frame {frame_n}')
        print(f'   max_depth = {round(output_um[0], 1)} um\n', 
              f'   max_length = {round(output_um[1], 1)} um\n',
              f'   depth_at_max_length = {round(output_um[2], 1)} um\n',
              f'   apperture_length = {round(output_um[3], 1)} um\n'
              )
        keyhole_dimensions_plot(im, trackid, frame_n, lengths_px, depths_px, min_col, max_col, min_row, max_row)
    return output_um
    
def keyhole_dimensions_plot(im, trackid, frame_n, lengths, depths, min_col, max_col, min_row, max_row):
    # Get image of keyhole cropped to its bounding box
    keyhole_cropped = im[min_row:max_row, min_col:max_col]
    (y_lim, x_lim) = keyhole_cropped.shape
    aspect = y_lim/x_lim
    
    # Initialise figure with title
    fig, ax = plt.subplots()
    fig.suptitle(f'{trackid}_frame_{frame_n}\nKeyhole dimensions')

    # Add keyhole image as subplot
    ax.imshow(keyhole_cropped, cmap='gray')

    # create new axes on the right and on the bottom of the current axes
    divider = make_axes_locatable(ax)
    ax_depth = divider.append_axes("bottom", size='100%', pad='3%', sharex=ax)
    ax_length = divider.append_axes("right", size='100%', pad='3%', sharey=ax)
    
    # Turn off axis ticks and labels for image
    ax.yaxis.set_tick_params(labelleft=False, size=0)
    ax.xaxis.set_tick_params(labelbottom=False, size=0)
    
    # Plot keyhole depth on axes below image
    ax_depth.plot(range(x_lim), depths)
    ax_depth.set_box_aspect(aspect)
    ax_depth.set_ylim(y_lim, 0)
    ax_depth.set_ylabel('Depth (px)')
    ax_depth.set_xlabel('X-pos. (px)')
    
    # Plot keyhole length on axes beside image
    ax_length.yaxis.set_tick_params(labelright=True)
    ax_length.set_box_aspect(aspect)
    ax_length.plot(lengths[:y_lim], range(y_lim))
    ax_length.yaxis.set_label_position("right")
    ax_length.yaxis.tick_right()
    ax_length.set_xlabel('Length (px)')
    ax_length.set_ylabel('Y-pos. (px)')

    plt.show()
    

if __name__ == "__main__":
	main()