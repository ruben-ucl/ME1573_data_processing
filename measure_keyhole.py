import h5py, glob, cv2, os, functools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from my_funcs import *

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

"""
CHANGELOG
    v0.1 - 
    
INTENDED CHANGES
    - 
"""

"""Controls"""
input_dset_name = 'keyhole_binary_yen_refined'
save_mode = 'save' # Set to 'preview' or 'save'
plot_keyholes = False

um_per_pix = 4.3
capture_framerate = 40000 # fps

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

username = 'MSM35_Admin' # Set to PC username so that correct Dropbox directory can be located
logbook_fpath = Path(f'C:/Users/{username}/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19/LTP 2 June 2022', '20220622_ME1573_Logbook_Final.xlsx')

def main():
    logbook = get_logbook(logbook_fpath)
    # Initialise dictionary for storing summary statistics
    keyhole_data_summary = {'trackid': [],
                            'area_mean': [],
                            'area_sd': [],
                            'max_depth_mean': [],
                            'max_depth_sd': [],
                            'max_width_mean': [],
                            'max_width_sd': [],
                            'depth_at_max_width_mean': [],
                            'depth_at_max_width_sd': [],
                            'apperture_width_mean': [],
                            'apperture_width_sd': [],
                            }
    # Iterate through files measuring keyholes frame by frame
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(file).name
        print('\nReading file %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            n_frames = len(dset)
            keyhole_data = {'time': [],
                            'centroid': [],
                            'area': [],
                            'max_depth': [],
                            'max_width': [],
                            'depth_at_max_width': [],
                            'apperture_width': []
                            }
            f1, f2 = get_start_end_frames(trackid, logbook, margin=3, start_frame_offset=5)
            frame_inds = range(f1, f2)
            len_frame_inds = len(frame_inds)
            for i in frame_inds:
                im = dset[i, :, :]
                # Get timestamp of frame
                time = i * 1/capture_framerate * 1000 # ms
                keyhole_data['time'].append(time)
                
                # Get keyhole region properties
                try:
                    props = measure.regionprops(im)
                    keyhole_data['centroid'].append(props[0].centroid)  # Assume keyhole is at index 1, 0 is background
                    keyhole_data['area'].append(props[0].area)
                    max_depth, max_width, depth_at_max_width, apperture_width = get_keyhole_measurements(im, trackid, i+1)
                    keyhole_data['max_depth'].append(max_depth)
                    keyhole_data['max_width'].append(max_width)
                    keyhole_data['depth_at_max_width'].append(depth_at_max_width)
                    keyhole_data['apperture_width'].append(apperture_width)
                except IndexError:
                    keyhole_data['centroid'].append(None)
                    keyhole_data['area'].append(0)
                    keyhole_data['max_depth'].append(0)
                    keyhole_data['max_width'].append(0)
                    keyhole_data['depth_at_max_width'].append(0)
                    keyhole_data['apperture_width'].append(0)
                
                if plot_keyholes == False: 
                    printProgressBar(i-f1+1, len_frame_inds, prefix='Measuring keyholes', suffix='Complete', length=50)
                
        keyhole_data = pd.DataFrame(keyhole_data)   # Convert to pandas dataframe
        keyhole_data_summary = generate_summary_stats(keyhole_data, keyhole_data_summary, trackid)
        if save_mode == 'save':
            output_folder = Path(filepath, 'keyhole_measurements')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            keyhole_data.to_csv(Path(output_folder, f'{trackid}_keyhole_measurements_v3.csv'))
            pass
        else:
            print(keyhole_data[100:110])
    pd.DataFrame(keyhole_data_summary).to_csv(Path(output_folder, f'{trackid}_keyhole_measurements_summary_v3.csv'))

def generate_summary_stats(keyhole_data, keyhole_data_summary, trackid):
    print('Calculating summary stats')
    keyhole_data_summary['trackid'].append(trackid)
    for col_name in ['area', 'max_depth', 'max_width', 'depth_at_max_width', 'apperture_width']:
        data = keyhole_data[col_name].to_numpy()
        if col_name == 'depth_at_max_width':
            data_nonzero = data
        else:
            data_nonzero = data[np.nonzero(data)]
        mean = data_nonzero.mean()
        sd = data_nonzero.std()
        keyhole_data_summary[f'{col_name}_mean'].append(mean)
        keyhole_data_summary[f'{col_name}_sd'].append(sd)
    return keyhole_data_summary
            
def get_keyhole_measurements(im, trackid, frame_n):
    # Initialise variables
    max_width_px = 0
    max_depth_px = 0
    apperture_width_px = 0
    depth_at_max_width_px = 0
    widths_px = []
    depths_px = []
    im_height, im_width = im.shape
    
    min_row, min_col, max_row, max_col = measure.regionprops(im)[0]['bbox']
    substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    slope = substrate_surface_df.at[trackid, 'm']
    intercept = substrate_surface_df.at[trackid, 'c']
    
    # Iterate through image rows to find max_width, depth_at_max_width and apperture_width 
    for r in range(min_row, max_row):
        row = im[r]
        try:
            width_px = pd.value_counts(row).at[255]   # Count white pixels in row to get keyhole width at that row
            if plot_keyholes == True:
                widths_px.append(width_px)
            if r == min_row:
                apperture_width_px = width_px
            if width_px >= max_width_px:
                max_width_px = width_px
                substrate_height = round(slope * min_col + intercept)
                depth_at_max_width_px = r - substrate_height
        except ValueError:
            if plot_keyholes == True:
                widths_px.append(0)
            
    # Iterate through columns to find max_depth
    for c in range(min_col, max_col):
        col = im.T[c]
        try:
            substrate_height = round(slope * c + intercept)
            depth_px = np.where(col == 255)[0][-1] - substrate_height   # Find distance between substrate top of last (lowest) occurance of 255 in column
            if plot_keyholes == True:
                depths_px.append(depth_px)
            if depth_px > max_depth_px:
                max_depth_px = depth_px
        except ValueError:
            if plot_keyholes == True:
                depths_px.append(0)
    
    # Convert from pixels to micrometers
    # output_um = (x * um_per_pix for x in [max_depth_px, max_width_px, depth_at_max_width_px, apperture_width_px])
    output_px = (max_depth_px, max_width_px, depth_at_max_width_px, apperture_width_px)
    
    if plot_keyholes == True:
        print(f'max_depth = {max_depth_px}\nmax_width = {max_width_px}\ndepth_at_max_width = {depth_at_max_width_px}\napperture_width = {apperture_width_px}')
        keyhole_dimensions_plot(im, trackid, frame_n, widths_px, depths_px, min_col, max_col, min_row, max_row)
    
    return output_px
    
def keyhole_dimensions_plot(im, trackid, frame_n, widths, depths, min_col, max_col, min_row, max_row):
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
    ax_width = divider.append_axes("right", size='100%', pad='3%', sharey=ax)
    
    # Turn off axis ticks and labels for image
    ax.yaxis.set_tick_params(labelleft=False, size=0)
    ax.xaxis.set_tick_params(labelbottom=False, size=0)
    
    # Plot keyhole depth on axes below image
    ax_depth.plot(range(x_lim), depths)
    ax_depth.set_box_aspect(aspect)
    ax_depth.set_ylim(y_lim, 0)
    ax_depth.set_ylabel('Depth (px)')
    ax_depth.set_xlabel('X-pos. (px)')
    
    # Plot keyhole width on axes beside image
    ax_width.yaxis.set_tick_params(labelright=True)
    ax_width.set_box_aspect(aspect)
    ax_width.plot(widths[:y_lim], range(y_lim))
    ax_width.yaxis.set_label_position("right")
    ax_width.yaxis.tick_right()
    ax_width.set_xlabel('Width (px)')
    ax_width.set_ylabel('Y-pos. (px)')

    plt.show()
    

if __name__ == "__main__":
	main()