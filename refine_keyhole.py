import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import morphology
from skimage import measure
from skimage import segmentation
import matplotlib.pyplot as plt
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bgs_prev_10/keyhole_binary'
output_dset_name = f'{input_dset_name}_refined'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

username = 'MSM35_Admin' # Set to PC username so that correct Dropbox directory can be located
logbook_fpath = Path(f'C:/Users/{username}/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19/LTP 2 June 2022', '20220622_ME1573_Logbook_Final.xlsx')
substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def morpho_ops(dset, trackid, f1=0, f2=-1):
    output_dset = np.zeros_like(dset)
    substrate_mask = get_substrate_mask(dset[0].shape, substrate_surface_measurements_fpath, trackid)
    background_mask = np.invert(substrate_mask)
    frame_inds = range(len(dset))[f1:f2]
    print('Executing morphological operations')
    for i in frame_inds:
        print(f'Working on frame {i}', end='\r')
        frame = dset[i, :, :]
        op1 = morphology.binary_opening(frame, footprint=morphology.disk(1))
        op1[background_mask] = 255
        op2 = morphology.binary_dilation(op1, footprint=morphology.disk(3))
        op3 = morphology.remove_small_holes(op2, area_threshold=100, connectivity=2)
        op4 = morphology.binary_erosion(op3, footprint=morphology.disk(3))    
        # op5 = morphology.binary_closing(op4, footprint=morphology.disk(2))
        op4[background_mask] = 0
        
        output_dset[i, :, :] = op4
    return output_dset
    
def get_largest_cc(dset, f1=0, f2=-1):
    print('Extracting largest connected component')
    keyhole_mask = np.zeros_like(dset).astype(bool)
    frame_inds = range(len(dset))[f1:f2]
    for i in frame_inds:
        print(f'Working on frame {i}', end='\r')
        frame = dset[i, :, :]
        labels = measure.label(frame, connectivity=2)
        props_df = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'area')))
        try:
            props_df.sort_values('area', ascending=False, inplace=True, ignore_index=True)        # Sort by area so that df row at index 0 contains largest cc
            if props_df.at[0, 'area'] > 20:                                     # Filter out small cc's pre- and post- laser onset
                largest_cc_label = props_df.at[0, 'label']
                keyhole_isolated = labels == largest_cc_label
                keyhole_mask[i, :, :] = keyhole_isolated
        except KeyError as e:
            pass
    output_dset = np.zeros_like(dset)
    output_dset[keyhole_mask] = 255
    return output_dset
    
def mask_frames(dset, f1, f2):    # Mask array from indices f1 to f2
    a = np.array(dset)
    print(f'Masking frames {f1} to {f2}')
    a[:f1, :, :] = 0
    a[f2:, :, :] = 0
    
    return a

def plot_result(file, dset, trackid, f1, f2):
    print('Plotting results preview')
    substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    
    z_project_binary = np.max(dset[::10], axis=0)
    edge_mask = segmentation.find_boundaries(z_project_binary, mode='outer', background=0)
    
    print('Calculating z-projection')
    bg_sub = file['ffc_bg_sub_prev_10_frames_hist_eq_med_filt_r3']  # Background image dataset to plot keyhole outlines over
    bg_sub[:f1, :, :] = 0
    bg_sub[f2:, :, :] = 0
    z_project = np.max(bg_sub[::10], axis=0)
    z_project[edge_mask] = 0
    
    fig, ax = plt.subplots()
    ax.imshow(z_project, cmap='gray')
    xx = range(dset.shape[2])
    yy = [m * x + c for x in xx]
    ax.plot(xx, yy, 'b--', linewidth=0.9)
    plt.show()

def main():
    logbook = get_logbook(logbook_fpath)
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(f).name
        print('\nReading %s' % fname)
        trackid = fname[:5] + '0' + fname[-6]
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                f1, f2 = get_start_end_frames(trackid, logbook, start_frame_offset=10)
                dset_masked_frames = mask_frames(dset, f1, f2)
                keyhole_refined = morpho_ops(dset_masked_frames, trackid, f1, f2)
                keyhole_isolated = get_largest_cc(keyhole_refined, f1, f2)
                
                plot_result(file, keyhole_isolated, trackid, f1, f2)
                
                file[output_dset_name] = keyhole_isolated
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            print(e)
            
if __name__ == "__main__":
	main()