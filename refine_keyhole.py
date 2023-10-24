import h5py, glob, functools
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import morphology, measure, segmentation
import matplotlib.pyplot as plt
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bs-p5-s5_lagrangian_meltpool_bin'
output_dset_name = 'keyhole_bin'
mode = 'cropped'    # Set to 'cropped' or 'full_frame'
save_output = True
preview = False
mask_top_n_rows = 4

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading data from {filepath}')

substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')

def morpho_ops(dset, trackid, f1=0, f2=-1):
    output_dset = np.zeros_like(dset)
    if mode == 'full_frame':
        substrate_mask = get_substrate_mask(dset[0].shape, substrate_surface_measurements_fpath, trackid)
        background_mask = np.invert(substrate_mask)
    frame_inds = range(len(dset))[f1:f2]
    print('Executing morphological operations')
    for i in frame_inds:
        print(f'Working on frame {i}', end='\r')
        frame = dset[i, :, :]
        # if mode == 'full_frame': frame[background_mask] = 255
        frame = morphology.binary_opening(frame, footprint=morphology.disk(1))
        frame = morphology.binary_dilation(frame, footprint=morphology.disk(3))
        frame = morphology.remove_small_holes(frame, area_threshold=500, connectivity=2)
        frame = morphology.binary_erosion(frame, footprint=morphology.disk(3))    
        if mode == 'full_frame': frame[background_mask] = 0
        
        output_dset[i, :, :] = frame
    return output_dset
    
def get_largest_cc(dset, f1=0, f2=-1, filter_by_pos=False, mask_top_rows=None):
    print('Extracting largest connected component')
    keyhole_mask = np.zeros_like(dset).astype(bool)
    frame_inds = range(len(dset))[f1:f2]
    if mask_top_rows != None:
        dset[:, 0:mask_top_rows, :] = 0
    for i in frame_inds:
        print(f'Working on frame {i}', end='\r')
        frame = dset[i, :, :]
        labels = measure.label(frame, connectivity=2)
        props_df = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'area', 'centroid')))
        if filter_by_pos == True:
            h_cent = props_df['centroid-1']
            v_cent = props_df['centroid-0']
            props_df = props_df.loc[np.logical_and(np.logical_and(h_cent > 228, h_cent < 281) , v_cent < 27)]
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

def plot_result(file, dset, trackid, f1=0, f2=-1):
    print('Plotting results preview')
    substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    
    z_project_binary = np.max(dset[::10], axis=0)
    edge_mask = segmentation.find_boundaries(z_project_binary, mode='outer', background=0)
    
    print('Calculating z-projection')
    bg_sub = file['bs-p5-s5']  # Background image dataset to plot keyhole outlines over
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
    files = sorted(glob.glob(str(Path(filepath, '*.hdf5'))))
    for f in files:
        fname = Path(f).name
        print('\nReading %s: %s' % (fname, input_dset_name)) 
        trackid = fname[:5] + '0' + fname[-6]
        with h5py.File(f, 'a') as file:
            if output_dset_name not in file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                
                keyhole_isolated = get_largest_cc(dset, filter_by_pos=True, mask_top_rows=mask_top_n_rows)
                keyhole_refined = morpho_ops(keyhole_isolated, trackid)
                keyhole_final = get_largest_cc(keyhole_refined)
                
                if preview == True:
                    # plot_result(file, keyhole_final, trackid)
                    view_histogram(keyhole_final)
                if save_output == True:
                    file[output_dset_name] = keyhole_final
                
                print('\nDone\n')
            else:
                print(f'Dataset \'{output_dset_name}\' already exists - skipping file\n')
                
if __name__ == "__main__":
	main()