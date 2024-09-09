import h5py, glob, cv2, os, functools, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import measure
from skimage import morphology

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_substrate_mask

filepath = get_paths()['hdf5']

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

"""
CHANGELOG
    v0.1 - Quantifies porosity from binary image datasets
           Saves images with detected pore locations highlighted with a coloured overlay
           Saves pore measurements (depth, area, approximate volume, x-position) to a .csv file
           Pore volumes calculated by oblate spheroid method
    v0.2 - Added dilation/erosion stage to smooth pore regions and fill holes
           
INTENDED CHANGES
    - 
"""

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

save_mode = 'preview' # Set to 'preview' or 'save'
um_per_pix = 4.3

substrate_surface_measurements_fpath = Path(filepath, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
    
dset_name = 'bs-f40_tri+35'
output_im_dset_name = 'bs-f40'

def con_comp(im, min_area=1, max_area=np.inf, connectivity=2):
    # Get labels and region properties using skimage connected component analysis functions
    label_im, num_labels = measure.label(im, return_num = True, connectivity=connectivity)
    print(f'num_labels: {num_labels}')
    props = measure.regionprops(label_im)
    labels = []
    
    compMask = np.zeros_like(im, dtype=bool)
    
    for i in range(num_labels):
        area = props[i].area
        x_cent = props[i].centroid[1]
        y_cent = props[i].centroid[0]
        conditions = [area >= min_area,
                      area <= max_area,
                      # y_cent >= 322 - x_cent * 0.01955,
                      # y_cent <= 450
                      ]
        
        if all(conditions):
            label_i = props[i].label
            labels.append(i)
            compMask[label_im == label_i] = True

            contour = measure.find_contours(label_im == label_i)[0]
            y, x = contour.T
    print(f'After filtering: {len(labels)}\n')
    
    return compMask, props, labels

def dilate_erode(binary_im):
    output_im = morphology.binary_dilation(binary_im, footprint=morphology.disk(2))
    output_im = morphology.remove_small_holes(output_im, area_threshold=100, connectivity=2)
    output_im = morphology.binary_erosion(output_im, footprint=morphology.disk(1))
    return output_im
    
def main():
    stats_df = pd.DataFrame(columns = ['track_id', 'pore_id', 'depth', 'area', 'volume', 'x_pos'])
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        print(f'Opening {Path(file).name}')
        trackid = Path(file).name[:-5]
        with h5py.File(file, 'a') as f:
            
            im = f[dset_name][-1, :, :]
            substrate_mask = get_substrate_mask(trackid, im.shape, substrate_surface_measurements_fpath)
            im, _, _ = con_comp(im, min_area=10)
            plt.imshow(im)
            plt.show()
            im[np.invert(substrate_mask)] = 1
            im = dilate_erode(im)
            im, props, labels = con_comp(im, max_area=100000)
            
            for i in labels:
                stats_i = pd.DataFrame()
                stats_i['track_id'] = [trackid]
                stats_i['pore_id'] = [i]
                stats_i['area'] = [props[i].area]
                axis_major_length = props[i].axis_major_length
                axis_minor_length = props[i].axis_minor_length
                x_cent = props[i].centroid[1]
                y_cent = props[i].centroid[0]
                
                """ Calculate depth, horizontal position and volume for each pore and save to dataframe """
                stats_i['depth'] = [(y_cent - (322 - x_cent * 0.01955)) * um_per_pix]
                stats_i['volume'] = [4/3 * np.pi * axis_major_length * axis_minor_length ** 2]
                stats_i['x_pos'] = [(x_cent - 38) * um_per_pix]
                
                stats_df = pd.concat([stats_df, stats_i])
            
            fig, ax = plt.subplots()
            output_im = f[output_im_dset_name][-1, :, :]
            ax.imshow(output_im, cmap='gray')
            ax.axis('off')
            
            scalebar = ScaleBar(dx=4.3, units='um', location='lower left', width_fraction=0.01)
            plt.gca().add_artist(scalebar)
            
            overlay = np.ones_like(output_im)
            overlay = np.ma.masked_where(np.invert(im), overlay)
            alphas = np.ones_like(output_im) * 0.5 
            ax.imshow(overlay, cmap='plasma', vmin=0, vmax=1, alpha=alphas)
            
            if save_mode == 'save':
                # Save measurments to csv file
                stats_df.to_csv(Path(filepath, 'pore_measurements.csv'))
                
                # Save overlayed image as png
                output_filename = f'{trackid}_pore_loc_overlay.png'
                output_folder = Path(filepath, 'Pore location overlay images')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_filepath = Path(output_folder, output_filename)
                plt.savefig(output_filepath, dpi=600, bbox_inches='tight')
            elif save_mode == 'preview':
                plt.show()
                
    
if __name__ == "__main__":
	main()