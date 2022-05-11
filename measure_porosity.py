import h5py, glob, cv2, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology

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

save_mode = 'preview' # Set to 'preview' or 'save'
um_per_pix = 4.7

with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
dset_name = 'bg_sub_first_30_frames_/bilateral_filt_r8_li-thresh'
output_im_dset_name = 'bg_sub_first_30_frames'

def con_comp(im, connectivity):
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
        conditions = [area > 2,
                      # area < 1000,
                      y_cent > 322 - x_cent * 0.01955,
                      y_cent < 450
                      ]
        
        if all(conditions):
            label_i = props[i].label
            labels.append(i)
            compMask[label_im == label_i] = True

            contour = measure.find_contours(label_im == label_i)[0]
            y, x = contour.T
    print(f'After filtering: {len(labels)}')
    
    return compMask, props, labels

def dilate_erode(binary_im):
    output_im = morphology.binary_dilation(binary_im, footprint=morphology.disk(3))
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
            im = dilate_erode(im)
            compMask, props, labels = con_comp(im, 2)
            
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
            
            output_im = f[output_im_dset_name][-1, :, :]
            overlay = np.ones_like(output_im)
            overlay = np.ma.masked_where(np.invert(compMask), overlay)
            alphas = np.ones_like(output_im) * 0.6 
            
            scalebar_text = '500 um'
            output_im = cv2.putText(output_im,                             # Original frame
                                    scalebar_text,                  # Text to add
                                    (890, 500),                     # Text origin
                                    cv2.FONT_HERSHEY_DUPLEX,        # Font
                                    0.9,                            # Fontscale
                                    (255, 255, 255),                # Font colour (BGR)
                                    1,                              # Line thickness
                                    cv2.LINE_AA                     # Line type
                                    )
            # Add scalebar to frame
            bar_originx = 889
            bar_originy = 470
            bar_length = int(500/4.3)
            bar_thickness = 4
            output_im = cv2.rectangle(output_im,                                            # Original frame
                                      (bar_originx, bar_originy),                           # Top left corner
                                      (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                                      (255, 255, 255),                                      # Colour (BGR)
                                      -1                                                    # Line thickness (-ve means fill shape inwards)
                                      )
                                      
            fig, ax = plt.subplots()
            ax.imshow(output_im, cmap='gray')
            ax.imshow(overlay, cmap='plasma', vmin=0, vmax=1, alpha=alphas)
            ax.axis('off')
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