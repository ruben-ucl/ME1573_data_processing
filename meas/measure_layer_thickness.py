import h5py, glob, functools, os, sys
import numpy as np
import pandas as pd
from skimage import filters, feature, measure, morphology, segmentation
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook

filepath = get_paths()['hdf5']

'''

'''

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

show_figs = False
save_figs = True
save_measurements = True

# Define and create output folder if necessary
output_folder = Path(filepath, 'layer_thickness_measurements')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read logbook and previous measurements into memory
logbook = get_logbook()
output_filepath = Path(output_folder, 'powder_layer_thickness.csv')
if os.path.exists(output_filepath):
    output_dict = pd.read_csv(output_filepath, index_col=0)
else:
    output_dict = pd.DataFrame(index=list(logbook['trackid']), columns=['layer_thickness_um'])
    print(output_dict)

def median_filter(im, x, y):
    """
    Simple function for applying a median filter to an image using an x by y rectangular kernel
    ...
    im : 2D array
        The image to be filtered as a 2 dimensional numpy array
    x, y : int
        Horizontal and veritical dimensions of the rectangular filter kernel
    ...
    Returns : filtered image of the same size as the input image
    """
    im_filt = filters.rank.median(im, np.ones((y, x), dtype=np.uint8))
    return im_filt

def main():
    # Iterate through files
    files = glob.glob(str(Path(filepath, '*.h*5')))
    for f in sorted(files):
        fname = Path(f).name
        print(f'\nReading {fname}')
        trackid = fname[:7]
        
        # Skip welding cases
        welding = logbook['Powder material'] == 'None'
        log_powder = logbook[np.invert(welding)]
        if trackid not in list(log_powder['trackid']):
            continue
            
        # if trackid != '0103_04':
            # continue
        
        with h5py.File(f, 'a') as file:
            im = np.array(file['bs-f40'][-1]) # Take last frame of background subtracted dataset as a 2d array
            im_0 = np.array(file['ff_corrected'][0])
        
        # Median filter to homogenize brightness of powder layer 
        kernel_width = 100
        kernel_height = 5
        im_filt = median_filter(im, kernel_width, kernel_height)  
        
        # Apply threshold
        thresh = filters.threshold_triangle(im_filt) + 15
        im_thresh = np.zeros_like(im)
        im_thresh[im_filt > thresh] = 255
        
        im_filt_inv = np.invert(im_filt)
        thresh_inv = filters.threshold_triangle(im_filt_inv) + 40
        im_thresh[im_filt_inv > thresh_inv] = 255
        
        im_thresh_orig = im_thresh
        
        # Crop to the central portion of the frame
        edge_mask = np.zeros_like(im).astype(bool)
        edge_mask[:, 100:-100] = True
        im_thresh[np.invert(edge_mask)] = 0
        
        # Refine segmentation
        im_thresh = morphology.binary_closing(im_thresh, footprint=morphology.disk(2))
        im_thresh = morphology.convex_hull_object(im_thresh)
        
        # Get connected component properties and extract longest component - corresponding to powder layer
        labels = measure.label(im_thresh, connectivity=2)
        props_df = pd.DataFrame(measure.regionprops_table(labels, properties=('label',
                                                                              'axis_major_length',
                                                                              'area'
                                                                              )))
        props_df.sort_values('axis_major_length', ascending=False, inplace=True, ignore_index=True)
        powder_thickness = None
        powder_label = None
        for i in props_df.index:
            if props_df.at[i, 'area'] < 50000:
                powder_label = props_df.at[i, 'label']
                powder_thickness = props_df.at[i, 'area']/props_df.at[i, 'axis_major_length'] * 4.3
                break
        result_print = f'Layer thickness = {round(powder_thickness, 1)} um' if powder_thickness != None else 'Measurment failed'
        print(result_print)
        
        # Crop out edges of the image
        powder_mask = np.zeros_like(im).astype(bool)
        powder_mask[labels == powder_label] = True
        im_thresh[np.invert(powder_mask)] = 0
        
        # Save result to output DataFrame
        output_dict.loc[trackid] = powder_thickness
        
        # Show/save figure depending on options
        if show_figs == True or save_figs == True:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10), dpi=300)
            fig.suptitle(trackid)
            im_outlined = segmentation.mark_boundaries(im_0, im_thresh)
            
            ax1.imshow(im_filt, cmap='gray')
            ax2.imshow(im_thresh_orig, cmap='gray')
            ax3.imshow(im_outlined, cmap='gray')
        
            if show_figs == True:
                plt.show()
            
            if save_figs == True:
                plt.savefig(Path(output_folder, f'{trackid}_powder_layer_detection.png'))
            
            plt.close('all')
    
    # Save measurements to .csv if set in options 
    if save_measurements == True:
        pd.DataFrame(output_dict).to_csv(output_filepath)
        
if __name__ == "__main__":
	main()