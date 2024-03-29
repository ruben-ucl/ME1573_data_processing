import h5py, glob, cv2, functools
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from skimage.morphology import disk, ball

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.5'
'''
CHANGELOG
    v0.1 - Image segmentation tool based on thresholding and filtering
    v0.2 - Restructured for readability and flexibility
    v0.3 - Made it easier to switch between plotting and processing entire datasets with 'mode' field
    v0.4 - Switched to triangle thresholding algorithm
    v0.4.1 - Moved data folder path storage to text file for ease of copying script to different machines
    v0.5 - Added Li thresholding
           
INTENDED CHANGES
    - Streamline changing between different filters and kernel dimensions
    
'''
print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bg_sub_prev_5_frames_crop_rotate_denoised'

mode = 'apply'              # Set to 'preview' or 'apply' to either preview a single image or apply to the entire dataset
thresh_mode = 'frame'       # 'frame' or 'global'
filter_mode = None          # none, 'median', 'gauss' or 'bilateral'
filter_radius = 3           # pixels
threshold_mode = 'triangle' # 'triangle' or 'li'

# Iterate through files and datasets to perform filtering and thresholding
def main(mode, filt):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('Shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                if mode == 'preview':
                    threshold_im(dset[200, :, :], filt)
                elif mode == 'apply':
                    if filter_mode != None:
                        output_dset_name = f'{input_dset_name}_/{filter_mode}_filt_r{filter_radius}_{threshold_mode}-thresh'
                    else:
                        output_dset_name = f'{input_dset_name}_/{threshold_mode}-thresh'
                    print(f'Output dataset name: {output_dset_name}')
                    output_dset = file.require_dataset(output_dset_name, shape=dset.shape, dtype=np.uint8)
                    dset_filt_seg = threshold_timeseries(dset, filt)
                    # transfer_attr(dset, output_dset, 'element_size_um')
                    output_dset[:, :, :] = dset_filt_seg
                print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')

def threshold_timeseries(dset, filt):
    if filt == 'gauss':
        print('Applying Gaussian filter')
        dset_filt = filters.gaussian(dset) * 255
    elif filt == 'median':
        print('Applying median filter')
        dset_filt = filters.rank.median(dset, footprint=get_kernel(dset, filter_radius))
    elif filt == 'bilateral':
        print('Applying bilateral filter')
        dset_filt = np.zeros_like(dset)
        for i, im in enumerate(dset):
            im_filt = cv2.bilateralFilter(im, filter_radius, 75, 75)
            dset_filt[i, :, :] = im_filt
    else:
        print('No filter applied')
        dset_filt = np.array(dset)
    
    binary = np.zeros_like(dset)
    if thresh_mode == 'global':
        print('Calculating threshold')
        if threshold_mode == 'triangle':
            thresh = filters.threshold_triangle(dset_filt)
        elif threshold_mode == 'li':
            thresh = filters.threshold_li(dset_filt, initial_guess=170)
        print(f'Applying {threshold_mode} threshold: {thresh}')
        mask = dset_filt > thresh
        binary[mask] = 255
    else:
        for i, im in enumerate(dset_filt):
            if threshold_mode == 'triangle':
                thresh = filters.threshold_triangle(im)
            elif threshold_mode == 'li':
                thresh = filters.threshold_li(im, initial_guess=170)
            mask = im > thresh
            binary[i][mask] = 255
            
    print('Removing outliers from binary image')
    binary = remove_outliers(binary)
    return binary
    
def threshold_im(im, filt):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)    # Initialise figure with three subplots
    
    create_subplot(im,
                   ax1,
                   'Flat field correction, background subtraction', cmap='gray',
                   scalebar=True
                   )
                   
    thresh = filters.threshold_triangle(im)
    binary = im > thresh
    if filt == 'median':   
        im_filt = filters.median(im, footprint=get_kernel(im, filter_radius))
    elif filt == 'gauss':
        im_filt = filters.gaussian(im) * 255
    elif filt == 'bilateral':
        im_filt = cv2.bilateralFilter(im, filter_radius, 75, 75)
    else:
        im_filt = im
    create_subplot(im_filt,
                   ax2,
                   f'{filter_mode} filter (radius={filter_radius})',
                   cmap='gray',
                   scalebar=True
                   )
    if threshold_mode == 'triangle':               
        thresh = filters.threshold_triangle(im_filt)
    elif threshold_mode == 'li':
        thresh = filters.threshold_li(im_filt, initial_guess=170)
    binary = im_filt > thresh
    create_subplot(remove_outliers(binary),
                   ax3,
                   f'{threshold_mode} threshold of filtered image, outliers removed',
                   scalebar=True
                   )
                   
    # fig.dpi = 600
    plt.show()
    
def remove_outliers(image, radius=2, threshold=0.5):
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    median_filtered = filters.median(image, footprint)
    outliers = (
        (image > median_filtered + threshold)
        | (image < median_filtered - threshold)
    )
    output = np.where(outliers, 0, image)
    return output

def create_subplot(im, ax, title, cmap=None, scalebar=False):
    ax.set_title(title)
    if scalebar == True:
        scalebar = ScaleBar(4.6,
                            "um",
                            length_fraction=0.2,
                            box_alpha=1.0,
                            color='black',
                            fixed_value=500,
                            location='lower right',
                            border_pad=-0.1,
                            # font_properties={'weight':'semibold'}
                            )
        ax.add_artist(scalebar)
        ax.axis('off')
    ax.imshow(im, cmap=cmap)

def transfer_attr(dset_1, dset_2, attr):    # Copy attribute from dset_1 to dset_2
    data = dset_1.attrs.get(attr)
    dset_2.attrs.create(attr, data)

def get_kernel(data, radius):
    footprint_function = disk if data.ndim == 2 else ball
    filter_kernel = footprint_function(radius=filter_radius)
    return filter_kernel

main(mode, filter_mode)