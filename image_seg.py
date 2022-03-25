import h5py, glob, cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from skimage.morphology import disk, ball

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.4.1'

'''
CHANGELOG
    v0.1 - Image segmentation tool based on thresholding and filtering
    v0.2 - Restructured for readability and flexibility
    v0.3 - Made it easier to switch between plotting and processing entire datasets with 'mode' field
    v0.4 - Switched to triangle thresholding algorithm
    v0.4.1 - Moved data folder path storage to text file for ease of copying script to different machines
           
INTENDED CHANGES
    - Streamline changing between different filters and kernel dimensions
    
'''
# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bg_sub_prev_10_frames'

mode = 'preview' # Set to 'preview' or 'apply' to either preview a single image or apply to the entire dataset

filter_mode = 'bilateral'

# Iterate through files and datasets to perform filtering and thresholding
def main(mode, filt):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                if mode == 'preview':
                    threshold_im(dset[262, :, :], filt)
                elif mode == 'apply':
                    output_dset_name = f'{input_dset_name}_/bilat_filt_r1_tri-thresh'
                    output_dset = file.require_dataset(output_dset_name, shape=dset.shape, dtype=np.uint8)
                    dset_filt_seg = threshold_timeseries(dset, filt)
                    transfer_attr(dset, output_dset, 'element_size_um')
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
        dset_filt = filters.rank.median(dset, footprint=np.ones((3, 3, 3)))
    elif filt == 'bilateral':
        print('Applying bilateral filter')
        dset_filt = np.zeros_like(dset)
        for i, im in enumerate(dset):
            im_filt = cv2.bilateralFilter(im, 8, 75, 75)
            dset_filt[i, :, :] = im_filt
    else:
        print('No filter applied')
        dset_filt = dset
    print('Calculating threshold')
    thresh = filters.threshold_triangle(dset_filt)
    print(f'Applying threshold: {thresh}')
    mask = dset_filt > thresh
    binary = np.zeros_like(dset)
    binary[mask] = 255
    print('Removing outliers from binary image')
    binary = remove_outliers(binary)
    return binary
    
def threshold_im(im, filt):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)    # Initialise figure with four subplots
    
    create_subplot(im,
                   ax1,
                   'Flat field correction, background subtraction', cmap='gray',
                   scalebar=True
                   )
                   
    thresh = filters.threshold_triangle(im)
    binary = im > thresh
    create_subplot(binary,
                   ax2,
                   'Triangle threshold',
                   scalebar=True
                   )
    if filt == 'median':   
        im_filt = filters.median(im, footprint=np.ones((3, 3)))
    elif filt == 'gauss':
        im_filt = filters.gaussian(im) * 255
    elif filt == 'bilateral':
        im_filt = cv2.bilateralFilter(im, 8, 75, 75)
    else:
        im_filt = im
    create_subplot(im_filt,
                   ax3,
                   'Flat field correction, background subtraction, median filter (radius=1)',
                   cmap='gray',
                   scalebar=True
                   )
                   
    thresh = filters.threshold_triangle(im_filt)
    binary = im_filt > thresh
    create_subplot(remove_outliers(binary),
                   ax4,
                   'Triangle threshold of filtered image, outliers removed',
                   scalebar=True
                   )
                   
    # fig.dpi = 300
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
    
main(mode, filter_mode)