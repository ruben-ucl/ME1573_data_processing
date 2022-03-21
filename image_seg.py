import h5py, glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from skimage.morphology import disk, ball

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.4'

'''
CHANGELOG
    v0.1 - Image segmentation tool based on thresholding and filtering
    v0.2 - Restructured for readability and flexibility
    v0.3 - Made it easier to switch between plotting and processing entire datasets with 'mode' field
    v0.4 - Switched to triangle thresholding algorithm
           
INTENDED CHANGES
    - Streamline changing between different filters and kernel dimensions
    
'''
# Input data informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_10_frames'

mode = 'apply' # Set to 'preview' or 'apply' to either preview a single image or apply to the entire dataset

# Iterate through files and datasets to perform filtering and thresholding
def main(mode):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                if mode == 'preview':
                    threshold_im(dset[262, :, :])
                elif mode == 'apply':
                    dset_filt_seg = threshold_timeseries(dset)
                    output_dset_name = 'bg_sub_prev_10_frames_/median_filt_r1_tri-thresh'
                    output_dset = file.require_dataset(output_dset_name, shape=dset.shape, dtype=np.uint8)
                    transfer_attr(dset, output_dset, 'element_size_um')
                    output_dset = dset_filt_seg.astype(np.uint8)
                print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')

def threshold_timeseries(dset):
    # dset_filt = filters.gaussian(dset) * 255
    dset_filt = filters.median(dset, footprint=np.ones((3, 3, 3)))
    thresh = filters.threshold_triangle(dset_filt)
    mask = dset_filt > thresh
    binary = np.zeros_like(dset)
    binary[mask] = 255
    return binary
    
def threshold_im(im):
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
                   
    im_filt = filters.median(im, footprint=np.ones((3, 3)))
    # im_filt = filters.gaussian(im) * 255
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
    
main(mode)