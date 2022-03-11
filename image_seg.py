import h5py, glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters


__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.3'

'''
CHANGELOG
    v0.1 - Image segmentation tool based on thresholding and filtering
    v0.2 - Restructured for readability and flexibility
    v0.3 - Made it easier to switch between plotting and processing entire datasets with 'mode' field
           
INTENDED CHANGES
    - Streamline changing between different filters and kernel dimensions
    
'''

# Set min and max pixel values for segmentation here
segmentation_min = 171
segmentation_max = 255

# Input data informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_10_frames'

mode = 'apply' # Set to 'preview' or 'apply' to either preview a single image or apply to the entire dataset


# Iterate through files and datasets to perform filtering and thresholding
def main(lower, upper, mode):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                if mode == 'preview':
                    threshold_im(dset[262, :, :], lower, upper)
                elif mode == 'apply':
                    dset_filt_seg = threshold_timeseries(dset, lower, upper)
                    file['bg_sub_prev_10_frames_/median_filt_r1_segmented_%s-%s' % (str(lower), str(upper))] = dset_filt_seg
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')

def threshold_timeseries(dset, lower, upper):
    # dset_filt = filters.gaussian(dset) * 255
    dset_filt = filters.median(dset, footprint=np.ones((3, 3, 3)))
    dset_filt_seg = np.zeros_like(dset)
    mask = np.logical_and(dset_filt > lower, dset_filt < upper)
    dset_filt_seg[mask] = 255
    return dset_filt_seg
    
    
def threshold_im(im, lower, upper):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)    # Initialise figure with four subplots
    
    create_subplot(im,
                   ax1,
                   'Flat field correction, background subtraction', cmap='gray',
                   scalebar=True
                   )
                   
    mask = np.logical_and(im > lower, im < upper)
    im_seg = np.zeros_like(im)
    im_seg[mask] = 1
    create_subplot(im_seg,
                   ax2,
                   'Segmented image (%s < pixel value < %s)' % (str(lower), str(upper)),
                   scalebar=True
                   )
                   
    im_filt = filters.median(im, footprint=np.ones((3, 3)))
    # im_filt = filters.gaussian(im) * 255
    create_subplot(im_filt,
                   ax3,
                   'Flat field correction, background subtraction, median filter (radius=3)',
                   cmap='gray',
                   scalebar=True
                   )
                   
    mask = np.logical_and(im_filt > lower, im_filt < upper)
    im_filt_seg = np.zeros_like(im)
    im_filt_seg[mask] = 1
    create_subplot(im_filt_seg,
                   ax4,
                   'Segmented image (%s < pixel value < %s)' % (str(lower), str(upper)),
                   scalebar=True
                   )
                   
    # fig.dpi = 300
    plt.show()

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
 
    
main(segmentation_min, segmentation_max, mode)