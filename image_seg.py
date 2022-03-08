import h5py, glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Image segmentation tool based on thresholding and filtering
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
filepath = 'C:/Users/rlamb/Desktop/ESRF ME1573 Python sandbox/hdf5 test sandbox/0103 AlSi10Mg/'
input_dset_name = 'bg_sub_first_10_frames'

def main(lower, upper):
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                threshold_im(dset[len(dset)//2-5, :, :], lower, upper)
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            
def threshold_im(im, lower, upper):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    
    im_seg = np.zeros_like(im)
    ax1.imshow(im, cmap='gray')
    mask = np.logical_and(im > lower, im < upper)
    im_seg[mask] = 255
    ax2.imshow(im_seg)
    
    im_filt_seg = np.zeros_like(im)
    # im_filt = filters.rank.median(im)
    im_filt = filters.gaussian(im)
    ax3.imshow(im_filt, cmap='gray')
    mask = np.logical_and(im_filt > lower/255, im_filt < upper/255)
    im_filt_seg[mask] = 255
    ax4.imshow(im_filt_seg)
    
    # fig.dpi = 300
    plt.show()
    
main(168, 220)