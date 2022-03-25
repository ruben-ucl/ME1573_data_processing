import h5py, glob, cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from skimage.morphology import disk, ball

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Tool for looking at the influence of bilateral filtering parameters on image segmentation
           
INTENDED CHANGES
    - 
    
'''
# Input data informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_10_frames'

kernel_diameters = [5, 10, 15]
sigma_values = [10, 30, 60]

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        with h5py.File(f, 'a') as file:
            dset = file[input_dset_name]
            print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
            create_plot(dset[262, :, :])
            print('Done\n')
  
def create_plot(im, threshold=True):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)    # Initialise figure with nine subplots
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            kernel_dia = kernel_diameters[j]
            sigma_c = sigma_values[i]
            sigma_s = 75
            im_filt = cv2.bilateralFilter(im, kernel_dia, sigma_c, sigma_s)
            if threshold == True:
                thresh = filters.threshold_triangle(im_filt)
                mask = im_filt > thresh
                binary = np.zeros_like(im_filt)
                binary[mask] = 255
                display_im = binary
            else:
                display_im = im_filt
            ax.imshow(display_im, cmap='gray')
            ax.axis('off')
            ax.set_title(f'dia = {kernel_dia}, sigma_c = {sigma_c}, sigma_s = {sigma_s}')
            
    plt.show()
    
main()