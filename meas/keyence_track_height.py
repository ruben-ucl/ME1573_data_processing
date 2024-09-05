import h5py, glob, functools, os
import numpy as np
import pandas as pd
from skimage import filters, feature, measure, morphology, segmentation
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from my_funcs import get_logbook

'''

'''

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

filepath = r'J:\AlSi10Mg_CW_powder_L1\mono'
    
def main():
    files = glob.glob(str(Path(filepath, '*.tif')))
    for fp in sorted(files):
        trackid, im = read_im(fp)
        
        coords = height_projection(im)
        print(profile.shape)
        
        
def read_im(fp):
    fname = Path(fp).name
    print(f'\nReading {fname}')
    trackid = fname[:7]
    im = np.array(Image.open(fp, mode='r').convert('L'))
    
    return trackid, im
   
def height_projection(im):
    x = np.arange(im.shape[1])
    y = np.argmax(im, axis=0)
    z = np.take_along_axis(im, y, axis=0)
    plt.plot(x, y, z)
    plt.show()
    print(im.shape)
    print(xy_coords.T)
    height = im[xy_coords]
    coords = np.stack(xy_coords, height)
    
    return coords

main()