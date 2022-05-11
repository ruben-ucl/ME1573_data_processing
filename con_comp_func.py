import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure
from skimage.draw import ellipse
from skimage.transform import rotate

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.2'

"""
CHANGELOG
    v0.1 - Connected component analysis of single binary image using opencv
    v0.2 - Switched from opencv to scikit-image connected component analysis for better compatibilty
           
INTENDED CHANGES
    - 
"""

def con_comp(im, connectivity):
    # Get labels and region properties using skimage connected component analysis functions
    label_im, num_labels = measure.label(im, return_num = True, connectivity=connectivity)
    print(f'num_labels = {num_labels}')
    props = measure.regionprops(label_im)
    
    compMask = np.zeros_like(im, dtype=bool)
    
    for i in range(num_labels):
        area_i = props[i].area
        print(f'Area {i} = {area_i}') 
        
        conditions = [area_i < 10000]
        
        if all(conditions):
            label_i = props[i].label
            compMask[label_im == label_i] = True

            contour = measure.find_contours(label_im == label_i)[0]
            y, x = contour.T
     
    return compMask, props
    
# Test image
image = np.zeros((600, 600))
rr, cc = ellipse(300, 350, 100, 220)
image[rr, cc] = 1
image = rotate(image, angle=15, order=0)
rr, cc = ellipse(100, 100, 60, 50)
image[rr, cc] = 1

compMask, props = con_comp(image, 2)

overlay = np.ones_like(image)
overlay = np.ma.masked_where(np.invert(compMask), overlay)

alphas = np.ones_like(image) * 0.75 

plt.imshow(image, cmap='gray')
plt.imshow(overlay, cmap='plasma', vmin=0, vmax=1, alpha=alphas)
plt.show()