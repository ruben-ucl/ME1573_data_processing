import h5py, glob, os, sys
import numpy as np
import pandas as pd
from skimage import filters, feature, measure
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

filepath = get_paths()['hdf5']

def get_kernel(x, y):
	return np.ones((y, x), dtype=np.uint8)
	
def get_image(dset_name, frame_n, file_n=0):
    with h5py.File(glob.glob(str(Path(filepath, '*.hdf5')))[file_n]) as file: # Open first HDF5 file in 'filepath'
        im = np.array(file[dset_name][frame_n])
    return im
    
def median_filter(im, kernel):
    im_filt = filters.rank.median(im, kernel)
    return im_filt
    
def show_im(im1, im2):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(im1, cmap='gray')
    ax2.imshow(im2, cmap='gray')
    plt.show()
    
def get_edge(im):
    labels = measure.label(im, connectivity=2)
    props_df = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'centroid', 'coords')))
    props_df.sort_values('centroid-0', ascending=False, inplace=True, ignore_index=True)
    edge_label = props_df.at[2, 'label']
    edge_coords = props_df.at[2, 'coords'].transpose()
    edge_mask = labels == edge_label
    m, c = np.polyfit(edge_coords[1], edge_coords[0], 1)
    return edge_mask, m, c, edge_coords
    
def main():
    im = get_image('ffc_bg_sub_first_40_frames', -1)    # Load last frame of dataset
    fig = plt.figure(dpi=300)
    kernel_width = 81 # default 81
    kernel_height = 5 # default 5
    im_filt = median_filter(im, get_kernel(kernel_width, kernel_height))      # Apply median filter with custom kernel
    plt.imshow(im_filt[200:350, 400:800], cmap='gray')
    kernel_box_x = [20, 20+kernel_width, 20+kernel_width, 20, 20]
    kernel_box_y = [20, 20, 20+kernel_height, 20+kernel_height, 20]
    plt.plot(kernel_box_x, kernel_box_y, color='black', linewidth=0.8)
    plt.text(30+kernel_width, 29, f'{kernel_width} x {kernel_height} kernel')
    plt.axis('off')
    plt.show()
    
    im_canny = feature.canny(im_filt, sigma=2, low_threshold=10)
    fig = plt.figure(dpi=300)
    canny_overlay = 0.7 * im_filt
    canny_overlay[im_canny] = 255
    # plt.imshow(canny_overlay, cmap='gray')
    # plt.show()
    
    im_edge, m, c, edge_coords = get_edge(im_canny)
    plt.imshow(canny_overlay, cmap='gray')
    plt.plot(edge_coords[1], edge_coords[0], 'b', linewidth=0.4)
    plt.show()
    
    xx = range(im.shape[1])
    yy = [m * x + c for x in xx]
    
    
    fig = plt.figure(dpi=300)
    plt.imshow(get_image('ffc_bg_sub_first_40_frames', 500), cmap='gray')
    plt.plot(xx, yy, 'b--', linewidth=1)
    plt.text(xx[-1]-350, yy[0]-40, f'y = {round(m, 3)} x + {round(c, 1)}', color='b')
    plt.show()
    
if __name__ == "__main__":
	main()