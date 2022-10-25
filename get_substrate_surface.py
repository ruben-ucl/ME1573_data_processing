import h5py, glob, functools, os
import numpy as np
import pandas as pd
from skimage import filters, feature, measure
import matplotlib.pyplot as plt
from pathlib import Path

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

def get_kernel(x, y):
	return np.ones((y, x), dtype=np.uint8)
    
def median_filter(im, kernel):
    im_filt = filters.rank.median(im, kernel)
    return im_filt
    
def get_edge(im):
    labels = measure.label(im, connectivity=2)
    edges = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'centroid', 'coords', 'feret_diameter_max', 'area')))
    long_edges = edges[edges['feret_diameter_max'] > 400] # Filter out short edges
    long_edges_sorted = long_edges.sort_values('centroid-0', ascending=False, inplace=False, ignore_index=True)
    print(long_edges_sorted.head())
    edge_label = long_edges_sorted.at[0, 'label']
    edge_coords = long_edges_sorted.at[0, 'coords'].transpose()
    m, c = np.polyfit(edge_coords[1], edge_coords[0], 1)
    return m, c, edge_coords # Returns intercept and slope of linear fit, and coordinates of actual edge points.
    
def get_surface(im):
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300)
    
    kernel_width = 81 # default 81
    kernel_height = 5 # default 5
    im_filt = median_filter(im, get_kernel(kernel_width, kernel_height))      # Apply median filter with custom kernel
    
    im_canny = feature.canny(im_filt, sigma=1, mode='nearest')
    
    m, c, edge_coords = get_edge(im_canny)
    ax1.imshow(im_canny, cmap='gray')
    ax1.plot(edge_coords[1], edge_coords[0], 'b', linewidth=0.4)
        
    xx = range(im.shape[1])
    yy = [m * x + c for x in xx]
    
    ax2.imshow(im, cmap='gray')
    ax2.plot(xx, yy, 'b--', linewidth=0.8)
    ax2.text(xx[-1]-400, yy[0]-60, f'y = {round(m, 3)} x + {round(c, 1)}', color='b', size='small')
    
    return m, c, fig
    
def main():
    substrate_surface_dict = {'trackid': [],
                            'm': [],
                            'c': []
                            } # Initialise dataframe for storing surface location equations
                            
    output_folder = Path(filepath, 'substrate_surface_measurements')    # Create folder to store output files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        fname = Path(f).name
        print(f'Reading {fname}')
        trackid = fname[:5] + '0' + fname[-6]
        
        with h5py.File(f, 'a') as file:
            im = np.array(file['ffc_bg_sub_first_40_frames'][-1]) # Take last frame of background subtracted dataset as a 2d array
        
        try:
            m, c, fig = get_surface(im)  # Get slope and left intercept of substrate top surface
            
            print(f'Recorded edge equation: y = {m}x + {c}')
            
            fig.suptitle(f'{trackid} substrate surface detection')
            plt.savefig(Path(output_folder, f'{trackid}_substrate_surface.png'))
            print('Figure saved\n')
            plt.close('all')
        except KeyError:
            print('ERR: Edge not detected\n')
            m = None
            c = None
        
        substrate_surface_dict['trackid'].append(trackid)
        substrate_surface_dict['m'].append(m)
        substrate_surface_dict['c'].append(c)
        
        pd.DataFrame(substrate_surface_dict).to_csv(Path(output_folder, 'substrate_surface_locations.csv'), index=False) # Save numerical results as .csv file
    
if __name__ == "__main__":
	main()