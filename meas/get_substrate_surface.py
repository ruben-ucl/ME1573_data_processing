import h5py, glob, functools, os
import numpy as np
import pandas as pd
from skimage import filters, feature, measure
import matplotlib.pyplot as plt
from pathlib import Path

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

filepath = get_paths()['hdf5']

show_figs = True
save_figs= True
save_measurements = True

def get_kernel(x, y):
	return np.ones((y, x), dtype=np.uint8)

def median_filter(im, kernel):
    im_filt = filters.rank.median(im, kernel)
    return im_filt

def get_edge_fit(im):
    labels = measure.label(im, connectivity=2)
    edges = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'centroid', 'coords', 'feret_diameter_max', 'area')))
    long_edges = edges[edges['feret_diameter_max'] > 350] # Filter out short edges
    long_edges_sorted = long_edges.sort_values('centroid-0', ascending=False, inplace=False, ignore_index=True)
    print(long_edges_sorted.head())
    edge_label = long_edges_sorted.at[1, 'label']
    edge_coords = long_edges_sorted.at[1, 'coords'].transpose()
    m, c = np.polyfit(edge_coords[1], edge_coords[0], 1)
    return m, c, edge_coords # Returns intercept and slope of linear fit, and coordinates of actual edge points.

def get_substrate_edge(im, trackid, output_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))
    
    kernel_width = 201 # default 81
    kernel_height = 3 # default 5
    im_filt = median_filter(im, get_kernel(kernel_width, kernel_height))      # Apply median filter with custom kernel    
    im_canny = feature.canny(im_filt, sigma=1, mode='nearest')
    mask = np.zeros_like(im_canny, dtype=bool)
    mask[:, 100:-100] = True
    im_canny_masked = im_canny
    im_canny_masked[np.invert(mask)] = 0
    
    m, c, edge_coords = get_edge_fit(im_canny_masked)

    ax1.imshow(im_filt, cmap='gray')
    ax2.imshow(im_canny, cmap='gray')
    ax2.scatter(edge_coords[1], edge_coords[0], c='b', s=0.4)
    
    xx = range(im.shape[1])
    yy = [m * x + c for x in xx]
    
    ax3.imshow(im, cmap='gray')
    ax3.plot(xx, yy, 'b--', linewidth=0.8)
    ax3.text(xx[-1]-400, yy[0]-60, f'y = {round(m, 3)} x + {round(c, 1)}', color='b', size='small')
    
    fig.suptitle(f'{trackid} substrate surface detection')
    
    if show_figs:
        plt.show()
    if save_figs:
        plt.savefig(Path(output_folder, f'{trackid}_substrate_surface_detection.png'))
        print('Figure saved')
    
    plt.close('all')
    
    return m, c

def main():
    substrate_surface_dict = {'trackid': [],
                            'm': [],
                            'c': []
                            } # Initialise dataframe for storing surface location equations
                            
    output_folder = Path(filepath, 'substrate_surface_measurements')    # Create folder to store output files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = glob.glob(str(Path(filepath, '*.h*5')))
    for f in sorted(files):
        fname = Path(f).name
        print(f'\nReading {fname}')
        trackid = fname[:7]
        
        with h5py.File(f, 'a') as file:
            im = np.array(file['bs-f40'][-1]) # Take last frame of background subtracted dataset as a 2d array
        
        try:
            m, c = get_substrate_edge(im, trackid, output_folder)  # Get slope and left intercept of substrate top surface
            print(f'Recorded edge equation: y = {m}x + {c}')
        except KeyError:
            print('ERR: Edge not detected\n')
            m = None
            c = None
        
        substrate_surface_dict['trackid'].append(trackid)
        substrate_surface_dict['m'].append(m)
        substrate_surface_dict['c'].append(c)
        
        if save_measurements:
            pd.DataFrame(substrate_surface_dict).to_csv(Path(output_folder, 'substrate_surface_locations.csv'), index=False) # Save numerical results as .csv file

if __name__ == "__main__":
	main()