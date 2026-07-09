import h5py, glob, os, imageio, functools, sys
import numpy as np
from pathlib import Path
from skimage import filters
from skimage.morphology import disk
from skimage import exposure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files and saves max z-projected images from specified image stack datasets
           
INTENDED CHANGES
    - 

'''
print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Input informaton
# Read data folder path from .txt file
filepath = get_paths()['hdf5']
# filepath = r'F:\AlSi10Mg single layer ffc'
    
input_dset_name = 'bs-p1_lagrangian'

frame_reduction_factor = 1                              # Set to 1 to use all frames
filter_radius = 2                                    # Median filter radius, set to None for no filter
norm = True
equalize = True
mode = 'save'                                           # Set to 'view' or 'save'
proj_mode = 'mean'                                      # 'median', 'mean', 'min' or 'max'
skip_frames_end = 'auto'                                    # For Lagrangian videos use 45, set to 0 to use all frames
skip_frames_start = 'auto'

reduction_txt = f'_x{frame_reduction_factor}_stack_reduction' if frame_reduction_factor != 1 else ''
folder_name = f'{input_dset_name}_z_projection_{proj_mode}{reduction_txt}'
folder_path = Path(filepath, 'z-project_images', folder_name)

def main():
    for file in sorted(glob.glob(str(Path(filepath, '*.h*5')))):
        with h5py.File(file, 'a') as f:
            # Extract data and trackid
            trackid = Path(file).name[:-5]
            dset = f[input_dset_name]
            
            if skip_frames_start == 'auto' or skip_frames_end == 'auto':
                fov_length = dset.shape[2]
                dset_flat = np.median(dset, axis=1)
            
            if skip_frames_start == 'auto':
                for i, val in enumerate(dset_flat):
                    if val[0] != 0:
                        i_start = i
                        # print(f'Automaticallly skipping first {i_start} frames')
                        break
            else:
                i_start = skip_frames_start
            
            if skip_frames_end == 'auto':
                for i, val in enumerate(dset_flat[::-1]):
                    if val[-1] != 0:
                        i_end = len(dset_flat) - i
                        # print(f'Automaticallly skipping last {len(dset_flat)-i_end} frames')
                        break
            else:
                i_end = skip_frames_end
            
            if mode == 'save':
                output_filename = f'{trackid}_{input_dset_name}_z_projection_{proj_mode}.png'
                output_filepath = Path(folder_path, output_filename)
                if os.path.exists(output_filepath):
                    print(f'Output file {output_filename} already exists, skipping file.')
                    continue
            
            print(f'Working on {trackid}')
            
            # Reduce frames by specified factor
            print(f'Reducing image stack by factor {frame_reduction_factor}')
            dset_reduced = dset[i_start:i_end:frame_reduction_factor, :, :]
            
            # Apply 2D median filter to each frame
            if filter_radius != None:
                print(f'Applying median filter with radius {filter_radius}')
                dset_filt = np.zeros_like(dset_reduced)
                for i, frame in enumerate(dset_reduced):
                    dset_filt[i, :, :] = filters.median(frame, footprint=disk(filter_radius))
            else:
                dset_filt = dset_reduced
            
            # Create z projection of max pixel values
            print('Creating z-projection')
            if proj_mode == 'max':
                output_im = np.amax(dset_filt, axis=0)
                # output_im = exposure.equalize_hist(output_im)
            elif proj_mode == 'min':
                output_im = np.amin(dset_filt, axis=0)
            elif proj_mode == 'median':
                output_im = np.median(dset_filt, axis=0)
            elif proj_mode == 'mean':
                output_im = np.mean(dset_filt, axis=0)
                
            if norm:
                norm_min = np.percentile(output_im, 0)
                norm_max = np.percentile(output_im, 100)
                img_norm = np.clip((output_im - norm_min) / (norm_max - norm_min), 0, 1)
                output_im = img_norm
                
            if equalize:
                output_im = exposure.equalize_adapthist(output_im, clip_limit=0.02)
            
            # Plot figure
            plt.rcParams.update({'font.size': 8})
            fig, ax = plt.subplots(figsize=(4, 2), dpi=600, tight_layout=True)
            im = ax.imshow(output_im, cmap='grey', vmin=0, vmax=1) # Keyhole default colour bar 110:140
            scalebar = ScaleBar(dx=4.3, units='um', location='lower left', width_fraction=0.02, box_alpha=0)
            plt.gca().add_artist(scalebar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax, ticks=[100, 150])
            cax.set_ylabel('Mean intensity')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.suptitle(trackid)
            
            # Save or display figure
            if mode == 'save':
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                plt.savefig(output_filepath)
                print(f'Image saved to {output_filepath}')
                plt.close()
            else:
                plt.show()
            
        print('Done\n')

if __name__ == "__main__":
	main()