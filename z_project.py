import h5py, glob, os, imageio, functools
import numpy as np
from pathlib import Path
from skimage import filters
from skimage.morphology import disk
from skimage import exposure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar

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
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bs-f40_lagrangian_meltpool'

frame_reduction_factor = 1                              # Set to 1 to use all frames
filter_radius = None                                    # Median filter radius, set to None for no filter
mode = 'save'                                        # Set to 'view' or 'save'
proj_mode = 'mean'                                       # 'median', 'mean' or 'max'
skip_frames_end = 45                                    # For Lagrangian videos use 45, set to 0 to use all frames
skip_start_frames = 40

reduction_txt = f'_x{frame_reduction_factor}_stack_reduction' if frame_reduction_factor != 1 else ''
folder_name = f'{input_dset_name}_z_projection_{proj_mode}{reduction_txt}'
folder_path = Path(filepath, 'z-project_images', folder_name)

def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name][skip_start_frames:-skip_frames_end]
            trackid = Path(file).name[:-5]
            if mode == 'save':
                output_filename = f'{trackid}_{input_dset_name}_z_projection_{proj_mode}.png'
                output_filepath = Path(folder_path, output_filename)
                if os.path.exists(output_filepath):
                    print(f'Output file {output_filename} already exists, skipping file.')
                    continue
            print(f'Working on {trackid}')
            
            # Reduce frames by specified factor
            print(f'Reducing image stack by factor {frame_reduction_factor}')
            dset_reduced = dset[::frame_reduction_factor, :, :]
            
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
            elif proj_mode == 'median':
                output_im = np.median(dset_filt, axis=0)
            elif proj_mode == 'mean':
                output_im = np.mean(dset_filt, axis=0)
            
            # Plot figure
            plt.rcParams.update({'font.size': 8})
            fig, ax = plt.subplots(figsize=(4, 2), dpi=600, tight_layout=True)
            im = ax.imshow(output_im, cmap='viridis', vmin=110, vmax=140)
            scalebar = ScaleBar(dx=4.3, units='um', location='lower left', width_fraction=0.02)
            plt.gca().add_artist(scalebar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax)
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
            else:
                plt.show()
            
        print('Done\n')

if __name__ == "__main__":
	main()