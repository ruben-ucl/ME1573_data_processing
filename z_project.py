import h5py, glob, os, imageio
import numpy as np
from pathlib import Path
from skimage import filters
from skimage.morphology import disk
from skimage import exposure
import matplotlib.pyplot as plt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files and saves max z-projected images from specified image stack datasets
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bg_sub_first_30_frames'

frame_reduction_factor = 1                               # Set to 1 to use all frames
filter_radius = 3
mode = 'save'                                            # Set to 'view' or 'save'

folder_name = f'{input_dset_name}_z_projection_images_stack_reduction_x{frame_reduction_factor}_t'
folder_path = Path(filepath, 'z-project_images', folder_name)

def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            trackid = Path(file).name[:-5]
            if mode == 'save':
                output_filename = f'{trackid}_{input_dset_name}_z_projection_max.png'
                output_filepath = Path(folder_path, output_filename)
                if os.path.exists(output_filepath):
                    print(f'Output file {output_filename} already exists, skipping file.')
                    continue
            print(f'Working on {trackid}')
            
            # Reduce frames by specified factor
            print(f'Reducing image stack by factor {frame_reduction_factor}')
            dset_reduced = dset[::frame_reduction_factor, :, :]
            
            # Apply 2D median filter to each frame
            print(f'Applying median filter with radius {filter_radius}')
            dset_filt = np.zeros_like(dset_reduced)
            for i, frame in enumerate(dset_reduced):
                dset_filt[i, :, :] = filters.median(frame, footprint=disk(filter_radius))
                
            # Create z projection of max pixel values
            print('Creating z-projection max image')
            output_im = np.amax(dset_filt, axis=0)
            # output_im = exposure.adjust_gamma(output_im, 3)
            output_im = exposure.equalize_hist(output_im)
            
            if mode == 'save':
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                imageio.imwrite(output_filepath, output_im)
                print(f'Image saved to {output_filepath}')
            else:
                plt.imshow(output_im, cmap='gray')
                plt.show()
                
        print('Done\n')

if __name__ == "__main__":
	main()