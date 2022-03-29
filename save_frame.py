import h5py, glob, os, imageio
import numpy as np
from pathlib import Path
from skimage import filters

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files and saves the spcified frame number from each as a still image
    v0.2 - Added function to apply threshold to frame before saving
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bg_sub_first_30_frames'
frame_no = -1
folder_name = f'{input_dset_name}_frame_{frame_no}_stills'

folder_path = Path(filepath, 'still_frames', folder_name)

make_binary = True  # Set to True to threshold frame using triangle algorithm

def main():
    print(f'Saving frame no. {frame_no} from dataset: {input_dset_name}\n')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            im = dset[frame_no]
            trackid = Path(file).name[:-5]
            print(f'Saving {trackid} frame {frame_no}')
            output_filename = f'{trackid}_{input_dset_name}_frame_{frame_no}.png'
            
            if make_binary == True:
                output_filename = f'{output_filename[:-4]}_li-thresh{output_filename[-4:]}'
                im_filt = filters.rank.median(im, footprint=np.ones((7, 7)))
                thresh = filters.threshold_li(im_filt, initial_guess=170)
                print(f'Applying threshold: {thresh}')
                mask = im_filt > thresh
                binary = np.zeros_like(im)
                binary[mask] = 255
                im = binary
                
            output_filepath = Path(folder_path, output_filename)
            imageio.imwrite(output_filepath, im)
        print('Done\n')

if __name__ == "__main__":
	main()