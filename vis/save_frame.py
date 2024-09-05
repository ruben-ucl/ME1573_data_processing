import h5py, glob, os, imageio, cv2, functools
import numpy as np
from pathlib import Path
from skimage import filters
from my_funcs import printProgressBar

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files and saves the spcified frame number from each as a still image
    v0.2 - Added function to apply threshold to frame before saving
           
INTENDED CHANGES
    - 
    
'''
# print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

# Input informaton
# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bs-f40'
frame_no = -1
folder_name = f'{input_dset_name}_frame_{frame_no}_stills'

add_scalebar = True
text_colour = 'white'

folder_path = Path(filepath, 'still_frames', folder_name)

make_binary = False  # Set to True to threshold frame using triangle algorithm

def create_overlay(im):
    if text_colour == 'black':
        bgr_colour = (0, 0, 0)
    elif text_colour == 'white':
        bgr_colour = (255, 255, 255)
    # Add scale bar text to frame
    scalebar_text = '500 um'
    im = cv2.putText(im,                      # Original frame
                     scalebar_text,                  # Text to add
                     (890, 500),                     # Text origin
                     cv2.FONT_HERSHEY_DUPLEX,        # Font
                     0.9,                            # Fontscale
                     bgr_colour,                     # Font colour (BGR)
                     1,                              # Line thickness
                     cv2.LINE_AA                     # Line type
                     )
    # Add scalebar to frame
    bar_originx = 889
    bar_originy = 470
    bar_length = int(500/4.3)
    bar_thickness = 4
    im = cv2.rectangle(im,                                            # Original frame
                       (bar_originx, bar_originy),                           # Top left corner
                       (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                       bgr_colour,                                           # Colour (BGR)
                       -1                                                    # Line thickness (-ve means fill shape inwards)
                       )
    return im

def main():
    print(f'Saving frame no. {frame_no} from dataset: {input_dset_name}\n')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    files = sorted(glob.glob(str(Path(filepath, '*.hdf5'))))
    n_files = len(files)
    for i, file in enumerate(files):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            im = dset[frame_no]
            trackid = Path(file).name[:-5]
            # print(f'Saving {trackid} frame {frame_no}')
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
                
            if add_scalebar == True:
                im = create_overlay(im)
            
            output_filepath = Path(folder_path, output_filename)
            imageio.imwrite(output_filepath, im)
            printProgressBar(i+1, n_files, prefix=f'Working on {trackid} | Total progress:', length=50)
    print('Done                                                      \n')

if __name__ == "__main__":
	main()