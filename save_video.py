import h5py, glob, os, cv2, time
import numpy as np
from skimage import morphology
from pathlib import Path
from my_funcs import *

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v1.0'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files in specified path and saves the specified dataset
           from each to a .mp4 video file with a timestamp and scalebar on each frame
    v0.2 - Added pore tracking function using cv2 to run connected component analysis - needs optimisation
    v1.0 - Removed connected component analysis functionalty to a seperate script.
           Generates videos with overlay and optional outlining of features defined by a second, corresponding binary input dataset

INTENDED CHANGES
    - Process list of datasets instead of one at a time
    - Replace cv2.putText() with a method that supports unicode characters
'''
"""Controls"""

input_dset_name = 'bs-p5-s5_lagrangian_meltpool'

binary_dset_name = None
binary_overlay_mode = 'outline'     # 'outline' or 'fill'
overlay_suffix = '_KH-overlay' if binary_dset_name != None else ''

output_name = f'{input_dset_name}{overlay_suffix}'

capture_framerate = 40000 # fps
output_framerate = 30 # fps
text_colour = 'white'   # 'black' or 'white'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
def main():
    print(f'Creating videos from dataset: {input_dset_name}')
    for file in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
        fname = Path(file).name
        trackid = fname[:5] + '0' + fname[-6]
        with h5py.File(file, 'a') as f:
            dset = np.array(f[input_dset_name])
            try:
                binary_dset = np.array(f[binary_dset_name])   # Get binary dataset of same dimensions as input to use as coloured overlay in output video
                if binary_overlay_mode == 'outline':
                    binary_dset = get_perimeter(binary_dset, 1)
                isRGB = True                        # If a binary dataset is enabled for a colour overlay, set to save vid in RGB
                start_frame_offset = len(dset) - len(binary_dset)
                dset = dset[start_frame_offset:]            # trim input dset to match length of ginary dset so frames match up
            except TypeError:
                binary_dset = None
                isRGB = False                       # If no colour overlay, set to save vid in greyscale
        fileext = '.mp4'
        vid_filename = f'{trackid}_{output_name}{fileext}'
        output_folder = 'videos'
        create_video_from_dset(dset, vid_filename, output_folder, binary_dset, isRGB)

def get_perimeter(dset, weight):
    perimeter_mask = np.zeros_like(dset)
    for i, frame in enumerate(dset):
        eroded = morphology.binary_erosion(frame, footprint=morphology.disk(weight))
        dilated = morphology.binary_dilation(frame, footprint=morphology.disk(weight))
        perimeter_i = np.not_equal(dilated, eroded)
        perimeter_mask[i, :, :][perimeter_i] = 255
    return perimeter_mask

def create_video_from_dset(dset, vid_filename, output_folder, binary_dset=None, isRGB=False, overlay=True):
    n_frames = len(dset)
    output_path = Path(filepath, output_folder, output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    frame_size = (dset.shape[-1], dset.shape[-2])
    output_filepath = Path(output_path, vid_filename)
    out = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*'mp4v'), output_framerate , frame_size, isRGB) # Add argument False to switch to greyscale
    for i, frame in enumerate(dset):
        if binary_dset is not None:
            binary_im = binary_dset[i, :, :]
            con_comp_box_mask = binary_im == 255
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame[con_comp_box_mask] = (0, 255, 0)
        if overlay == True:
            frame = create_overlay(i, frame)
        out.write(frame)
        printProgressBar(i + 1, n_frames, prefix=vid_filename[:7], suffix='Complete', length=50)
    out.release()

def create_overlay(i, frame, fontscale=1, scalebar_length=200, scale_txt_from_right=160, bottom_margin=20, left_margin=20):
    '''
    For full frame video:
    fontscale = 1
    scalebar_length = 200
    scale_txt_from_right = 160
    bottom_margin = 20
    left_margin = 20
    
    For Lagrangian cropped video:
    fontscale = 0.5
    scalebar_length = 100
    scale_txt_from_right = 70
    bottom_margin = 10
    left_margin = 5
    '''
    if text_colour == 'black':
        bgr_colour = (0, 0, 0)
    elif text_colour == 'white':
        bgr_colour = (255, 255, 255)
    timestamp = str(format(i * 1/capture_framerate * 1000, '.3f')) + ' ms'        # in milliseconds
    height = frame.shape[0]
    width = frame.shape[1]
    bottom_left = (left_margin, height-bottom_margin)
    bottom_right = (width-scale_txt_from_right, height-bottom_margin)
    # Add timestamp to frame
    new_frame = cv2.putText(frame,                          # Original frame
                            timestamp,                      # Text to add
                            bottom_left,                       # Text origin
                            cv2.FONT_HERSHEY_DUPLEX,        # Font
                            fontscale,                            # Fontscale
                            bgr_colour,                     # Font colour (BGR)
                            1,                              # Line thickness
                            cv2.LINE_AA                     # Line type
                            )
    # Add scale bar text to frame
    scalebar_text = f'{str(scalebar_length)} um'
    new_frame = cv2.putText(new_frame,                      # Original frame
                            scalebar_text,                  # Text to add
                            bottom_right,                   # Text origin
                            cv2.FONT_HERSHEY_DUPLEX,        # Font
                            fontscale,                      # Fontscale
                            bgr_colour,                     # Font colour (BGR)
                            1,                              # Line thickness
                            cv2.LINE_AA                     # Line type
                            )
    # Add scalebar to frame
    bar_length = int(scalebar_length/4.3)
    scale_bar_h_offset = int((scale_txt_from_right - bar_length) / 2 - left_margin // 2)
    bar_originx = bottom_right[0] + scale_bar_h_offset
    bar_originy = bottom_right[1] + 10
    bar_thickness = 3
    new_frame = cv2.rectangle(new_frame,                                            # Original frame
                              (bar_originx, bar_originy),                           # Top left corner
                              (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                              bgr_colour,                                           # Colour (BGR)
                              -1                                                    # Line thickness (-ve means fill shape inwards)
                              )
    return new_frame

if __name__ == "__main__":
	main()
