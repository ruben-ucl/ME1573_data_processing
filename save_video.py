import h5py, glob, os, cv2
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files in specified path and saves the specified dataset
           from each to a .mp4 video file with a timestamp and scalebar on each frame
           
INTENDED CHANGES
    - Process list of datasets instead of one at a time
    - Replace cv2.putText() with a method that supports unicode characters
    
'''
# Input informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_20_frames_/median_filt_r1_tri-thresh'
output_name = 'bg_sub_prev_20_frames_median_filt_r1_tri-thresh'
capture_framerate = 40000
output_framerate = 30
text_colour = 'black'   # 'black' or 'white'

def main():
    print(f'Creating videos from dataset: {input_dset_name}\n')
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            trackid = Path(file).name[:-5]
            print(trackid)
            fileext = '.mp4'
            vid_filename = f'{trackid}_{output_name}{fileext}'
            output_folder = 'videos'
            create_video_from_dset(dset, vid_filename, output_folder)
            print('Done\n')

def create_video_from_dset(dset, vid_filename, output_folder, overlay=True):
    output_path = Path(filepath, output_folder, output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    frame_size = (dset.shape[-1], dset.shape[-2])
    output_filepath = Path(output_path, vid_filename)
    out = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*'mp4v'), output_framerate , frame_size, False)
    for i, frame in enumerate(dset):
        if overlay == True:
            frame = create_overlay(i, frame)
        out.write(frame)
    out.release()

def create_overlay(i, frame):
    if text_colour == 'black':
        bgr_colour = (0, 0, 0)
    elif text_colour == 'white':
        bgr_colour = (255, 255, 255)
    timestamp = str(format(i * 1/capture_framerate * 1000, '.3f')) + ' ms'        # in milliseconds
    # Add timestamp to frame
    new_frame = cv2.putText(frame,                          # Original frame
                            timestamp,                      # Text to add
                            (10, 32),                       # Text origin
                            cv2.FONT_HERSHEY_DUPLEX,        # Font
                            0.9,                            # Fontscale
                            bgr_colour,                     # Font colour (BGR)
                            1,                              # Line thickness
                            cv2.LINE_AA                     # Line type
                            )
    # Add scale bar text to frame
    scalebar_text = '500 um'
    new_frame = cv2.putText(new_frame,                      # Original frame
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
    new_frame = cv2.rectangle(new_frame,                                            # Original frame
                              (bar_originx, bar_originy),                           # Top left corner
                              (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                              bgr_colour,                                           # Colour (BGR)
                              -1                                                    # Line thickness (-ve means fill shape inwards)
                              )
    return new_frame
    
if __name__ == "__main__":
	main()