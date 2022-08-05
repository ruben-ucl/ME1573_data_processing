import h5py, glob, os, cv2, time
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.2'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files in specified path and saves the specified dataset
           from each to a .mp4 video file with a timestamp and scalebar on each frame
    v0.2 - Added pore tracking function using cv2 to run connected component analysis - needs optimisation
           
INTENDED CHANGES
    - Process list of datasets instead of one at a time
    - Replace cv2.putText() with a method that supports unicode characters
    
'''
# Input informaton

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
input_dset_name = 'bg_sub_first_50_frames'
binary_dset_name = 'bg_sub_first_30_frames_/bilateral_filt_r8_li-thresh'
output_name = 'bg_sub_first_50_frame'
capture_framerate = 40000
output_framerate = 30
text_colour = 'white'   # 'black' or 'white'

def main():
    print(f'Creating videos from dataset: {input_dset_name}')
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            # binary_dset = f[binary_dset_name]
            trackid = Path(file).name[:-5]
            fileext = '.mp4'
            vid_filename = f'{trackid}_{output_name}{fileext}'
            output_folder = 'videos'
            create_video_from_dset(dset, vid_filename, output_folder)

def create_video_from_dset(dset, vid_filename, output_folder, binary_dset=None, con_comp_label=False, overlay=True):
    n_frames = len(dset)
    output_path = Path(filepath, output_folder, output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    frame_size = (dset.shape[-1], dset.shape[-2])
    output_filepath = Path(output_path, vid_filename)
    isRGB = con_comp_label  # Sets mode video mode to RGB if saving con comp labels (in colour), else greyscale
    out = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*'mp4v'), output_framerate , frame_size, isRGB) # Add argument False to switch to greyscale
    for i, frame in enumerate(dset):
        if con_comp_label == True:
            binary_im = binary_dset[i, :, :]
            con_comp_box_mask, _ = con_comp(binary_im, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if True in con_comp_box_mask:
                frame[con_comp_bx_mask] = (0, 255, 0)           
        if overlay == True:
            frame = create_overlay(i, frame)
        out.write(frame)
        printProgressBar(i + 1, n_frames, prefix=vid_filename[:7], suffix='Complete', length=50)
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

def con_comp(im, connectivity):
    
    # Get connected components with stats
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(im, connectivity, cv2.CV_32S)
    
    con_comp_dict = {'numLabels': numLabels,
                     'labels': labels,
                     'stats': stats,
                     'centroid': centroids
                     }
    
    # Initialise mask to store component locations and RGB image to store bounding boxes
    mask = np.zeros_like(im)
    output_im = np.zeros_like(im)
    output_box_mask = np.zeros_like(im)
    
    # Loop through all components
    for i in range(1, numLabels):
        # Get stats for component i
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        # Set selection criteria for components
        keepWidth = w > 2 and w < 50
        keepHeight = h > 2 and h < 50
        keepArea = area > 5 and area < 1000
        keepY = y > 320 and y < 450
        
        # For components that satisfy conditions: print details, add to mask, and add bounding box to output image
        if all((keepWidth, keepHeight, keepArea, keepY)):
            # print(f'Keeping component {i+1}/{numLabels}\nx: {x}, y: {y}, w: {w}, h: {h}, area: {area}, centroid: {(cX, cY)}')
            componentMask = (labels == i).astype("uint8") * 255
            mask += componentMask
            output_im = cv2.rectangle(output_im, (x, y), (x + w, y + h), 255, 1)
            output_box_mask = output_im == 255
    
    return output_box_mask, con_comp_dict
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == "__main__":
	main()
