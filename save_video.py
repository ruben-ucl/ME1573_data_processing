import h5py, glob, os, cv2, time
import numpy as np
from skimage import morphology
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
"""Controls"""

input_dset_name = 'ff_corrected_crop'

binary_dset_name = 'keyhole_binary_refined'
binary_overlay_mode = 'outline'     # 'outline' or 'fill'

output_name = f'{input_dset_name}_keyhole_overlay'

capture_framerate = 40000 # fps
output_framerate = 30 # fps
text_colour = 'white'   # 'black' or 'white'


# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
def main():
    print(f'Creating videos from dataset: {input_dset_name}')
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
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
        trackid = Path(file).name[:-5]
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

def create_overlay(i, frame):
    if text_colour == 'black':
        bgr_colour = (0, 0, 0)
    elif text_colour == 'white':
        bgr_colour = (255, 255, 255)
    timestamp = str(format(i * 1/capture_framerate * 1000, '.3f')) + ' ms'        # in milliseconds
    height, width, _ = frame.shape
    bottom_left = (5, height-10)
    bottom_right = (width-112, height-12)
    # Add timestamp to frame
    new_frame = cv2.putText(frame,                          # Original frame
                            timestamp,                      # Text to add
                            bottom_left,                       # Text origin
                            cv2.FONT_HERSHEY_DUPLEX,        # Font
                            0.7,                            # Fontscale
                            bgr_colour,                     # Font colour (BGR)
                            1,                              # Line thickness
                            cv2.LINE_AA                     # Line type
                            )
    # Add scale bar text to frame
    scalebar_text = '500 um'
    new_frame = cv2.putText(new_frame,                      # Original frame
                            scalebar_text,                  # Text to add
                            bottom_right,                     # Text origin
                            cv2.FONT_HERSHEY_DUPLEX,        # Font
                            0.7,                            # Fontscale
                            bgr_colour,                     # Font colour (BGR)
                            1,                              # Line thickness
                            cv2.LINE_AA                     # Line type
                            )
    # Add scalebar to frame
    bar_originx = bottom_right[0] - 14
    bar_originy = bottom_right[1] + 7
    bar_length = int(500/4.3)
    bar_thickness = 3
    new_frame = cv2.rectangle(new_frame,                                            # Original frame
                              (bar_originx, bar_originy),                           # Top left corner
                              (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                              bgr_colour,                                           # Colour (BGR)
                              -1                                                    # Line thickness (-ve means fill shape inwards)
                              )
    return new_frame

''' ***DEPRECATED***
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
'''
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
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
