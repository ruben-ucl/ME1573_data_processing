import h5py, numpy, glob, cv2, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

"""
CHANGELOG
    v0.1 - Connected component analysis of single binary image, returns image with filtered compoenents visually identified
           
INTENDED CHANGES
    - 
"""

with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
dset_name = 'bg_sub_first_30_frames_/bilateral_filt_r8_li-thresh'
output_im_dset_name = dset_name
    
def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        print(f'Opening {Path(file).name}')
        with h5py.File(file, 'a') as f:
            dset = f[dset_name]
            im = np.array(dset[-1])
            output = cv2.connectedComponentsWithStats(im, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            
            mask = np.zeros_like(im)
            output_im = cv2.cvtColor(f[output_im_dset_name][-1], cv2.COLOR_GRAY2RGB)
            # loop over the number of unique connected component labels
            for i in range(1, numLabels):
                # if this is the first component then we examine the
                # *background* (typically we would just ignore this
                # component in our loop)
                if i == 0:
                    text = "examining component {}/{} (background)".format(
                        i + 1, numLabels)
                # otherwise, we are examining an actual connected component
                else:
                    text = "examining component {}/{}".format( i + 1, numLabels)
                # print a status message update for the current connected
                # component
                
                # print("[INFO] {}".format(text))
                
                # extract the connected component statistics and centroid for
                # the current label
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]
                
                # print(f'x: {x}\ny: {y}\nw: {w}\nh: {h}\narea: {area}\ncentroid: {(cX, cY)}')
                
                # output_im = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB)
                # output_im = cv2.rectangle(output_im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # output_im = cv2.circle(output_im, (int(cX), int(cY)), 4, (0, 0, 255), -1)
                
                keepWidth = w > 2 and w < 50
                keepHeight = h > 2 and h < 50
                keepArea = area > 5 and area < 1000
                keepY = y > 320 and y < 450
                
                if all((keepWidth, keepHeight, keepArea, keepY)):
                    print("[INFO] keeping connected component '{}'".format(i))
                    print(f'x: {x}\ny: {y}\nw: {w}\nh: {h}\narea: {area}\ncentroid: {(cX, cY)}')
                    componentMask = (labels == i).astype("uint8") * 255
                    mask = np.add(mask, componentMask)
                    output_im = cv2.rectangle(output_im, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # output_im = cv2.circle(output_im, (int(cX), int(cY)), 2, (255, 0, 0), -1)
                    # plt.imshow(componentMask)
                    # plt.show()
                
            # plt.imshow(mask)
            scalebar_text = '500 um'
            output_im = cv2.putText(output_im,                      # Original frame
                                    scalebar_text,                  # Text to add
                                    (890, 500),                     # Text origin
                                    cv2.FONT_HERSHEY_DUPLEX,        # Font
                                    0.9,                            # Fontscale
                                    (255, 255, 255),                # Font colour (BGR)
                                    1,                              # Line thickness
                                    cv2.LINE_AA                     # Line type
                                    )
            # Add scalebar to frame
            bar_originx = 889
            bar_originy = 470
            bar_length = int(500/4.3)
            bar_thickness = 4
            output_im = cv2.rectangle(output_im,                                            # Original frame
                                      (bar_originx, bar_originy),                           # Top left corner
                                      (bar_originx+bar_length, bar_originy-bar_thickness),  # Bottom right corner
                                      (255, 255, 255),                                      # Colour (BGR)
                                      -1                                                    # Line thickness (-ve means fill shape inwards)
                                      )
            plt.imshow(output_im)
            # cv2.imshow("Connected Component", componentMask)
            plt.show()
                
if __name__ == "__main__":
	main()