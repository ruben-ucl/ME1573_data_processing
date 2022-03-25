import h5py, glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.3'

'''
CHANGELOG
    v0.1 - Measures keyhole depth in segmented images as the lowest white pixel in each frame
    v0.2 - Added subplot for photodiode signal to compare with keyhole depth
    v0.2.1 - Moved filepath storage to external text file called 'data_path.txt'
    v0.3 - Corrected for image slant in depth calculation
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

input_dset_name = 'bg_sub_prev_10_frames_/median_filt_r1_tri-thresh'

# For image slant compensation (WIP)
# Meaaure the y value at the top of the powder surface on the left and right extremes of the image
# Used for calculating keyhole depth
# left_top_edge = 295
# right_top_edge = 276
# surface_slant = (right_top_edge - left_top_edge) / 1024

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        trackid = Path(f).name[:-5]
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                n_frames = len(dset)
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                time = []
                exp_t = 1/40000 # in seconds
                depth = []
                for i, frame in enumerate(dset):
                    time.append(i * exp_t)
                    d = measure_depth(i, frame, n_frames)
                    depth.append(d)
                    
                AMPM_time = list(file['AMPM_data/Time'])
                AMPM_pd = list(file['AMPM_data/Photodiode1Bits'])
                AMPM_pwr = list(file['AMPM_data/BeamDumpDiodeNormalised'])
                t, pd = trim_pd(AMPM_time, AMPM_pd, AMPM_pwr)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                
                ax1.set_title(trackid)
                ax1.scatter(t, pd, s=1, c='r')
                ax1.set_ylim(0, 1000)
                ax1.set_ylabel('Photodiode signal strength (-)', color='r')
                ax1.set_xlim(0, time[-1])
                
                ax2.scatter(time, depth, s=1, c='b')
                ax2.set_ylim(-800, 0)
                ax2.set_ylabel('Keyhole depth (um)', color='b')
                ax2.set_xlabel('Time (ms)')
                
                # plt.show()
                plt.savefig(str(Path(filepath, '%s keyhole depth plot.png' % trackid)))
                
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            
def trim_pd(t, pd, pwr):
    start = 0
    for i, val in enumerate(pwr):        
        if (val > 100):
            start = i - 125
            break
    t = t[start:] - t[start]
    pd = pd[start:]
    return t, pd

def measure_depth(i, frame, n_frames):
    depth = 0
    for i, row in enumerate(frame):
        if np.mean(row) != 0:
            depth = i
    # Correct for slanted image
    # if i <= 50:                     # Before laser onset
        # surface_y = left_top_edge
    # elif i >= n_frames - 50:        # After scan finished
        # surface_y = right_top_edge
    # else:                           # During scanning
        # surface_y = surface_slant * ((i - 50) / (n_frames - 100)) * 1024 * 0.91 + left_top_edge     # Factor 0.91 to account for track length only being 91% of frame width
    surface_y = 283
    depth -= surface_y      # Adjust to measure from top of substrate at the given point
    depth *= -4.3           # Convert to microns and make negative
    return depth

main()
    