import h5py, glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Measures keyhole depth in segmented images as the lowest white pixel in each frame
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_10_frames_/median_filt_r1_segmented_171-255'


def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        trackid = Path(f).name[:-5]
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Calculating output')
                time = []
                exp_t = 1/40000 # in seconds
                depth = []
                for i, frame in enumerate(dset):
                    time.append(i * exp_t)
                    d = measure_depth(frame)
                    depth.append(d)
                    
                fig, ax1 = plt.subplots()
                ax1.plot(time, depth, 'b-')
                ax1.set_ylim(-500, 0)
                ax1.set_xlim(0, time[-1])
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('Keyhole depth (um)')
                ax1.set_title(trackid)
                
                # ax2 = ax1.twinx()
                # AMPM_time = list(file['AMPM_data/Time'])
                # AMPM_pd = list(file['AMPM_data/Photodiode1Normalised'])
                # ax2.plot(trim_pd(AMPM_time, AMPM_pd), 'r-')
                # ax2.set_ylabel('Photodiode signal strength (-)')
                
                plt.show()
                # plt.savefig(str(Path(filepath, '%s keyhole depth plot.png' % trackid)))
                
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file')
            
def trim_pd(t, pd):
    for i, val in enumerate(pd):
        start = 0
        if val > 100:
            start = i - 20
            break
    t = np.subtract(t[start:], t[start])
    pd = pd[start:]
    return t, pd

def measure_depth(frame):
    depth = 0
    for i, row in enumerate(frame):
        if np.mean(row) != 0:
            depth = i
    depth -= 310    # Adjust to measure from top of substrate
    depth *= -4.3    # Convert to microns and make negative
    return depth

main()
    