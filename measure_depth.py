import h5py, glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.2'

'''
CHANGELOG
    v0.1 - Measures keyhole depth in segmented images as the lowest white pixel in each frame
    v0.2 - Added subplot for photodiode signal to compare with keyhole depth
           
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
                    
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.scatter(time, depth, s=1, c='b')
                ax1.set_ylim(-800, 0)
                ax1.set_xlim(0, time[-1])
                ax1.set_ylabel('Keyhole depth (um)', color='b')
                ax1.set_title(trackid)
                
                AMPM_time = list(file['AMPM_data/Time'])
                AMPM_pd = list(file['AMPM_data/Photodiode1Bits'])
                AMPM_pwr = list(file['AMPM_data/BeamDumpDiodeNormalised'])
                t, pd = trim_pd(AMPM_time, AMPM_pd, AMPM_pwr)
                ax2.scatter(t, pd, s=1, c='r')
                ax2.set_ylim(0, 1000)
                ax2.set_ylabel('Photodiode signal strength (-)', color='r')
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
            start = i - 20
            break
    t = t[start:] - t[start]
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
    