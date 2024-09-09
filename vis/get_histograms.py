import h5py, glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

filepath = get_paths()['hdf5']

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - On one set of axes, plot full-stack histogram curves for specified datsets
    
INTENDED CHANGES
    - 
    
'''

# Input informaton
filepath = get_paths()['hdf5']

""" Controls """
input_dset_name = 'xray_images'
include_tracks = ['0103_01', '0103_02', '0103_03', '0103_04', '0103_05', '0103_06']
save_to_csv = False
generate_plots = True
""" -------- """

histograms = pd.DataFrame()
histograms['value'] = [i for i in range(256)] # First column contains integers 0-255 for 8-bit greyscale values

def main():
    for f in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
        trackid = Path(f).name[:-5]
        if trackid not in include_tracks:
            continue
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = np.array(file[input_dset_name])
                hist, bin_edges = np.histogram(dset, bins = 256, range = [0, 256])
            col_name = f'{trackid}_count'
            histograms[col_name] = hist
        except Exception as e:
            print(e)
    if save_to_csv == True:        
        histograms.to_csv(Path(filepath, f'{input_dset_name}_histograms.csv'), index=False)    
    if generate_plots == True:
        fig, ax = plt.subplots(figsize = [9, 6])
        x = histograms['value']
        for col in histograms.columns:
            if col != 'value':
                ax.plot(x, histograms[col], label = col[:7])
        ax.set_xlabel('pixel value')
        ax.set_ylabel('count')
        ax.legend()
        fig.set_tight_layout(True)
        plt.show()
    
if __name__ == "__main__":
	main()