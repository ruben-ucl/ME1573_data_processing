import h5py, glob, os, imageio
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - Iterates through hdf5 files and saves the spcified frame number from each as a still image
           
INTENDED CHANGES
    - 
    
'''
# Input informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'
input_dset_name = 'bg_sub_prev_10_frames'
frame_no = 100
folder_name = f'{input_dset_name}_frame_{frame_no}_stills'

folder_path = Path(filepath, folder_name)

def main():
    print(f'Saving frame no. {frame_no} from dataset: {input_dset_name}\n')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            dset = f[input_dset_name]
            trackid = Path(file).name[:-5]
            print(trackid)
            output_filename = f'{trackid}_{input_dset_name}_frame_{frame_no}.png'
            imageio.imwrite(output_filename, dset[frame_no])
        print('Done\n')

if __name__ == "__main__":
	main()