import h5py, glob, cv2, functools, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, compare_histograms

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_names = ['bs-p5-s5_lagrangian_keyhole',
                    'bs-p5-s5_tri+35_lagrangian_keyhole',
                    'bs-p5-s5_tri+35_lagrangian_keyhole_refined',
                    ]
                    
frame_pos = 0.5 # 0 is first frame, 1 is last frame

# Read data folder path from .txt file
filepath = get_paths()['hdf5']

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        trackid = Path(f).name[:-5]
        try:
            with h5py.File(f, 'a') as file:
                im_dict = {}
                for dset_name in input_dset_names:
                    frame_n = round(frame_pos * len(file[dset_name])) - 24
                    im_dict[dset_name] = file[dset_name][frame_n]
                    
                compare_histograms(im_dict, trackid)
        except Exception as e:
            print(e)
            
if __name__ == "__main__":
	main()