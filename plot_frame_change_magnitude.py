import h5py, glob, pathlib, os
from skimage.io import imshow, imsave
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Additional steps for processing to see keyhole: Increase contrast, invert colouring, filter out noise somehow

output_folder = 'prev_frame_bg_sub_rect/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

window = 100 # number of frames for rolling avg window

for f in glob.glob('*.hdf5'):
    print(f)
    with h5py.File(f, 'r') as file:
        dsets = list(file.keys())
        for dset_name in dsets:
            print(dset_name)
            dset = np.array(file[dset_name])
            print(dset.shape)
            print(dset.dtype)
    diff_sum_list = []
    diff_rolling_avg = []
    for i, frame in enumerate(dset):
        prev_frame = dset[i-1, :, :]
        diff_array = np.divide(frame, prev_frame)
        diff_sum = np.sum(diff_array)
        diff_sum_list.append(diff_sum)
        if i >= window:
            diff_rolling_avg.append(np.sum(diff_sum_list[i-window:i])/window)
    plt.plot(diff_rolling_avg[1:])
    plt.show()
    input('Press any key to continue')

	
