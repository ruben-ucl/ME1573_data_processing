import h5py, glob, pathlib, os
from skimage.io import imshow, imsave
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Additional steps for processing to see keyhole: Increase contrast, invert colouring, filter out noise somehow

start_frame = 800
end_frame = 1300

output_folder = 'prev_frame_bg_sub_rect/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for f in glob.glob('*.hdf5'):
    print(f)
    with h5py.File(f, 'r') as file:
        dsets = list(file.keys())
        for dset_name in dsets:
            print(dset_name)
            dset = np.array(file[dset_name])
            print(dset.shape)
            print(dset.dtype)
            output_shape = [end_frame - start_frame + 1] + list(dset.shape)[1:] 
            output_dset = np.zeros(output_shape, dtype='uint8')
    for i, frame in enumerate(dset):
        prev_frame = dset[i-1, :, :]
        if i == 0 or i <= start_frame or i > end_frame:
            continue
        else:
            output_frame = np.subtract(np.maximum(frame / prev_frame, 1), 1) * 255
            output_dset[i - start_frame, :, :] = output_frame
    filename = f[0:7] + '-prev_frame_bg_sub_rect.hdf5'
    print('Saving ' + filename)
    try:
        with h5py.File(output_folder + filename, 'x') as file:
            file['images'] = output_dset
    except FileExistsError as e:
        print('%s - Skipping file' % str(e))
    print('')

	
