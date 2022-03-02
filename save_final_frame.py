import h5py, glob, pathlib, os
from skimage.io import imshow, imsave
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

for f in glob.glob('*.hdf5'):
    print(f)
    with h5py.File(f, 'r') as file:
        dsets = list(file.keys())
        for dset in dsets:
            print('Dataset:')
            print(dset)
            print(file[dset].shape)
            print(file[dset].dtype)
            fig = plt.figure(figsize = (4, 2), dpi=300)
            im = file[dset][-1, :, :]
            # plt.imshow(im, cmap='gray', origin='lower')
            # plt.show()
            im_name = f[0:6] + '-final_frame.tif'
            print(im_name)
            output_folder = 'Final frames/'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            imsave(output_folder+im_name, im)
	
