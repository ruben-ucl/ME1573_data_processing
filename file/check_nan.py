import h5py
import numpy as np

filename = input('Enter name of hdf5 file to check\n')
with h5py.File(filename, 'r') as file:
    print('Datasets:')
    print(file.keys())
    dsetname = input('\nEnter name of dataset to check\n')
    array = file[dsetname]
    nan_check = np.isnan(np.sum(array))
    print(nan_check)
    if nan_check == True:
        print('Dataset contains NaN values')
    else:
        print('Dataset does not contain Nan values')
    

