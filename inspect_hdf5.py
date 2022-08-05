import h5py, glob
import pandas as pd
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.2'

'''
CHANGELOG
    v0.1 - simple loop that prints dataset names and gives option to delete them
    v0.2 - prints dataset attributes in a table
           
INTENDED CHANGES
    - 
    
'''

# Input data informaton
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            inspect_and_delete(f)
    print('\nFinished - no more files')
        
def inspect_and_delete(f):
    print(f, ' Datasets:\n-----------------------------------------------------------------------------------')
    dset_props = {}
    for dset_name in f.keys():
        dset = f[dset_name]
        dset_props[dset_name] = [dset.shape, dset.dtype, dset.nbytes]
    print('{:<30} {:<25} {:<15} {:<10}'.format('Name', 'Shape', 'Datatype', 'Gigabytes'))
    print('-----------------------------------------------------------------------------------')
    for k, v in dset_props.items():
        shape, dtype, nbytes = v
        print('{:<30} {:<25} {:<15} {:<10}'.format(k, str(shape), str(dtype), str(round(nbytes/(10**9), 6))))
    try:
        dset_to_delete = input('\nEnter name of dataset you would like to delete, or \'c\' to continue\n')
        if dset_to_delete == 'c':
            cont = 'y'
        else:
            del f[dset_to_delete]
            cont = input('Move on to next file? (y/n)\n')
        if cont == 'n':
            inspect_and_delete(f)
        else:
            pass
    except KeyError:
        pass
        
main()