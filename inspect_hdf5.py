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
    
repeat_for_all = False
dset_to_delete_all = 'ffcorr'

col_w = [50, 25, 15, 10]
total_w = np.sum(col_w) + 3
col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
tab_rule = '-'*total_w

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            inspect_and_delete(f)
    print('\nFinished - no more files')
        
def inspect_and_delete(f):
    print(f, f' Datasets:\n{tab_rule}')
    dset_props = {}
    for i in f.keys():
        dset = f[i]
        try:
            dset_props[i] = [str(dset.shape), str(dset.dtype), str(round(dset.nbytes/(10**9), 6))]
        except AttributeError:
            dset_props[i+'/'] = ['', 'Group', '']
            for j in f[i].keys():
                subset = f[i][j]
                try:
                    dset_props['    '+j] = [str(subset.shape), str(subset.dtype), str(round(subset.nbytes/(10**9), 6))]
                except AttributeError:
                    dset_props[j] = ['', '    Sub-group', '']
                        for k in f[i][j].keys():
                        subsubset = f[i][j][k]
                        try:
                            dset_props['        '+j] = [str(subsubset.shape), str(subsubset.dtype), str(round(subsubset.nbytes/(10**9), 6))]
    print(col_format.format('Name', 'Shape', 'Datatype', 'Gigabytes'))
    print(tab_rule)
    for k, v in dset_props.items():
        shape, dtype, nbytes = v
        print(col_format.format(k, shape, dtype, nbytes))
    try:
        if repeat_for_all != True:
            dset_to_delete = input('\nEnter name of dataset you would like to delete or \'c\' to continue\n')
            if dset_to_delete == 'c':
                cont = 'y'
            else:
                del f[dset_to_delete]
                cont = input('Move on to next file? (y/n)\n')
            if cont == 'n':
                inspect_and_delete(f)
            else:
                pass
        else:
            del f[dset_to_delete_all]
    except KeyError:
        pass
        
main()