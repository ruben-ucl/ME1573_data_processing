import h5py, glob
import pandas as pd
import numpy as np
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v1.0'

'''
CHANGELOG
    v0.1 - simple loop that prints dataset names and gives option to delete them
    v0.2 - prints dataset attributes in a table
    v1.0 - multilevel tables for displaying up to two levels of data subgroups
         - improved control loop for automatically deleting a dataset from all files
         - added loop break with 'x' input
           
INTENDED CHANGES
    - 
    
'''

# Input data informaton
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'\nReading from {filepath}')
    
repeat_for_all = True
# dset_to_delete_all = 'bs-f40_lagrangian_meltpool_bin'
dset_to_delete_all = 'keyhole_bin'

col_w = [50, 25, 15, 10]
total_w = np.sum(col_w) + 3
col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
tab_rule = '-'*total_w

# Do not modify
global skip_input
skip_input = False

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for file in sorted(glob.glob(str(Path(filepath, '*.h*5')))):
        with h5py.File(file, 'a') as f:
            status = 0           # 0 = stay on current file, 1 = continue to next file, 2 = exit
            while status == 0:
                status = inspect_and_delete(f)
            if status == 2:
                break
            else:
                pass
                
    print('\nDone')
        
def inspect_and_delete(f):
    if globals()['skip_input'] != True:
        print()
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
                        dset_props[f'    {j}'] = [str(subset.shape), str(subset.dtype), str(round(subset.nbytes/(10**9), 6))]
                    except AttributeError:
                        dset_props[f'    {j}/'] = ['', 'Sub-group', '']
                        for k in f[i][j].keys():
                            subsubset = f[i][j][k]
                            try:
                                dset_props[f'        {k}'] = [str(subsubset.shape), str(subsubset.dtype), str(round(subsubset.nbytes/(10**9), 6))]
                            except AttributeError:
                                pass
        print(col_format.format('Name', 'Shape', 'Datatype', 'Gigabytes'))
        print(tab_rule)
        for set_name, vals in dset_props.items():
            shape, dtype, nbytes = vals
            print(col_format.format(set_name, shape, dtype, nbytes))
    else:
        print(f)
        
    try:
        if repeat_for_all != True:
            dset_to_delete = input('\nEnter name of dataset you would like to delete, \'c\' to continue or \'x\' to exit.\n')
            if dset_to_delete == 'c':
                return 1
            elif dset_to_delete =='x':
                return 2
            else:
                del f[dset_to_delete]
                cont = input('Move on to next file? (y/n)\n')
            if cont == 'n':
                return 0
            else:
                return 1
        else:
            if globals()['skip_input'] != True:
                confirm_del = input(f'Are you sure you want to delete all datasets \'{dset_to_delete_all}\'? (y/n)\n')
                if  confirm_del == 'y':
                    globals()['skip_input'] = True
            if globals()['skip_input'] == True:
                del f[dset_to_delete_all]
            else:
                return 2
    except KeyError:
        if repeat_for_all == False:
            print('\nInput not recognised, try again.')
            return 0
        else:
            print('Dataset deleted')
            return 1
            
main()