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
    print(f'\nReading from {filepath}')
    
repeat_for_all = False
dset_to_delete_all = 'bs-p5-s5'

col_w = [50, 25, 15, 10]
total_w = np.sum(col_w) + 3
col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
tab_rule = '-'*total_w

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for file in glob.glob(str(Path(filepath, '*.hdf*'))):
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
            confirm_del = input(f'Are you sure you want to delete all datasets \'{dset_to_delete_all}\'? (y/n)\n)
            if confirm_del == 'y':
                del f[dset_to_delete_all]
            else:
                return 2
            print(f'Dataset \'{dset_to_delete_all}\' deleted from all files in directory.')
    except KeyError:
        print('\nInput not recognised, try again.')
        return 0
        
main()