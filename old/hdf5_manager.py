import h5py, glob
from pathlib import Path

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v0.1'

'''
CHANGELOG
    v0.1 - simple loop that prints dataset names and gives option to delete them
           
INTENDED CHANGES
    - 
    
'''

# Input data informaton
filepath = r'C:\Users\rlamb\Dropbox (UCL)\PhD students\Rubén Lambert-Garcia\ESRF ME1573 Python sandbox\hdf5 test sandbox\0103 AlSi10Mg'

# Iterate through files and datasets to perform filtering and thresholding
def main():
    for file in glob.glob(str(Path(filepath, '*.hdf5'))):
        with h5py.File(file, 'a') as f:
            inspect_and_delete(f)
        
def inspect_and_delete(f):
    print(f)
    print('Datasets:')
    print(f.keys())
    try:
        dset_to_delete = input('Enter name of dataset you would like to delete\n')
        del f[dset_to_delete]
        cont = input('Move on to next file? (y/n)\n')
        if cont == 'n':
            inspect_and_delete(f)
        else:
            pass
    except KeyError:
        pass
        
main()