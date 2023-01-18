import h5py, glob, functools
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'bs-p5'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

"""Input function to execute on dataset here"""
def get_means(dset):
    means = np.mean(dset, axis=(1, 2))
    print(means.shape)
    return means

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        with h5py.File(f, 'a') as file:
            dset = file[input_dset_name]
            print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
            
            means = get_means(dset)
            t = [i/40 for i in range(len(means))]

            means_fft = abs(np.fft.fft(means))
            freq = np.fft.fftfreq(len(means)) * 40000
            
            fig, (ax1, ax2) = plt.subplots(1, 2,
                                           tight_layout=True,
                                           figsize=(8, 3))
            ax1.plot(range(len(means)), means)
            ax1.set_xlim(100, 130)
            ax1.set_xlabel('Frame [-]')
            ax1.set_ylabel('Mean grey value [-]')
            
            ax2.plot(freq / 40000, means_fft.real)
            ax2.set_ylim(0, 1500)
            ax2.set_xlim(0, None)
            ax2.set_xlabel('Frequency [1/frame]')
            ax2.set_ylabel('Amplitude [-]')
            plt.show()
            
        print('Done\n')
            
if __name__ == "__main__":
	main()