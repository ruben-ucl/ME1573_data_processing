import h5py, glob, cv2, functools
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

mode = 'apply' # 'apply' or 'preview'
input_dset_name = 'bg_sub_prev_5_frames_crop_rotate'
output_dset_name = input_dset_name + '_denoised'

filepath = get_paths()['hdf5']

def denoise(dset):
    if mode == 'preview':
        input_img = dset[200]
        output_img = cv2.fastNlMeansDenoising(input_img, None, 10, 7, 21) # args: img, output, strength, templateWindowSize, searchWindowSize
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(output_img, cmap='gray')
        plt.show()
    else:
        output_dset = np.zeros_like(dset)
        for i, im in enumerate(dset):
            output_dset[i] = cv2.fastNlMeansDenoising(im, None, 10, 7, 21) # args: img, output, strength, templateWindowSize, searchWindowSize
        return output_dset
        
def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print('Denoising...')
                denoised_dset = denoise(dset)
                if mode == 'apply':
                    file[output_dset_name] = denoised_dset
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()