import h5py, glob, cv2, functools
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

rot_angle = 0 # degrees
keep_pixels = [0, 1024, 308, 388] # [x_min, x_max, y_min, y_max]
keep_frames = [400, 800] # [start_frame, end_frame]

input_dset_name = 'ff_corrected'
output_dset_name = input_dset_name + '_crop'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

def rotation_correction(image, angle):
    image_centre = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_centre[::-1], angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
    return rotated_image
    
def crop_trim(dset, frames, pixels):
    cropped_dset = dset[frames[0]:frames[1], pixels[2]:pixels[3], pixels[0]:pixels[1]]
    return cropped_dset

def main():
    for f in glob.glob(str(Path(filepath, '*.hdf5'))):
        print('Reading %s' % Path(f).name)
        try:
            with h5py.File(f, 'a') as file:
                dset = file[input_dset_name]
                print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
                print(f'Rotating by {rot_angle} degree, cropping to {keep_pixels[1]-keep_pixels[0]} x {keep_pixels[3]-keep_pixels[2]}, trimming to {keep_frames[1]-keep_frames[0]} frames.')
                trimmed_dset = crop_trim(dset, keep_frames, keep_pixels)
                output_dset = np.zeros_like(trimmed_dset)
                print(f'Trimmed shape: {trimmed_dset.shape}')
                for i, frame in enumerate(trimmed_dset):
                    output_dset[i, :, :] = rotation_correction(frame, rot_angle)
                print(f'Rotated shape: {output_dset.shape}')
                file[output_dset_name] = output_dset
            print('Done\n')
        except OSError as e:
            print('Error: output dataset with the same name already exists - skipping file\n')
            
if __name__ == "__main__":
	main()