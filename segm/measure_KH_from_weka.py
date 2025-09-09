import os, h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def import_tiff_folder_to_numpy(folder_path):
    """
    Import all TIFF images from a folder into a single numpy array.
    
    Parameters:
        folder_path (str): Path to the folder containing TIFF images
        
    Returns:
        numpy.ndarray: A 3D array of shape (n, height, width) containing all images
                       with 8-bit unsigned integer data type (uint8)
    """
    # List all TIFF files in the folder
    folder_path = os.path.abspath(folder_path)
    tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    # Sort the files to ensure consistent ordering
    tiff_files.sort()
    
    # Get dimensions from the first image
    first_img_path = os.path.join(folder_path, tiff_files[0])
    first_img = Image.open(first_img_path)
    img_width, img_height = first_img.size
    
    # Create an empty array to hold all images
    all_images = np.empty((len(tiff_files), img_height, img_width), dtype=np.uint8)
    
    # Load each image into the array
    for i, file_name in enumerate(tiff_files):
        img_path = os.path.join(folder_path, file_name)
        with Image.open(img_path) as img:
            # Check if dimensions match the first image
            if img.size != (img_width, img_height):
                raise ValueError(f"Image {file_name} has different dimensions than the first image")
            
            # Convert to 8-bit if needed and add to array
            img_array = np.array(img)
            
            # Handle different bit depths
            if img_array.dtype != np.uint8:
                # If input image has higher bit depth, scale down to 8-bit
                if img_array.dtype == np.uint16:
                    img_array = (img_array / 256).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            all_images[i] = img_array
    
    return all_images
    
def save_measurements_to_hdf5(time_data, measurement_data, output_path, measurement_name="measurements"):
    """
    Save time series measurements to an HDF5 file.
    
    Parameters:
        time_data (list or array): Array of time points
        measurement_data (list or array): Array of measurements corresponding to time points
        output_path (str): Path where the HDF5 file will be saved
        measurement_name (str): Name for the measurement dataset (default: "measurements")
        
    Returns:
        str: Path to the saved HDF5 file
    """
    # Convert inputs to numpy arrays if they aren't already
    time_array = np.array(time_data)
    measurement_array = np.array(measurement_data)
    
    # Verify that arrays have the same length
    if len(time_array) != len(measurement_array):
        raise ValueError("Time and measurement arrays must have the same length")
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Create a group for the time series data
        time_series_group = hf.create_group("meas")
        
        # Add datasets to the group
        time_series_group.create_dataset("time", data=time_array)
        time_series_group.create_dataset(measurement_name, data=measurement_array)
        
        # Add attributes with metadata
        time_series_group.attrs["num_points"] = len(time_array)
        time_series_group.attrs["time_unit"] = "ms"  # Change this to the appropriate unit
        time_series_group.attrs["measurement_type"] = measurement_name
        
        # Add some basic statistics as attributes
        time_series_group.attrs["measurement_mean"] = float(np.mean(measurement_array))
        time_series_group.attrs["measurement_std"] = float(np.std(measurement_array))
        time_series_group.attrs["measurement_min"] = float(np.min(measurement_array))
        time_series_group.attrs["measurement_max"] = float(np.max(measurement_array))
    
    return output_path

def main():
    image_folder = r"E:\sim_segmented_350W\Weka_segmented_tiffs"
    substrate_height = 89
    resolution = 0.858
    framerate = 100000
    images = import_tiff_folder_to_numpy(image_folder)
    time = []
    KH_depth = []
    for i, im in enumerate(images):
        print('im ', i)
        for j, row in enumerate(im[::-1, :]):
            if 3 not in row:
                pass
            else:
                print('row ', j)
                KH_bottom = im.shape[0] - j
                time.append(i * 1000 * 1/framerate)
                KH_depth.append((KH_bottom - substrate_height) * resolution)
                break
        
    KH_depth = np.clip(KH_depth, 0, None)
    output = {'time_ms': time, 'KH_depth_um': KH_depth}
    df = pd.DataFrame(output)
    df.to_csv(image_folder + r'\KH_depth.csv')
    save_measurements_to_hdf5(df['time_ms'],
        df['KH_depth_um'],
        image_folder + r'\KH_depth.hdf5',
        'KH_depth_um')
    
    
if __name__ == "__main__":
    main()