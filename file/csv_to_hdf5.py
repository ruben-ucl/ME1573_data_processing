import h5py
import numpy as np
import pandas as pd

def main():
    # Inputs
    input_path = 'E:/sim_segmented_300W_800mm_s/simulation_data.csv'    # Old .csv file to read
    time_col = 'Time'                                           # Collumn name for time data
    measurment_col = 'EnergyAbsorbed_W'                         #  Column name for measurement data
    output_path = 'absorption.hdf5'                             # New .hdf5 file to write
    
    csv_to_hdf5(input_path, output_path, time_col, measurment_col)
    
    # Verify by reading the data back
    # data = read_measurements_from_hdf5(output_path)
    # print(f"Loaded {data['attributes']['num_points']} measurements")
    
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
        time_series_group.attrs["time_unit"] = "steps"  # Change this to the appropriate unit
        time_series_group.attrs["measurement_type"] = measurement_name
        
        # Add some basic statistics as attributes
        time_series_group.attrs["measurement_mean"] = float(np.mean(measurement_array))
        time_series_group.attrs["measurement_std"] = float(np.std(measurement_array))
        time_series_group.attrs["measurement_min"] = float(np.min(measurement_array))
        time_series_group.attrs["measurement_max"] = float(np.max(measurement_array))
    
    return output_path

# Example usage with a dictionary or pandas DataFrame
def save_dict_or_df_to_hdf5(data, output_path, time_key="time", measurement_key="measurement"):
    """
    Convert a dictionary or DataFrame into an HDF5 file.
    
    Parameters:
        data (dict or pd.DataFrame): Data containing time and measurement columns
        output_path (str): Path where the HDF5 file will be saved
        time_key (str): Key or column name for time data
        measurement_key (str): Key or column name for measurement data
        
    Returns:
        str: Path to the saved HDF5 file
    """
    if isinstance(data, dict):
        time_data = data[time_key]
        measurement_data = data[measurement_key]
    elif isinstance(data, pd.DataFrame):
        time_data = data[time_key].values
        measurement_data = data[measurement_key].values
    else:
        raise TypeError("Data must be either a dictionary or a pandas DataFrame")
    
    return save_measurements_to_hdf5(time_data, measurement_data, output_path, measurement_name=measurement_key)
    
def csv_to_hdf5(csv_path, output_path, time_column="time", measurement_column="measurement"):
    """
    Read data from a CSV file and save it to an HDF5 file.
    
    Parameters:
        csv_path (str): Path to the input CSV file
        output_path (str): Path where the HDF5 file will be saved
        time_column (str): Name of the time column in the CSV
        measurement_column (str): Name of the measurement column in the CSV
        
    Returns:
        str: Path to the saved HDF5 file
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Check if the required columns exist
    if time_column not in df.columns or measurement_column not in df.columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Required columns not found. Available columns: {available_columns}")
    
    # Convert the DataFrame to an HDF5 file
    return save_dict_or_df_to_hdf5(df, output_path, time_key=time_column, measurement_key=measurement_column)

# Additional function to read back the data
def read_measurements_from_hdf5(file_path):
    """
    Read time series measurements from an HDF5 file.
    
    Parameters:
        file_path (str): Path to the HDF5 file
        
    Returns:
        dict: Dictionary containing the time and measurement data and attributes
    """
    result = {}
    
    with h5py.File(file_path, 'r') as hf:
        time_series_group = hf["time"]
        
        # Read datasets
        result["time"] = time_series_group[time_series_group.keys()[0]][:]
        
        # Find the measurement dataset (excluding "time")
        for key in time_series_group.keys():
            if key != "time":
                result[key] = time_series_group[key][:]
        
        # Read attributes
        result["attributes"] = dict(time_series_group.attrs)
    
    return result
    
if __name__ == "__main__":
    main()