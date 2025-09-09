#!/usr/bin/env python

"""
This script trains and evaluates res-net ML models for classifying
photodiode process monitoring signals

Author:     Rubén Lambert-Garcia
Version:    0.1
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Package imports
import sys, h5py, keras, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from pathlib import Path
from skimage import io
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2

# Local imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, get_logbook_data, get_paths, printProgressBar

# Read log book
log = get_logbook()

# Signals to use (dataset names)
dset_names = ['AMPM/Time',
              'AMPM/Photodiode1Bits',
              'AMPM/Photodiode2Bits']
              
# Settings
sr = 100 # Sampling rate in kHz
wl = 1  # Window length in ms
wo = 0.8 # Window offset in ms
bit_depth = 16 # Bit depth of greyscale training images
img_channels = 1 # Colour channels in image (1 or 3)
trim = 500 # Number of points to trim from start/end of data (default 500 for AMPM)
raw_signal_min = 54
raw_signal_max = 2440

filter_log = True
generate_signal_overview_plots = False
show_figs = False
save_figs = False
generate_training_data = False  
label_images = False
train = True
debug = False

n_samples = sr * wl

# Tracks to exclude due to corrupted PD signal
exclude = ['0514_02',
           '0514_04',
           '0514_05',
           '0515_01',
           '0515_02',
           '0515_03',
           '0515_04',
           '0515_05',
           '0515_06',
           '0516_01',
           '0516_02',
           '0516_03',
           '0516_04',
           '0516_05',
           '0516_06']
              
# Function to read data from hdf5 file and return datsets as a dictionary
def read_hdf5(fpath, dset_names):
    data = {}
    with h5py.File(fpath, 'r') as file:
        for i, name in enumerate(dset_names):
            data[i] = np.array(file[name])
    
    return data

# Function returns a filtered list of trackids based on specidied conditions
def filter_logbook(log):
    # filters for welding or powder melting
    welding = log['Powder material'] == 'None'
    powder = np.invert(welding)
    
    # filters for CW or PWM laser mode
    cw = log['Point jump delay [us]'] == 0
    pwm = np.invert(cw)
    
    # filter for Layer 1 tracks only
    L1 = log['Layer'] == 1
    
    # filter for presence of KH pores
    pores = log['n_pores'] > 2
    
    # filter by layer thickness
    thin_layer = log['measured_layer_thickness [um]'] <= 80
    very_thin_layer = log['measured_layer_thickness [um]'] <= 35
    
    # filter by scan speed
    speed = log['Scan speed [mm/s]'] == 400
    
    # filter by beamtime
    ltp1 = log['Beamtime'] == 1
    ltp2 = log['Beamtime'] == 2
    ltp3 = log['Beamtime'] == 3
    
    # filter by substrate
    s0514 = log['Substrate No.'] == '0514'
    s0515 = log['Substrate No.'] == '0515'
    
    # filter by material
    AlSi10Mg = log['Substrate material'] == 'AlSi10Mg'
    Al7A77 = log['Substrate material'] == 'Al7A77'
    Al = log['Substrate material'] == 'Al'
    Ti64 = log['Substrate material'] == 'Ti64'
    lit = np.logical_or(Ti64, Al7A77)
    
    # filter by regime
    not_flickering = log['Melting regime'] != 'keyhole flickering'
    not_cond = log['Melting regime'] != 'conduction'
    
    if filter_log == False:
        log_red = log[L1]
    else:
        log_red = log[AlSi10Mg & L1 & cw & powder]
    
    track_list = log_red['trackid'].tolist()
    return track_list
    
# Function for transforming the data to better represnt the distribution of values
def transform_data(data):
    # Take natural log of signal amplitudes
    data = np.log(data)
    return data

# Function to generate the overview plot showing original signals and signal window images
def create_window_plot(trackid, n_windows, data, windows, n_samples, total_samples, sr, wl):
    # Figure size, DPI and text size settings for A4 full-width high resolution
    plt.rcParams.update({
        'font.size': 7,          # Base font size
        'figure.titlesize': 10    # Figure title size
    })

    # Create figure with subplots
    # A4 width is 8.27 inches, making it slightly narrower for margins
    fig = plt.figure(figsize=(6.3, 4), dpi=300)  # Increased height for more space
    fig.suptitle(trackid)
    gs0 = gs.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    gs01 = gs0[1].subgridspec(n_windows, 1, hspace=0.4)  # Increased hspace for title space

    # Plot original data
    axs = []
    axs.append(fig.add_subplot(gs0[0]))

    # Define signal labels
    signal_labels = ['PD1', 'PD2']
    plot_colors = ['#f98e09', '#57106e']

    # Plot signals with labels
    for j in range(data.shape[0]):
        # Define x and y to plot
        y = data[j]
        x = range(total_samples)
        # Convert x to ms
        x = np.divide(x, sr)
        # Plot signal
        axs[0].plot(x, y,
            linewidth=0.5,
            c=plot_colors[j],   # Add custom line colours
            label=signal_labels[j])  # Add labels for the legend

    # Find global min and max for consistent colormaps and signal plot y-axis range
    # Apply the same transformation used on the raw dataset
    vmin = transform_data(raw_signal_min)
    vmax = transform_data(raw_signal_max)
    axs[0].set_ylim(vmin, vmax)

    # Add labels and legend to original signal plot
    axs[0].legend(loc='best')
    axs[0].set_title('Photodiode signals')
    axs[0].set_ylabel('log(Amplitude)')
    axs[0].set_xlabel('Time [ms]')

    # Draw window boundaries and labels over the original signal plot
    for j in range(n_windows+1):
        # Plot boundaries as vertical lines
        axs[0].plot([j*wl, j*wl],
            [vmin, vmax],
            'k--',
            linewidth=0.5,
            alpha=0.4)
        if j < n_windows:    
            # Add window number labels
            axs[0].text((j+0.5)*wl, vmax,
                f'\nw{j+1}',
                ha='center',
                va='top',
                alpha=0.4)

    # Create axes for displaying windowed data beside original signal
    for j in range(n_windows):
        # Create axes for displaying signal windows
        axs.append(fig.add_subplot(gs01[j]))

    # Plot extracted windows as images
    for j in range(n_windows):
        # Add title to top window subplot and add initial window boundary line
        if j == 0:
            axs[j+1].set_title('Signal windows')
            
            axs[0].plot([j*wl, j*wl],
            [vmin, vmax],
            'k--',
            linewidth=0.5,
            alpha=0.4)
        
        # Hide x ticks and labels except for bottom image which get the axis title
        if j < n_windows - 1:
            axs[j+1].get_xaxis().set_visible(False)
        else:
            axs[j+1].set_xlabel('Time [ms]')
        
        # Use same vmin and vmax for consistent colormap across all subplots
        img = axs[j+1].imshow(windows[j],
            cmap='gray',
            aspect='4',
            vmin=vmin,
            vmax=vmax)
            
        # Customise x tick locations and display in ms units
        n_ticks = 5
        axs[j+1].set_xticks([n_samples*i/n_ticks for i in range(n_ticks+1)],
            [wl*i/n_ticks for i in range(n_ticks+1)])
        
        # Move 'Window [j]' to subplot title
        axs[j+1].set_ylabel(f'w{j+1}',
            rotation=0,
            labelpad=10,
            va='center')
        
        # Add y-axis labels that match signal labels
        axs[j+1].set_yticks(range(len(signal_labels)))
        axs[j+1].set_yticklabels(signal_labels)

    # Add colorbar for the image plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label('Amplitude')

    # Adjust spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    # For saving the figure
    if save_figs:
        fig_folder = Path(im_root_folder, '..', 'signal_overview_plots')
        if not os.path.exists(fig_folder):
                    os.makedirs(fig_folder)
        save_path = Path(fig_folder, f'{trackid}_{wl}ms_PD_windows.png')
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    
    if show_figs:
        plt.show()
    
    plt.close()

# Function returns filtered list of file paths if corresponding trackids are in trackid_list
def filter_fpaths(fpath_list, trackid_list):
    filtered_fpath_list = []
    for trackid in trackid_list:
        filtered_fpath_list = filtered_fpath_list + [p for p in fpath_list if trackid in p]
    print(f'Using {len(filtered_fpath_list)} files out of {len(fpath_list)}')
    return filtered_fpath_list

# Sort images into label folders by logbook attribute
def sort_images(im_root_folder, track_list):
    im_fpaths = sorted(glob.glob(f'{im_root_folder}/*.tiff'))
    im_fpaths = filter_fpaths(im_fpaths, track_list)
    n_files = len(im_fpaths)
    
    for i, im_fpath in enumerate(im_fpaths):
        fname = Path(im_fpath).name
        trackid = fname[:7]
        
        if debug:
            print(trackid)
        # Print progress to terminal
        else:
            printProgressBar(i,
                n_files-1,
                prefix='Progress:',
                suffix=f'Working on {trackid}',
                length=50)            
        
        # Skip if track ID in exclude list
        if trackid in exclude:
            continue
        
        # Read image
        im = io.imread(im_fpath)
        
        # Read label
        label = get_logbook_data(log, trackid)['melting_regime']
        if debug:
            print(label)
        label_folder = Path(im_root_folder, label)
        save_path = Path(label_folder, fname)
        
        # Create label folder if it does not already exist
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Save image to label folder
        io.imsave(save_path, im)

# Compile training dataset from image files
def get_training_dataset(im_root_folder):
    SIZE_X = 100 
    SIZE_Y = 2
    img_size = (SIZE_X, SIZE_Y)
    
    dataset = [] 
    label = []
    # Create image and label lists
    # Change the image type

    L0_images = sorted(glob.glob(str(Path(im_root_folder, '0', '*.tiff'))))
    for i, image_path in enumerate(L0_images):
        image = io.imread(image_path, as_gray=True)
        dataset.append(image)
        label.append(0)

    L1_images = sorted(glob.glob(str(Path(im_root_folder, '1', '*.tiff'))))
    for i, image_path in enumerate(L1_images):
        image = io.imread(image_path, as_gray=True)
        dataset.append(image)
        label.append(1)
    
    # Record number of images and train/test split
    global N0
    N0 = label.count(0)
    global N1
    N1 = label.count(1)
    global TS
    TS = 0.2
    
    # Convert to np array
    dataset = np.array(dataset)
    label = np.array(label)

    # Split the dataset as needed
    X_train, X_test, y_train, y_test = train_test_split(dataset, 
        label,
        test_size = TS,
        random_state = 0,
        stratify = label)

    #Data normalization (0,1) to help convergence
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    
    return X_train, X_test, y_train, y_test

# Augment dataset
def data_augmentation_generator(X, y, batch_size=32):
    """
    Generate batches of augmented data
    
    Args:
        X: Input data with shape (samples, sequence_length, channels)
        y: Target labels (one-hot encoded)
        batch_size: Number of samples per batch
        
    Yields:
        Batches of augmented data
    """
    num_samples = X.shape[0]
    
    while True:
        # Shuffle the data
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices].copy()
            batch_y = y[batch_indices].copy()
            
            # Apply augmentation
            for j in range(len(batch_X)):
                # Apply random augmentation with certain probability
                if np.random.random() < 0.5:
                    # Apply one or more augmentations
                    augmentation_choice = np.random.random()
                    
                    if augmentation_choice < 0.5:
                        # Add calibrated noise
                        noise_level = np.random.uniform(0.01, 0.05)
                        batch_X[j] += np.random.normal(0, noise_level, size=batch_X[j].shape)
                        
                    else:
                        # Scale amplitude
                        scale_factor = np.random.uniform(0.95, 1.05)
                        batch_X[j] *= scale_factor
            
            yield batch_X, batch_y

# Define ML model parameters
def build_model_v1(input_shape):
    model = Sequential()
    
    # First convolutional block
    model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Second convolutional block
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Third convolutional block
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer - adjust units and activation based on your task
    # For classification:
    model.add(Dense(1, activation='sigmoid')) # Class number is 1 for binary classification
    
    return model

def build_model_v2(input_shape=(100, 2), num_classes=1):
    """
    Creates a CNN model adapted for small datasets of multi-channel 1D signals.
    
    Args:
        input_shape: Tuple specifying input dimensions (sequence_length, channels)
        num_classes: Number of signal classes to classify
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block with strong regularization
    model.add(Conv1D(filters=16, kernel_size=7, 
                     activation='relu', 
                     padding='same',
                     kernel_regularizer=l2(0.001),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    # Second convolutional block
    model.add(Conv1D(filters=32, kernel_size=5, 
                     activation='relu', 
                     padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', 
                   kernel_regularizer=l2(0.001),
                   activity_regularizer=l1(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='sigmoid' if num_classes==1 else 'softmax'))
    
    # Compile model with Adam optimizer
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# For automatic version numbering    
def get_latest_version(model_dir):
    """Get the latest version number from model files in directory."""
    model_list = sorted(glob.glob(str(Path(model_dir, 'PD_timeseries_classifier_*.h5'))))
    
    if not model_list:
        return '000'
    
    # Extract version numbers from filenames and find the highest
    versions = []
    for model_file in model_list:
        filename = Path(model_file).stem
        # Look for patterns like "_001", "_v001", "_123" at the end of filename
        import re
        version_match = re.search(r'[_v]?([0-9]{3})$', filename)
        if version_match:
            try:
                version_num = int(version_match.group(1))
                versions.append(version_num)
            except ValueError:
                continue
    
    if versions:
        next_version = max(versions) + 1
        return f'{next_version:03d}'
    else:
        return '001'  # Start from 001 if no valid versions found

# Define callbacks for training
def get_training_callbacks(model_dir, v_num):
    """
    Returns a list of callbacks for training the model
    """
    callbacks = [
        #Use CSVlogger to record training process
        CSVLogger(Path(model_dir, f'{v_num}_info', f'CWT_image_binary_classification_{v_num}_training_log.csv'),
            append=True),
            
        # Stop training when validation loss stops improving
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        
        # Save best model during training
        ModelCheckpoint(Path(model_dir, f'PD_timeseries_classifier_{v_num}.h5'),
            save_best_only=True,
            monitor='val_loss')
    ]
    return callbacks

# Resize test and train datasets for Keras compatability
def batch_resize(datasets, size):
    resized_datasets = []
    
    for dset in datasets:
        print(dset.shape)
        dset = dset.reshape(size)
        resized_datasets.append(dset)
    
    return resized_datasets

# Run model training
def train_model(X_train, X_test, y_train, y_test, model, im_root_folder):
    SIZE_X = X_train.shape[1]
    SIZE_Y = X_train.shape[2]
    
    model_dir = Path(im_root_folder, '..', 'models')
    # Get latest model version in the folder and increment by 1
    v_num = get_latest_version(model_dir)
    v_num = f'{int(v_num)+1:03}'
    
    # Create output folder for supporting file
    if not os.path.exists(Path(model_dir, f'{v_num}_info')):
        os.makedirs(Path(model_dir, f'{v_num}_info'))
    
    # Get training callbacks
    callbacks = get_training_callbacks(model_dir, v_num)
    
    # Learning rate
    learning_rate = 0.0001

    # Optimizer
    Adam_optimizer = Adam(learning_rate=learning_rate)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Model compile
    model.compile(optimizer= Adam_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
    # Augment training data
    batch_size = 8
    train_generator = SignalDataGenerator(X_train, y_train, batch_size=batch_size, augment=True)
    val_generator = SignalDataGenerator(X_test, y_test, batch_size=batch_size, augment=False)
    
    # Model training
    history = model.fit(train_generator,
        validation_data = val_generator,
        steps_per_epoch = len(X_train) // batch_size,
        verbose = 1,
        epochs = 300,
        shuffle = False,
        callbacks = callbacks)
    
    # Save model
    # model.save(Path(model_dir, f'PD_timeseries_classifier_{v_num}.h5'))
    
    # Save model metadata
    with open(Path(model_dir, f'{v_num}_info', f'CWT_image_binary_classification_{v_num}_info.txt'), 'w') as f:
        l1 = f'Version: {v_num}\n'
        l2 = f'Image source: {str(im_root_folder)}\n'
        l3 = f'Image shape: ({SIZE_X}, {SIZE_Y}, {img_channels})\n'
        l4 = f'Number of images: {N0} \'0\' label, {N1} \'1\' label ({N0+N1} total)\n'
        l5 = f'Train/test split: {int((1-TS)*100)}/{int(TS*100)}\n\n'
        
        f.writelines([l1, l2, l3, l4, l5])
        with redirect_stdout(f):
            model.summary()
        
        f.write('\n\nNotes:')

class SignalDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices].copy()
        
        if self.augment:
            # Apply augmentation
            for i in range(len(batch_X)):
                # Apply random augmentation with certain probability
                if np.random.random() < 0.5:
                    # Apply one or more augmentations
                    augmentation_choice = np.random.random()
                    
                    if augmentation_choice < 0.5:
                        # Add calibrated noise
                        noise_level = np.random.uniform(0.01, 0.05)
                        batch_X[i] += np.random.normal(0, noise_level, size=batch_X[i].shape)
                        
                        # clip <---------------------------------------------------------------------------
                        
                    else:
                        # Scale amplitude
                        scale_factor = np.random.uniform(0.95, 1.05)
                        batch_X[i] *= scale_factor
                        
                        # clip <---------------------------------------------------------------------------
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)

# Main loop iterates through files to compile training data
def main():
    # Get list of tracks to use, based on set filters
    track_list = filter_logbook(log)
    if debug:
        print('Filtered logbook:')
        print(track_list)
    
    # Define root folder for images
    data_folder = get_paths()['hdf5']
    im_root_folder = Path(data_folder, f'Photodiode_{wl}ms_window_plots', f'window_plots_{bit_depth}bit')
    
    if generate_signal_overview_plots or generate_training_data:        
        # Read local data locations
        hdf5_fpaths = sorted(glob.glob(f'{data_folder}/*.h*5'))
        hdf5_fpaths = filter_fpaths(hdf5_fpaths, track_list)
        n_files = len(hdf5_fpaths)
        
        # Starting values for finding global min and max signal intensities
        overall_min = 1000
        overall_max = 0
        
        # Manually defined min and max if previously found
        vmin = transform_data(raw_signal_min)
        vmax = transform_data(raw_signal_max)
        
        for i, fpath in enumerate(hdf5_fpaths):
            # Read track ID from file name
            trackid = Path(fpath).name[:7]
            
            # Skip if track ID in exclude list
            if trackid in exclude:
                continue
            
            # Print progress to terminal
            if not debug:
                printProgressBar(i,
                    n_files-1,
                    prefix='Progress:',
                    suffix=f'Working on {trackid}',
                    length=50)
            else:
                print(trackid)
            
            # Read specified datasets from hdf5 file as a dictionary
            data = read_hdf5(fpath, dset_names)
            
            # Convert to numpy array of dimensions (n_samples, len(dset_names))
            data = np.array(list(data.values()))
            
            # Trim data to laser start/stop
            data = data[:, trim+int((wo/wl)*n_samples):-trim-1]
            
            # Extract time column
            time = data[0]
            
            # Take logarithm of PD signals, discarding time column, to deal with extreme spikes
            data = transform_data(data[1:])
            
            # Print min/max of each signal if it is a global min/max
            if debug:
                dat_min = np.min(data)
                if dat_min < overall_min:
                    overall_min = dat_min
                    print(f'Min: {overall_min}')
                dat_max = np.max(data)
                if dat_max > overall_max:
                    overall_max = dat_max
                    print(f'Max: {overall_max}')
                
            # Calculate number of windows that can be extracted from signal
            total_samples = data.shape[1]
            n_windows = total_samples//n_samples
            
            # Create dict to store windowed data
            windows = {}
            
            # Iterate through signal, slicing into designated window lengths
            for j in range(n_windows):
                # Assign PD 1 and 2 signals to a new array and slice to window length
                w_j = data[:, j*n_samples:(j+1)*n_samples]
                windows[j] = w_j     
                
                if generate_training_data:
                    # Rescale to desired image bit depth
                    w_j_rescaled = np.multiply(np.divide(np.subtract(w_j, vmin), vmax-vmin), 65535).astype(np.uint16)
                    
                    # Trim to window bounds
                    start = str(round(1000 * time[j*n_samples], 3))
                    stop = str(round(1000 * time[(j+1)*n_samples], 3))
                    
                    # Set output location and file name
                    filename = f'{trackid}_PD_slice_{start}-{stop}ms.tiff'
                    if not os.path.exists(im_root_folder):
                        os.makedirs(im_root_folder)
                    save_path = Path(im_root_folder, filename)
                    
                    # Save signal sindow as image
                    io.imsave(save_path, w_j_rescaled)
            
            # Visualise data (for development/debugging)
            if generate_signal_overview_plots:
                create_window_plot(trackid, n_windows, data, windows, n_samples, total_samples, sr, wl)
        
    # Sort images into label folders
    if label_images:
        sort_images(im_root_folder, track_list)
    
    # Train ML model on training dataset
    if train:
        # Get train/test data from folder
        X_train, X_test, y_train, y_test = get_training_dataset(im_root_folder)
        
        # Resize training data for Keras compatability
        X_train, X_test = batch_resize([X_train, X_test], [-1, 100, 2])
        
        # Get model
        model = build_model_v2((100, 2))
        
        # Train model
        train_model(X_train, X_test, y_train, y_test, model, im_root_folder)
    
         
if __name__ == '__main__':
    main()