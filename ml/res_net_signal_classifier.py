#!/usr/bin/env python

"""
This script trains and evaluates res-net ML models for classifying
photodiode process monitoring signals

Author:     Rubén Lambert-Garcia
Version:    0.1
"""

# Package imports
import os, sys, h5py, keras, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from pathlib import Path
from skimage import io

# Local imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, get_paths, printProgressBar

# Read local data
data_folder = get_paths()['hdf5']
fpaths = sorted(glob.glob(f'{data_folder}/*.h*5'))
log = get_logbook()

# Signals to use (dataset names)
dset_names = ['AMPM/Time',
              'AMPM/Photodiode1Bits',
              'AMPM/Photodiode2Bits']
              
# Settings
sr = 100 # Sampling rate in kHz
wl = 1  # Window length in ms
trim = 500 # Number of points to trim from start/end of data (default 500 for AMPM)
debug = True
show_figs = False
save_figs = False
train = True

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
def get_data(fpath, dset_names):
    data = {}
    with h5py.File(fpath, 'r') as file:
        for i, name in enumerate(dset_names):
            data[i] = np.array(file[name])
    
    return data

# Function for transforming the data to better represnt the distribution of values
def transform_data(data):
    # Take natural log of signal amplitudes
    data[1:] = np.log(data[1:])
    return data

# Finction to generate the overview plot showing original signals and signal window images
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
    vmin = transform_data(54)
    vmax = transform_data(2440)
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
        save_folder = Path(data_folder, f'Photodiode_{wl}ms_window_plots')
        if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
        save_path = Path(save_folder, f'{trackid}_{wl}ms_PD_windows.png')
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    
    if show_figs:
        plt.show()
    
    plt.close()
    

# Function to read signal window images into memory for model training
def build_training_dataset():
    
    pass

# Function to train ML model
def train_model(training_data):
    
    pass

# Main loop iterates through files to compile training data
def main():
    n_files = len(fpaths)
    overall_min = 1000
    overall_max = 0
    for i, fpath in enumerate(fpaths):
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
        data = get_data(fpath, dset_names)
        
        # Convert to numpy array of dimensions (n_samples, len(dset_names))
        data = np.array(list(data.values()))
        
        # Trim data to laser start/stop
        data = data[:, trim:-trim]
        
        # Take logarithm to deal with extreme spikes
        data = transform_data(data)
        
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
            w_j = data[1:, j*n_samples:(j+1)*n_samples]
            windows[j] = w_j
            print(w_j)
            
            if train:
                start = str(round(1000 * data[0][j*n_samples], 3)).replace('.', '_')
                stop = str(round(1000 * data[0][(j+1)*n_samples], 3)).replace('.', '_')
                filename = f'{trackid}_PD_slice_{start}-{stop}ms.tiff'
                print(filename)
                io.imsave(filename, w_j)
        
        # Visualise data (for development/debugging)
        if debug:
            create_window_plot(trackid, n_windows, data, windows, n_samples, total_samples, sr, wl)
            
        input()
        
        # Label and save windows as images, with folder corresponding to label
        # Labels are read from logbook
        
        # build_training_dataset()
        
        # if train:
            # train_model()
        
        
        
            
    print(f'Overall - Min: {overall_min}, Max: {overall_max}')      
if __name__ == '__main__':
    main()