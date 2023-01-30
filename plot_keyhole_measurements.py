import functools, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

mode = 'preview'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
data_path = Path(filepath, 'keyhole_measurements_lagrangian')
logbook = get_logbook()

data_comp = {}

files = glob.glob(str(Path(data_path, '*_keyhole_measurements.csv')))
for csv_file in files:
    trackid = Path(csv_file).name[:7]       # Get trackid from file name
    print(f'Reading {trackid}')
    data = pd.read_csv(csv_file)            # Read track data from csv
    data_comp[trackid] = data               # Add data table to dictionary
    col_names = data.keys()                 # Save collumn names to list

trackids = sorted(data_comp)
keyhole_regimes = []
for trackid in trackids:
    _, _, _, regime = get_logbook_data(logbook, trackid)
    keyhole_regimes.append(regime)
    
print('Data imported\n\nCreating figures')

for col_name in col_names[3:]:      # Iterate through collumn names except 0, 1 and 2 (index, time and centroid)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig_data = []
    
    for trackid in trackids:
        col_data = data_comp[trackid][col_name][:-45]
        fig_data.append(col_data)
        
    violins = ax.violinplot(fig_data,
                            showextrema = False,
                            showmedians = False,
                            )
    for i, pc in enumerate(violins['bodies']):
        if keyhole_regimes[i] == 'unstable keyhole':
            pc.set_facecolor('#fde725')
            pc.set_label('unstable keyhole')
        elif keyhole_regimes[i] == 'quasi-stable keyhole':
            pc.set_facecolor('#5ec962')
            pc.set_label('quasi-stable keyhole')
        elif keyhole_regimes[i] == 'keyhole flickering':
            pc.set_facecolor('#21918c')
            pc.set_label('keyhole flickering')
        elif keyhole_regimes[i] == 'quasi-stable vapour depression':
            pc.set_facecolor('#3b528b')
            pc.set_label('quasi-stable vapour depression')
        elif keyhole_regimes[i] == 'conduction':
            pc.set_facecolor('#440154')
            pc.set_label('conduction')
        else:
            print(f'{trackids[i]} keyhole regime definition invalid')
        pc.set_alpha(0.65)
        
    ax.boxplot(fig_data,
               showfliers = False    ,
               medianprops = {'color': 'k'}
               )
    
    x_inds = np.arange(1, len(trackids)+1)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(trackids)
    ax.set_xlabel('Track ID')
    ax.set_ylabel('Area (μ$\mathregular{m^3}$)' if col_name == 'area' else 'Distance (μm)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    fig.suptitle(col_name)
    
    if mode == 'save':
        output_filename = f'keyhole_{col_name}_stats.png'
        output_filepath = Path(data_path, output_filename) 
        plt.savefig(output_filepath)
        print(f'{col_name} figure saved to {output_filepath}')
    else:
        plt.show()
    