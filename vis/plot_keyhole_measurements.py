import functools, glob, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import *

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

mode = 'save'

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')
    
data_path = Path(filepath, 'keyhole_measurements_lagrangian')
logbook = get_logbook()

colour_key = {'unstable keyhole': '#fde725',
              'quasi-stable keyhole': '#5ec962',
              'keyhole flickering': '#3b528b',
              'quasi-stable vapour depression': '#21918c',
              'conduction': '#440154'
              }

keyhole_data_comp = {}
files = glob.glob(str(Path(data_path, '*_keyhole_measurements.csv')))
for csv_file in files:
    trackid = Path(csv_file).name[:7]       # Get trackid from file name
    print(f'Reading {trackid}')
    keyhole_data = pd.read_csv(csv_file)            # Read track data from csv
    keyhole_data_comp[trackid] = keyhole_data               # Add data table to dictionary
    col_names = keyhole_data.keys()                 # Save collumn names to list

trackids = keyhole_data_comp.keys()
keyhole_regimes = []
LEDs = []
for trackid in trackids:
    track_data = get_logbook_data(logbook, trackid)
    keyhole_regimes.append(track_data['keyhole_regime'])
    LEDs.append(track_data['LED'])
track_df = pd.DataFrame({'trackid': trackids,
                         'keyhole_regime': keyhole_regimes,
                         'LED': LEDs,
                         }).sort_values('LED').reset_index(drop=True)
print(track_df)

print('Data imported\n\nCreating figures')

for col_name in col_names[3:]:      # Iterate through collumn names except 0, 1 and 2 (index, time and centroid)
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, tight_layout=True)
    
    for i, row in track_df.iterrows():
        trackid = row['trackid']
        regime = row['keyhole_regime']
        LED = row['LED']
        col_data = keyhole_data_comp[trackid][col_name][:-45]
        if col_name != 'area':
            col_data = [x * 4.3 for x in col_data]  # convert distances from pixels to um
        else:
            col_data = [math.sqrt(x) * 4.3 for x in col_data]    # convert area from pixels to um^2
        violin = ax.violinplot(col_data,
                               # positions = [i+1],
                               positions = [LED],
                               # widths = 0.6,
                               widths = 40,
                               showextrema = False,
                               showmedians = False
                               )
        ax.scatter(LED,
                   # i+1,
                   np.median(col_data),
                   marker='+',
                   c='k',
                   linewidths = mpl.rcParams['axes.linewidth']
                   )
        body = violin['bodies'][0]
        try:
            body.set_facecolor(colour_key[regime])
            body.set_edgecolor('black')
            body.set_linewidth(mpl.rcParams['axes.linewidth'])
            body.set_label(regime)
            body.set_alpha(1)
        except KeyError:
            print(f'{trackid} keyhole regime definition invalid')
        # ax.boxplot(col_data,
                   # positions = [i],
                   # showfliers = False,
                   # medianprops = {'color': 'k'}
                   # )
        
    # x_inds = np.arange(1, len(trackids)+1)
    # ax.set_xticks(x_inds)
    # ax.set_xticklabels(track_df['trackid'], rotation=45, ha='right')
    # ax.set_xlabel('Track ID')
    
    # ax.set_xticks(track_df['LED'])
    # ax.set_xticklabels(track_df['LED'], rotation=45, ha='right')
    ax.set_xlabel('LED (J/m)')
    
    ax.set_ylabel('Area (μ$\mathregular{m^2}$)' if col_name == 'area' else 'Distance (μm)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    fig.suptitle(col_name.replace('_', ' ').replace('max', 'max.'), ha='center', position=(0.57, 0.85))
    
    if mode == 'save':
        output_filename = f'keyhole_{col_name}_stats_4x4.png'
        output_filepath = Path(data_path, output_filename) 
        plt.savefig(output_filepath)
        print(f'{col_name} figure saved to {output_filepath}')
    else:
        plt.show()
    