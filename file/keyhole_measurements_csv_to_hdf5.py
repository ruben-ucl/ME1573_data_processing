import glob, functools, h5py, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, interpolate_low_quality_data, validate_timeseries_quality

csv_path = get_paths()['KH_meas']
hdf5_path = get_paths()['hdf5']

mode = 'append' # 'append' or 'overwrite'

# Safety check before overwriting data
if mode == 'overwrite':
    confirm = input('Are you sure you want to run in \'overwrite\' mode? (y/n)')
    if confirm != 'y':
        mode = 'append'

for file in sorted(glob.glob(f'{csv_path}/*keyhole_measurements_nofilt.csv')):
    print('\n')
    df = pd.read_csv(file, index_col=0, keep_default_na=False)
    fname = Path(file).name
    trackid = fname[:7]
    print('--------------------------------------')
    print(f'========== trackid: {trackid} ==========')
    print('--------------------------------------\n')
    
    try:
        hdf5_fpath = glob.glob(fr'{hdf5_path}\{trackid}.hdf5')[0]
        print('hdf5 file: ', hdf5_fpath)
    except IndexError:
        print(f'No hdf5 found for {trackid}')
        continue
    
    # Shift time so laser switches on at t=0 and switch from ms to s
    if 'kh_' in str(csv_path):
        df['time'] -= 0.05
         # Convert area to um2
        df['area'] *= 4.3**2
    
    t_ms = df['time'].copy()   # keep for plotting
    df['time'] *= 0.001 # convert from ms to s
    
    data_clean = {}
    # Interpolate points with poor fits
    for dataset in ['area', 'max_depth', 'max_length', 'depth_at_max_length']:
        print(f'\n---------- Dataset: {dataset} ----------')
        corrected_data, interpolated_mask = interpolate_low_quality_data(df[dataset], None)
        data_clean[dataset] = corrected_data
        
        stats = validate_timeseries_quality(df[dataset], None)
        print(f"\nQuality statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    df_clean = pd.DataFrame(data_clean)
    
    # corrected_data, interpolated_mask = interpolate_low_quality_data(df['fkw_angle'], df['n_points_fit'])
    
    # Get quality statistics
    # stats = validate_timeseries_quality(df['fkw_angle'], df['n_points_fit'])
    # print(f"\nQuality statistics:")
    # for key, value in stats.items():
        # print(f"  {key}: {value}")
    
    # Extract centroid coordinates
    # cx = []
    # cy = []
    # for e in df['centroid'].copy():
        # if e == '':
            # cx.append(np.nan)
            # cy.append(np.nan)
        # else:
            # coords = e[1:-1].split(', ')
            # cx.append(float(coords[1]))
            # cy.append(float(coords[0]))
    # df['centroid_x'] = cx
    # df['centroid_y'] = cy
    
    # Plot data
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(6.3, 3.15), dpi=300, tight_layout=True)
    fig.suptitle(f'{trackid} KH measurements')
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    ax1.set_xlabel('Time [ms]')
    # ax1.set_ylabel('Angle [degrees]')
    ax1.set_ylabel('Distance [μm]')
    # ax1.plot(t_ms, df['fkw_angle'], lw=0.5, ls='--', label='FKW angle')
    # ax1.plot(t_ms, corrected_data, lw=0.5, label='FKW angle_corrected')
    ax1.plot(t_ms, df['max_depth'], lw=0.5, label='KH depth')
    ax1.plot(t_ms, df['max_length'], lw=0.5, label='KH length')
    ax1.legend(loc='upper left')
    
    # ax2.set_ylabel('N points')
    # ax2.plot(t_ms, df['n_points_fit'], 'k', lw=0.5, ls=':', label='N points')
    ax2.set_ylabel('Area [μm^2]')
    ax2.plot(t_ms, df['area'], 'k', lw=0.5, label='KH area')
    ax2.legend(loc='upper right')
    
    # plt.savefig(Path(csv_path, 'plots', f'{trackid}_KH_measurements.png'))
    plt.show()
    plt.close()
    
    with h5py.File(hdf5_fpath, 'r+') as hdf5_file:
        for dset in ['time', 'area', 'max_depth', 'max_length', 'depth_at_max_length']:
        # for dset in ['fkw_angle']:
            if mode == 'overwrite':
                try:
                    del hdf5_file[f'KH/{dset}']
                except KeyError:
                    pass
            try:
                hdf5_file[f'KH/{dset}'] = corrected_data
                # hdf5_file[f'KH/{dset}'] = df[dset]
            except OSError:
                print(f'Error: \'{dset}\' HDF5 dataset already exists')
                
        # plt.plot(np.array(hdf5_file[f'AMPM/Time']), np.array(hdf5_file[f'AMPM/Photodiode1Bits']), lw=0.5)
        # plt.show()
        