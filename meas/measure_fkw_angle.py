import h5py, glob, functools, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, segmentation
from my_funcs import get_logbook

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

input_dset_name = 'keyhole_bin'

filter_by_regime = False
preview_edge_fits = False
preview_final_result = True
save_result_figures = True
save_result_data = True

framerate = 504000 # fps
ignore_last_n_frames = 100
# ignore_laser_onset = 0 # mm # Changed below

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    print(f'Reading from {filepath}\n')

output_folder = Path(filepath, 'fkw_angle_measurements')
if not os.path.exists(output_folder): os.makedirs(output_folder)

log = get_logbook() # Read logbook

if filter_by_regime == True:
    log_filt = log[np.logical_or(log['Melting regime'] == 'unstable keyhole', # Filter logbook
                                 log['Melting regime'] == 'quasi-stable keyhole',
                                 # log['Melting regime'] == 'quasi-stable vapour depression'
                                 )
                   ]
else:
    log_filt = log

def get_fkw_angle(dset):
    dset_edge_mask = np.zeros_like(dset)
    fkw_measurements = {'time': [],
                        'fkw_angle': [],
                        'n_points_fit': [],
                        'm': [],
                        'c': []
                        }
    for i, im in enumerate(dset):
        edge_mask_i = np.zeros_like(im)
        
        time = i * 1/framerate * 1000 # ms
        fkw_measurements['time'].append(time)
        
        try:
            min_row, min_col, max_row, max_col = measure.regionprops(im)[0]['bbox']
        except IndexError:
            for k in fkw_measurements:
                if k != 'time':
                    fkw_measurements[k].append(0)
            continue
        
        crop_margin = int(0.25 * (max_row - min_row))
        
        for r in range(min_row+crop_margin, max_row-crop_margin):
            row = im[r]
            right_edge_ind = max(i for i, val in enumerate(row) if val != 0)
            edge_mask_i[r, right_edge_ind] = 255
        
        edge_points = np.where(edge_mask_i == 255)
        n_points_fit = len(edge_points[0])
        fkw_measurements['n_points_fit'].append(n_points_fit)
        
        [m, c] = np.polyfit(edge_points[0], edge_points[1], 1)
        fkw_measurements['m'].append(m)
        fkw_measurements['c'].append(c)
        
        X = range(im.shape[0])
        Y = [m * x + c for x in X]
        fkw_angle = np.arctan(m) * 360 / (2 * np.pi) + 90
        fkw_measurements['fkw_angle'].append(fkw_angle)
        
        dset_edge_mask[i] = edge_mask_i
        
        if preview_edge_fits == True:
            preview_im = im / 2
            preview_im[edge_mask_i == 255] = 255
            plt.imshow(preview_im)
            plt.plot(Y, X, 'w--', lw=0.5)
            plt.text(150, 25, f'θ_fkw = {round(fkw_angle, 2)}°\nFitted from {n_points_fit} points', c='w')
            plt.show()
    
    return pd.DataFrame(fkw_measurements)

def main():
    results = {'trackid': [],
               'fkw_angle_mean': [],
               'fkw_angle_stdev': [],
               'n_samples': [],
               'm': [],
               'c': []
               }
    for f in sorted(glob.glob(str(Path(filepath, '*.hdf5')))):
        fname = Path(f).name
        trackid = fname[:7]
        
        if trackid not in log_filt['trackid'].tolist():
            continue
            
        print('\nReading %s' % fname)
        
        # Read keyhole binary dataset, if none found record blank row in results table and continue to next file
        with h5py.File(f, 'r') as file:
            try:
                dset = file[input_dset_name]
            except KeyError:
                for key in results:
                    if key == 'trackid':
                        results[key].append(trackid)
                    else:
                        results[key].append(None)
                continue
                
            print('shape: %s, dtype: %s'% (dset.shape, dset.dtype))
            scan_speed = log_filt.loc[log_filt['trackid']==trackid, 'Scan speed [mm/s]'].iloc[0]
            # ignore_first_n_frames = int(framerate * ignore_laser_onset / scan_speed)
            ignore_first_n_frames = 70
            
            fkw_measurements = get_fkw_angle(dset[ignore_first_n_frames:-ignore_last_n_frames])
            len_raw = len(fkw_measurements)
            points_filter = fkw_measurements['n_points_fit'] > 4
            angle_filter = np.logical_and(fkw_measurements['fkw_angle'] < 100, fkw_measurements['fkw_angle'] > 10)
            measurements_filtered = fkw_measurements[points_filter & angle_filter]
            len_filt = len(measurements_filtered)
            
            if save_result_data == True:
                pd.DataFrame(measurements_filtered).to_csv(Path(output_folder, f'{trackid}_fkw_angle_measurements_filtered.csv'))
                pd.DataFrame(fkw_measurements).to_csv(Path(output_folder, f'{trackid}_fkw_angle_measurements_raw.csv'))
            
            m_mean = np.mean(measurements_filtered['m'])
            c_mean = np.mean(measurements_filtered['c'])
            fkw_angle_mean = np.mean(measurements_filtered['fkw_angle'])
            fkw_angle_stdev = np.std(measurements_filtered['fkw_angle'])
            
            results['trackid'].append(trackid)
            results['fkw_angle_mean'].append(fkw_angle_mean)
            results['fkw_angle_stdev'].append(fkw_angle_stdev)
            results['n_samples'].append(len_filt)
            results['m'].append(m_mean)
            results['c'].append(c_mean)
            
            if preview_final_result == True or save_result_figures == True:
                fig, ax = plt.subplots(figsize=(3.15, 3.15), dpi=300)
                fig.suptitle(trackid)
                ax.imshow(np.mean(file['bs-f40_lagrangian'][50:-100], axis=0), vmin=100, vmax=150, cmap='gray')
                ax.set_axis_off()
                X = range(dset.shape[1])
                Y = [m_mean * x + c_mean for x in X]
                ax.plot(Y, X, 'k--', lw=0.6)
                ax.text(Y[-1]+10, X[-1]-5,
                         f'θ_fkw = {round(fkw_angle_mean, 2)}°',
                         c='k',
                         fontsize='small'
                         )
                # ax.text(Y[-1]+10, X[-1]-5, 'θ_fkw')
                
                if preview_final_result == True:
                    print(f'Used {len_filt}/{len_raw} measurements')
                    plt.show()
                    
                if save_result_figures == True:
                    plt.savefig(Path(output_folder, f'{trackid}_fkw_angle_measurement.png'))
        
        plt.close('all')
        
    if save_result_data == True:
        pd.DataFrame(results).to_csv(Path(output_folder, 'fkw_angle_measurements.csv'))
    
if __name__ == "__main__":
	main()