import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, functools, glob
import pywt

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes


def get_paths():
    path_dict = {}
    for file in glob.glob('dirs/*.txt'):
        with open(file, encoding='utf8') as f:
            path_dict[Path(file).stem] = Path(f.read())
    return path_dict

def get_logbook():
    logbook_path = get_paths()['logbook']
    print(f'Trying to read logbook: {logbook_path.name}')
    try:
        logbook = pd.read_excel(logbook_path,
            sheet_name='Logbook',
            converters={'Substrate No.': str, 'Sample position': str},
            keep_default_na = False,
            na_values = ['', 'NaN', 'nan', 'null']
            )
        # logging.info('Logbook data aquired from %s' % logbook_path)
        print('Logbook read successfully', end='\n\n')
    
        return logbook
    
    except Exception as e:
        print('Error: Failed to read logbook')
        print(e)
        # logging.info('Failed to read logbook - unable to continue')
        # logging.debug(str(e))
        raise
        
def get_logbook_data(logbook, trackid, layer_n=1):  # Get scan speed and framerate from logbookprint('Reading scan speed and framerate from logbook')
    track_row = logbook.loc[(logbook['trackid'] == trackid) &
        (logbook['Layer'] == layer_n)
        ]
    track_data = {}
    track_data['peak_power'] = int(track_row['Power [W]'].iloc[0])
    track_data['avg_power'] = int(track_row['Avg. power [W]'].iloc[0])
    track_data['pt_dist'] = int(track_row['Point distance [um]'].iloc[0])
    track_data['exp_time'] = int(track_row['Exposure time [us]'].iloc[0])
    track_data['pt_jump_delay'] = int(track_row['Point jump delay [us]'].iloc[0])
    track_data['scan_speed'] = int(track_row['Scan speed [mm/s]'].iloc[0])
    track_data['LED'] = int(track_row['LED [J/m]'].iloc[0])
    track_data['framerate'] = int(track_row['Frame rate (kHz)'].iloc[0] * 1000)
    track_data['laser_onset_frame'] = int(track_row['Laser onset frame #'].iloc[0])
    track_data['melting_regime'] = track_row['Melting regime'].iloc[0]
    
    return track_data

def define_collumn_labels():
    # Dict item structure:
    # label: [logbook header, axis label]
    col_dict = {'power':            ['Avg. power [W]',
                                     'Power [W]'
                                     ],
        'pt_dist':                  ['Point distance [um]',
                                     'Point distance [μm]'
                                     ],
        'exp_t':                    ['Exposure time [us]',
                                     'Exposure time [μs]'
                                     ],
        'scan_speed':               ['Scan speed [mm/s]',
                                     'Scan speed [mm/s]'
                                     ],
        'LED':                      ['LED [J/m]',
                                     'LED [J/m]'
                                     ],
        'regime':                   ['Melting regime',
                                     'Melting regime'
                                     ],
        'n_pores':                  ['n_pores',
                                     'Keyhole pore count'
                                     ],
        'pore_density':              ['pore_density [/mm3]',
                                      'Keyhole porosity [/mm\u00b3]'
                                      ],
        'pore_vol':                 ['pore_vol_mean [um^3]',
                                     'Mean pore volume [μm\u00b3]'
                                     ],
        'pore_angle':               ['pore_angle_mean [°]',
                                     r'Mean pore angle [$\degree$]'
                                     ],
        'pore_roundness':           ['pore_roundness_mean',
                                     'Mean pore roundness'
                                     ],
        'eot_depression':           ['end_of_track_depression',
                                     'End of track\ndepression'
                                     ],
        'eot_depression_depth':     ['end_of_track_depression_depth',
                                     'End of track depression depth [μm]'
                                     ],
        'h_pores':                  ['hydrogen_pores',
                                     'Hydrogen porosity'
                                     ],
        'MP_depth':                 ['melt_pool_depth [um]',
                                     'Melt pool depth [μm]'
                                     ],
        'MP_length':                ['melt_pool_length [um]',
                                     'Melt pool length [μm]'
                                     ],
        'MP_width':                 ['track_width_mean [um]',
                                     'Melt pool width [μm]'
                                     ],
        'track_height':             ['track_height [um]',
                                     'Track height [μm]'
                                     ],
        'MP_vol':                   ['total_melt_volume [mm^3]',
                                     r'Melt pool volume, $\it{V}$ [mm\u00b3]'
                                     ],
        'MP_vol_err':               ['melt_pool_volume_error [mm^3]',
                                     'Melt pool volume error [mm\u00b3]'
                                     ],
        'MP_rear_wall_angle':       ['rear_melt_pool_wall_angle [deg]',
                                     r'Melt pool rear wall angle [$\degree$]'
                                     ],
        'melting_efficiency':       ['melting_efficiency',
                                     r'Melting efficiency, $\it{η}$'
                                     ],
        'R':                        ['R [mm/s]',
                                     'Solidification rate, R [mm/s]'
                                     ],
        'G1':                       ['G1 [K/mm]',
                                     'Temperature gradient @ D1 [K/mm]'
                                     ],
        'G2':                       ['G2 [K/mm]',
                                     'Temperature gradient @ D2 [K/mm]'
                                     ],
        'G3':                       ['G3 [K/mm]',
                                     'Temperature gradient @ D3 [K/mm]'
                                     ],
        'G_rear':                   ['G_rear [K/mm]',
                                     'Temperature gradient @ rear tip [K/mm]'
                                     ],
        'dT/dt1':                   ['dT/dt1 [K/s]',
                                     'Cooling rate @ D1 [K/s]'
                                     ],
        'dT/dt2':                   ['dT/dt2 [K/s]',
                                     'Cooling rate @ D2 [K/s]'
                                     ],                             
        'dT/dt3':                   ['dT/dt3 [K/s]',
                                     'Cooling rate @ D3 [K/s]'
                                     ],   
        'dT/dt_rear':               ['dT/dt_rear [K/s]',
                                     'Cooling rate @ rear tip [K/s]'
                                     ],
        'KH_depth':                 ['keyhole_max_depth_mean [um]',
                                     'Keyhole depth [μm]'
                                     ],
        'max_depth':                ['keyhole_max_depth_mean [um]',
                                     'Keyhole depth [μm]'
                                     ],
        'KH_depth_sd':              ['keyhole_max_depth_sd [um]',
                                     'Keyhole depth std. dev. [μm]'
                                     ],
        'KH_length':                ['keyhole_max_length_mean [um]',
                                     'Keyhole length [μm]'
                                     ],
        'max_length':               ['keyhole_max_length_mean [um]',
                                     'Keyhole length [μm]'
                                     ],
        'KH_area':                  ['keyhole_area_mean [um^2]',
                                     'Keyhole area [μm\u00b2]'
                                     ],
        'area':                     ['keyhole_area_mean [um^2]',
                                     'Keyhole area [μm\u00b2]'
                                     ],
        'KH_depth_at_max_length':   ['keyhole_depth_at_max_length_mean [um]',
                                     'Keyhole depth at max. length [μm]'
                                     ],
        'depth_at_max_length':      ['keyhole_depth_at_max_length_mean [um]',
                                     'Keyhole depth at max. length [μm]'
                                     ],
        'layer_thickness':          ['measured_layer_thickness [um]',
                                     'Powder layer thickness [μm]'
                                     ],
        'KH_depth_w_powder':        ['KH_depth_w_powder',
                                     'Normalised keyhole (from powder) [μm]'
                                     ],
        'fkw_angle':                ['fkw_angle_mean [deg]',
                                     r'FKW angle, $\it{\theta_{FKW}}$ [$\degree$]'
                                     ],
        'tan_fkw_angle':            ['tan_fkw_angle',
                                     'FKW angle tangent'
                                     ],
        'fkw_angle_sd':             ['fkw_angle_sd [deg]',
                                     r'FKW angle standard deviation [$\degree$]'
                                     ],
        'fkw_angle_n_samples':      ['fkw_angle_n_samples',
                                     'FKW angle sample count'
                                     ],
        'norm_H_prod':              ['Normalised enthalpy product',
                                     r'Normalised enthalpy product, $\it{\Delta H/h_m \dot L_{th}^*}$'
                                     ],
        'KH_AR':                    ['keyhole_aspect_ratio',
                                     'Keyhole aspect ratio'
                                     ],
        'PD_1_mean':                ['PD_1_mean [bits]',
                                     'PD 1 mean signal intensity'
                                     ],
        'PD_1_std':                 ['PD_1_std [bits]',
                                     'PD 1 signal intensity st. dev.'
                                     ],
        'PD_1_min':                 ['PD_1_min [bits]',
                                     'PD 1 signal intensity min.'
                                     ],
        'PD_1_max':                 ['PD_1_max [bits]',
                                     'PD 1 signal intensity max.'
                                     ],
        'Bo':                       ['Bo',
                                     'Bond number'
                                     ],
        'Ca':                       ['Ca',
                                     'Capillary number'
                                     ],                             
        'La':                       ['La',
                                     'Laplace number'
                                     ],                             
        'St':                       ['St',
                                     'Strouhal number'
                                     ],
        }
    return col_dict

def get_AMPM_channel_names():  
    ChannelNames = ['GalvoXDemandBits', # 0
        'GalvoXDemandCartesian',        # 1
        'GalvoYDemandBits',             # 2
        'GalvoYDemandCartesian',        # 3
        'FocusDemandBits',              # 4
        'FocusDemandCartesian',         # 5
        'GalvoXActualBits',             # 6
        'GalvoXActualCartesian',        # 7
        'GalvoYActualBits',             # 8
        'GalvoYActualCartesian',        # 9
        'FocusActualBits',              # 10
        'FocusActualCartesian',         # 11
        'Modulate',                     # 12
        'BeamDumpDiodeBits',            # 13
        'BeamDumpDiodeWatts',           # 14
        'Photodiode1Bits',              # 15
        'Photodiode1Watts',             # 16
        'Photodiode2Bits',              # 17
        'Photodiode2Watts',             # 18
        'PSDPositionXBits',             # 19
        'PSDPositionYBits',             # 20
        'PSDIntensity',                 # 21
        'PowerValue1',                  # 22
        'PowerValue2',                  # 23
        'Photodiode1Normalised',        # 24
        'Photodiode2Normalised',        # 25
        'BeamDumpDiodeNormalised',      # 26
        'LaserBackReflection',          # 27
        'OutputPowerMonitor']           # 28
    return ChannelNames

def get_start_end_frames(trackid, logbook, margin=50, start_frame_offset=0):
    track_data = get_logbook_data(logbook, trackid)
    framerate = track_data['framerate']
    scan_speed = track_data['scan_speed']
    start_frame = track_data['laser_onset_frame']
    
    n_frames = round(framerate * 4 / scan_speed) # based on track length of 4 mm
    f1 = start_frame - margin - start_frame_offset
    f2 = start_frame + n_frames + margin - start_frame_offset 
    
    return f1, f2
    
def get_substrate_mask(trackid, shape, substrate_surface_measurements_fpath):   # Generates a 2d mask of the substrate
    substrate_mask = np.zeros(shape, dtype=bool)
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    print(f'Substrate edge equation retrieved: y = {round(m, 3)}x + {round(c)}')
    n_rows, n_cols = substrate_mask.shape
    print('Calculating mask dimensions')
    for x in range(n_cols):
        surface_height = int(m * x + c)
        substrate_mask[surface_height:, x] = True
    
    return substrate_mask

def get_substrate_surface_coords(shape, substrate_surface_measurements_fpath, trackid):
    substrate_surface_df = pd.read_csv(substrate_surface_measurements_fpath, index_col='trackid')
    m = substrate_surface_df.at[trackid, 'm']
    c = substrate_surface_df.at[trackid, 'c']
    xx = range(shape[2])
    yy = [round(m * x + c) for x in xx]
    
    return xx, yy
    
def median_filt(dset, kernel):
    from skimage import filters
    if dset.dtype == np.uint8:
        median_filter = filters.rank.median_filt # faster method for 8-bit integers
    else:
        median_filter = filters.median
    
    tic = time.perf_counter()
    try:
        if dset.ndim == 3:
            output_dset = np.zeros_like(dset)
            for i, frame in enumerate(dset):
                print(f'Median filtering frame {i}', end='\r')
                output_dset[i] = median_filter(frame, kernel)
        elif dset.ndim == 2:
            output_dset = median_filter(dset, kernel)
        else:
            raise UnexpectedShape
        
        toc = time.perf_counter()
        print(f'Median filter duration: {toc-tic:0.4f} seconds')
        
        return output_dset
    except UnexpectedShape:
        print(f'Expected dataset of 2 or 3 dimensions, but received dataset with {dset.ndim} dimension(s)')

def view_histogram(a, title=None, show_std=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    if title != None:
        fig.suptitle(title)
    if a.ndim == 3:
        # Create panel of 4 frames from stack to display
        l = len(a)
        r1 = np.concatenate((a[0], a[l//3]), axis=1)
        r2 = np.concatenate((a[2*l//3], a[-1]), axis=1)
        im = np.concatenate((r1, r2), axis=0)
        im[im.shape[0]//2-1:im.shape[0]//2+1, :] = 255
        im[:, im.shape[1]//2-1:im.shape[1]//2+1] = 255
    else:
        im = a
    ax1.imshow(im, cmap='gray')
    frame_labels = {1: (0.01, 0.52),
        l//3+1: (0.51, 0.52),
        2*l//3+1: (0.01, 0.02),
        l+1: (0.51, 0.02)
        }
    for i in frame_labels:
        ax1.annotate(f'frame {i}', xy=frame_labels[i], xycoords='axes fraction', color='w')
    im_aspect = im.shape[0] / im.shape[1]
    ax2.set_box_aspect(im_aspect)
    ax2.hist(a.ravel(), bins=255, density=True, zorder=0)
    ax2.annotate(f'max: {np.max(a)}\nmin: {np.min(a)}\ndtype: {a.dtype}', xy=(0.01, 0.75), xycoords='axes fraction')
    
    if show_std:
        a_mean = np.mean(a)
        a_sigma = np.std(a)
        a_std_intervals = [a_mean + i * a_sigma for i in range(-3, 4)]
        marker_h = ax2.get_ylim()[1] / 30
        ax2.scatter(a_std_intervals, [marker_h for i in range(7)], c='k', marker='v')
        ax2.scatter((np.min(a), np.max(a)), (marker_h, marker_h), c='r', marker='v')
        txt_h = ax2.get_ylim()[1] / 15
        ax2.annotate('min', (np.min(a), txt_h), ha='center', color='r')
        ax2.annotate('max', (np.max(a), txt_h), ha='center', color='r')
        sigma_text = ['-3\u03C3', '-2\u03C3', '-\u03C3', '\u00B5', '\u03C3', '2\u03C3', '3\u03C3']
        for i, x in enumerate(a_std_intervals):
            ax2.annotate(sigma_text[i], (x, txt_h), ha='center')
    
    plt.show()
    
def compare_histograms(im_dict, fig_title=None):
    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(len(im_dict), 2, figsize=(5, len(im_dict)*2), dpi=300, tight_layout=True, sharex='col')
    # fig, axs = plt.subplots(len(im_dict), 2, figsize=(4, 4), dpi=300, tight_layout=True, sharex='col')
    if fig_title != None:
        fig.suptitle(fig_title)
    for i, key in enumerate(im_dict):
        im = im_dict[key]
        axs[i, 0].imshow(im, cmap='gray')
        axs[i, 0].title.set_text(key)
        axs[i, 0].axis('off')
        im_aspect = im.shape[0] / im.shape[1]
        axs[i, 1].set_box_aspect(im_aspect)
        axs[i, 1].set_xlim(0, 255)
        axs[i, 1].hist(im.ravel(), bins=255, density=True)
        axs[i, 1].annotate(f'max: {np.max(im)}\nmin: {np.min(im)}\ndtype: {im.dtype}', xy=(0.05, 0.6), xycoords='axes fraction')
    axs[len(im_dict)-1, 1].set_xlabel('greyscale value [0, 255]')
    
    plt.show()    

def hist_eq(dset):
    from skimage import exposure
    tic = time.perf_counter()
    # output_dset = np.zeros_like(dset, dtype=np.uint8)
    # for i, im in enumerate(dset):
        # output_dset[i] = (exposure.equalize_hist(im) * 255).astype(np.uint8)
    output_dset = (exposure.equalize_hist(dset) * 255).astype(np.uint8)
    toc = time.perf_counter()
    print(f'Histogram equalisation duration: {toc-tic:0.4f} seconds')
    
    return output_dset
    
# Scales optimization
def get_cwt_scales(wavelet, num=512, fmin=1000, fmax=50000, sampling_rate=100000):
    """
    Generate CWT scales for a given wavelet with dynamic frequency range calculation.
    
    Args:
        wavelet (str): Name of the wavelet
        num (int): Number of scales to generate (default: 512)
        fmin (float): Minimum frequency in Hz (default: 1000 Hz = 1 kHz)
        fmax (float): Maximum frequency in Hz (default: 50000 Hz = 50 kHz) 
        sampling_rate (float): Sampling rate in Hz (default: 100000)
    
    Returns:
        tuple: (scales array, vmax for visualization)
    """
    # BACKUP: Original hardcoded scale_lims dictionary (for reference)
    scale_lims_backup = {'cmor1.5-1.0': [1, 7, 50],    #           150 | 50
        'cmor2.5-0.5': [0, 6, 150],             #           150 |
        'cmor3.0-0.5': [0, 6, 150],             #           150 |
        'cmor10.0-0.3': [-0.67808, 5.4, 150],   #           150 |
        'mexh': [-1, 5, 100],                   #           300 | 100
        'morl': [0.7, 6.3, 300],                #           Corrected: Morlet (was [0.70043, 6.7, 300])
        'gaus1': [-1.32194, 5, 300],            #           300 |
        'gaus2': [-1, 5, 150],                  #           New: 2nd Gaussian derivative
        "fbsp2-1.0-1.0": [3, 9, 300],           #           300 |
        "fbsp4-0.6-1.0": [3, 9, 300],           #           300 |
        "fbsp1-1.5-1.0": [-0.9, 4.8, 200],     #           New: frequency B-spline (corrected for 1-50 kHz)
        "shan1.5-1.0": [-0.9, 4.8, 200],       #           New: Shannon wavelet (corrected for 1-50 kHz)
        "cgau8": [0.5, 6.1, 150]                #           New: complex Gaussian (corrected for 1-50 kHz)
        }
    
    # vmax values for visualization (wavelet-dependent)
    vmax_dict = {
        'cmor1.5-1.0': 50, 'cmor2.5-0.5': 150, 'cmor3.0-0.5': 150, 'cmor10.0-0.3': 150,
        'mexh': 100, 'morl': 300, 'gaus1': 300, 'gaus2': 150,
        'fbsp2-1.0-1.0': 300, 'fbsp4-0.6-1.0': 300, 'fbsp1-1.5-1.0': 200,
        'shan1.5-1.0': 200, 'cgau8': 150
    }
    
    # Return list of supported wavelet names
    if wavelet == None:
        return vmax_dict.keys()
    
    try:
        # Dynamic calculation of scale limits based on frequency range using pywt.frequency2scale
        scale_min = pywt.frequency2scale(wavelet, fmax / sampling_rate)  # Higher frequency = smaller scale
        scale_max = pywt.frequency2scale(wavelet, fmin / sampling_rate)  # Lower frequency = larger scale
        
        # Generate logarithmic scale array
        scales = np.logspace(np.log2(scale_min),
                            np.log2(scale_max),
                            base=2,
                            num=num,
                            endpoint=True)
        
        # Get vmax for visualization
        vmax = vmax_dict.get(wavelet, 150)  # Default vmax if not found
        
        return scales, vmax
        
    except Exception as e:
        print(f'get_cwt_scales() error for wavelet {wavelet}: {e}')
        print('Using fallback scale range')
        
        # Fallback to safe default range
        scales = np.logspace(1, 7, base=2, num=num, endpoint=True)
        vmax = 150
        
        return scales, vmax

def interpolate_low_quality_data(fkw_angle: np.ndarray, 
                                n_points_fit: np.ndarray | None, 
                                min_score: int = 3,
                                min_consecutive_zeros: int = 6,
                                max_isolated_nonzeros: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate timeseries data points with low quality scores using linear interpolation.
    Also cleans up isolated non-zero points surrounded by zeros.
    
    Parameters:
    -----------
    fkw_angle : np.ndarray
        1D array of timeseries data values
    n_points_fit : np.ndarray
        1D array of quality scores corresponding to each data point
    min_score : int, default=5
        Minimum acceptable quality score
    min_consecutive_zeros : int, default=4
        Minimum number of consecutive zeros required to be considered valid.
        Isolated zero sequences shorter than this will be interpolated.
    max_isolated_nonzeros : int, default=4
        Maximum number of consecutive non-zero points that will be set to zero
        if they are surrounded by zeros on both sides.
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - Corrected fkw_angle array with interpolated values and cleaned zeros
        - Boolean mask indicating which values were modified (True = modified)
    
    Notes:
    ------
    - Values with score < min_score are interpolated
    - Zero values are preserved only if they form sequences of min_consecutive_zeros or more
    - Isolated zero sequences (< min_consecutive_zeros) are treated as low quality and interpolated
    - Isolated non-zero sequences (≤ max_isolated_nonzeros) surrounded by zeros are set to zero
    - If no good neighbors exist for interpolation, the original value is kept
    - Edge cases (first/last points) use the nearest good value or extrapolation
    """
    
    # Input validation
    if (n_points_fit != None) and (len(fkw_angle) != len(n_points_fit)):
        raise ValueError("fkw_angle and n_points_fit must have the same length")
    
    if len(fkw_angle) == 0:
        return np.array([]), np.array([], dtype=bool)
    
    # Create copies to avoid modifying original arrays
    corrected_data = fkw_angle.copy()
    modified_mask = np.zeros(len(fkw_angle), dtype=bool)
    
    # Step 1: Clean up isolated non-zero points surrounded by zeros
    zero_mask = (fkw_angle == 0)
    nonzero_mask = ~zero_mask
    
    if np.any(nonzero_mask):
        # Find sequences of consecutive non-zero values
        nonzero_diff = np.diff(np.concatenate(([False], nonzero_mask, [False])).astype(int))
        nonzero_starts = np.where(nonzero_diff == 1)[0]
        nonzero_ends = np.where(nonzero_diff == -1)[0]
        
        # Check each non-zero sequence
        for start, end in zip(nonzero_starts, nonzero_ends):
            sequence_length = end - start
            
            # Only process short sequences that could be isolated
            if sequence_length <= max_isolated_nonzeros:
                # Check if surrounded by zeros (or at edges)
                left_is_zero = (start <= 1) or (corrected_data[start - 1] == 0 and corrected_data[start - 2] == 0)
                right_is_zero = (end >= len(corrected_data) - 1) or (corrected_data[end] == 0 and corrected_data[end + 1] == 0)
                
                # If surrounded by zeros on both sides, set to zero
                if left_is_zero and right_is_zero:
                    corrected_data[start:end] = 0
                    modified_mask[start:end] = True
    
    # Step 2: Find sequences of consecutive zeros (after cleaning)
    zero_mask = (corrected_data == 0)
    valid_zeros = np.zeros(len(corrected_data), dtype=bool)
    
    if np.any(zero_mask):
        # Find start and end of zero sequences
        zero_diff = np.diff(np.concatenate(([False], zero_mask, [False])).astype(int))
        zero_starts = np.where(zero_diff == 1)[0]
        zero_ends = np.where(zero_diff == -1)[0]
        
        # Mark zeros that are part of sequences >= min_consecutive_zeros
        for start, end in zip(zero_starts, zero_ends):
            if end - start >= min_consecutive_zeros:
                valid_zeros[start:end] = True
    
    # Step 3: Identify points that need interpolation
    # Low quality points, but exclude zeros that are part of valid sequences
    if n_points_fit != None:
        needs_interpolation = (n_points_fit < min_score) & (~valid_zeros)
    else:
        needs_interpolation = ~valid_zeros & (corrected_data == 0)
    
    if not np.any(needs_interpolation):
        return corrected_data, modified_mask
    
    # Find indices of good quality points
    good_points = ~needs_interpolation
    good_indices = np.where(good_points)[0]
    
    if len(good_indices) == 0:
        # No good points available, return original data
        return corrected_data, interpolated_mask
    
    # Process each bad point
    bad_indices = np.where(needs_interpolation)[0]
    
    for bad_idx in bad_indices:
        # Find the nearest good neighbors
        left_neighbors = good_indices[good_indices < bad_idx]
        right_neighbors = good_indices[good_indices > bad_idx]
        
        if len(left_neighbors) > 0 and len(right_neighbors) > 0:
            # Normal case: interpolate between left and right neighbors
            left_idx = left_neighbors[-1]  # Closest left neighbor
            right_idx = right_neighbors[0]  # Closest right neighbor
            
            # Linear interpolation
            x0, x1 = left_idx, right_idx
            y0, y1 = corrected_data[left_idx], corrected_data[right_idx]
            
            # Interpolate at bad_idx
            corrected_data[bad_idx] = y0 + (y1 - y0) * (bad_idx - x0) / (x1 - x0)
            
        elif len(left_neighbors) > 0:
            # Only left neighbors available (extrapolate or use nearest)
            if len(left_neighbors) >= 2:
                # Extrapolate using last two good points
                x0, x1 = left_neighbors[-2], left_neighbors[-1]
                y0, y1 = corrected_data[x0], corrected_data[x1]
                slope = (y1 - y0) / (x1 - x0)
                corrected_data[bad_idx] = y1 + slope * (bad_idx - x1)
            else:
                # Use the single left neighbor
                corrected_data[bad_idx] = corrected_data[left_neighbors[-1]]
                
        elif len(right_neighbors) > 0:
            # Only right neighbors available (extrapolate or use nearest)
            if len(right_neighbors) >= 2:
                # Extrapolate using first two good points
                x0, x1 = right_neighbors[0], right_neighbors[1]
                y0, y1 = corrected_data[x0], corrected_data[x1]
                slope = (y1 - y0) / (x1 - x0)
                corrected_data[bad_idx] = y0 + slope * (bad_idx - x0)
            else:
                # Use the single right neighbor
                corrected_data[bad_idx] = corrected_data[right_neighbors[0]]
        
        # Mark as interpolated
        modified_mask[bad_idx] = True
    
    return corrected_data, modified_mask


def validate_timeseries_quality(fkw_angle: np.ndarray, 
                               n_points_fit: np.ndarray | None, 
                               min_score: int = 5,
                               min_consecutive_zeros: int = 4,
                               max_isolated_nonzeros: int = 4) -> dict:
    """
    Analyze the quality of timeseries data and provide statistics.
    
    Parameters:
    -----------
    fkw_angle : np.ndarray
        1D array of timeseries data values
    n_points_fit : np.ndarray
        1D array of quality scores
    min_score : int, default=5
        Minimum acceptable quality score
    min_consecutive_zeros : int, default=4
        Minimum number of consecutive zeros required to be considered valid
    max_isolated_nonzeros : int, default=4
        Maximum number of consecutive non-zero points that will be set to zero
        if surrounded by zeros
    
    Returns:
    --------
    dict
        Dictionary containing quality statistics
    """
    
    total_points = len(fkw_angle)
    
    # Count isolated non-zeros that will be set to zero
    zero_mask = (fkw_angle == 0)
    nonzero_mask = ~zero_mask
    isolated_nonzeros = 0
    
    if np.any(nonzero_mask):
        nonzero_diff = np.diff(np.concatenate(([False], nonzero_mask, [False])).astype(int))
        nonzero_starts = np.where(nonzero_diff == 1)[0]
        nonzero_ends = np.where(nonzero_diff == -1)[0]
        
        for start, end in zip(nonzero_starts, nonzero_ends):
            sequence_length = end - start
            if sequence_length <= max_isolated_nonzeros:
                left_is_zero = (start == 0) or (fkw_angle[start - 1] == 0)
                right_is_zero = (end == len(fkw_angle)) or (fkw_angle[end] == 0)
                if left_is_zero and right_is_zero:
                    isolated_nonzeros += sequence_length
    
    # Find valid zero sequences (after cleaning isolated non-zeros)
    # For statistics, we simulate the cleaning process
    temp_data = fkw_angle.copy()
    if np.any(nonzero_mask):
        nonzero_diff = np.diff(np.concatenate(([False], nonzero_mask, [False])).astype(int))
        nonzero_starts = np.where(nonzero_diff == 1)[0]
        nonzero_ends = np.where(nonzero_diff == -1)[0]
        
        for start, end in zip(nonzero_starts, nonzero_ends):
            sequence_length = end - start
            if sequence_length <= max_isolated_nonzeros:
                left_is_zero = (start == 0) or (temp_data[start - 1] == 0)
                right_is_zero = (end == len(temp_data)) or (temp_data[end] == 0)
                if left_is_zero and right_is_zero:
                    temp_data[start:end] = 0
    
    # Now analyze the cleaned data
    zero_mask_cleaned = (temp_data == 0)
    valid_zeros = np.zeros(len(temp_data), dtype=bool)
    
    if np.any(zero_mask_cleaned):
        zero_diff = np.diff(np.concatenate(([False], zero_mask_cleaned, [False])).astype(int))
        zero_starts = np.where(zero_diff == 1)[0]
        zero_ends = np.where(zero_diff == -1)[0]
        
        for start, end in zip(zero_starts, zero_ends):
            if end - start >= min_consecutive_zeros:
                valid_zeros[start:end] = True
    
    valid_zero_count = np.sum(valid_zeros)
    isolated_zeros = np.sum(zero_mask_cleaned & ~valid_zeros)
    if n_points_fit != None:
        low_quality_non_zero = np.sum((n_points_fit < min_score) & ~zero_mask_cleaned)
    else:
        low_quality_non_zero = ~valid_zeros & (fkw_angle == 0)
    total_interpolated = low_quality_non_zero + isolated_zeros
    total_modified = total_interpolated + isolated_nonzeros
    good_quality = total_points - total_modified - valid_zero_count
    
    return {
        'total_points': total_points,
        'good_quality_points': good_quality,
        'low_quality_non_zero': low_quality_non_zero,
        'isolated_zeros': isolated_zeros,
        'isolated_nonzeros_to_zero': isolated_nonzeros,
        'valid_zero_sequences': valid_zero_count,
        'total_interpolated': total_interpolated,
        'total_modified': total_modified,
        'percentage_good': (good_quality / total_points * 100) if total_points > 0 else 0,
        'percentage_modified': (total_modified / total_points * 100) if total_points > 0 else 0
    }

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
class UnexpectedShape(Exception):
    "Raised when input data is not the expected shape"
    pass