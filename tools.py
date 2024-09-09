import pandas as pd
import numpy as np
from pathlib import Path
from skimage import filters, exposure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, functools, glob

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes


def get_paths():
    path_dict = {}
    for file in glob.glob('dirs/*.txt'):
        with open(file, encoding='utf8') as f:
            path_dict[Path(file).stem] = Path(f.read())
            print(f'Reading filepath from {file}\n')
    return path_dict

# def get_logbook(logbook_path = Path('J:\Logbook_Al_ID19_combined_RLG.xlsx')):
def get_logbook():
    logbook_path = get_paths['logbook']
    print(f'Trying to read logbook: {logbook_path.name}')
    try:
        logbook = pd.read_excel(logbook_path,
            sheet_name='Logbook',
            # usecols='C, D, E, F, I, J, M, O, P, Q, R, S, T, U, W, AM, AP, AS, AT, AU, AV, AW, AX, AY, AZ, BA, BG, BN, BP, BV, BW, BX, BY, BZ, CA, CB, CC, CD, CE, CF, CG, CH, CI, CJ, CK, CL',
            converters={'Substrate No.': str, 'Sample position': str}
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
    # print(track_row)
    track_data = {}
    track_data['peak_power'] = int(track_row['Power [W]'])
    track_data['avg_power'] = int(track_row['Avg. power [W]'])
    track_data['pt_dist'] = int(track_row['Point distance [um]'])
    track_data['exp_time'] = int(track_row['Exposure time [us]'])
    track_data['pt_jump_delay'] = int(track_row['Point jump delay [us]'])
    track_data['scan_speed'] = int(track_row['Scan speed [mm/s]'])
    track_data['LED'] = int(track_row['LED [J/m]'])
    track_data['framerate'] = int(track_row['Frame rate (kHz)'] * 1000)
    track_data['laser_onset_frame'] = int(track_row['Laser onset frame #'])
    track_data['keyhole_regime'] = track_row['Melting regime'].values[0]
    
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
                                     'Mean pore angle [$\degree$]'
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
                                     'Melt pool volume [mm\u00b3]'
                                     ],
        'MP_vol_err':               ['melt_pool_volume_error [mm^3]',
                                     'Melt pool volume error [mm\u00b3]'
                                     ],
        'MP_rear_wall_angle':       ['rear_melt_pool_wall_angle [deg]',
                                     'Melt pool rear wall angle [$\degree$]'
                                     ],
        'melting_efficiency':       ['melting_efficiency',
                                     'Melting efficiency, η'
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
                                     r'FKW angle, $\theta_{FKW}$ [$\degree$]'
                                     ],
        'tan_fkw_angle':            ['tan_fkw_angle',
                                     'FKW angle tangent'
                                     ],
        'fkw_angle_sd':             ['fkw_angle_sd [deg]',
                                     'FKW angle standard deviation [$\degree$]'
                                     ],
        'fkw_angle_n_samples':      ['fkw_angle_n_samples',
                                     'FKW angle sample count'
                                     ],
        'norm_H_prod':              ['Normalised enthalpy product',
                                     r'Normalised enthalpy product, $\Delta H/h_m \dot L_{th}^*$'
                                     ],
        'KH_aspect':                ['keyhole_aspect_ratio',
                                     'Keyhole aspect ratio'
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
    tic = time.perf_counter()
    # output_dset = np.zeros_like(dset, dtype=np.uint8)
    # for i, im in enumerate(dset):
        # output_dset[i] = (exposure.equalize_hist(im) * 255).astype(np.uint8)
    output_dset = (exposure.equalize_hist(dset) * 255).astype(np.uint8)
    toc = time.perf_counter()
    print(f'Histogram equalisation duration: {toc-tic:0.4f} seconds')
    
    return output_dset
    
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