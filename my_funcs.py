import pandas as pd
import numpy as np
from pathlib import Path

# Read data folder path from .txt file
with open('data_path.txt', encoding='utf8') as f:
    filepath = fr'{f.read()}'
    # print(f'Reading data from {filepath}\n')

def get_logbook(logbook_path = Path('J:\Logbook_Al_ID19_combined_RLG.xlsx')):
    print(f'Trying to read logbook: {logbook_path.name}')
    try:
        logbook = pd.read_excel(logbook_path,
                                sheet_name='Logbook',
                                usecols='C, D, Q, AK, AN',
                                converters={'Substrate No.': str, 'Sample position': str}
                                )
        # logging.info('Logbook data aquired from %s' % logbook_path)
        print('Logbook read successfully')
    
        return logbook
    
    except Exception as e:
        print('Error: Failed to read logbook')
        print(e)
        # logging.info('Failed to read logbook - unable to continue')
        # logging.debug(str(e))
        raise
        
def get_logbook_data(logbook, trackid):  # Get scan speed and framerate from logbook
    print('Reading scan speed and framerate from logbook')
    substrate_no = trackid[1:4]
    track_no = trackid[-1]
    track_row = logbook.loc[(logbook['Substrate No.'] == substrate_no) & (logbook['Sample position'] == track_no)]
    scan_speed = int(track_row['scan speed [mm/s]'])
    framerate = int(track_row['Frame rate (kHz)'] * 1000)
    laser_onset_frame = int(track_row['Laser onset frame #'])
    
    return framerate, scan_speed, laser_onset_frame
    
def get_start_end_frames(trackid, logbook, margin=50, start_frame_offset=0):
    framerate, scan_speed, start_frame = get_logbook_data(logbook, trackid)
    
    n_frames = round(framerate * 4 / scan_speed) # based on track length of 4 mm
    f1 = start_frame - margin - start_frame_offset
    f2 = start_frame + n_frames + margin - start_frame_offset 
    
    return f1, f2
    
def get_substrate_mask(shape, substrate_surface_measurements_fpath, trackid):   # Generates a 2d mask of the substrate
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