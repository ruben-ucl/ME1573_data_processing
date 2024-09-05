import glob, read_AMPM, functools, h5py
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from my_funcs import get_logbook, get_logbook_data

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

AMPM_path = 'J:\AMPM'
hdf5_path = r'J:\AlSi10Mg single layer ffc'

mode = 'append' # 'append' or 'overwrite'

# Safety check before overwriting data
if mode == 'overwrite':
    confirm = input('Are you sure you want to run in \'overwrite\' mode? (y/n)')
    if confirm != 'y':
        mode = 'append'

logbook = get_logbook()

for file in sorted(glob.glob(f'{AMPM_path}/0109*AMPM*L4_100K*.dat')):
    print()
    AMPM_df = read_AMPM.readAMPMdat(file)
    fname = Path(file).name
    trackid = fname[:4] + '_0' + fname[fname.find('AMPM_') + 5]
    print('trackid: ' + trackid)
    
    try:
        hdf5_fpath = glob.glob(f'{hdf5_path}\{trackid}.hdf5')[0]
        print('hdf5 file: ', hdf5_fpath)
    except IndexError:
        print(f'No hdf5 found for {trackid}')
        continue
    
    track_data = get_logbook_data(logbook, trackid)
    scan_speed = track_data['scan_speed']
    scan_duration = 4/scan_speed
    npoints = int(100000*scan_duration)
    
    t = AMPM_df['Time']
    PD = AMPM_df['Photodiode1Bits']
    mod = AMPM_df['Modulate']
    pwr = AMPM_df['PowerValue1']
    dmp = AMPM_df['BeamDumpDiodeBits']
    dmp_1st_der = np.gradient(dmp)
    i_rising = np.where(dmp > np.max(dmp)*0.2)[0][0] - 1
    
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(6.3, 9.45), dpi=300, tight_layout=True)
    fig.suptitle(f'{trackid} AMPM')
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax4 = ax3.twinx()
    
    a = i_rising - 500
    b = i_rising + npoints + 500
    
    # Shift time so laser switches on at t=0
    t -= t[i_rising]
    t_ms = t*1000
    
    ax1.plot(t_ms, PD, lw=0.5, label='Photodiode1Bits', zorder=0)
    ax1.scatter(t_ms[i_rising], PD[i_rising], marker='x', c='k', linewidths=1, s=20, label='laser on')
    ax1.scatter((t_ms[a]), (PD[a]), marker=4, c='k', linewidths=1, s=20, label='crop start')
    ax1.scatter((t_ms[b]), (PD[b]), marker=5, c='k', linewidths=1, s=20, label='crop end')
    ax1.legend()
    
    ax2.plot(t_ms[a:b], PD[a:b], lw=0.5, zorder=0)
    ax2.scatter(t_ms[i_rising], PD[i_rising], marker='x', c='k', linewidths=1, s=20)
    
    ax3.plot(t_ms[i_rising-20:i_rising+40], PD[i_rising-20:i_rising+40], lw=0.5, zorder=1)
    ax3.scatter(t_ms[i_rising], PD[i_rising], marker='x', c='k', linewidths=1, s=20)
    ax3.set_xlabel('Time [ms]')
    ax4.plot(t_ms[i_rising-20:i_rising+40], dmp[i_rising-20:i_rising+40], 'r--', lw=0.5, label='BeamDumpDiodeBits', zorder=0)
    ax4.legend()
    
    plt.savefig(Path(AMPM_path, 'plots', f'{trackid}_AMPM_crop.png'))
    plt.close()
        
    with h5py.File(hdf5_fpath, 'r+') as hdf5_file:
        for dset in AMPM_df:
            try:
                if mode == 'overwrite':
                    del hdf5_file[f'AMPM/{dset}']
            except KeyError:
                pass
            try:
                hdf5_file[f'AMPM/{dset}'] = AMPM_df[dset][a:b]
            except OSError:
                print(f'Error: \'{dset}\' HDF5 dataset already exists')
        # plt.plot(np.array(hdf5_file[f'AMPM/Time']), np.array(hdf5_file[f'AMPM/Photodiode1Bits']), lw=0.5)
        # plt.show()
        