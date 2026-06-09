#!/usr/bin/env python3
"""
Quick analysis of window distribution per track to inform block size selection.
"""
import pandas as pd
import numpy as np

# Read the label file
df = pd.read_csv(r'F:\AlSi10Mg single layer ffc\CWT_labelled_windows\1.0ms-window_0.2ms_offset_data_labels.csv', encoding='utf-8')

# Get windows per track
windows_per_track = df.groupby('trackid')['window_n'].count()

print('=' * 80)
print('WINDOW DISTRIBUTION PER TRACK ANALYSIS')
print('=' * 80)
print(f'\nTotal tracks: {len(windows_per_track)}')
print(f'Total windows: {len(df)}')
print(f'\nWindows per track statistics:')
print(f'  Mean: {windows_per_track.mean():.1f}')
print(f'  Median: {windows_per_track.median():.1f}')
print(f'  Min: {windows_per_track.min()}')
print(f'  Max: {windows_per_track.max()}')
print(f'  Std: {windows_per_track.std():.1f}')

print(f'\nPercentiles:')
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f'  {p}th percentile: {int(np.percentile(windows_per_track, p))}')

print(f'\nDistribution:')
print(f'  < 10 windows: {(windows_per_track < 10).sum()} tracks ({(windows_per_track < 10).sum() / len(windows_per_track) * 100:.1f}%)')
print(f'  < 20 windows: {(windows_per_track < 20).sum()} tracks ({(windows_per_track < 20).sum() / len(windows_per_track) * 100:.1f}%)')
print(f'  < 30 windows: {(windows_per_track < 30).sum()} tracks ({(windows_per_track < 30).sum() / len(windows_per_track) * 100:.1f}%)')
print(f'  < 40 windows: {(windows_per_track < 40).sum()} tracks ({(windows_per_track < 40).sum() / len(windows_per_track) * 100:.1f}%)')
print(f'  >= 40 windows: {(windows_per_track >= 40).sum()} tracks ({(windows_per_track >= 40).sum() / len(windows_per_track) * 100:.1f}%)')

print(f'\nTracks with < 40 windows (minimum for B_train=24 pattern):')
short_tracks = windows_per_track[windows_per_track < 40]
if len(short_tracks) > 0:
    print(f'  Count: {len(short_tracks)} / {len(windows_per_track)} ({len(short_tracks) / len(windows_per_track) * 100:.1f}%)')
    print(f'  Windows in these tracks: {short_tracks.sum()} / {windows_per_track.sum()} ({short_tracks.sum() / windows_per_track.sum() * 100:.1f}%)')
else:
    print('  None')

# Calculate expected retention for different block sizes
print(f'\n' + '=' * 80)
print('EXPECTED DATA RETENTION FOR DIFFERENT BLOCK SIZES')
print('=' * 80)
print(f'\nFormula: Discard% = 8 / (B_train + 8) %')
print(f'         Cycle = 1.25×B_train + 10')
print(f'\nNote: Tracks shorter than one full cycle cannot be properly split\n')

for B_train in [12, 16, 20, 24, 28, 32, 40, 48]:
    B_val = int(B_train * 0.25)
    cycle_length = int(1.25 * B_train + 10)
    discard_pct = 8 / (B_train + 8) * 100

    # Count tracks that can fit at least one full cycle
    viable_tracks = (windows_per_track >= cycle_length).sum()
    viable_pct = viable_tracks / len(windows_per_track) * 100

    # Windows in viable tracks
    viable_windows = windows_per_track[windows_per_track >= cycle_length].sum()
    viable_windows_pct = viable_windows / windows_per_track.sum() * 100

    print(f'B_train={B_train:2d} (B_val={B_val}, cycle={cycle_length:2d}): '
          f'Discard={discard_pct:4.1f}%, Viable tracks={viable_tracks:3d}/{len(windows_per_track)} ({viable_pct:5.1f}%), '
          f'Viable windows={viable_windows:4d}/{windows_per_track.sum()} ({viable_windows_pct:5.1f}%)')

print(f'\n' + '=' * 80)
print('RECOMMENDATION')
print('=' * 80)
print(f'\nBased on this analysis, the optimal B_train should:')
print(f'  1. Maximize viable tracks (cycle_length should be small)')
print(f'  2. Minimize discard percentage (B_train should be large)')
print(f'  3. Balance between these competing objectives')
