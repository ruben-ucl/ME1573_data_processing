"""
Report all groups of duplicate tracks in the logbook.

Duplicates are defined as tracks sharing the same:
  - Power [W]
  - Scan speed [mm/s]
  - laser mode (cw / pwm, derived from Point jump delay [us])
  - base type (powder / welding, derived from Powder material)
  - Substrate material
  - Layer

For each group, prints the shared parameters and a table of:
  trackid | exp_time_us | measured_layer_thickness [um] | Melting regime
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tools import get_logbook, filter_logbook_tracks

group_min_n = 3

LOGBOOK_FILTERS = {
    'material': 'AlSi10Mg',
}

def laser_mode(row):
    return 'cw' if row['Point jump delay [us]'] == 0 else 'pwm'


def base_type(row):
    return 'welding' if str(row['Powder material']) == 'None' else 'powder'


def main():
    logbook = get_logbook()
    logbook, _ = filter_logbook_tracks(logbook, LOGBOOK_FILTERS)

    logbook = logbook.dropna(subset=['trackid'])

    logbook['_laser_mode'] = logbook.apply(laser_mode, axis=1)
    logbook['_base_type']  = logbook.apply(base_type, axis=1)

    group_cols = ['Power [W]', 'Scan speed [mm/s]', '_laser_mode', '_base_type',
                  'Substrate material', 'Layer']
    detail_cols = ['trackid', 'Exposure time [us]', 'measured_layer_thickness [um]', 'Melting regime']

    # Only keep rows that have all grouping columns populated
    logbook_clean = logbook.dropna(subset=['Power [W]', 'Scan speed [mm/s]',
                                           'Substrate material', 'Layer'])

    groups = logbook_clean.groupby(group_cols, dropna=False)
    duplicate_groups = {k: v for k, v in groups if len(v) >= group_min_n}

    if not duplicate_groups:
        print('No duplicate tracks found.')
        return

    print(f'Found {len(duplicate_groups)} group(s) of duplicate tracks.\n')
    print('=' * 72)

    for i, (key, members) in enumerate(duplicate_groups.items(), 1):
        power, speed, mode, base, material, layer = key
        print(f'\nGroup {i}')
        print(f'  Power:     {int(power)} W')
        print(f'  Speed:     {int(speed)} mm/s')
        print(f'  Mode:      {mode}')
        print(f'  Base:      {base}')
        print(f'  Material:  {material}')
        print(f'  Layer:     {int(layer)}')
        print(f'  Tracks ({len(members)}):')

        sub = members[detail_cols].copy()
        sub = sub.rename(columns={
            'Exposure time [us]': 'exp_time_us',
            'measured_layer_thickness [um]': 'layer_thickness_um',
            'Melting regime': 'regime',
        })
        sub = sub.reset_index(drop=True)

        col_w = {
            'trackid': max(len('trackid'), sub['trackid'].astype(str).str.len().max()),
            'exp_time_us': max(len('exp_time_us'), sub['exp_time_us'].astype(str).str.len().max()),
            'layer_thickness_um': max(len('layer_thickness_um'),
                                      sub['layer_thickness_um'].astype(str).str.len().max()),
            'regime': max(len('regime'), sub['regime'].astype(str).str.len().max()),
        }

        header = (f"    {'trackid':<{col_w['trackid']}}  "
                  f"{'exp_time_us':>{col_w['exp_time_us']}}  "
                  f"{'layer_thickness_um':>{col_w['layer_thickness_um']}}  "
                  f"{'regime':<{col_w['regime']}}")
        print(header)
        print('    ' + '-' * (sum(col_w.values()) + 6))

        for _, row in sub.iterrows():
            et = row['exp_time_us']
            et_str = f'{int(et)}' if pd.notna(et) else 'N/A'
            lt = row['layer_thickness_um']
            lt_str = f'{lt:.1f}' if pd.notna(lt) else 'N/A'
            print(f"    {str(row['trackid']):<{col_w['trackid']}}  "
                  f"{et_str:>{col_w['exp_time_us']}}  "
                  f"{lt_str:>{col_w['layer_thickness_um']}}  "
                  f"{str(row['regime']):<{col_w['regime']}}")

        print()

    print('=' * 72)


if __name__ == '__main__':
    main()
