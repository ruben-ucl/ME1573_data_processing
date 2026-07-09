"""
Collate curve-fit coefficients from PWM multi-curve plot folders into a single CSV.

Reads all *__fit.txt files produced by vis/3d_plot_multi_curve.py from folders
matching PWM1* – PWM5* inside a given root directory, then writes a sectioned CSV
where each section corresponds to one measured quantity (y-axis of the plots).

Usage:
    python file/collate_fit_coefficients.py <multi_curve_plots_dir> [--output OUTPUT]

Example:
    python file/collate_fit_coefficients.py \
        "C:/Users/lbn38569/UCL Dropbox/PhD students/Rubén Lambert-Garcia/Parameter space plots/Multi curve plots"
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_fit_txt(filepath):
    """Return (equation_str, list_of_series_dicts) from a __fit.txt file."""
    text = filepath.read_text(encoding='utf-8')
    lines = text.splitlines()

    equation = None
    series_list = []
    current = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Curve function:'):
            equation = stripped[len('Curve function:'):].strip()
        elif stripped.startswith('Series:'):
            if current is not None:
                series_list.append(current)
            current = {'_raw': stripped[len('Series:'):].strip(), '_failed': False}
        elif stripped == 'fit failed':
            if current is not None:
                current['_failed'] = True
        elif '=' in stripped and current is not None and not current['_failed']:
            key, _, val = stripped.partition('=')
            current[key.strip()] = val.strip()

    if current is not None:
        series_list.append(current)

    return equation, series_list


def parse_series_tag(raw):
    """Parse 'power=350, category=1.00' → (z_type, z_value, duty_cycle)."""
    m = re.match(r'(avg_power|power)=(\S+?)(?:,\s*category=(\S+))?$', raw)
    if m:
        return m.group(1), m.group(2), m.group(3) or ''
    return '', raw, ''


def parse_filename(name):
    """Extract y_quantity, z_type, z_value from a fit filename.

    Handles filenames like:
        scan_speed_vs_melting_efficiency_vs_power_350__1.00_0.89_0.80__fit.txt
        scan_speed_vs_MP_vol_vs_avg_power_400__1.00_0.89_0.80__err_bars__fit.txt
    """
    stem = re.sub(r'__fit\.txt$', '', name)
    stem = re.sub(r'__err_bars', '', stem)
    m = re.match(r'.+_vs_(.+)_vs_(avg_power|power)_(\d+)(?:__.+)?$', stem)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def collect_records(base_path):
    pwm_dirs = sorted(
        d for d in base_path.iterdir()
        if d.is_dir() and re.match(r'PWM[1-5]', d.name)
    )

    if not pwm_dirs:
        print(f'No PWM1–PWM5 folders found in: {base_path}')
        sys.exit(1)

    records = []
    for folder in pwm_dirs:
        folder_label = re.sub(r'^PWM\d+\s*', '', folder.name).strip()
        for fit_file in sorted(folder.glob('*__fit.txt')):
            y_qty, _z_type_from_name, _z_val_from_name = parse_filename(fit_file.name)
            if y_qty is None:
                print(f'  Skipping unrecognised filename: {fit_file.name}')
                continue

            equation, series_list = parse_fit_txt(fit_file)

            for s in series_list:
                if s['_failed']:
                    continue
                z_type, z_value, duty_cycle = parse_series_tag(s['_raw'])
                records.append({
                    'folder_label': folder_label,
                    'y_quantity': y_qty,
                    'equation': equation or '',
                    'z_type': z_type,
                    'z_value_W': z_value,
                    'duty_cycle': duty_cycle,
                    'a': s.get('a', ''),
                    'b': s.get('b', ''),
                    'R2': s.get('R²', ''),
                })

    return pd.DataFrame(records)


def write_sectioned_csv(df, out_path):
    """Write one section per y_quantity, with power rows grouped by duty cycle."""
    col_headers = ['z_type', 'z_value_W', 'duty_cycle', 'a', 'b', 'R²']
    output_rows = []

    # Preserve folder order from the dataframe
    seen = {}
    ordered_keys = []
    for _, row in df.iterrows():
        key = (row['folder_label'], row['y_quantity'])
        if key not in seen:
            seen[key] = True
            ordered_keys.append(key)

    for folder_label, y_qty in ordered_keys:
        group = df[(df['folder_label'] == folder_label) & (df['y_quantity'] == y_qty)]
        equation = group['equation'].iloc[0]

        output_rows.append([f'{y_qty}  ({folder_label})'] + [''] * (len(col_headers) - 1))
        output_rows.append([f'Equation: {equation}'] + [''] * (len(col_headers) - 1))
        output_rows.append(col_headers)

        for _, row in group.iterrows():
            output_rows.append([
                row['z_type'],
                row['z_value_W'],
                row['duty_cycle'],
                row['a'],
                row['b'],
                row['R2'],
            ])

        output_rows.append([''] * len(col_headers))  # blank separator

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_path, index=False, header=False, encoding='utf-8')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Collate curve-fit coefficients from PWM multi-curve plot folders.'
    )
    parser.add_argument(
        'input_dir',
        help='Path to the "Multi curve plots" directory containing PWM1–PWM5 folders',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV path (default: <input_dir>/fit_coefficients.csv)',
    )
    args = parser.parse_args()

    base_path = Path(args.input_dir)
    if not base_path.is_dir():
        print(f'Error: directory not found: {base_path}')
        sys.exit(1)

    out_path = Path(args.output) if args.output else base_path / 'fit_coefficients.csv'

    print(f'Scanning: {base_path}')
    df = collect_records(base_path)

    if df.empty:
        print('No fit data found.')
        sys.exit(1)

    print(f'Found {len(df)} series across {df["y_quantity"].nunique()} quantity groups.')
    write_sectioned_csv(df, out_path)


if __name__ == '__main__':
    main()
