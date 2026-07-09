"""
Plot curve-fit coefficients (a, b, R²) vs laser power from fit_coefficients.csv.

For each quantity group (melting_efficiency, MP_vol, etc.) and each coefficient,
produces one figure with three series — one per duty cycle — plotted against
laser power.  Average-power entries are excluded.

Output: vis/plot_fit_coefficients/<y_quantity>__coeff_<name>.pdf/.png

Usage:
    python vis/plot_fit_coefficients.py <fit_coefficients_csv>
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

FONT_SIZE = 9
TICK_SIZE = 8
DPI = 300
FIG_W = 4.13   # 1/2 A4 width, inches
FIG_H = 3.0

DUTY_STYLES = {
    '1.00': {'color': '#1f77b4', 'marker': 'o', 'ls': '-',  'label': 'D = 1.00 (CW)'},
    '0.89': {'color': '#d95f02', 'marker': 's', 'ls': '--', 'label': 'D = 0.89'},
    '0.80': {'color': '#2ca02c', 'marker': '^', 'ls': ':',  'label': 'D = 0.80'},
}

COEFF_META = {
    'a':  {'symbol': 'a',  'ylabel': 'Coefficient  a'},
    'b':  {'symbol': 'b',  'ylabel': 'Coefficient  b'},
    'R2': {'symbol': 'R²', 'ylabel': 'Goodness of fit  R²'},
}

QUANTITY_LABELS = {
    'melting_efficiency': 'Melting efficiency',
    'MP_vol':             'Melt pool volume',
    'MP_depth':           'Melt pool depth',
    'MP_length':          'Melt pool length',
    'MP_width':           'Melt pool width',
}


def quantity_label(y_qty):
    return QUANTITY_LABELS.get(y_qty, y_qty.replace('_', ' '))


def parse_csv(csv_path):
    """Parse sectioned CSV into a list of group dicts, each with a 'df' key."""
    df_raw = pd.read_csv(csv_path, header=None, dtype=str, encoding='utf-8').fillna('')

    groups = []
    current = None
    data_rows = []
    in_data = False

    for _, row in df_raw.iterrows():
        cells = [str(c).strip() for c in row]
        first = cells[0]

        if all(c == '' for c in cells):
            if current is not None and data_rows:
                current['df'] = pd.DataFrame(
                    data_rows,
                    columns=['z_type', 'z_value_W', 'duty_cycle', 'a', 'b', 'R2'],
                )
                groups.append(current)
            current = None
            data_rows = []
            in_data = False

        elif first == 'z_type':
            in_data = True

        elif first.startswith('Equation:'):
            if current is not None:
                current['equation'] = first[len('Equation:'):].strip()

        elif in_data and first in ('power', 'avg_power'):
            data_rows.append(cells[:6])

        elif '(' in first and ')' in first and not in_data:
            m = re.match(r'^(.+?)\s+\((.+)\)$', first)
            if m:
                current = {
                    'y_qty': m.group(1).strip(),
                    'folder_label': m.group(2).strip(),
                    'equation': '',
                }

    # flush trailing group if CSV has no final blank row
    if current is not None and data_rows:
        current['df'] = pd.DataFrame(
            data_rows,
            columns=['z_type', 'z_value_W', 'duty_cycle', 'a', 'b', 'R2'],
        )
        groups.append(current)

    return groups


def plot_coeff(group, coeff_key, out_dir, r2_min=None):
    y_qty = group['y_qty']
    equation = group['equation']
    meta = COEFF_META[coeff_key]

    df = group['df'].copy()
    df = df[df['z_type'] == 'power'].copy()
    df['z_value_W'] = pd.to_numeric(df['z_value_W'])
    df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
    df[coeff_key] = pd.to_numeric(df[coeff_key], errors='coerce')

    if r2_min is not None:
        n_before = len(df)
        df = df[df['R2'] >= r2_min]
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f'    R^2 < {r2_min}: dropped {n_dropped} point(s)')

    plt.rcParams.update({'font.size': FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, tight_layout=True)

    for dc, style in DUTY_STYLES.items():
        sub = df[df['duty_cycle'] == dc].sort_values('z_value_W').dropna(subset=[coeff_key])
        if sub.empty:
            continue
        ax.plot(
            sub['z_value_W'], sub[coeff_key],
            color=style['color'],
            marker=style['marker'],
            ls=style['ls'],
            linewidth=0.9,
            markersize=4.5,
            markeredgecolor='k',
            markeredgewidth=0.4,
            label=style['label'],
        )

    ax.set_xlabel('Laser power (W)', fontsize=FONT_SIZE)
    ax.set_ylabel(meta['ylabel'], fontsize=FONT_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xticks([300, 350, 400, 450, 500])

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    qty_lbl = quantity_label(y_qty)
    ax.set_title(
        f'{qty_lbl}  —  {meta["symbol"]}  in  {equation}',
        fontsize=FONT_SIZE,
    )

    leg = ax.legend(
        title='Duty cycle',
        title_fontsize=FONT_SIZE - 1,
        fontsize=FONT_SIZE - 1,
        labelspacing=0.25,
        fancybox=False,
        framealpha=0,
        edgecolor='inherit',
    )
    leg.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

    r2_tag = f'__r2min{r2_min}' if r2_min is not None else ''
    fname = f'{y_qty}__coeff_{coeff_key}{r2_tag}'
    plt.savefig(out_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.savefig(out_dir / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    print(f'  Saved: {fname}.[pdf/png]')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot curve-fit coefficients vs power for each quantity group.',
    )
    parser.add_argument(
        'csv_path',
        help='Path to fit_coefficients.csv (from file/collate_fit_coefficients.py)',
    )
    parser.add_argument(
        '--r2_min',
        type=float,
        default=None,
        metavar='THRESHOLD',
        help='Exclude points where R² is below this value (e.g. 0.5)',
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f'Error: file not found: {csv_path}')
        sys.exit(1)

    groups = parse_csv(csv_path)
    if not groups:
        print('No groups parsed from CSV.')
        sys.exit(1)

    out_dir = Path(__file__).parent / Path(__file__).stem
    out_dir.mkdir(exist_ok=True)

    for group in groups:
        y_qty = group['y_qty']
        print(f'\nGroup: {y_qty}')
        df_power = group['df'][group['df']['z_type'] == 'power']
        for coeff_key in ('a', 'b', 'R2'):
            if (df_power[coeff_key] != '').any():
                plot_coeff(group, coeff_key, out_dir, r2_min=args.r2_min)

    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
