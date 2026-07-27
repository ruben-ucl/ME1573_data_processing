#!/usr/bin/env python3
"""
Regenerate the channel attribution figure for a trained multi-channel CWT model.

Reads the pre-computed attribution CSVs from the version's gradcam_analysis directory
and saves a single-panel bar chart (mean ± std) alongside the existing CSVs.

Output: ml/outputs/cwt/<version>/test_evaluation/gradcam_analysis_<version>/
        channel_attribution_analysis_<version>.{pdf,png}

Usage:
    python ml/channel_attribution_plot.py --version v213
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import CWT_OUTPUTS_DIR, format_version

THIRD_A4   = 2.28   # inches
LABEL_SIZE = 9      # pt
TICK_SIZE  = 8      # pt
DPI        = 300

CHANNEL_COLORS = {
    'PD1_cmor1.5-1.0': '#2171b5',
    'PD1_mexh':         '#e6550d',
    'PD2_cmor1.5-1.0': '#74c476',
    'PD2_mexh':         '#756bb1',
}
_DEFAULT_COLOR = '#636363'


def load_data(version_str: str):
    test_eval_dir = CWT_OUTPUTS_DIR / version_str / 'test_evaluation'
    analysis_dir  = test_eval_dir / f'gradcam_analysis_{version_str}'
    summary_path  = analysis_dir / f'channel_attribution_summary_{version_str}.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Not found: {summary_path}")
    return pd.read_csv(summary_path), test_eval_dir


def make_figure(summary: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        'font.size':         TICK_SIZE,
        'axes.labelsize':    LABEL_SIZE,
        'axes.titlesize':    LABEL_SIZE,
        'xtick.labelsize':   TICK_SIZE,
        'ytick.labelsize':   TICK_SIZE,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3.0,
        'ytick.major.size':  3.0,
        'axes.edgecolor':    'black',
    })

    channels = summary['channel'].tolist()
    n_ch     = len(channels)
    colors   = [CHANNEL_COLORS.get(ch, _DEFAULT_COLOR) for ch in channels]
    x        = np.arange(n_ch)
    rotate   = 30 if n_ch > 3 else 0

    fig, ax = plt.subplots(figsize=(THIRD_A4, 2.5))

    bars = ax.bar(
        x, summary['mean_attribution'],
        yerr=summary['std_attribution'],
        width=0.45, color=colors, alpha=0.85,
        error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='#333333'),
        zorder=3,
    )
    ax.axhline(
        1.0 / n_ch, color='#888888', linestyle='--', linewidth=0.8, zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=rotate, ha='right' if rotate else 'center')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean attribution score')
    ax.set_ylim(0, min(1.05, summary['mean_attribution'].max() + summary['std_attribution'].max() + 0.14))

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.98, 1.0 / n_ch + 0.015, f'{1.0 / n_ch:.1f}',
            transform=trans, color='#888888', fontsize=TICK_SIZE,
            ha='right', va='bottom')

    for bar, val, std in zip(bars, summary['mean_attribution'], summary['std_attribution']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.015,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=TICK_SIZE,
        )

    ax.yaxis.grid(True, linewidth=0.4, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Regenerate channel attribution figure from saved CSVs')
    parser.add_argument('--version', required=True, help='Model version, e.g. v213 or 213')
    args = parser.parse_args()

    version_str = format_version(args.version)
    summary, analysis_dir = load_data(version_str)

    stem = f'channel_attribution_analysis_{version_str}'
    fig  = make_figure(summary)

    for ext, kwargs in [('.pdf', {}), ('.png', {'dpi': DPI})]:
        path = analysis_dir / f'{stem}{ext}'
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
