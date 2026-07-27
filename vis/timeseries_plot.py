"""
Multi-trackid timeseries plot.

One subplot per trackid, stacked vertically at 1/2 A4 width.
Datasets are plotted as line/scatter series; optional secondary y-axis per dataset.
X-axis can be time (ms) or displacement (mm) relative to laser onset.

Edit the CONFIG section below to customise tracks, datasets and appearance.
"""
import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import get_paths, get_logbook, get_logbook_data

# ─────────────────────────── CONFIG ──────────────────────────────────────────

TRACKIDS = [
    # '0325_01', # repeatability set A
    # '0325_02', # repeatability set A
    # '0325_03', # repeatability set A
    # '0109_06', # repeatability set C
    # '0109_03', # repeatability set C
    # '0108_06', # repeatability set C
    # '0108_03', # repeatability set C
    '0102_03',   # PWM demo set powder
    '0557_03',   # PWM demo set powder
    '0104_04',   # PWM demo set powder
    '0514_06',   # PWM demo set powder

]

# Each dataset dict keys:
#   group       : HDF5 group (e.g. 'AMPM', 'KH')
#   name        : HDF5 dataset name inside group
#   label       : legend label
#   time_group  : HDF5 group containing the time vector
#   time_name   : HDF5 name of the time vector dataset
#   time_units  : 's', 'ms', 'us'  (default 's')
#   y_axis      : 'primary' or 'secondary'  (default 'primary')
#   offset_ms   : manual time offset in ms applied before x conversion (default 0)
#   binary      : if True, normalise data to [0, 1] and show only 0/1 y-ticks
DATASETS = [
    # dict(group='AMPM', name='Photodiode1Bits',  label='PD1',
         # time_group='AMPM', time_name='Time', time_units='s', y_axis='primary'),
    # dict(group='AMPM', name='Photodiode2Bits',  label='PD2',
         # time_group='AMPM', time_name='Time', time_units='s', y_axis='primary'),
    dict(group='KH',   name='max_depth',            label='KH depth [μm]',
         time_group='KH',   time_name='time', time_units='s', y_axis='secondary'),
    # dict(group='KH',   name='area',            label='KH area [μm$^{2}$]',
         # time_group='KH',   time_name='time', time_units='s', y_axis='secondary'),
    dict(group='AMPM',   name='Modulate',            label='Modulation',
         time_group='AMPM',   time_name='Time', time_units='s', y_axis='primary', binary=True),
]

# X-axis mode: 'time' (ms) or 'displacement' (mm, clipped to PLOT_LENGTH_MM from laser onset)
X_MODE        = 'displacement'   # 'time' | 'displacement'
PLOT_LENGTH_MM = 3.5             # only used when X_MODE == 'displacement'

# Crop the start of the plot (in the same units as X_MODE: mm or ms). Set to 0 to disable.
X_START       = 2.5

# ── Style ──────────────────────────────────────────────────────────────────
LINE_WIDTH   = 0.7    # pt
MARKER       = 'None' # 'None' for lines only, 'o', '.', etc. for markers
MARKER_SIZE  = 2.0    # pt (ignored when MARKER == 'None')
LINE_STYLE   = '-'    # '-', '--', ':', '-.'

# Colorblind-safe Okabe-Ito palette (one colour per dataset, reused across trackids)
COLORS = [
    '#320a5e',  # Primary
    '#bc3754',  # Secondary
]

HSPACE      = 0.08    # vertical space between subplots (figure fraction)
SUBPLOT_H   = 1.2     # inches per subplot

# Hide x tick labels and axis title on all but the bottom subplot so that
# constrained_layout keeps inter-subplot gaps tight.
SHARE_X_AXIS_LABELS = True

SHARE_Y_PRIMARY   = True   # force all primary y-axes to the same min/max
SHARE_Y_SECONDARY = True   # force all secondary y-axes to the same min/max

# ── Smoothing (Savitzky-Golay) ─────────────────────────────────────────────
# Window is specified in ms so it scales correctly across signals with different
# sampling rates. Set SMOOTH_WINDOW_MS = 0 to disable smoothing.
SMOOTH_WINDOW_MS  = 0     # ms  (0 = off)
SMOOTH_POLYORDER  = 3     # polynomial order (must be < window in samples)

# ── Output ─────────────────────────────────────────────────────────────────
# Figures are saved into a subfolder named after this script.
# The filename is built from the dataset labels so each configuration
# produces a unique, self-describing file.
_SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR   = Path(__file__).parent / _SCRIPT_NAME

# ─────────────────────────── HELPERS ─────────────────────────────────────────

_TIME_FACTORS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}


def _smooth(data: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Apply Savitzky-Golay filter with a time-based window. No-op if SMOOTH_WINDOW_MS == 0."""
    if SMOOTH_WINDOW_MS <= 0 or len(time_s) < 2:
        return data
    sampling_rate = 1.0 / np.mean(np.diff(time_s))
    window = max(SMOOTH_POLYORDER + 1, int(SMOOTH_WINDOW_MS * 1e-3 * sampling_rate))
    if window % 2 == 0:
        window += 1
    if len(data) < window:
        return data
    return savgol_filter(data, window, SMOOTH_POLYORDER)


def _sci_formatter() -> mticker.ScalarFormatter:
    """ScalarFormatter that switches to scientific notation outside ±10³."""
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    return fmt


def _read_dataset(f: h5py.File, ds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, time_seconds) arrays from an open HDF5 file."""
    data = np.array(f[f"{ds['group']}/{ds['name']}"], dtype=float)
    tg   = ds.get('time_group') or ds['group']
    tn   = ds.get('time_name',  'Time')
    tu   = ds.get('time_units', 's')
    time = np.array(f[f"{tg}/{tn}"], dtype=float) * _TIME_FACTORS.get(tu, 1.0)
    return data, time


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    paths   = get_paths()
    hdf5_dir = paths['hdf5']
    logbook = get_logbook()

    n_tracks = len(TRACKIDS)
    label_col_w = 0.25                              # inches for trackid label column
    fig_w = 4.13 + label_col_w
    fig_h = n_tracks * SUBPLOT_H

    fig = plt.figure(figsize=(fig_w, fig_h), layout='constrained')
    gs = fig.add_gridspec(
        n_tracks, 2,
        width_ratios=[label_col_w / fig_w, 1 - label_col_w / fig_w],
        hspace=HSPACE,
    )
    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_tracks)]
    axes       = [fig.add_subplot(gs[i, 1]) for i in range(n_tracks)]

    # ── Binary-axis flags (pre-computed from DATASETS) ────────────────────
    _pri_binary = any(ds.get('binary') and ds.get('y_axis', 'primary') != 'secondary'
                      for ds in DATASETS)
    _sec_binary = any(ds.get('binary') and ds.get('y_axis') == 'secondary'
                      for ds in DATASETS)

    # ── Per-dataset linestyles and axis colours ────────────────────────────
    _ls_cycle = [LINE_STYLE, '--', ':']
    _ds_linestyle = {}
    _pri_ax_color = 'black'
    _sec_ax_color = 'black'
    for axis_key in ('primary', 'secondary'):
        group = [(i, ds) for i, ds in enumerate(DATASETS)
                 if (ds.get('y_axis', 'primary') == axis_key)
                 or (axis_key == 'primary' and ds.get('y_axis') != 'secondary')]
        # deduplicate (the two conditions above can overlap)
        seen = set()
        deduped = []
        for i, ds in group:
            if i not in seen:
                seen.add(i)
                deduped.append((i, ds))
        for j, (i, ds) in enumerate(deduped):
            _ds_linestyle[i] = _ls_cycle[min(j, len(_ls_cycle) - 1)]
        if deduped:
            first_i = deduped[0][0]
            if axis_key == 'primary':
                _pri_ax_color = COLORS[first_i % len(COLORS)]
            else:
                _sec_ax_color = COLORS[first_i % len(COLORS)]

    for lax, trackid in zip(label_axes, TRACKIDS):
        lax.axis('off')
        lax.text(0.5, 0.5, trackid,
                 transform=lax.transAxes,
                 fontsize=9, ha='center', va='center',
                 rotation=90)

    sec_axes = []  # collect secondary axes for shared-y post-processing

    for ax, trackid in zip(axes, TRACKIDS):
        hdf5_path = Path(hdf5_dir) / f'{trackid}.hdf5'
        if not hdf5_path.exists():
            print(f'WARNING: {hdf5_path} not found – skipping {trackid}')
            ax.set_visible(False)
            continue

        track_data = get_logbook_data(logbook, trackid)
        scan_speed_mm_s = track_data['scan_speed']    # mm/s

        ax2 = None  # secondary y-axis, created on demand

        with h5py.File(hdf5_path, 'r') as f:
            for i, ds in enumerate(DATASETS):
                color = COLORS[i % len(COLORS)]
                try:
                    data, time_s = _read_dataset(f, ds)
                except KeyError as e:
                    print(f'WARNING: {trackid} – dataset not found: {e}')
                    continue

                data = _smooth(data, time_s)

                if ds.get('binary'):
                    lo, hi = data.min(), data.max()
                    data = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)

                time_s = time_s + ds.get('offset_ms', 0) * 1e-3

                if X_MODE == 'displacement':
                    x = time_s * scan_speed_mm_s          # mm, t=0 already at laser onset
                    mask = (x >= X_START) & (x <= PLOT_LENGTH_MM)
                    x, data = x[mask], data[mask]
                    xlabel = 'Displacement [mm]'
                else:
                    x = time_s * 1e3                      # ms, t=0 already at laser onset
                    mask = (x >= X_START) & (x <= x[-1])
                    x, data = x[mask], data[mask]
                    xlabel = 'Time [ms]'

                target_ax = ax
                if ds.get('y_axis') == 'secondary':
                    if ax2 is None:
                        ax2 = ax.twinx()
                        sec_axes.append(ax2)
                        for spine in ax2.spines.values():
                            spine.set_linewidth(0.6)
                            spine.set_color('black')
                    target_ax = ax2

                target_ax.plot(
                    x, data,
                    color=color,
                    lw=LINE_WIDTH,
                    ls=_ds_linestyle[i],
                    marker=MARKER if MARKER != 'None' else None,
                    ms=MARKER_SIZE,
                    label=ds['label'],
                )

        # ── Axis styling ───────────────────────────────────────────────────
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8)
        if _pri_binary:
            ax.yaxis.set_major_locator(mticker.FixedLocator([0, 1]))
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='both'))
            ax.yaxis.set_major_formatter(_sci_formatter())
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('black')

        pri_labels = [ds['label'] for ds in DATASETS if ds.get('y_axis') != 'secondary']
        sec_labels = [ds['label'] for ds in DATASETS if ds.get('y_axis') == 'secondary']
        if pri_labels:
            ax.set_ylabel(', '.join(pri_labels), fontsize=9)

        if ax2 is not None:
            ax2.tick_params(labelsize=8)
            if _sec_binary:
                ax2.yaxis.set_major_locator(mticker.FixedLocator([0, 1]))
                ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
                ax2.set_ylim(-0.1, 1.1)
            else:
                ax2.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='both'))
                ax2.yaxis.set_major_formatter(_sci_formatter())
            if sec_labels:
                ax2.set_ylabel(', '.join(sec_labels), fontsize=9)

        # ── Colour y-axes to match first curve on each axis ────────────────
        ax.spines['left'].set_color(_pri_ax_color)
        ax.tick_params(axis='y', colors=_pri_ax_color)
        ax.yaxis.label.set_color(_pri_ax_color)
        if ax2 is not None:
            ax2.spines['right'].set_color(_sec_ax_color)
            ax2.tick_params(axis='y', colors=_sec_ax_color)
            ax2.yaxis.label.set_color(_sec_ax_color)

    # ── Hide x-axis labels on all but the bottom subplot ─────────────────
    if SHARE_X_AXIS_LABELS:
        for ax in axes[:-1]:
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel('')

    # ── Shared y-axis ranges ───────────────────────────────────────────────
    if SHARE_Y_PRIMARY and axes and not _pri_binary:
        lims = [ax.get_ylim() for ax in axes]
        y_min, y_max = min(l[0] for l in lims), max(l[1] for l in lims)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    if SHARE_Y_SECONDARY and sec_axes and not _sec_binary:
        lims = [ax.get_ylim() for ax in sec_axes]
        y_min, y_max = min(l[0] for l in lims), max(l[1] for l in lims)
        for ax in sec_axes:
            ax.set_ylim(y_min, y_max)

    # ── Single shared legend above the first subplot ───────────────────────
    first_ax = axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    # Collect secondary-axis handles from all subplots (same datasets, so first is enough)
    for child in first_ax.get_shared_x_axes().get_siblings(first_ax):
        if child is not first_ax:
            h2, l2 = child.get_legend_handles_labels()
            handles += h2
            labels  += l2
    leg = first_ax.legend(handles, labels,
                          fontsize=8, loc='lower left',
                          bbox_to_anchor=(0, 1.01),
                          bbox_transform=first_ax.transAxes,
                          ncols=len(DATASETS),
                          framealpha=0.7, handlelength=1.2,
                          labelspacing=0.3, columnspacing=1.0,
                          fancybox=False, edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = '_'.join(ds['label'].replace(' ', '_') for ds in DATASETS) \
            + '__' + '_'.join(TRACKIDS)
    stem  = OUTPUT_DIR / fname
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved {stem}.pdf / .png')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
