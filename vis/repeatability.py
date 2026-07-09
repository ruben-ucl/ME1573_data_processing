import functools, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, define_column_labels


__author__ = 'Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True)

###########################################################################################################################

### Figure settings ###
#----------------------
font_size = 9
figsize = (6.30, 4.5)
dpi = 300
cmap_name = 'tab10'
mean_marker_size = 40   # scatter s (pt²)
track_marker_size = 12
jitter_width = 0.15     # x-axis jitter for individual track points
save_figure = True

### Group display labels ###
#---------------------------
GROUP_LABELS = {
    'cw_powder_1': 'CW powder #1',
    'pwm_weld':    'PWM weld',
    'cw_powder_2': 'CW powder #2',
}

### Metrics to plot (col_dict keys) ###
#---------------------------------------
METRICS = [
    'KH_depth',
    'KH_length',
    # 'KH_AR',
    'KH_area',
    'fkw_angle',
    'MP_depth',
    'MP_width',
    'MP_vol',
    # 'track_height',
    'melting_efficiency',
]

###########################################################################################################################

def load_tracks():
    tracks_path = Path(__file__).parent.parent / 'vis' / 'repeatability_test_tracks.txt'
    ns = {}
    exec(tracks_path.read_text(), ns)
    return ns['repeat_tracks']


def load_logbook(repeat_tracks):
    all_tracks = [t for tracks in repeat_tracks.values() for t in tracks]
    log = get_logbook()
    return log[log['trackid'].isin(all_tracks)]


def compute_group_stats(repeat_tracks, log, col_dict):
    stats = {}
    for group_key, track_ids in repeat_tracks.items():
        stats[group_key] = {}
        group_rows = log[log['trackid'].isin(track_ids)]
        for metric_key in METRICS:
            if metric_key not in col_dict:
                continue
            col_name = col_dict[metric_key][0]
            if col_name not in group_rows.columns:
                continue
            vals = group_rows[col_name].values.astype(float)
            n_valid = np.sum(~np.isnan(vals))
            stats[group_key][metric_key] = {
                'mean': np.nanmean(vals) if n_valid > 0 else np.nan,
                'std':  np.nanstd(vals, ddof=1) if n_valid > 1 else np.nan,
                'vals': vals,
            }
    return stats


def metric_has_data(stats, metric_key):
    return any(
        metric_key in grp and not np.isnan(grp[metric_key]['mean'])
        for grp in stats.values()
    )


def style_axis(ax):
    ax.tick_params(labelsize=font_size - 1)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')


def plot_metric(ax, metric_key, stats, group_keys, group_colors, col_dict):
    ax.set_ylabel(col_dict[metric_key][1], fontsize=font_size)
    ax.set_xlim(-0.5, len(group_keys) - 0.5)
    ax.set_xticks(range(len(group_keys)))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(len(group_keys))],
                       fontsize=font_size - 1)

    rng = np.random.default_rng(seed=42)

    for ix, group_key in enumerate(group_keys):
        color = group_colors[ix]
        if metric_key not in stats[group_key]:
            continue
        s = stats[group_key][metric_key]
        vals = s['vals']
        mean = s['mean']
        std  = s['std']

        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            jitter = rng.uniform(-jitter_width, jitter_width, size=len(valid))
            ax.scatter(np.full(len(valid), ix) + jitter, valid,
                       s=track_marker_size, color=color, alpha=0.55,
                       edgecolors='k', linewidths=0.3, zorder=3)

        if not np.isnan(mean):
            ax.errorbar(ix, mean,
                        yerr=std if not np.isnan(std) else 0,
                        fmt='o', color=color,
                        markersize=np.sqrt(mean_marker_size),
                        markeredgecolor='k', markeredgewidth=0.6,
                        ecolor='k', elinewidth=0.8, capsize=3, capthick=0.8,
                        zorder=5)

    style_axis(ax)

    summary = ', '.join(
        f'{GROUP_LABELS.get(k, k)}={stats[k][metric_key]["mean"]:.2f}'
        f'+/-{stats[k][metric_key]["std"]:.2f}'
        if (metric_key in stats[k] and not np.isnan(stats[k][metric_key]['mean']))
        else f'{GROUP_LABELS.get(k, k)}=N/A'
        for k in group_keys
    )
    print(f'  {metric_key}: {summary}')


def main():
    repeat_tracks = load_tracks()
    col_dict = define_column_labels()
    log = load_logbook(repeat_tracks)

    stats = compute_group_stats(repeat_tracks, log, col_dict)
    group_keys = list(repeat_tracks.keys())

    cmap = mpl.colormaps[cmap_name]
    group_colors = [mpl.colors.to_hex(cmap(i)) for i in range(len(group_keys))]

    active_metrics = [m for m in METRICS if metric_has_data(stats, m)]
    print(f'Plotting {len(active_metrics)}/{len(METRICS)} metrics with data\n')

    n_cols = 4
    n_rows = int(np.ceil(len(active_metrics) / n_cols))
    plt.rcParams.update({'font.size': font_size})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True)
    axes = np.array(axes).flatten()

    for i, metric_key in enumerate(active_metrics):
        plot_metric(axes[i], metric_key, stats, group_keys, group_colors, col_dict)

    for ax in axes[len(active_metrics):]:
        ax.set_visible(False)

    if save_figure:
        out_dir = Path(__file__).parent / Path(__file__).stem
        out_dir.mkdir(exist_ok=True)
        plt.savefig(out_dir / 'repeatability.pdf', bbox_inches='tight')
        plt.savefig(out_dir / 'repeatability.png', dpi=dpi, bbox_inches='tight')
        print(f'\nFigure saved: {out_dir / "repeatability"}.[pdf/png]')

    plt.show()


if __name__ == '__main__':
    main()
