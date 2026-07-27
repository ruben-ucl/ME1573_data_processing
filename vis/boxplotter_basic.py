import functools, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, get_paths, define_column_labels, apply_filter, filter_logbook_tracks, fmt_sigfigs_padded, read_hdf5_signal


__author__ = 'Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True)

###########################################################################################################################

### Figure settings ###
#----------------------
font_size = 8           # point
figsize = (2.1, 1.6)      # inch (page width = 6.3)
dpi = 300
plot_bg = 'w'
cat_cmap_name = 'inferno_r' # colormap for category colour coding
cat_colours = None          # explicit colour list (e.g. ['#f6d746', '#f37819']); None → auto from cmap
alpha = 0.8
box_width = 0.7         # width of each individual box
cluster_gap = 0.8       # whitespace between clusters (box_width units)
show_fliers = True      # show outlier markers on boxes
show_points = False     # overlay individual data points on boxes
show_n = True           # show sample size label below each box

include_legend = True
save_figure = True
xtick_sigfigs = 2       # significant figures for x-axis tick labels, or None to use str()
DEBUG = True            # print per-track bucketing and signal diagnostics

### Data source ###
#------------------
DATA_SOURCE = 'hdf5'   # 'logbook' | 'hdf5'

# Used when DATA_SOURCE == 'hdf5': samples from each track's signal are pooled into the box
HDF5_DATASET = dict(
    group='KH',
    name='max_depth',
    label='',
)
HDF5_TRIM = None          # samples to discard from each end (None = group default: KH→0, AMPM→500)

### Y-axis settings ###
#----------------------
if True:
    ploty = 'KH_depth'    # col_dict key — used when DATA_SOURCE == 'logbook'
    ylim = None
    yticks = None

### X-axis settings ###
#----------------------
if True:
    plotx = 'duty_cycle'
    # xvals = [300, 350, 400, 450, 500]
    xvals = [1.0, 0.8]
    xval_atol = 0.1          # ±tolerance for x-axis bucketing; 0 for exact match

### Logbook filter settings ###
#------------------------------
LOGBOOK_FILTERS = {
    'base_type': 'powder',
    # 'trackids': [
                # '0304_04', '0304_05', '0304_06', '0305_01', '0305_02',    # Weld D=0.80
                # '0305_03', '0305_04', '0305_05', '0305_06', '0306_01',    # Weld D=0.86
                # '0306_02', '0306_03', '0306_04', '0306_05', '0306_06',    # Weld D=0.89
                # '0110_01', '0110_02', '0110_03',                           # Weld D=1.0
                # '0102_01', '0102_02', '0102_03', '0102_04', '0102_05',    # Powder D=0.8
                # '0557_05', '0557_06', '0557_03', '0558_02', '0557_01',    # Powder D=0.86
                # '0104_02', '0104_03', '0104_04', '0104_05', '0104_06',    # Powder D=0.89
                # '0516_05', '0323_01', '0323_02', '0323_03',               # Powder D=1.0
                # ],
    'trackids': ['0515_01', '0555_06']
}

### Category settings ###
#------------------------
# col: col_dict key for the column used to split rows into categories
# op:  comparison operator ('==', '!=', '>', '<', '~=')
# val: value to compare against
# ls:  kept for compatibility with 3d_plot_multi_curve.py (not used here)
categories = [
    # {'label': '1.00', 'col': 'duty_cycle', 'op': '~=', 'val': 1.0,   'ls': '-'},
    # {'label': '0.89', 'col': 'duty_cycle', 'op': '~=', 'val': 8/9,   'ls': '--'},
    # {'label': '0.86', 'col': 'duty_cycle', 'op': '~=', 'val': 6/7,   'ls': ':'},
    # {'label': '0.80', 'col': 'duty_cycle', 'op': '~=', 'val': 0.8,   'ls': '-.'},
    ]

def filter_logbook():
    log = get_logbook()
    log_red, _ = filter_logbook_tracks(log, LOGBOOK_FILTERS)
    log_red.reset_index(inplace=True)
    return log_red

def set_up_figure(col_dict):
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = fig.add_subplot()
    if plot_bg is not None:
        ax.set_facecolor(plot_bg)
    ax.set_ylabel(col_dict[ploty][1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(labelsize=font_size - 1)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    return fig, ax

def _get_y_data(log_red_i_c, col_dict, hdf5_dir):
    if DATA_SOURCE == 'hdf5':
        segments = []
        for _, row in log_red_i_c.iterrows():
            sig = read_hdf5_signal(
                row['trackid'], hdf5_dir,
                HDF5_DATASET['group'], HDF5_DATASET['name'],
                trim=HDF5_TRIM,
            )
            if DEBUG:
                print(f'    {row["trackid"]}: {len(sig)} samples')
            if len(sig):
                segments.append(sig)
            elif str(row.get('Melting regime', '')).lower() == 'conduction':
                if DEBUG:
                    print(f'    {row["trackid"]}: conduction mode, no KH data → substituting 0')
                segments.append(np.array([0.0]))
        return np.concatenate(segments) if segments else np.array([]), len(segments)
    else:
        ycol = col_dict[ploty][0]
        vals = log_red_i_c[ycol].dropna().values
        if DEBUG:
            for _, row in log_red_i_c.iterrows():
                v = row[ycol]
                print(f'    {row["trackid"]}: {v}')
        return vals, len(vals)

def _composite_on_white(color, a):
    """Return the opaque RGB equivalent of `color` at alpha `a` on a white background."""
    c = np.array(mpl.colors.to_rgb(color))
    return mpl.colors.to_hex(a * c + (1 - a))

def plot_boxplots(ax, log_red, col_dict, cat_colors, hdf5_dir):
    _cats = categories or [{'label': '', 'col': None, 'op': None, 'val': None, 'ls': '-'}]
    n_cats = len(_cats)
    xcol = col_dict[plotx][0]
    cluster_step = n_cats * box_width + cluster_gap

    n_labels = []  # collect (x_pos, n) to draw after axis limits settle

    for ix, xval in enumerate(xvals):
        xfilt = apply_filter(log_red, xcol, '~=' if xval_atol > 0 else '==', xval, xval_atol)
        log_red_i = log_red[xfilt]
        cluster_center = ix * cluster_step

        if DEBUG:
            matched = log_red_i['trackid'].tolist()
            xcol_vals = log_red_i[xcol].tolist()
            print(f'\nxval={xval} (±{xval_atol}): {len(matched)} tracks matched')
            for tid, xv in zip(matched, xcol_vals):
                print(f'  {tid}  {xcol}={xv}')

        for ic, cat in enumerate(_cats):
            if cat['col'] is not None:
                col = col_dict[cat['col']][0]
                cfilt = apply_filter(log_red_i, col, cat['op'], cat['val'], cat.get('atol', 0.01))
                log_red_i_c = log_red_i[cfilt]
            else:
                log_red_i_c = log_red_i

            if DEBUG:
                print(f'  category={cat["label"]}: {log_red_i_c["trackid"].tolist()}')

            y_data, n_src = _get_y_data(log_red_i_c, col_dict, hdf5_dir)

            # centre of this box within its cluster
            x_pos = cluster_center + (ic - (n_cats - 1) / 2) * box_width
            n_labels.append((x_pos, n_src))

            print(f'  xval={xval}, {cat["label"]}: N={n_src}'
                  + (f', median={np.nanmedian(y_data):.2f}' if len(y_data) else ''))

            if len(y_data) == 0:
                continue

            ax.boxplot(y_data,
                       positions=[x_pos],
                       widths=box_width * 0.85,
                       patch_artist=True,
                       showfliers=show_fliers,
                       medianprops={'color': 'k', 'linewidth': 1.2},
                       boxprops={'facecolor': cat_colors[ic], 'alpha': alpha, 'linewidth': 0.7},
                       whiskerprops={'linewidth': 0.7},
                       capprops={'linewidth': 0.7},
                       flierprops={'marker': 'o', 'markersize': 2,
                                   'markerfacecolor': _composite_on_white(cat_colors[ic], alpha - 0.2),
                                   'markeredgecolor': 'k', 'markeredgewidth': 0.3},
                       )

            if show_points:
                rng = np.random.default_rng(seed=ix * 100 + ic)
                jitter = rng.uniform(-box_width * 0.15, box_width * 0.15, size=len(y_data))
                ax.scatter(np.full(len(y_data), x_pos) + jitter, y_data,
                           s=6, color=cat_colors[ic], alpha=alpha-0.2, zorder=3,
                           edgecolors='k', linewidths=0.3)

    # x-axis: one tick per cluster labelled with xval
    cluster_centers = [ix * cluster_step for ix in range(len(xvals))]
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(fmt_sigfigs_padded(xvals, xtick_sigfigs))
    ax.set_xlabel(col_dict[plotx][1])
    ax.set_xlim(cluster_centers[0]  - cluster_step / 2,
                cluster_centers[-1] + cluster_step / 2)

    # light vertical separators between clusters
    for ix in range(len(xvals) - 1):
        sep_x = (cluster_centers[ix] + cluster_centers[ix + 1]) / 2
        ax.axvline(sep_x, color='lightgray', linewidth=0.5, zorder=0)

    # n labels as a secondary x-axis row below the main axis
    if show_n:
        xdata_yaxes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        below_axis = mtransforms.ScaledTranslation(0, -30/72, ax.figure.dpi_scale_trans)
        trans = xdata_yaxes + below_axis
        for x_pos, n in n_labels:
            ax.text(x_pos, 0, str(n),
                    transform=trans,
                    ha='center', va='top',
                    fontsize=font_size - 2, color='gray',
                    clip_on=False)
        # "N" axis label at the left end of the row
        n_row_trans = ax.transAxes + below_axis
        ax.text(0, 0, 'N =',
                transform=n_row_trans,
                ha='right', va='top',
                fontsize=font_size - 2, color='gray',
                clip_on=False)

def create_legend(ax, col_dict, cat_colors):
    header = mpl.patches.Patch(visible=False, label=col_dict[categories[0]['col']][1])
    handles = [header] + [
        mpl.patches.Patch(facecolor=cat_colors[ic], edgecolor='k',
                          linewidth=0.7, alpha=alpha, label=cat['label'])
        for ic, cat in enumerate(categories)
    ]
    legend = ax.legend(handles, [h.get_label() for h in handles],
                       ncol=1, fontsize='medium',
                       fancybox=False, framealpha=1.00, edgecolor='w')
    legend.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

def main():
    paths = get_paths()
    hdf5_dir = paths.get('hdf5')
    log_red = filter_logbook()
    col_dict = define_column_labels()
    fig, ax = set_up_figure(col_dict)

    n_cats = len(categories)
    if cat_colours is not None:
        cat_colors = [mpl.colors.to_hex(c) for c in cat_colours]
    else:
        cmap = mpl.colormaps[cat_cmap_name]
        cat_colors = [mpl.colors.to_hex(cmap.resampled(n_cats)(ic)) for ic in range(n_cats)] if categories else ['#555555']

    plot_boxplots(ax, log_red, col_dict, cat_colors, hdf5_dir)

    if include_legend and categories:
        create_legend(ax, col_dict, cat_colors)

    if save_figure:
        y_part = HDF5_DATASET['name'] if DATA_SOURCE == 'hdf5' else ploty
        cat_part = '__' + '_'.join(cat['label'] for cat in categories) if categories else ''
        fname = f'{plotx}_vs_{y_part}{cat_part}'
        out_dir = Path(__file__).parent / Path(__file__).stem
        out_dir.mkdir(exist_ok=True)
        plt.savefig(out_dir / f'{fname}.svg', bbox_inches='tight')
        plt.savefig(out_dir / f'{fname}.png', dpi=dpi, bbox_inches='tight')
        print(f'\nFigure saved: {out_dir / fname}.[svg/png]')

    plt.show()

if __name__ == '__main__':
    main()
