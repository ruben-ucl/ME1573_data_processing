import functools, inspect, math, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from curve_fitter import fit_curve

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append ('D:\\ME1573_data_processing')
from tools import get_logbook, define_column_labels, apply_filter, filter_logbook_tracks, fmt_sigfigs_padded


__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

###########################################################################################################################

### Figure settings ###
#----------------------
    # Format # 
save_figure = True
font_size = 9           # point
figsize = (3.15, 2.8)     # inch (page width = 6.3)
dpi = 300
plot_bg = 'w'
cmap_name = 'inferno_r'    # colormap for z-value colour coding
marker_size = 30        # scatter marker area (points²)
marker_edge_width = 0.5 # marker edge linewidth (points)
fit_lw = 0.7            # curve fit line width

    # Visual #
colour_curves = False    # False → all fit curves drawn in black
regime_point_colours = False
regime_point_shapes = False
label_points = False
include_hline = None

    # Data #
pop_nans = True
show_r2 = False          # show R² label at end of each fit curve
show_r2_in_legend = True # show R² in parentheses after category label in legend
include_error_bars = True
log_log_axes = False     # True → both axes use log scale

    # Curve fitting #
fit_curves = True
fit_p0 = [0.1, 1, 1, 1] # initial parameter guess (None → auto-compute via log-linear OLS)
fit_bounds = None       # (lower, upper) bounds — None disables bounded solver and DE fallback
fit_linearize = True   # bootstrap p0 via log-linear OLS (power-exponential models)
fit_auto_select = True  # True → fit every candidate below; plot the one with the best R²
fit_candidates = [       # evaluated only when fit_auto_select = True
    # {
        # 'func':      lambda x, a, b, c, d: a * x**b * c**(-x/d),
        # 'bounds':    ([0, 0, 1.001, 0], [1e8, 10, 1e6, 1e6]),
        # 'p0':        None,
        # 'linearize': True,
        # 'label':     'y = a·x^b·c^(−x/d)',
        # 'symbol':    '‡',
    # },
    # {
        # 'func':      lambda x, a, b, c, d: a * x**b * d**c / (x + d)**(b + c),
        # 'bounds':    ([0, 0, 0, 0], [1e8, 10, 10, 1e6]),
        # 'p0':        [1e5, 0.5, 0.5, 800],
        # 'linearize': False,
        # 'label':     'y = a·x^b·d^c / (x+d)^(b+c)',
        # 'symbol':    '‖',
    # },
    {
        'func':      lambda x, a, b: a * x**b,
        'bounds':    ([0, -10], [1e9, 10]),
        'p0':        None,
        'linearize': True,
        'label':     'y = a·x^b',
        'symbol':    '†',
    },
    # {
        # 'func':      lambda x, a, b, c: a + b**(-c*x),
        # 'bounds':    ([0, 1.001, 0], [1, 1e6, 1e6]),
        # 'p0':        [0.05, 2.0, 0.001],
        # 'linearize': False,
        # 'label':     'y = a+b^(−c·x)',
        # 'symbol':    '§',
    # },
    # {
        # 'func':      lambda x, a, b: a * x + b,
        # 'bounds':    ([-1e8, -1e8], [1e8, 1e8]),
        # 'p0':        None,
        # 'linearize': False,
        # 'label':     'y = a·x+b',
        # 'symbol':    '*',
    # },
]

    # Legend #
include_legend = True
stack_legends  = False            # True → single stacked legend; False → two independent legends
legend_loc_z   = 'upper right'  # location of z-value legend (or combined legend when stacked)
legend_loc_cat = 'lower left'   # location of category legend (ignored when stacked)
legend_z_sigfigs = 2             # significant figures for z-value legend labels, or None to use str()

### X-axis settings ###
#----------------------
if True:
    plotx = 'scan_speed'
    xerr_col = None         # col_dict key for x error bars, or None
    xlim = [300, 2100]
    # xticks = [400, 800, 1200, 1600, 2000]
    # xlim = None
    xticks = None
    # xlim = [275, 525]
    # xticks = [300, 400, 500]
    # xlim = [0, 0.13]
    # xticks = [0, 0.05, 0.1]

### Y-axis settings ###
#----------------------
if True:
    ploty = 'MP_length'
    yerr_col = ploty + '_err' # col_dict key for y error bars, or None
    # yerr_col = None # col_dict key for y error bars, or None
    # ylim = [-0.001, 0.071]
    # yticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    # ylim = [-0.01, 0.33]
    # yticks = [0, 0.1, 0.2, 0.3]
    # yticks = [0, 10, 20, 30, 40, 50]
    # ylim = [-2, 56]
    # ylim = [-0.001, 0.027]
    # yticks = [0, 0.01, 0.02]
    # ylim = [-10, 180]
    # yticks = [0, 50, 100, 150]
    ylim = [0, 1490]
    # ylim = [-10, 450]
    # yticks = [0, 100, 200, 300, 400]
    # ylim = [-20, 520]
    # ylim = [-0.001, 0.066]
    # ylim = [4000, 110000]
    # ylim = None
    yticks = None

### Z-axis settings ###
#----------------------
zcdict = {500: '#fcffa4',
              450: '#f98e09',
              400: '#bc3754',
              350: '#781c6d',
              300: '#2d0b59'
              }

if True:
    plotz = 'power'
    zunit = 'W'
    # zvals = [400, 800, 1200, 1600]
    # zvals = [1.0, 8/9, 6/7, 0.8]
    # zvals = [1.0, 8/9, 0.8]
    # zvals = [500, 450, 400, 350, 300]
    zvals = [500]
    # zvals = [0.80, 0.89, 1.00]
    # zcolours = ['#fcffa4', '#f98e09', '#bc3754', '#781c6d', '#2d0b59']
    # zcolours = ['#f98e09']     # None → auto from cmap; or e.g. ['#e41a1c', '#377eb8', '#4daf4a']
    zcolours = [zcdict[zvals[0]]]
    # zcolours = None
    
### Logbook filter settings ###
#------------------------------
LOGBOOK_FILTERS = {
    'material':   'AlSi10Mg',
    'layer':      1,
    'base_type':  'powder',
    # 'laser_mode': 'cw',
    # 'beamtime':   3,
    # 'material':   ['AlSi10Mg', 'Al7A77', 'Ti64'],  # multi-material
    # 'regime':     'not_cond',
    # 'trackids':     [
                    # '0304_04', '0304_05', '0304_06', '0305_01', '0305_02',    # Weld D=0.80
                    # '0305_03', '0305_04', '0305_05', '0305_06', '0306_01',     # Weld D=0.86
                    # '0306_02', '0306_03', '0306_04', '0306_05', '0306_06',     # Weld D=0.89
                    # '0110_01', '0507_05', '0507_01', '0110_03',  #'0110_02',     # Weld D=1.0
                    # '0102_01', '0102_02', '0102_03', '0102_04', '0102_05',     # Powder D=0.8
                    # '0557_05', '0557_06', '0557_03', '0558_02', '0557_01',     # Powder D=0.86
                    # '0104_02', '0104_03', '0104_04', '0104_05', '0104_06',     # Powder D=0.89
                    # '0516_05', '0514_05', '0514_06', '0323_02', '0323_03', #'0323_01',     # Powder D=1.0
                    # ] 
}

### Category settings ###
#------------------------
# col:        col_dict key for the column used to split rows into categories
# op:         comparison operator ('==', '!=', '>', '<')
# val:        value to compare against
# ls:         line style for curve fit
# connect_ls: line style for straight lines connecting scatter points in z-colour (omit or None for scatter only)
categories = [
    # {'label': 'powder', 'col': 'base_type', 'op': '==', 'val': 'powder', 'ls': '-'},
    # {'label': 'weld',   'col': 'base_type', 'op': '==', 'val': 'weld', 'ls': '--'},
    {'label': '1.00',   'col': 'duty_cycle', 'op': '~=', 'val': 1.0, 'ls': '-'},
    {'label': '0.89',   'col': 'duty_cycle', 'op': '~=', 'val': 0.89, 'ls': '--'},
    # {'label': '0.86',   'col': 'duty_cycle', 'op': '~=', 'val': 0.86, 'ls': ':'},
    {'label': '0.80',   'col': 'duty_cycle', 'op': '~=', 'val': 0.8, 'ls': ':'},

]

TRACKID_EXCLUDE_PREFIXES = ['0106']

def filter_logbook():
    log = get_logbook()
    log_red, _ = filter_logbook_tracks(log, LOGBOOK_FILTERS)
    if TRACKID_EXCLUDE_PREFIXES:
        mask = ~log_red['trackid'].str.startswith(tuple(TRACKID_EXCLUDE_PREFIXES))
        log_red = log_red[mask]
    log_red.reset_index(inplace=True)
    return log_red

def set_up_figure(col_dict):
    # Set up figure with two or three axes
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = fig.add_subplot()

    if plot_bg != None: ax.set_facecolor(plot_bg)

    if log_log_axes:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(col_dict[plotx][1])
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylabel(col_dict[ploty][1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])

    if log_log_axes:
        def _minor_label(x, _):
            exp = int(np.floor(np.log10(abs(x))))
            mantissa = int(round(x / 10**exp))
            if mantissa not in (5,):
                return ''
            return rf'$\mathdefault{{{mantissa}\times10^{{{exp}}}}}$'
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0,)))
            axis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
            axis.set_minor_formatter(mpl.ticker.FuncFormatter(_minor_label))
    else:
        if xticks != None: ax.set_xticks(xticks)
        if yticks != None: ax.set_yticks(yticks)

    ax.tick_params(which='major', labelsize=font_size - 1)
    if log_log_axes:
        ax.tick_params(which='minor', labelsize=font_size - 2)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    return fig, ax

def define_point_formats():
    if zcolours is not None:
        colours = [mpl.colors.to_hex(c) for c in zcolours]
    else:
        cmap = mpl.colormaps[cmap_name].resampled(len(zvals))
        colours = [mpl.colors.to_hex(cmap(iz)) for iz in range(len(zvals))]
    marker_dict = {iz: {'mp': 'o', 'mw': 'x', 'c': colours[iz]}
                   for iz in range(len(zvals))}
    marker_dict.update({
        'unstable keyhole':               {'m': 'o', 'c': '#fde725'},
        'keyhole flickering':             {'m': 's', 'c': '#3b528b'},
        'quasi-stable keyhole':           {'m': '^', 'c': '#5ec962'},
        'quasi-stable vapour depression': {'m': 'D', 'c': '#21918c'},
        'conduction':                     {'m': 'v', 'c': '#440154'},
        'Al7A77 (Huang et al., 2022)':    {'m': 'd', 'c': 'k'},
        'Ti64 (Zhao et al., 2020)':       {'m': 'd', 'c': 'lightgray'},
        'Ti64 (Cunningham et al., 2019)': {'m': 'd', 'c': 'gray'},
    })
    return marker_dict

def plot_data(ax, iz, log_red, marker_dict, cat, col_dict):
    # Initialise lists for storing point coordinates
    x = np.zeros((len(log_red), 1))
    y = np.zeros_like(x)

    # Add points to plot by iterating through the logbook row by row
    for i, row in log_red.iterrows():
        trackid = row['trackid']
        regime = row['Melting regime']

        # Set variables to plot
        x[i] = row[col_dict[plotx][0]]
        y[i] = row[col_dict[ploty][0]]

        ax.scatter(x[i], y[i],
                   label = zvals[iz],
                   facecolors = marker_dict[iz]['c'],
                   edgecolors = 'k',
                   linewidths = marker_edge_width,
                   linestyles = cat['ls'],
                   marker = marker_dict[regime]['m'] if regime_point_shapes == True else marker_dict[iz]['mp'],
                   s = marker_size,
                   )
                   
        if label_points == True:
            ax.text(x[i]+10, y[i],
                    trackid,
                    va = 'center',
                    ha = 'center',
                    fontsize = 'xx-small',
                    )
        if include_error_bars:
            xerr = row[col_dict[xerr_col][0]] if xerr_col is not None else None
            yerr = row[col_dict[yerr_col][0]] if yerr_col is not None else None
            if xerr is not None or yerr is not None:
                ax.errorbar(x[i], y[i], xerr=xerr, yerr=yerr,
                            ecolor='k', elinewidth=0.6, capsize=3.2, capthick=0.6, zorder=0)
    
    return x.T, y.T

def draw_hline(ax, hliney):
    ax.axhline(hliney, c='gray', ls='--', lw=0.7, zorder=0)
    ax.text(0.98, hliney, f'η = {hliney}',
            transform=ax.get_yaxis_transform(),
            c='gray', va='bottom', ha='right')

def draw_connecting_lines(ax, iz, marker_dict, cat, x, y):
    ls = cat.get('connect_ls')
    if ls is None:
        return
    sort_idx = np.argsort(x)
    xs = np.array(x)[sort_idx]
    ys = np.array(y)[sort_idx]
    ax.plot(xs, ys, color=marker_dict[iz]['c'], linestyle=ls, linewidth=fit_lw, zorder=0)

def curve_function(x, a, b, c, d):
    # return a*x**3 + b*x**2 + c*x + d
    # return a*np.exp(-(x-b)**2/(2*c**2))+d
    # return a*x**2 + b*x + c
    # return a*b**x
    # return a + b*np.log(x)
    # return a*np.exp(b*x)
    # return np.arctan(a*(x+b))*(180/np.pi)
    return a * x**b #* c**(-x/d)
    # return a * x + b

curve_function_str = next(
    f'y = {line.strip()[len("return "):]}'
    for line in inspect.getsource(curve_function).splitlines()
    if line.strip().startswith('return')
)

def draw_curve_fit(ax, iz, marker_dict, ls, xx, yy):
    xx_arr = np.array(xx, dtype=float)
    yy_arr = np.array(yy, dtype=float)

    if fit_auto_select:
        best_result, best_candidate = None, None
        for cand in fit_candidates:
            n_cand = len(cand['bounds'][0])
            if len(xx_arr) < n_cand:
                print(f'  [auto-select] Skipping "{cand["label"]}" \u2014 only {len(xx_arr)} points, need {n_cand}.')
                continue
            result = fit_curve(
                cand['func'], xx_arr, yy_arr,
                p0=cand.get('p0'),
                bounds=cand['bounds'],
                linearize=cand.get('linearize', True),
                verbose=True,
            )
            if result.converged and (best_result is None or result.r2 > best_result.r2):
                best_result, best_candidate = result, cand
        if best_result is None:
            print(f'\nAll candidate fits failed for z-index {iz}.')
            return None, None, None
        result     = best_result
        func_used  = best_candidate['func']
        fit_info   = {'label': best_candidate['label'], 'symbol': best_candidate['symbol']}
        print(f'  [auto-select] Best: "{best_candidate["label"]}"  R\u00b2={result.r2:.4f}  [{best_candidate["symbol"]}]')
    else:
        result = fit_curve(
            curve_function, xx_arr, yy_arr,
            p0=fit_p0,
            bounds=fit_bounds,
            linearize=fit_linearize,
            verbose=True,
        )
        if not result.converged:
            print(f'\nCurve fit failed for z-index {iz}:')
            for w in result.warnings:
                print(f'  {w}')
            return None, None, None
        func_used = curve_function
        fit_info  = None

    popt = result.popt
    r2   = result.r2
    X = np.geomspace(min(xx), max(xx), 50) if log_log_axes else np.linspace(min(xx), max(xx), 50)
    Y = func_used(X, *popt)
    curve_colour = marker_dict[iz]['c'] if colour_curves else 'k'
    ax.plot(X, Y, c=curve_colour, ls=ls, lw=fit_lw, zorder=0)
    if show_r2:
        symbol = fit_info['symbol'] if fit_info else ''
        ax.text(X[-1], Y[-1], f' R\u00b2={round(r2, 2)}{symbol}',
                color=curve_colour,
                fontsize=font_size - 1,
                va='center', ha='left',
                clip_on=False)
    return popt, r2, fit_info

def create_legend(ax, col_dict, marker_dict, fit_results=None):
    def section_header(label):
        return mpl.patches.Patch(visible=False, label=label)

    def _place_legend(handles, loc):
        """Add a legend; loc may be a string or an (x, y) axes-fraction tuple."""
        shared = dict(handles=handles,
                      labels=[h.get_label() for h in handles],
                      fontsize=font_size - 1,
                      labelspacing=0.25,
                      fancybox=False, framealpha=0, edgecolor='inherit')
        if isinstance(loc, tuple):
            leg = ax.legend(**shared, bbox_to_anchor=loc, loc='upper left')
        else:
            leg = ax.legend(**shared, loc=loc)
        leg.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])
        return leg

    legend_markersize = np.sqrt(marker_size)  # scatter s is area (pt²), legend markersize is diameter (pt)
    zval_labels = fmt_sigfigs_padded(zvals, legend_z_sigfigs)
    zval_handles = [section_header(col_dict[plotz][1])] + [
        mpl.lines.Line2D([], [], color=marker_dict[iz]['c'],
                         marker='o', linestyle='none',
                         markeredgecolor='k', markeredgewidth=marker_edge_width,
                         markersize=legend_markersize, label=zval_labels[iz])
        for iz, zv in enumerate(zvals)
    ]
    if categories:
        cat_col_label = col_dict[categories[0]['col']][1]
        cat_r2 = {}  # {category: [(r2_float, symbol_str), ...]}
        if show_r2_in_legend and fit_results:
            for r in fit_results:
                if r['r2'] is not None:
                    entry = (r['r2'], r.get('fit_symbol', ''))
                    cat_r2.setdefault(r['category'], []).append(entry)
        def _cat_label(cat):
            label = cat['label']
            if show_r2_in_legend and label in cat_r2:
                entries = cat_r2[label]
                r2_str = ', '.join(f'{v:.2f}{sym}' for v, sym in entries)
                return f'{label} (R²={r2_str})'
            return label
        style_handles = [section_header(cat_col_label)] + [
            mpl.lines.Line2D([], [], color='k', linestyle=cat['ls'],
                             linewidth=fit_lw, label=_cat_label(cat))
            for cat in categories
        ]
        if stack_legends:
            combined = zval_handles + [section_header('')] + style_handles
            _place_legend(combined, legend_loc_z)
        else:
            leg_z = _place_legend(zval_handles, legend_loc_z)
            ax.add_artist(leg_z)
            _place_legend(style_handles, legend_loc_cat)
    else:
        _place_legend(zval_handles, legend_loc_z)

def print_data_summary(x, y, z, tag=None):
    col_w = [10, 10, 10, 10]
    total_w = np.sum(col_w) + 3
    col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
    tab_rule = '-'*total_w
    tag_note = f'{tag}, ' if tag != None else ''
    print(f'\n{plotz} = {z} ({tag_note}N = {len(x)}):')
    if len(x) == 0: return
    print(tab_rule)
    print(col_format.format('Axis', 'Min', 'Mean', 'Max'))
    print(tab_rule)
    print(col_format.format(f'x', round(np.min(x), 4), round(np.mean(x), 4), round(np.max(x), 4)))
    print(col_format.format(f'y', round(np.min(y), 4), round(np.mean(y), 4), round(np.max(y), 4)))

def remove_nan_values(data):
    # Get indices of NaN values in all lists
    data = (list(data[0]), list(data[1]))
    nan_indices = []
    for dset in data:
        for i, e in enumerate(dset):
            if math.isnan(e) and i not in nan_indices:
                nan_indices.append(i)
    # Remove values at NaN indices from all lists
    output_data = data
    for dset in output_data:
        for i in sorted(nan_indices, reverse=True):
            del dset[i]
    print(f'\nRemoved {len(nan_indices)} datapoints that contained NaN values')        
    return output_data

def main():
    log_red = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_column_labels()
    fig, ax = set_up_figure(col_dict)
    
    _cats = categories or [{'label': '', 'col': None, 'op': None, 'val': None, 'ls': '-'}]
    fit_results = []
    for iz, zval in enumerate(zvals):
        zfilt = apply_filter(log_red, col_dict[plotz][0], '~=', zval)
        log_red_i = log_red[zfilt]
        for cat in _cats:
            if cat['col'] is not None:
                col = col_dict[cat['col']][0]
                cfilt = apply_filter(log_red_i, col, cat['op'], cat['val'], cat.get('atol', 0.01))
                log_red_i_c = log_red_i[cfilt]
            else:
                log_red_i_c = log_red_i
            log_red_i_c.reset_index(inplace=True)
            x, y = plot_data(ax, iz, log_red_i_c, marker_dict, cat, col_dict)
            (x, y) = remove_nan_values((x[0], y[0]))
            draw_connecting_lines(ax, iz, marker_dict, cat, x, y)
            popt, r2, fit_info = None, None, None
            if fit_auto_select:
                n_params_min = min(len(c['bounds'][0]) for c in fit_candidates)
            else:
                n_params_min = len(fit_p0) if fit_p0 is not None else len(fit_bounds[0])
            if fit_curves and len(x) >= n_params_min:
                popt, r2, fit_info = draw_curve_fit(ax, iz, marker_dict, cat['ls'], x, y)
            fit_results.append({
                'zval': zval, 'category': cat['label'], 'popt': popt, 'r2': r2,
                'fit_label':  fit_info['label']  if fit_info else curve_function_str,
                'fit_symbol': fit_info['symbol'] if fit_info else '',
            })
            print_data_summary(x, y, zval, cat['label'])

    if include_hline != None:
        draw_hline(ax, include_hline)

    if include_legend == True:
        create_legend(ax, col_dict, marker_dict, fit_results)

    if save_figure:
        z_part = plotz + (f'_{zvals[0]}' if len(zvals) == 1 else '')
        err_part = '__err_bars' if include_error_bars and (xerr_col or yerr_col) else ''
        cat_part = '__' + '_'.join(cat['label'] for cat in categories) if categories else ''
        fname = f'{plotx}_vs_{ploty}_vs_{z_part}{cat_part}{err_part}'
        out_dir = Path(__file__).parent / Path(__file__).stem
        out_dir.mkdir(exist_ok=True)
        plt.savefig(out_dir / f'{fname}.pdf', bbox_inches='tight')
        plt.savefig(out_dir / f'{fname}.png', dpi=dpi, bbox_inches='tight')
        print(f'\nFigure saved: {out_dir / fname}.[pdf/png]')
        if fit_curves:
            zval_labels = fmt_sigfigs_padded(zvals, legend_z_sigfigs)
            zval_label_map = {zv: lbl for zv, lbl in zip(zvals, zval_labels)}
            if fit_auto_select:
                lines = ['Fit candidates (auto-selected by R²):']
                for cand in fit_candidates:
                    lines.append(f'  {cand["symbol"]}  {cand["label"]}')
                lines.append('')
            else:
                lines = [f'Curve function: {curve_function_str}', '']
            for r in fit_results:
                zv_str = zval_label_map[r['zval']]
                tag = f'{plotz}={zv_str}' + (f', category={r["category"]}' if r['category'] else '')
                lines.append(f'Series: {tag}')
                if r['popt'] is not None:
                    if fit_auto_select:
                        lines.append(f'  Selected: {r["fit_label"]} [{r["fit_symbol"]}]')
                    param_names = list('abcdefghij')[:len(r['popt'])]
                    for name, val in zip(param_names, r['popt']):
                        lines.append(f'  {name} = {val:.6g}')
                    lines.append(f'  R² = {r["r2"]:.4f}')
                else:
                    lines.append('  fit failed')
                lines.append('')
            info_path = out_dir / f'{fname}__fit.txt'
            info_path.write_text('\n'.join(lines), encoding='utf-8')
            print(f'Fit info saved:  {info_path}')

    plt.show()

if __name__ == '__main__':
    main()