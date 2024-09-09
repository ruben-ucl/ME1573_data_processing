import functools, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize as optimize
from scipy.interpolate import Rbf
from sklearn.metrics import r2_score

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, define_collumn_labels


__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

###########################################################################################################################

### Figure settings ###
#----------------------
font_size = 9       # point
figsize = (4, 4)    # inch (page width = 6.3)
dpi = 300
plot_bg = 'w'

pop_nans = True
regime_point_colours = True
regime_point_shapes = True
label_points = False
include_hline = None
include_error_bars = False
include_legend = True
fit_curves = True


### X-axis settings ###
#----------------------
if True:
    plotx = 'scan_speed'
    xlim = [300, 2100]
    xticks = [400, 800, 1200, 1600, 2000]

### Y-axis settings ###
#----------------------
if True:
    ploty = 'MP_vol'
    ylim = None
    yticks = None

### Z-axis settings ###
#----------------------
if True:
    plotz = 'power'
    # zunit = 'W'
    # zvals = [400, 800, 1200, 1600]
    zvals = [500, 450, 400, 350, 300]

def filter_logbook():
    log = get_logbook()

    # filters for welding or powder melting
    welding = log['Powder material'] == 'None'
    powder = np.invert(welding)

    # filters for CW or PWM laser mode
    cw = log['Point jump delay [us]'] == 0
    pwm = np.invert(cw)

    # filter for Layer 1 tracks only
    L1 = log['Layer'] == 1

    # filter by material
    AlSi10Mg = log['Substrate material'] == 'AlSi10Mg'
    Al7A77 = log['Substrate material'] == 'Al7A77'
    Al = log['Substrate material'] == 'Al'
    Ti64 = log['Substrate material'] == 'Ti64'
    lit = np.logical_or(Ti64, Al7A77)

    # Apply combination of above filters to select parameter subset to plot
    # log_red = log[np.logical_or(AlSi10Mg, lit) & L1 & cw & powder]
    log_red = log[AlSi10Mg & L1 & cw]
    
    # filter by regime classified
    log_red = log_red[log_red['Melting regime'].notna()]
    
    # reset index
    log_red.reset_index(inplace=True)
    
    # print(log_red)
    return log_red

def set_up_figure(col_dict):
    # Set up figure with two or three axes
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = fig.add_subplot()

    if plot_bg != None: ax.set_facecolor(plot_bg)
    
    ax.set_xlabel(col_dict[plotx][1])
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if xticks != None: ax.set_xticks(xticks)
    
    ax.set_ylabel(col_dict[ploty][1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if yticks != None: ax.set_yticks(yticks)
        
    return fig, ax

def define_point_formats():
                 
    marker_dict = {0: {'mp': 'o', 'mw': 'x', 'c': '#fcffa4'},
                   1: {'mp': 'o', 'mw': 'x', 'c': '#fca50a'},
                   2: {'mp': 'o', 'mw': 'x', 'c': '#dd513a'},
                   3: {'mp': 'o', 'mw': 'x', 'c': '#932667'},
                   4: {'mp': 'o', 'mw': 'x', 'c': '#420a68'},
                   'unstable keyhole': {'m': 'o', 'c': '#fde725'},
                   'keyhole flickering': {'m': 's', 'c': '#3b528b'},
                   'quasi-stable keyhole': {'m': '^', 'c': '#5ec962'},
                   'quasi-stable vapour depression': {'m': 'D', 'c': '#21918c'},
                   'conduction': {'m': 'v', 'c': '#440154'},
                   'Al7A77 (Huang et al., 2022)': {'m': 'd', 'c': 'k'},
                   'Ti64 (Zhao et al., 2020)': {'m': 'd', 'c': 'lightgray'},
                   'Ti64 (Cunningham et al., 2019)': {'m': 'd', 'c': 'gray'},
                   }
                   
    # if regime_point_colours == False:
        # for k in marker_dict:
            # marker_dict[k]['c'] = 'w'
            
    # if regime_point_shapes == False:
        # for k in marker_dict:
            # marker_dict[k]['m'] = 'o'
            
    return marker_dict

def plot_data(ax, iz, log_red, marker_dict, pw, col_dict):
    # Initialise lists for storing point coordinates    
    x = np.zeros((len(log_red), 1))
    y = np.zeros_like(x)

    style = {'powder': {'m': 'o', 'l': '-', 'z': 0},
             'weld':   {'m': 'x', 'l': '--', 'z': 1}
             }
    
    # Add points to plot by iterating through the logbook row by row
    for i, row in log_red.iterrows():
        trackid = row['trackid']
        regime = row['Melting regime']
            
        # Set variables to plot
        x[i] = row[col_dict[plotx][0]]
        y[i] = row[col_dict[ploty][0]]
        
        # ax.scatter(x[i], y[i],
                   # label = f"{pw} {zvals[iz]} mm/s",
                   # c = marker_dict[iz]['c'],
                   # marker = marker_dict[iz][m],
                   # edgecolors = 'k',
                   # linewidths = 0.5,
                   # s = 25,
                   # )
                   
        ax.plot(x[i], y[i],
                label = zvals[iz],
                # label = f"{pw} {zvals[iz]} {zunit}",
                c = marker_dict[iz]['c'],
                marker = marker_dict[iz]['mp'],
                markeredgecolor = 'k',
                markeredgewidth = 0.8,
                ls = style[pw]['l'],
                lw = 0,
                ms = 7.5,
                # zorder = style[pw]['z']
                )
                   
        if label_points == True:
            ax.text(x[i]+10, y[i],
                    trackid[1],
                    va = 'center',
                    ha = 'center',
                    fontsize = 'xx-small',
                    )
        if include_error_bars == True:
            err = row[col_dict['fkw_angle_sd'][0]]/np.sqrt(row[col_dict['fkw_angle_n_samples'][0]])
            ax.errorbar(x[i], y[i], xerr=err, yerr=16, ecolor='k', elinewidth=0.6, capsize=3.2, capthick=0.6, zorder=0)
    
    return x.T, y.T

def draw_hline(ax, hliney):
    
    ax.plot(ax.get_xlim(), (hliney, hliney), c='gray', ls='--', lw=0.7, zorder=0)
    ax.text(1400, 0.185, f'η = {hliney}', c='gray')

def curve_function(x, a, b, c, d):
    # return a*x**3 + b*x**2 + c*x + d
    # return a*np.exp(-(x-b)**2/(2*c**2))+d
    # return a*x**2 + b*x + c
    # return a*b**x
    # return a + b*np.log(x)
    # return a*np.exp(b*x)
    # return np.arctan(a*(x+b))*(180/np.pi)
    return a*x**b
    pass

def draw_curve_fit(ax, iz, marker_dict, ls, xx, yy):
    # Remove value pairs that include NaN entries
    # xx = [x for x, y in zip(xx, yy) if not math.isnan(y)]
    # yy = [y for y in yy if not math.isnan(y)]
    
    popt, _ = optimize.curve_fit(curve_function, xx, yy, p0=[1, 0, 1000, 0])
    # popt = (0.29, -0.2)
    
    X = np.linspace(min(xx), max(xx), 50)
    Y = curve_function(X, *popt)
    
    residuals = yy - curve_function(np.array(xx), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r2 = 1 - (ss_res/ss_tot)
    a, b, c, d = [round(e, 3) for e in popt]
    # ax.text(np.min(xx), np.max(yy), 'a={a}, b={b}, R\u00b2 = {r2}'
        # .format(a=a, b=b, r2=round(r2, 2)))
    ax.text(np.min(xx), np.max(yy), f'R\u00b2 = {round(r2, 2)}')
    ax.plot(X, Y, c = marker_dict[iz]['c'], ls = ls, lw=1, zorder=0)

def create_legend(ax, col_dict):
    # Get handles and labels for points
    handles, labels = plt.gca().get_legend_handles_labels()
    # Combine into dictionary to eliminate duplicates
    by_label = dict(zip(labels, handles))
    # Re-order legend entries
    order = [0, 1, 2, 3, 4, 5]
    # order = [0, 1, 2]
    order = order[:len(by_label)]
    handles = [list(by_label.values())[i] for i in order]
    labels = [list(by_label.keys())[i] for i in order]
    # Create legend
    legend = ax.legend(handles,
                       labels,
                       title = col_dict[plotz][1],
                       # loc = 'upper center',
                       # bbox_to_anchor = (0.5, -0.2),
                       ncol = 1,
                       fontsize = 'medium',
                       fancybox = False,
                       framealpha = 0,
                       edgecolor = 'inherit'
                       )
    legend.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

def print_data_summary(x, y, z):
    col_w = [10, 10, 10, 10]
    total_w = np.sum(col_w) + 3
    col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
    tab_rule = '-'*total_w
    print(f'\n{plotz} = {z}:')
    print(col_format.format('Axis', 'Min', 'Mean', 'Max'))
    print(tab_rule)
    print(col_format.format(f'x', round(np.min(x), 2), round(np.mean(x), 2), round(np.max(x), 2)))
    print(col_format.format(f'y', round(np.min(y), 2), round(np.mean(y), 2), round(np.max(y), 2)))

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
    print(f'Removed {len(nan_indices)} datapoints that contained NaN values')        
    return output_data

def main():
    log_red = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_collumn_labels()
    fig, ax = set_up_figure(col_dict)
    
    for iz, zval in enumerate(zvals):
        zfilt = log_red[col_dict[plotz][0]] == zval
        log_red_i = log_red[zfilt]
        # log_red_i.reset_index(inplace=True)
        wfilt = log_red_i['Powder material'] == 'None'
        log_red_i_w = log_red_i[wfilt]
        log_red_i_w.reset_index(inplace=True)
        print(log_red_i_w)
        
        pfilt = log_red_i['Powder material'] != 'None'
        log_red_i_p = log_red_i[pfilt]
        log_red_i_p.reset_index(inplace=True)
        print(log_red_i_p)
        
        x, y = plot_data(ax, iz, log_red_i_p, marker_dict, 'powder', col_dict)
        (x, y) = remove_nan_values((x[0], y[0]))
        
        if fit_curves == True: draw_curve_fit(ax, iz, marker_dict, '-', x, y)
        
        # x, y = plot_data(ax, iz, log_red_i_w, marker_dict, 'weld', col_dict)
        # (x, y) = remove_nan_values((x[0], y[0]))
        # draw_curve_fit(ax, iz, marker_dict, '--', x, y)
        
        print_data_summary(x, y, zval)
        
    if include_hline != None:
        draw_hline(ax, include_hline)
    
    if include_legend == True:
        create_legend(ax, col_dict)
    
    plt.show()

if __name__ == '__main__':
    main()