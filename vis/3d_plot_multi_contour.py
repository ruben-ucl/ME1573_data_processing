import functools, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import get_logbook
from my_funcs import define_collumn_labels
import scipy.optimize as optimize
from scipy.interpolate import Rbf
from sklearn.metrics import r2_score

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

###########################################################################################################################

### Figure settings ###
#----------------------
font_size = 9       # point
figsize = (6.3, 3.15)    # inch
dpi = 300
projection = '2d'
plot_bg = 'w'
n_subplots = 1

pop_nans = True
regime_point_colours = False
regime_point_shapes = True
label_points = False
point_stems_3d = False
include_hline = None
include_error_bars = False
include_legend = False

include_curve_fit = False
include_surface_fit = False

LED_contours = False
include_contours = True
include_cbar = True
contour_levels = 16
contour_extend = None
contour_line = None
contour_label = r'$\eta$'
contour_unit = ''
contour_text_loc = [1100, 465]
contour_line_color = 'k'

### X-axis settings ###
#----------------------
if True:
    plotx = 'scan_speed'
    # xlim = [150, 1300]
    # xlim = [0, 0.4]
    xlim = [300, 2100]                      # scan speed
    xticks = [400, 800, 1200, 1600, 2000]   # scan speed
    # xticks = [40, 50, 60, 70, 80, 90]
    # xticks = [10, 20, 30, 40]
    # xticks = None

### Y-axis settings ###
#----------------------
if True:
    ploty = 'power'
    # ylim = [150, 1400]
    # ylim = [1200, 6200]
    # ylim = [-6, 86]
    # ylim= [-0.1, 1.1]
    ylim = [230, 520]                           # power
    yticks = [250, 300, 350, 400, 450, 500]     # power
    # yticks = [30, 45, 60, 75, 90]
    # yticks = [0, 40, 80]
    # yticks = None

### Z-axis settings ###
#----------------------
if True:
    plotz = 'pore_density'
    # zlim = [0, 2]
    # zticks = [0, 1, 2, 3]
    zlim = None
    zticks = None
    # zlim = [0, 18000]                                                       # G
    # zticks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000] # G
    # zticks = [140, 180, 220, 260, 300, 340, 380]  # R
    # zlim = [0, 5500000] # dT/dt
    # zticks = [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000] # dT/dt
    bins_per_tick = None
    if bins_per_tick != None and zticks != None: contour_levels = (len(zticks) - 1) * bins_per_tick + 1

    plotz2 = 'eot_depression_depth'

### Additional axes settings ###
#-------------------------------
if True:
    plotx2 = 'LED'
    ploty2 = 'eot_depression_depth'
    y2lim = [-10, 150]
    y2ticks = [0, 70, 140]
    
    
    plotx3 = 'LED'
    ploty3 = 'h_pores'
    y3lim = [-0.07, 1.07]
    y2ticks = [0, 1]
    plotz3 = None

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
    log_red = log[AlSi10Mg & L1 & cw & powder]
    
    # filter by regime classified
    log_red = log_red[log_red['Melting regime'].notna()]
    
    # reset index
    log_red.reset_index(inplace=True)
    
    # print(log_red)
    return log_red

def set_up_figure(col_dict):
    proj_dict = {'2d': 'rectilinear',
                 '3d': '3d'
                 }
    # Set up figure with two or three axes
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    axs = []
    for i in range(n_subplots):
        ax = fig.add_subplot(projection=proj_dict[projection])
        axs.append(ax)

    if plot_bg != None:
        for ax in axs:
            ax.set_facecolor(plot_bg)
    
    ax.set_xlabel(col_dict[plotx][1])
    ax.set_xlim(xlim[0], xlim[1])
    if xticks != None: ax.set_xticks(xticks)
    
    ax.set_ylabel(col_dict[ploty][1])
    ax.set_ylim(ylim[0], ylim[1])
    if yticks != None: ax.set_yticks(yticks)
    
    if projection == '2d' and LED_contours == True:
        S, P = np.mgrid[xlim[0]:xlim[1]+1, ylim[0]:ylim[1]+1]
        Z = np.clip(1000 * P / S, 100, 1450)
        print(np.min(Z), np.max(Z))
        cs = ax.contourf(S, P, Z, 13, cmap='Greys_r', alpha=0.55, vmin=100, vmax=1450)
        if include_cbar == True:
            cbar = fig.colorbar(cs)
            cbar.ax.set_ylabel('LED [J/m]')
            cbar.set_ticks([100, 400, 700, 1000, 1300])
            
    elif projection == '3d':      
        ax.set_zlabel(col_dict[plotz][1])
        ax.set_zlim(zlim[0], zlim[1])
        if zticks != None: ax.set_zticks(zticks)
        
    return fig, axs

def define_point_formats():
    # Define marker formats based on melting regime
    marker_dict = {'unstable keyhole': {'m': 'o', 'c': '#fde725'},
                   'keyhole flickering': {'m': 's', 'c': '#3b528b'},
                   'quasi-stable keyhole': {'m': '^', 'c': '#5ec962'},
                   'quasi-stable vapour depression': {'m': 'D', 'c': '#21918c'},
                   'conduction': {'m': 'v', 'c': '#440154'},
                   'Al7A77 (Huang et al., 2022)': {'m': 'd', 'c': 'k'},
                   'Ti64 (Zhao et al., 2020)': {'m': 'd', 'c': 'lightgray'},
                   'Ti64 (Cunningham et al., 2019)': {'m': 'd', 'c': 'gray'},
                   }
                   
    if regime_point_colours == False:
        for k in marker_dict:
            marker_dict[k]['c'] = 'w'
            
    if regime_point_shapes == False:
        for k in marker_dict:
            marker_dict[k]['m'] = 'o'
            
    return marker_dict

def plot_data(ax, log_red, marker_dict, col_dict):
    # Initialise lists for storing point coordinates    
    x = np.zeros((len(log_red)))
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    z2 = np.zeros_like(x)
    
    # Add points to plot by iterating through the logbook row by row
    for i, row in log_red.iterrows():
        trackid = row['trackid']
        regime = row['Melting regime']
        
        # Do not plot point if regime not categorised
        if regime not in marker_dict:
            for arr in [x, y, z]:
                np.delete(arr, obj=i, axis=0)
            continue
            
        # Set variables to plot
        x[i] = row[col_dict[plotx][0]]
        y[i] = row[col_dict[ploty][0]]
        z[i] = row[col_dict[plotz][0]]
        z2[i] = row[col_dict[plotz2][0]]

        if projection == '2d':
            ax.scatter(x[i], y[i],
                       label = regime,
                       c = marker_dict[regime]['c'],
                       # c = z,
                       marker = marker_dict[regime]['m'],
                       edgecolors = 'k',
                       linewidths = 0.5,
                       s = 25,
                       # cmap = 'hot',
                       # vmin = 1,
                       # vmax = 5
                       )
                       
            if label_points == True:
                ax.text(x[i], y[i]-15,
                        trackid,
                        va = 'center',
                        ha = 'center',
                        fontsize = 'xx-small',
                        )
            if include_error_bars == True:
                err = row[col_dict['fkw_angle_sd'][0]]/np.sqrt(row[col_dict['fkw_angle_n_samples'][0]])
                ax.errorbar(x[i], y[i], xerr=err, yerr=16, ecolor='k', elinewidth=0.6, capsize=3.2, capthick=0.6, zorder=0)
            
        elif projection == '3d':            
            if point_stems_3d == True:
                markerline, stemlines, baseline = ax.stem([x[i]], [y[i]], [z[i]],
                                                              bottom = 0,
                                                              linefmt = '--',
                                                              basefmt = 'none',
                                                              markerfmt = 'none'
                                                              )
                stemlines.set(linewidth = 0.7,
                              color = 'grey'
                              )
            ax.scatter(x[i], y[i], z[i],
                           label = regime,
                           c = marker_dict[regime]['c'],
                           marker = marker_dict[regime]['m'],
                           edgecolors = 'k',
                           linewidths = 0.5
                           )
    
    return x, y, z, z2

def remove_nan_values(data):
    # Get indices of NaN values in all lists
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

def draw_contours(fig, ax, col_dict, xx, yy, zz, zlim, contour_levels, zticks, label_var, contour_extend=None, cmap='Greys', alpha=1):
    levels = np.linspace(zlim[0], zlim[1], contour_levels) if zlim != None else np.linspace(min(zz), max(zz), contour_levels)
    contours = ax.tricontourf(xx, yy, zz, levels=levels, cmap=cmap, zorder=0, extend=contour_extend, alpha=alpha)
    if include_cbar == True:
        cbar = fig.colorbar(contours, location='right', ticks=zticks, label=col_dict[label_var][1], shrink=1)
        if zticks != None: cbar.set_ticks(zticks)
    
    if contour_line != None:
        if contour_text_loc[0] == None: contour_text_loc[0], contour_text_loc[1] = ((max(xx)+min(xx))/2, (max(yy)+min(yy))/2)
        levels = [contour_line]
        contours = ax.tricontour(xx, yy, zz, levels=levels, zorder=0, linestyles= '--', linewidths=0.7, colors=contour_line_color)
        contour_text = fr'{contour_label} = {contour_line}{contour_unit}' if contour_label != '' else fr'{contour_line}{contour_unit}'
        ax.text(contour_text_loc[0], contour_text_loc[1], contour_text, c=contour_line_color)

def draw_hline(ax, hliney):
    ax.plot((0, 1400), (hliney, hliney), c='gray', ls='--', lw=0.7, zorder=0)
    ax.text(1000, 0.185, f'η = {hliney}', c='gray')

def surf_function(data, a, b, c, d, e, f, g, h, i, j):
    x = data[0]
    y = data[1]
    # return (a*x**3 + b*x**2 + c*x + d) * (e*y**3 + f*y**2 + g*y * h)
    # return (b*x**2 + c*x + d) * (f*y**2 + g*y * h)
    # return a**x * (b*y**3 + c*y**2 + d*y * e)
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**2*y + h*x*y**2 + i*x**3 + j*y**3

def curve_function(x, a, b, c, d):
    # return a*x**3 + b*x**2 + c*x + d
    # return a*x**2 + b*x + c
    return a*x + b
    # return a**(x + b) + c
    # return np.arctan(a*(x+b))*(180/np.pi)
    pass

def draw_curve_fit(ax, xx, yy):
    # Remove value pairs that include NaN entries
    xx = [x for x, y in zip(xx, yy) if not math.isnan(y)]
    yy = [y for y in yy if not math.isnan(y)]
    
    popt, _ = optimize.curve_fit(curve_function, xx, yy)
    # popt = (0.29, -0.2)
    
    X = np.linspace(min(xx), max(xx), 50)
    Y = curve_function(X, *popt)
    
    residuals = yy - curve_function(np.array(xx), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r2 = 1 - (ss_res/ss_tot)
    a, b, c, d = [round(e, 3) for e in popt]
    # ax.text(max(xx)*0.5, max(yy), f'y = {a}x\u00b2 + {b}x + {c}\nR\u00b2 = {round(r2, 3)}')
    # ax.text(max(xx)*0.35, max(yy)*0.55,
            # r'$\theta_{FKW} = tan^{-1}\left[a \dot \left(\frac{\Delta H}{h_m} \dot L_{th}^*+b\right)\right]$'+f'\na = {a}, b = {b}\nR\u00b2 = {round(r2, 3)}', 
            # fontsize = 'small'
            # )
    ax.text((max(xx)+min(xx))*0.5, max(yy)*0.8, f'R\u00b2 = {round(r2, 2)}')
    ax.plot(X, Y, 'k--', lw=0.75, zorder=0)

def draw_surface_fit(fig, ax, xx, yy, zz):
    # print(pd.DataFrame({'x': xx, 'y': yy, 'z': zz}))
    # Remove value pairs that include NaN entries
    xx = [x for x, z in zip(xx, zz) if not math.isnan(z)]
    yy = [y for y, z in zip(yy, zz) if not math.isnan(z)]
    zz = [z for z in zz if not math.isnan(z)]
    
    popt, _ = optimize.curve_fit(surf_function, np.array([xx, yy]), zz)
    
    model_x_data = np.linspace(min(xx), max(xx), 50)
    model_y_data = np.linspace(min(yy), max(yy), 50)
    X, Y = np.meshgrid(model_x_data, model_y_data)
    Z = surf_function(np.array([X, Y]), *popt)
    # rbf = Rbf(xx, yy, zz, function='cubic', smooth=0)
    # Z = rbf(X, Y)
    
    for i in np.arange(len(X)):
        for j in np.arange(len(Y)):
            xi = model_x_data[i]
            yj = model_y_data[j]
            if yj > 0.1*xi + 350:
                Z[i, j] = np.nan
            # t1 = yj < 0.04 * xi + 230
            # t2 = yj > 0.2 * xi + 250
            # if t1 or t2:
                # Z[i, j] = np.nan
    
    residuals = zz - surf_function(np.array([xx, yy]), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((zz-np.mean(zz))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    ax.text(max(xx)/2, max(yy), max(zz)*0.75, f'R\u00b2 = {round(r2, 3)}')
    surf = ax.plot_surface(X, Y, Z, alpha=0.8, cmap='hot', vmin=min(zz), vmax=max(zz))
    if include_cbar == True:
        cbar = fig.colorbar(surf, shrink=0.4, location='left')

def create_legend(ax):
    # Get handles and labels for points
    handles, labels = plt.gca().get_legend_handles_labels()
    # Combine into dictionary to eliminate duplicates
    by_label = dict(zip(labels, handles))
    # Re-order legend entries
    order = [0, 1, 4, 3, 2, 5, 6, 7]
    order = order[:len(by_label)]
    handles = [list(by_label.values())[i] for i in order]
    labels = [list(by_label.keys())[i] for i in order]
    # Create legend
    legend = ax.legend(handles,
                       labels,
                       # loc = 'upper center',
                       # bbox_to_anchor = (0.5, -0.2),
                       ncol = 1,
                       fontsize = 'small',
                       fancybox = False,
                       framealpha = 0,
                       edgecolor = 'inherit'
                       )
    legend.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

def print_data_summary(x, y, z, z2):
    col_w = [10, 10, 10, 10]
    total_w = np.sum(col_w) + 3
    col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
    tab_rule = '-'*total_w
    print()
    print(col_format.format('Axis', 'Min', 'Mean', 'Max'))
    print(tab_rule)
    for i in range(n_subplots):
        print(col_format.format('x', round(min(x), 2), round(np.mean(x), 2), round(max(x), 2)))
        print(col_format.format('y', round(min(y), 2), round(np.mean(y), 2), round(max(y), 2)))
        print(col_format.format('z', round(min(z), 2), round(np.mean(z), 2), round(max(z), 2)))
        print(col_format.format('z2', round(min(z2), 2), round(np.mean(z2), 2), round(max(z2), 2)))

def main():
    log_red = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_collumn_labels()
    fig, axs = set_up_figure(col_dict)
    x, y, z, z2 = plot_data(axs[0], log_red, marker_dict, col_dict)
    
    if pop_nans == True:
        data = (x, y, z)
        (x, y, z) = remove_nan_values(data)
        
    if include_contours == True and projection == '2d':
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_under(color='w', alpha=0)
        zlim = [0, 150]
        contour_levels = 7
        zticks = [0, 50, 100, 150]
        z_mod = z2
        z_mod[z2 == 0] = -1
        draw_contours(fig, axs[0], col_dict, x, y, z_mod, zlim, contour_levels, zticks, label_var=plotz2, contour_extend=None, cmap=cmap, alpha=0.7)
        
        cmap = plt.cm.get_cmap('Reds').copy()
        cmap.set_under(color='w', alpha=0)
        zlim = [0, 180]
        contour_levels = 7
        zticks = [0, 60, 120, 180]
        z_mod = z
        z_mod[z == 0] = -30
        draw_contours(fig, axs[0], col_dict, x, y, z_mod, zlim, contour_levels, zticks, label_var=plotz, contour_extend=None, cmap=cmap, alpha=0.7)
        
        # cmap = plt.cm.get_cmap('Blues').copy()
        # cmap.set_under(color='w', alpha=0)
        # zlim = [0, 2]
        # contour_levels = 3
        # zticks = [0, 1, 2]
        # draw_contours(fig, ax, col_dict, xx, yy, zz, zlim, contour_levels, zticks, label_var=plotz, contour_extend=None, cmap=cmap, alpha=0.4)
        
    if include_hline != None and projection == '2d':
        draw_hline(axs[0], include_hline)
        
    if include_curve_fit == True and projection == '2d':
        # for data in [(xx, yy), (xx2, yy2), (xx3, yy3)]:
            # draw_curve_fit(data[0], data[1])
        draw_curve_fit(axs[0], x[0], y[0])
    
    if include_surface_fit == True and projection == '3d':
        draw_surface_fit(fig, axs[0], x[0], y[0], z[0])
    
    if include_legend == True:
        create_legend(axs[0])
    
    print_data_summary(x, y, z, z2)
    
    plt.show()

if __name__ == '__main__':
    main()