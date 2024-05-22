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
font_size = 8
# figsize = (3.15, 2.5) # page width = 6.3
figsize = (3.15, 2.5)
dpi = 300
projection = '2d'
plot_bg = 'w'

pop_nans = True                     # bool
regime_point_colours = True         # bool
regime_point_shapes = True          # bool
colour_points_by_z = False          # bool
label_points = False                # bool
point_stems_3d = False              # bool
include_hline = None                # float
include_error_bars = None    # string or None
include_legend = False              # bool

include_curve_fit = False           # bool
include_surface_fit = False         # bool

LED_contours = False                # bool
include_contours = False            # bool
contour_cmap = 'Blues'              # string
contour_levels = 10                 # int
contour_alpha = 0.7                 # float
include_cbar = True                 # bool
contour_extend = None               # 
contour_line = None                 # float
contour_label = r'$AR_{KH}$'        # string
# contour_label = r'$\theta_{FKW}$'
contour_unit = '°'                  # string
contour_text_loc = (1150, 470)      # tuple
contour_line_color = 'k'            # string

### X-axis settings ###
#----------------------
if True:
    plotx = 'MP_depth'
    # xlim = [150, 1300]
    # xlim = [0, 0.4]
    # xlim = [300, 2100]                      # scan speed
    # xticks = [400, 800, 1200, 1600, 2000]   # scan speed
    # xticks = [40, 50, 60, 70, 80, 90]
    # xticks = [10, 20, 30, 40]
    xlim = None
    xticks = None

### Y-axis settings ###
#----------------------
if True:
    ploty = 'MP_vol'
    # ylim = [150, 1400]
    # ylim = [1200, 6200]
    # ylim = [-6, 86]
    # ylim= [-0.1, 1.1]
    # ylim = [230, 520]                           # power
    # yticks = [250, 300, 350, 400, 450, 500]     # power
    # yticks = [30, 45, 60, 75, 90]
    # yticks = [0, 40, 80]
    ylim = None
    yticks = None

### Z-axis settings ###
#----------------------
if True:
    plotz = 'pore_vol'
    # zlim = [0, 2]
    # zticks = [0, 1, 2, 3]
    zlim = None
    zticks = None
    # zlim = [0, 18000]                                                       # G
    # zticks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000] # G
    # zticks = [140, 180, 220, 260, 300, 340, 380]  # R
    # zlim = [0, 5500000] # dT/dt
    # zticks = [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000] # dT/dt
    bins_per_tick = 1
    if bins_per_tick != None and zticks != None: contour_levels = (len(zticks) - 1) * bins_per_tick + 1

def filter_logbook():
    log = get_logbook()
    
    if True:
        # filters for welding or powder melting
        welding = log['Powder material'] == 'None'
        powder = np.invert(welding)

        # filters for CW or PWM laser mode
        cw = log['Point jump delay [us]'] == 0
        pwm = np.invert(cw)

        # filter for Layer 1 tracks only
        L1 = log['Layer'] == 1
        
        # filter for presence of KH pores
        pores = log['n_pores'] > 2
        
        # filter by scan speed
        speed = log['Scan speed [mm/s]'] == 400

        # filter by material
        AlSi10Mg = log['Substrate material'] == 'AlSi10Mg'
        Al7A77 = log['Substrate material'] == 'Al7A77'
        Al = log['Substrate material'] == 'Al'
        Ti64 = log['Substrate material'] == 'Ti64'
        lit = np.logical_or(Ti64, Al7A77)

    # Apply combination of above filters to select parameter subset to plot
    # log_red = log[np.logical_or(AlSi10Mg, lit) & L1 & cw & powder]
    log_red = log[AlSi10Mg & L1 & cw & powder]
    # print(log_red)
    return log_red

def set_up_figure(col_dict):
    proj_dict = {'2d': 'rectilinear',
                 '3d': '3d'
                 }
    # Set up figure with two or three axes
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = fig.add_subplot(projection=proj_dict[projection])
    if plot_bg != None: ax.set_facecolor(plot_bg)
    
    ax.set_xlabel(col_dict[plotx][1])
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if xticks != None: ax.set_xticks(xticks)
    
    ax.set_ylabel(col_dict[ploty][1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if yticks != None: ax.set_yticks(yticks)
    # ax.set_yticklabels(['N', 'S', 'A'])
    
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Draw LED contours in P-V map background
    if projection == '2d' and LED_contours == True:
        S, P = np.mgrid[xlim[0]:xlim[1]+1, ylim[0]:ylim[1]+1]
        Z = np.clip(1000 * P / S, None, 1450)
        cs = ax.contourf(S, P, Z, 13, cmap='hot', alpha=0.7)
        if include_cbar == True:
            cbar = fig.colorbar(cs)
            cbar.ax.set_ylabel('LED [J/m]')
            cbar.set_ticks([100, 300, 500, 700, 900, 1100, 1300, 1500])
            
    elif projection == '3d':      
        ax.set_zlabel(col_dict[plotz][1])
        ax.set_zlim(zlim[0], zlim[1])
        if zticks != None: ax.set_zticks(zticks)
        
    return fig, ax

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
            marker_dict[k]['c'] = 'k'
            
    if regime_point_shapes == False:
        for k in marker_dict:
            marker_dict[k]['m'] = 'o'
            
    return marker_dict

def plot_data(fig, ax, log_red, marker_dict, col_dict):
    print('plot_data()')
    # Initialise lists for storing point coordinates
    xx = []
    yy = []
    zz = []
    # Add points to plot by iterating through the logbook row by row
    for _, row in log_red.iterrows():
        trackid = row['trackid']
        regime = row['Melting regime']
        
        # Do not plot point if regime not categorised
        if regime not in marker_dict:
            continue
        
        # Set variables to plot
        x = row[col_dict[plotx][0]]
        y = row[col_dict[ploty][0]]#+row[col_dict['layer_thickness'][0]]
        z = row[col_dict[plotz][0]]
        xx.append(x)
        yy.append(y)
        zz.append(z)
        
        if projection == '2d':
            scatter = ax.scatter(x, y,
                                 label = regime,
                                 c = z if colour_points_by_z == True else marker_dict[regime]['c'],
                                 marker = marker_dict[regime]['m'],
                                 edgecolors = 'k',
                                 linewidths = 0.5,
                                 s = 30,      # 30
                                 cmap = 'Reds',
                                 vmin = 70,
                                 vmax = 120
                                 )
                       
            if label_points == True:
                ax.text(x, y,
                        trackid,
                        va = 'top',
                        ha = 'left',
                        fontsize = 'xx-small',
                        )
            if include_error_bars != None:
                err = row[col_dict[include_error_bars][0]]
                ax.errorbar(x, y, xerr=None, yerr=err, ecolor='k', elinewidth=0.6, capsize=3.2, capthick=0.6, zorder=0)
            
        elif projection == '3d':            
            if point_stems_3d == True:
                markerline, stemlines, baseline = ax.stem([x], [y], [z],
                                                          bottom = 0,
                                                          linefmt = '--',
                                                          basefmt = 'none',
                                                          markerfmt = 'none'
                                                          )
                stemlines.set(linewidth = 0.7,
                              color = 'grey'
                              )
            ax.scatter(x, y, z,
                       label = regime,
                       c = marker_dict[regime]['c'],
                       marker = marker_dict[regime]['m'],
                       edgecolors = 'k',
                       linewidths = 0.5
                       )
    # Add colourmap for point colours                   
    if colour_points_by_z == True:
        cbar = fig.colorbar(scatter, location='top', ticks=zticks, label=col_dict[plotz][1], pad=0.1, aspect=25)
        if zticks != None: cbar.set_ticks(zticks)
    
    return xx, yy, zz

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
    # return a*x**b
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
    ax.text(0.5, 0.75, f'R\u00b2 = {round(r2, 2):1.2f}', transform=ax.transAxes)
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

def print_data_summary(xx, yy, zz):
    col_w = [10, 10, 10, 10]
    total_w = np.sum(col_w) + 3
    col_format = '{:<'+str(col_w[0])+'} {:<'+str(col_w[1])+'} {:<'+str(col_w[2])+'} {:<'+str(col_w[3])+'}'
    tab_rule = '-'*total_w
    print()
    print(col_format.format('Axis', 'Min', 'Mean', 'Max'))
    print(tab_rule)
    print(col_format.format('x', round(min(xx), 2), round(np.mean(xx), 2), round(max(xx), 2)))
    print(col_format.format('y', round(min(yy), 2), round(np.mean(yy), 2), round(max(yy), 2)))
    print(col_format.format('z', round(min(zz), 2), round(np.mean(zz), 2), round(max(zz), 2)))

def main():
    log_red = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_collumn_labels()
    fig, ax = set_up_figure(col_dict)
    xx, yy, zz = plot_data(fig, ax, log_red, marker_dict, col_dict)
    
    if pop_nans == True:
        data = (xx, yy, zz)
        (xx, yy, zz) = remove_nan_values(data)
        
    if include_contours == True and projection == '2d':
        draw_contours(fig, ax, col_dict, xx, yy, zz, zlim, contour_levels, zticks, label_var=plotz, contour_extend=contour_extend, cmap=contour_cmap, alpha=contour_alpha)
    
    if include_hline != None and projection == '2d':
        draw_hline(ax, include_hline)
        
    if include_curve_fit == True and projection == '2d':
        # for data in [(xx, yy), (xx2, yy2), (xx3, yy3)]:
            # draw_curve_fit(data[0], data[1])
        draw_curve_fit(ax, xx, yy)
    
    if include_surface_fit == True and projection == '3d':
        draw_surface_fit(fig, ax, xx, yy, zz)
    
    if include_legend == True:
        create_legend(ax)
    
    print_data_summary(xx, yy, zz)
    
    plt.show()
    
if __name__ == '__main__':
    main()