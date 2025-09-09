import functools, math, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize as optimize
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, define_collumn_labels

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

###########################################################################################################################

### Figure settings ###
#----------------------
font_size = 8       # point
figsize = (6.3, 3.15)    # inch
dpi = 300
projection = '2d'
plot_bg = 'w'
n_subplots = 1

pop_nans = True
smooth_data = True
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
    
    plotz3 = 'melting_efficiency'

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
    
    # filter by beamtime
    ltp1 = log['Beamtime'] == 1
    ltp2 = log['Beamtime'] == 2
    ltp3 = log['Beamtime'] == 3
    
    # filter by substrate
    s0514 = log['Substrate No.'] == '514'
    s0515 = log['Substrate No.'] == '515'

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
    z3 = np.zeros_like(x)
    
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
        z3[i] = row[col_dict[plotz3][0]]

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
    
    return x, y, z, z2, z3

def remove_nan_values(data):
    # Get indices of NaN values in all lists
    nan_indices = []
    for dset in data:
        for i, e in enumerate(dset):
            if math.isnan(e) and i not in nan_indices:
                nan_indices.append(i)
                
    # Remove values at NaN indices from all lists
    output_data = np.delete(data, nan_indices, axis=1)
    
    print(f'Removed {len(nan_indices)} datapoints that contained NaN values')        
    
    return output_data

def draw_contours_old(fig, ax, col_dict, x, y, z, zlim, contour_levels, zticks,
                  label_var, contour_extend=None, cmap='Greys', alpha=1, filled=True,
                  resolution=2, smooth_factor=None):
                      
    # Define contour function depending on whether it needs to be filled or not
    contour_func = ax.tricontourf if filled == True else ax.tricontour
    
    # Define the number of levels depending on whether it is given explicitly or not
    levels = np.linspace(zlim[0], zlim[1], contour_levels) if zlim != None else np.linspace(min(z), max(z), contour_levels)
    
    # Create regular grid
    xi = np.linspace(np.min(x), np.max(x), resolution)
    yi = np.linspace(np.min(y), np.max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate scattered data to regular grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Apply smoothing
    if smooth_factor != None: zi = gaussian_filter(zi, sigma=smooth_factor)
    
    # Draw contours
    contours = contour_func(xi, yi, zi, levels=levels, cmap=cmap, zorder=0, extend=contour_extend, alpha=alpha, linewidths=0.5)
    
    # Include colourbar - optional
    if include_cbar == True:
        cbar = fig.colorbar(contours, location='right', ticks=zticks, label=col_dict[label_var][1], shrink=1)
        if zticks != None: cbar.set_ticks(zticks)
        
    # Draw single contour line at specified level - optional
    if contour_line != None:
        if contour_text_loc[0] == None: contour_text_loc[0], contour_text_loc[1] = ((max(x)+min(x))/2, (max(y)+min(y))/2)
        levels = [contour_line]
        contours = ax.tricontour(x, y, z, levels=levels, zorder=0, linestyles= '--', linewidths=0.7, colors=contour_line_color)
        contour_text = fr'{contour_label} = {contour_line}{contour_unit}' if contour_label != '' else fr'{contour_line}{contour_unit}'
        ax.text(contour_text_loc[0], contour_text_loc[1], contour_text, c=contour_line_color)



###############################################################


def fit_polynomial(x, y, degree=2):
    """Fit a polynomial of specified degree to x, y data."""
    coeffs = np.polyfit(x, y, degree)
    return coeffs

def eval_polynomial(x, coeffs):
    """Evaluate a polynomial with given coefficients at x points."""
    return np.polyval(coeffs, x)

def draw_contours(fig, ax, col_dict, x, y, z, zlim=None, contour_levels=20, zticks=None,
                 label_var=None, contour_extend=None, cmap='Greys', alpha=1.0, 
                 filled=True, resolution=20, smooth_factor=0, include_cbar=True, labels_on_plot=False,
                 contour_line=None, contour_line_color='k', contour_label='',
                 contour_unit='', contour_text_loc=(None, None), fit_contours=False,
                 fit_degree=3, min_points=10, r2_threshold=0.9, fit_color='r',
                 fit_style='--', fit_width=0.5, fit_alpha=0.7):
    """
    Draw smoothed contours from scattered data with optional features.
    
    Args:
        fig: matplotlib figure object
        ax: matplotlib axes object
        x, y, z: coordinate and value arrays
        zlim: tuple of (min, max) for z range, or None for auto-range
        contour_levels: number of contour levels
        zticks: custom ticks for colorbar, or None for auto-ticks
        label_var: label for colorbar
        contour_extend: how to extend contours beyond data range
        cmap: colormap name
        alpha: transparency of contours
        filled: whether to use filled contours
        resolution: grid resolution for smoothing
        smooth_factor: smoothing intensity
        include_cbar: whether to include colorbar
        contour_line: value for single contour line, or None
        contour_line_color: color for single contour line
        contour_label: label for single contour line
        contour_unit: unit for contour line label
        contour_text_loc: (x, y) location for contour line label
        fit_contours: whether to fit and plot polynomial curves to contour lines
        fit_contours: whether to fit polynomials to contour lines
        fit_degree: degree of polynomial fit (default: 2)
        min_points: minimum number of points required for fitting
        r2_threshold: minimum R² value for accepting a fit
        fit_color: color of fitted curves
        fit_style: line style of fitted curves
        fit_width: line width of fitted curves
        fit_alpha: transparency of fitted curves
    
    Returns:
        contours: contour plot object
        cbar: colorbar object (if include_cbar=True)
        fit_results: dict of fitting results if fit_contours=True
    """
    # Create regular grid for smoothing
    xi = np.linspace(np.min(x), np.max(x), resolution)
    yi = np.linspace(np.min(y), np.max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate scattered data to regular grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Apply smoothing
    zi = gaussian_filter(zi, sigma=smooth_factor, mode='nearest')
    
    # Define contour levels
    if zlim is not None:
        levels = np.linspace(zlim[0], zlim[1], contour_levels)
    else:
        levels = np.linspace(np.nanmin(z), np.nanmax(z), contour_levels)
    
    # Create contour plot
    contour_func = ax.contourf if filled else ax.contour
    contours = contour_func(
        xi, yi, zi,
        levels=levels,
        cmap=cmap,
        zorder=0,
        extend=contour_extend,
        alpha=alpha,
        linewidths=0.5
    )
    
    # Initialize fit results
    fit_results = {
        'levels': levels,
        'coefficients': [],
        'r2_scores': [],
        'n_points': []
    }
    
    # Fit polynomials to contour lines if requested
    if fit_contours:
        # Get contour line data
        contour_lines = ax.contour(
            xi, yi, zi,
            levels=levels,
            colors='k',
            linewidths=0.5,
            alpha=0.3,
            zorder=1
        )
        
        # Fit polynomial to each contour line
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        
        for i, collection in enumerate(contour_lines.collections):
            level = levels[i]
            paths = collection.get_paths()
            
            best_fit = None
            best_r2 = -np.inf
            best_x = None
            best_y = None
            
            # Try fitting each path segment
            for path in paths:
                vertices = path.vertices
                x_data = vertices[:, 0]
                y_data = vertices[:, 1]
                
                # Skip if too few points
                if len(x_data) < min_points:
                    continue
                
                try:
                    # Sort points by x-coordinate
                    sort_idx = np.argsort(x_data)
                    x_data = x_data[sort_idx]
                    y_data = y_data[sort_idx]
                    
                    # Fit polynomial
                    coeffs = fit_polynomial(x_data, y_data, degree=fit_degree)
                    y_fit = eval_polynomial(x_data, coeffs)
                    
                    # Calculate R² score
                    r2 = r2_score(y_data, y_fit)
                    
                    # Update best fit if this one is better
                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit = coeffs
                        best_x = x_data
                        best_y = y_data
                
                except Exception as e:
                    continue
            
            # Store and plot the best fit if it meets the threshold
            if best_fit is not None and best_r2 >= r2_threshold:
                fit_results['coefficients'].append(best_fit)
                fit_results['r2_scores'].append(best_r2)
                fit_results['n_points'].append(len(best_x))
                
                # Plot fitted curve
                y_fit = eval_polynomial(x_fit, best_fit)
                ax.plot(x_fit, y_fit, fit_style, color=fit_color, 
                       linewidth=fit_width, alpha=fit_alpha,
                       label=f'Level {level:.2f}, R²={best_r2:.2f}')
            else:
                fit_results['coefficients'].append(None)
                fit_results['r2_scores'].append(None)
                fit_results['n_points'].append(None)
    
    # Add colorbar if requested
    cbar = None
    if include_cbar:
        cbar = fig.colorbar(
            contours,
            ax=ax,
            location='right',
            ticks=zticks if zticks is not None else None
        )
        if label_var is not None:
            cbar.set_label(col_dict[label_var][1])
    
    # Add contour labels along the contour lines if requested
    if labels_on_plot == True:
        ax.clabel(contours)
    
    # Add single contour line if requested
    if contour_line is not None:
        # Set default text location to center if not specified
        if contour_text_loc[0] is None:
            text_x = (np.max(x) + np.min(x)) / 2
            text_y = (np.max(y) + np.min(y)) / 2
        else:
            text_x, text_y = contour_text_loc
        
        # Draw contour line
        single_contour = ax.contour(
            xi, yi, zi,
            levels=[contour_line],
            colors=contour_line_color,
            linestyles='--',
            linewidths=0.7,
            zorder=1
        )
        
        # Add contour label
        contour_text = (f'{contour_label} = {contour_line}{contour_unit}' 
                       if contour_label else f'{contour_line}{contour_unit}')
        ax.text(text_x, text_y, contour_text, color=contour_line_color)
    
    return contours, cbar

###############################################################



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

def print_data_summary(x, y, z, z2, z3):
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
        print(col_format.format('z3', round(min(z3), 2), round(np.mean(z3), 2), round(max(z3), 2)))

def main():
    log_red = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_collumn_labels()
    fig, axs = set_up_figure(col_dict)
    x, y, z, z2, z3 = plot_data(axs[0], log_red, marker_dict, col_dict)
    data = np.stack((x, y, z, z2, z3), axis=0)
    
    if pop_nans == True:
        data = remove_nan_values(data)
    
    if smooth_data == True:
        pass
    
    x = data[0]
    y = data[1]
    z = data[2]
    z2 = data[3]
    z3 = data [4]
        
    if include_contours == True and projection == '2d':
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_under(color='w', alpha=1)
        zlim = [0, 150]
        contour_levels = 7
        zticks = [0, 50, 100, 150]
        z_mod = z2
        print(z_mod)
        z_mod[z2 == 0] = -50
        draw_contours(fig,
            axs[0],
            col_dict,
            x,
            y,
            z_mod,
            zlim,
            contour_levels,
            zticks,
            label_var=plotz2,
            resolution=18,
            contour_extend='max',
            cmap=cmap,
            alpha=0.7)
        
        cmap = plt.get_cmap('Reds').copy()
        cmap.set_under(color='w', alpha=1)
        zlim = [0, 180]
        contour_levels = 7
        zticks = [0, 60, 120, 180]
        z_mod = z
        z_mod[z == 0] = -30
        draw_contours(fig,
            axs[0],
            col_dict,
            x,
            y,
            z_mod,
            zlim,
            contour_levels,
            zticks,
            label_var=plotz,
            resolution=18,
            contour_extend=None,
            cmap=cmap,
            alpha=0.7)
            
        cmap = plt.get_cmap('Greys')
        zlim = [0.05, 0.35]
        contour_levels = 7
        zticks = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        draw_contours(fig,
            axs[0],
            col_dict,
            x,
            y,
            z3,
            zlim,
            contour_levels,
            zticks,
            label_var=plotz3,
            resolution=18,
            cmap=cmap,
            alpha=1,
            filled=False,
            labels_on_plot=True,
            include_cbar=False)
    
    axs[0].set_facecolor('grey')

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
    
    print_data_summary(x, y, z, z2, z3)
    
    plt.show()

if __name__ == '__main__':
    main()