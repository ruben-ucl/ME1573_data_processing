import functools, math, os, re, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize as optimize
from scipy.interpolate import Rbf, griddata, RBFInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.path import Path as mpl_Path
from sklearn.metrics import r2_score

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, define_column_labels, apply_filter, filter_logbook_tracks

__author__ ='Rubén Lambert-Garcia'
__version__ = '1.0'

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

###########################################################################################################################

### Figure ###
font_size  = 9
figsize    = (3.15, 2.5)   # inch  (page width = 6.3)
# figsize  = (4, 4)
dpi        = 300
projection = '3d'           # '2d' or '3d'
plot_bg    = None           # background colour, or None for transparent
save_figure = True

### Scatter points ###
point_size = 20                 # Default 30
regime_point_colours = False
regime_point_shapes  = True
colour_points_by_z   = False
label_points         = False
point_stems_3d       = False

### Overlays ###
include_hline    = None     # float or None
include_curve_fit   = False
include_surface_fit = False
include_legend   = False

### Contours ###
LED_contours          = False
include_contours      = True
contour_interpolation = 'rbf'               # 'rbf', 'cubic', or 'natural_neighbor' / 'natural_neighbour'
rbf_kernel            = 'cubic' # 'thin_plate_spline', 'linear', 'cubic', 'gaussian'
rbf_smoothing         = 2                 # 0 = exact fit; higher = looser/smoother
rbf_epsilon           = 5                 # shape param for 'gaussian' — larger = more localised
nn_sigma              = 7                  # Gaussian blur sigma for 'natural_neighbor' (grid cells; higher = smoother boundaries)
contour_cmap          = 'inferno'
contour_alpha         = 1.0
cbar_n_ticks          = 5    # number of colorbar ticks (auto nice numbers via MaxNLocator)
contour_subdivisions  = 3    # contour bands per tick interval; total = (cbar_n_ticks-1) × this
cap_at_zlim           = False # clip auto ticks to zlim; False allows MaxNLocator to round beyond
include_cbar          = True
contour_extend        = 'neither'           # 'neither', 'min', 'max' or 'both'
contour_line          = None                # draw a single contour line at this z value
contour_label         = r'$\eta$'          # label for contour line
# contour_label       = r'$\theta_{FKW}$'
contour_unit          = ''
contour_text_loc      = (1150, 470)
contour_line_color    = 'k'

### Data ###
pop_nans         = True
axis_sci_not     = []    # List with any combination of 'x', 'y', 'z' (or empty for none)
yerr_col         = None     # col_dict key for y error bars, or None
skip_tracks      = ['0106_02', '0106_03', '0106_06']

### Regime contour order ###
REGIME_ORDER = [            # low → high energy input; sets colour ordering in regime fill
    'conduction',
    'keyhole flickering',
    'quasi-stable vapour depression',
    'quasi-stable keyhole',
    'unstable keyhole',
]

### Logbook filters ###
LOGBOOK_FILTERS = {
    'material':   'AlSi10Mg',
    'layer':      1,
    'base_type':  'powder',
    # 'laser_mode': 'cw',
    # 'regime':     'not_cond',
    # 'substrate_no': '0514',
    'custom': [
        # {'col': 'Duty cycle', 'op': '~=', 'val': 1.0, 'label': 'dc=1.0'},
        # {'col': 'Duty cycle', 'op': '~=', 'val': 0.89, 'label': 'dc=0.89'},
        {'col': 'Duty cycle', 'op': '~=', 'val': 0.8, 'label': 'dc=0.8'},
    ],
}

### X-axis ###
if True:
    plotx = 'scan_speed'
    # xlim = [150, 1300]
    # xlim = [0, 0.4]
    xlim = [300, 2100]                      # scan speed
    xticks = [400, 800, 1200, 1600, 2000]   # scan speed
    # xticks = [40, 50, 60, 70, 80, 90]
    # xticks = [10, 20, 30, 40]
    # xlim = None
    # xticks = None

### Y-axis ###
if True:
    ploty = 'power'
    # ylim = [150, 1400]
    # ylim = [1200, 6200]
    # ylim = [-6, 86]
    # ylim= [-0.1, 1.1]
    ylim = [225, 515]                           # power
    yticks = [250, 300, 350, 400, 450, 500]     # power
    # yticks = [30, 45, 60, 75, 90]
    # yticks = [0, 40, 80]
    # ylim = None
    # yticks = None

### Z-axis (or contour variable) ###
if True:
    plotz = 'KH_depth'
    zlim = [0, 400]
    # zlim = [30, 120]
    # zlim = [0, 18000]                                                       # G
    # zlim = [0, 5500000]                                                     # dT/dt
    zticks = None               # explicit tick positions — overrides cbar_n_ticks
    # zticks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000] # G
    # zticks = [140, 180, 220, 260, 300, 340, 380]                            # R
    # zticks = [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000] # dT/dt

def filter_logbook():
    log = get_logbook()
    log_red, active_filters = filter_logbook_tracks(log, LOGBOOK_FILTERS)
    log_red.reset_index(inplace=True)
    return log_red, active_filters

def set_up_figure(col_dict):
    proj_dict = {'2d': 'rectilinear',
                 '3d': '3d'
                 }
    # Set up figure with two or three axes
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = fig.add_subplot(projection=proj_dict[projection])
    if plot_bg != None: ax.set_facecolor(plot_bg)
    else:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    
    ax.set_xlabel(col_dict[plotx][1])
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if xticks != None: ax.set_xticks(xticks)
    
    ax.set_ylabel(col_dict[ploty][1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if yticks != None: ax.set_yticks(yticks)
    # ax.set_yticklabels(['N', 'S', 'A'])
    
    _sci_axes = set(axis_sci_not)
    for _a in _sci_axes - {'z'}:
        ax.ticklabel_format(axis=_a, style='sci', scilimits=(0,0))

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
        if zlim != None: ax.set_zlim(zlim[0], zlim[1])
        if zticks != None: ax.set_zticks(zticks)
        if 'z' in _sci_axes:
            ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    if projection == '2d':
        ax.tick_params(labelsize=font_size - 1)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

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
            marker_dict[k]['c'] = 'w'
            
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
        
        if trackid in skip_tracks:
            continue

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
                                 s = point_size,      # 30 for half page width figure
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
            if yerr_col is not None:
                err = row[col_dict[yerr_col][0]]
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

def draw_contours(fig, ax, col_dict, xx, yy, zz, zlim, contour_subdivisions, zticks, label_var, contour_extend=None, cmap='Greys', alpha=1, hull_xy=None, cbar_tick_labels=None):
    x_min, x_max = (xlim[0], xlim[1]) if xlim is not None else (min(xx), max(xx))
    y_min, y_max = (ylim[0], ylim[1]) if ylim is not None else (min(yy), max(yy))
    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    if contour_interpolation == 'rbf':
        try:
            rbf = RBFInterpolator(np.column_stack([xx, yy]), zz, kernel=rbf_kernel, smoothing=rbf_smoothing, epsilon=rbf_epsilon)
        except np.linalg.LinAlgError:
            fallback = max(rbf_smoothing, 0.5)
            print(f'RBF singular matrix with smoothing={rbf_smoothing}, retrying with smoothing={fallback}')
            rbf = RBFInterpolator(np.column_stack([xx, yy]), zz, kernel=rbf_kernel, smoothing=fallback, epsilon=rbf_epsilon)
        grid_z = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    elif contour_interpolation in ('natural_neighbor', 'natural_neighbour'):
        # Voronoi partition (each grid cell takes the value of its nearest data point),
        # then Gaussian blur to smooth the hard boundaries between regions.
        nn = NearestNDInterpolator(np.column_stack([xx, yy]), zz)
        grid_z = nn(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        grid_z = gaussian_filter(grid_z.astype(float), sigma=nn_sigma)
    else:
        grid_z = griddata((xx, yy), zz, (grid_x, grid_y), method='cubic')

    # Mask grid to convex hull of all scatter points to suppress extrapolation
    hx, hy = hull_xy if hull_xy is not None else (xx, yy)
    hull_pts = np.column_stack([hx, hy])
    hull_path = mpl_Path(hull_pts[ConvexHull(hull_pts).vertices])
    outside = ~hull_path.contains_points(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    grid_z[outside.reshape(grid_x.shape)] = np.nan
    v_min = zlim[0] if zlim is not None else np.nanmin(grid_z)
    v_max = zlim[1] if zlim is not None else np.nanmax(grid_z)
    if cbar_tick_labels is not None and zticks is not None:
        # Discrete/categorical colormap (regime): zticks are the band boundaries
        levels = zticks
        tick_positions = None
    else:
        if zticks is not None:
            tick_positions = np.array(zticks)
        else:
            tp = MaxNLocator(nbins=cbar_n_ticks - 1).tick_values(v_min, v_max)
            tick_positions = tp[(tp >= v_min) & (tp <= v_max)] if cap_at_zlim else tp
        levels = np.concatenate([np.linspace(tick_positions[i], tick_positions[i + 1], contour_subdivisions + 1)[:-1]
                                 for i in range(len(tick_positions) - 1)] + [tick_positions[-1:]])
    contours = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, zorder=0, extend=contour_extend, alpha=alpha)
    if include_cbar == True:
        cbar = fig.colorbar(contours, location='right', label=col_dict[label_var][1], shrink=1)
        if cbar_tick_labels is not None:
            cbar.set_ticks(np.arange(len(cbar_tick_labels)))
            cbar.set_ticklabels(cbar_tick_labels, fontsize=font_size - 1)
        else:
            cbar.set_ticks(tick_positions)
        if 'z' in set(axis_sci_not):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((0, 0))
            cbar.formatter = fmt
            cbar.update_ticks()
    
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
    # return a*x + b
    # return a*x**b
    # return a*(1-b**(-c*x))+d
    # return a/(1+np.exp(-b*(x-c)))
    # return a+b*np.log(x)
    return 0.3 * (1 - np.exp(-x/b))
    # return a * (x**b / (c + x**b))

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
    a, b, c, d = [round(e, 2) for e in popt]
    ax.text(0.45, 0.55,
            # r'$\theta_{FKW} = tan^{-1}\left[a \dot \left(\frac{\Delta H}{h_m} \dot L_{th}^*+b\right)\right]$'+f'\na = {a}, b = {b}\nR\u00b2 = {round(r2, 3)}' 
            # fr'$\eta={a}+{b}\log(V)$'
            fr'$\eta={a}\frac{{V^{{{b}}}}}{{{c}+V^{{{b}}}}}$'
            '\n'
            '$\it{R\u00b2} = $' + f'${round(r2, 2):1.2f}$',
            fontsize = 'small',
            transform=ax.transAxes,
            color='k'
            )
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
                       loc = 'upper center',
                       bbox_to_anchor = (0.5, -0.2),
                       ncol = 5,
                       fontsize = 'small',
                       fancybox = False,
                       framealpha = 0,
                       edgecolor = 'inherit',
                       columnspacing = 1
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
    print(col_format.format('x', round(min(xx), 4), round(np.mean(xx), 4), round(max(xx), 4)))
    print(col_format.format('y', round(min(yy), 4), round(np.mean(yy), 4), round(max(yy), 4)))
    print(col_format.format('z', round(min(zz), 4), round(np.mean(zz), 4), round(max(zz), 4)))

def main():
    log_red, active_filters = filter_logbook()
    marker_dict = define_point_formats()
    col_dict = define_column_labels()
    fig, ax = set_up_figure(col_dict)

    xx, yy, zz = plot_data(fig, ax, log_red, marker_dict, col_dict)
    xx_all, yy_all = xx[:], yy[:]  # full scatter positions before NaN removal

    if plotz == 'regime':
        _regime_int = {r: i for i, r in enumerate(REGIME_ORDER)}
        zz = [_regime_int.get(r, np.nan) for r in zz]

    if pop_nans == True:
        data = (xx, yy, zz)
        (xx, yy, zz) = remove_nan_values(data)

    if include_contours == True and projection == '2d':
        if plotz == 'regime':
            _regime_colors = [marker_dict[r]['c'] for r in REGIME_ORDER if r in marker_dict]
            _cmap   = mpl.colors.ListedColormap(_regime_colors)
            _zticks = np.arange(-0.5, len(_regime_colors))
            _extend = 'neither'
            _tick_labels = [r for r in REGIME_ORDER if r in marker_dict]
        else:
            _cmap, _zticks, _extend, _tick_labels = contour_cmap, zticks, contour_extend, None
        draw_contours(fig, ax, col_dict, xx, yy, zz, zlim, contour_subdivisions, _zticks,
                      label_var=plotz, contour_extend=_extend, cmap=_cmap,
                      alpha=contour_alpha, hull_xy=(xx_all, yy_all), cbar_tick_labels=_tick_labels)
    
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
    
    if save_figure:
        err_part = '_err_bars' if yerr_col is not None else ''
        safe_filters = re.sub(r'[<>:"/\\|?*\s]', '', '_'.join(str(f) for f in active_filters))
        fname = f'{plotx}_vs_{ploty}_vs_{plotz}_{safe_filters}{err_part}'
        out_dir = Path(__file__).parent / Path(__file__).stem
        out_dir.mkdir(exist_ok=True)
        transparent = plot_bg is None
        plt.savefig(out_dir / f'{fname}.pdf', bbox_inches='tight', transparent=transparent)
        plt.savefig(out_dir / f'{fname}.png', dpi=dpi, bbox_inches='tight', transparent=transparent)
        print(f'\nFigure saved: {out_dir / fname}.[pdf/png]')
    plt.show()
    
if __name__ == '__main__':
    main()