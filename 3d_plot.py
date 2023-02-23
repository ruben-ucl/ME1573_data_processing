import functools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import get_logbook

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

projection = '3d'
LED_contours = False
# Define trackids to label on plot:
# trackid_list = ['0105_04', '0323_01', '0323_02', '0323_03', '0323_04', '0323_05', '0323_06', '0324_01', '0324_03', '0324_06']
trackid_list = None
regime_point_formats = True
include_legend = False
label_points = False
point_stems_3d = True

log = get_logbook()

# filters for welding or powder melting
welding = log['Powder material'] == 'None'
powder = np.invert(welding)

# filters for CW or PWM laser mode
cw = log['Point jump delay [us]'] == 0
pwm = np.invert(cw)

# filter for Layer 1 tracks only
L1 = log['Layer'] == 1

# filter for Al10SiMg only
Al10SiMg = log['Substrate material'] == 'Al10SiMg'

# Apply combination of above filters to select parameter subset to plot
log_red = log[Al10SiMg & L1 & pwm & welding]
               
# Set up figure with two or three axes
if projection == '3d':
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(4, 4), dpi=300, tight_layout=True)
    ax = fig.add_subplot(projection=projection)
    ax.set_xlabel('Exposure time [μs]')
    ax.set_ylabel('Point distance [μm]')
    ax.set_zlabel('Avg. power [W]')
    ax.set_zlim(200, 500)
    # ax.set_xlabel('Scan speed [mm/s]')
    # ax.set_ylabel('Power [W]')
    # ax.set_zlabel('End-of-track depression')
    # ax.yticks = [20, 40, 60, 80, 100]
    # ax.set_xticks([400, 800, 1200, 1600])
    # ax.set_yticks([300, 350, 400, 450, 500])
    # ax.set_zticks([0, 1])
    
elif projection == '2d':
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(5, 3), dpi=300, tight_layout=True)
    ax = fig.add_subplot()
    if LED_contours == True:
        S, P = np.mgrid[200:2501, 150:551]
        Z = 1000 * P / S
        cs = ax.contourf(S, P, Z, 28, cmap='hot', alpha=0.7)
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('LED [J/m]')
    ax.set_xlabel('LED [J/m]')
    ax.set_ylabel('Pore number')
    # ax.set_xlim(200, 2200)
    # ax.set_ylim(175, 525)
    # ax.set_ylim([-0.2, 1.2])
    # ax.set_yticks([200, 250, 300, 350, 400, 450, 500])
    # ax.set_yticks([0, 1])
    # ax.set_xticks([400, 800, 1200, 1600, 2000])

# Define marker formats based on melting regime
if regime_point_formats == True:
    marker_dict = {'unstable keyhole': {'m': 'o', 'c': '#fde725'},
                   'keyhole flickering': {'m': 's', 'c': '#3b528b'},
                   'quasi-stable keyhole': {'m': '^', 'c': '#5ec962'},
                   'quasi-stable vapour depression': {'m': 'D', 'c': '#21918c'},
                   'conduction': {'m': 'v', 'c': '#440154'}
                   }
else:
    marker_dict = {'unstable keyhole': {'m': 'o', 'c': 'k'},
                   'keyhole flickering': {'m': 'o', 'c': 'k'},
                   'quasi-stable keyhole': {'m': 'o', 'c': 'k'},
                   'quasi-stable vapour depression': {'m': 'o', 'c': 'k'},
                   'conduction': {'m': 'o', 'c': 'k'}
                   }
               
# Add points to plot
for _, row in log_red.iterrows():
    trackid = row['trackid']
    regime = row['Melting regime']
    
    # Don not plot point if regime not categorised OR if trackid_list is specified and does not contain current trackid
    if regime not in marker_dict.keys() or \
       trackid_list is not None and trackid not in trackid_list:
        continue
    
    power = row['Avg. power [W]']
    pt_dist = row['Point distance [um]']
    exp_t = row['Exposure time [us]']
    scan_speed = row['Scan speed [mm/s]']
    LED = row['LED [J/m]']
    n_pores = row['n_pores']
    end_of_track_depression = row['end_of_track_depression']
    
    if projection == '2d':
        # Set variables to plot
        x = LED
        y = n_pores
        ax.scatter(x, y,
                   label = regime,
                   c = marker_dict[regime]['c'],
                   marker = marker_dict[regime]['m'],
                   edgecolors = 'k',
                   linewidths = 0.5
                   )
        if label_points == True:
            ax.text(x, y-10,
                    trackid,
                    va = 'top',
                    ha = 'center',
                    fontsize = 'small'
                    )
        
    elif projection == '3d':
        # Set variables to plot
        x = exp_t
        y = pt_dist
        z = power
        
        if point_stems_3d == True:
            markerline, stemlines, baseline = ax.stem([x], [y], [z],
                                                      bottom = 200,
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


# Final formatting of figure        
if include_legend == True:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(),
                       by_label.keys(),
                       # loc = 'upper center',
                       # bbox_to_anchor = (0.6, 1.3),
                       ncol = 1,
                       fontsize = 'medium',
                       fancybox = False,
                       framealpha = 1.0,
                       edgecolor = 'inherit'
                       )
    legend.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

plt.show()
