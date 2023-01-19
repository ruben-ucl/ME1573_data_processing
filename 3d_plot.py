import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from my_funcs import get_logbook

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

log = get_logbook()

fig = plt.figure()
projection = '3d'
ax = fig.add_subplot()
ax = fig.add_subplot(projection=projection)

# filter only PWM welding cases
welding = log['Powder material'] == 'None'

# filter only CW powder cases
cw = log['Point jump delay [us]'] == 0

# filter only Layer 1 cases
L1 = log['Layer'] == 1

# filter only Al10SiMg cases
Al10SiMg = log['Substrate material'] == 'Al10SiMg'

log_red = log[Al10SiMg & L1]
# print(log_red)

regime_markers = [('unstable keyhole', 'o'),
                  ('keyhole flickering', 's'),
                  ('quasi-stable keyhole', '^'),
                  ('quasi-stable vapour depression', 'D'),
                  ('conduction', 'v')
                  ]

for regime, m in regime_markers:
    reg_bool = log_red['Melting regime'] == regime
    log_red_reg = log_red[reg_bool]
    print(f'\n{regime}\n' + '-'*len(regime))
    print(log_red_reg)
    reg_pwr = log_red_reg['Avg. power [W]']
    
    if projection == '3d':
        xs = log_red_reg['Exposure time [us]']
        ys = log_red_reg['Point distance [um]']
        zs = log_red_reg['Avg. power [W]']
    
        ax.scatter(xs, ys, zs, marker=m, c=reg_pwr, cmap='jet', vmin=250, vmax=500)
    
        ax.set_xlabel('Exposure time [μs]')
        ax.set_ylabel('Point distance [μm]')
        ax.set_zlabel('Avg. power [W]')
        
    else:
        xs = log_red_reg['Scan speed [mm/s]']
        ys = log_red_reg['Power [W]']
        
        ax.scatter(xs, ys, marker = m, c=reg_pwr, cmap='jet', vmin=250, vmax=500)
        
        ax.set_xlabel('Scan speed [mm/s]')
        ax.set_ylabel('Power [W]')

ax.zticks = [300, 350, 400, 450, 500]
ax.yticks = [20, 40, 60, 80, 100]

plt.show()
