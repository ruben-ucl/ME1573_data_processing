import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path(r'C:\Users\rlamb\Documents\UCL\Experiments\ESRF LTP 2 logbook AlSi10Mg only.xlsx')

log = pd.read_excel(log_path)

fig = plt.figure()
projection = '2d'
ax = fig.add_subplot()
# ax = fig.add_subplot(projection=projection)

# filter only PWM welding cases
welding = log['Powder material'] == 'None'

# filter only CW powder cases
cw = log['Point jump delay [μs]'] == 0

# filter only Layer 1 cases
L1 = log['Layer'] == 1

# filter only Al10SiMg cases
Al10SiMg = log['Substrate material'] == 'Al10SiMg'

# log_red = log[welding]
log_red = log[cw & Al10SiMg]
# log_red = log[L1]

regime_markers = [('unstable keyhole', 'o'),
                  ('keyhole flickering', 's'),
                  ('quasi-stable keyhole', '^'),
                  ('quasi-stable vapour depression', 'D'),
                  ('conduction', 'v')
                  ]

for regime, m in regime_markers:
    reg_bool = log_red['Melting regime'] == regime
    log_red_reg = log_red[reg_bool]
    reg_pwr = log_red_reg['Power [W]']
    
    if projection == '3d':
        xs = log_red_reg['Exposure time [μs]']
        ys = log_red_reg['Point distance [μm]']
        zs = log_red_reg['Power [W]']
    
        ax.scatter(xs, ys, zs, marker=m, c=reg_pwr, cmap='jet')
    
        ax.set_xlabel('Exposure time [μs]')
        ax.set_ylabel('Point distance [μm]')
        ax.set_zlabel('Avg. power [W]')
        
    else:
        xs = log_red_reg['Scan speed [mm/s]']
        ys = log_red_reg['Power [W]']
        
        ax.scatter(xs, ys, marker = m, c=reg_pwr, cmap='jet')
        
        ax.set_xlabel('Scan speed [mm/s]')
        ax.set_ylabel('Power [W]')

ax.zticks = [300, 350, 400, 450, 500]
ax.yticks = [20, 40, 60, 80, 100]

plt.show()
