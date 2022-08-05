import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path(r"C:\Users\rlamb\Documents\UCL\Experiments\ESRF LTP 2 logbook AlSi10Mg only.xlsx")

log = pd.read_excel(log_path)

fig = plt.figure()
projection = '2d'
ax = fig.add_subplot()
# ax = fig.add_subplot(projection=projection)

# filter only PWM welding cases
welding = log["Powder material"] == "none"

# filter only CW powder cases
cw = log["point jump delay [μs]"] == 0

# filter only Layer 5 cases
L5 = log["Layer"] == 5

# log_red = log[welding]
log_red = log[cw]
# log_red = log[L5]

for power, m in [(300, "v"), (325, "*"), (350, "."), (375, "+"), (400, "D"), (450, "s"), (475, "x"), (500, "^")]:
    log_red_pwr = log_red[log_red["Power [W]"] == power]
    
    if projection == '3d':
        xs = log_red_pwr["Exposure time [μs]"]
        ys = log_red_pwr["Point distance [μm]"]
        zs = log_red_pwr["Power [W]"]
    
        ax.scatter(xs, ys, zs, marker = m)
    
        ax.set_xlabel('Exposure time [μs]')
        ax.set_ylabel('Point distance [μm]')
        ax.set_zlabel('Power [W]')
        
    else:
        xs = log_red_pwr["scan speed [mm/s]"]
        ys = log_red_pwr["Power [W]"]
        
        ax.scatter(xs, ys, marker = m)
        
        ax.set_xlabel("Scan speed [mm/s]")
        ax.set_ylabel("Power [W]")

ax.zticks = [300, 350, 400, 450, 500]
ax.yticks = [20, 40, 60, 80, 100]

plt.show()