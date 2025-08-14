import numpy as np
import matplotlib.pyplot as plt

def wave_position(t, L, f0, vt):
    alpha = vt / (2 * L * f0)
    tau = t * f0 * (1 - alpha**2)
    tau_mod = tau % 1
    
    forward_fraction = (1 + alpha) / 2
    
    x = np.where(tau_mod <= forward_fraction,
                 L * 2 * tau_mod / (1 + alpha),
                 L * (1 - 2 * (tau_mod - forward_fraction) / (1 - alpha)))
    return x

# Example parameters
L = 0.00035      # tank length
f0 = 2250     # tank frame frequency
vt = 0.8     # tank velocity
alpha = vt / (2 * L * f0)  # = 0.25

t = np.linspace(0, 0.002, 1000)
x = wave_position(t, L, f0, vt)

plt.plot(t*1000, x*1000000)
plt.xlabel('Time [ms]')
plt.ylabel('Wave Position [um]')
plt.title(f'Asymmetric Wave Motion (α = {alpha:.2f})')
plt.show()