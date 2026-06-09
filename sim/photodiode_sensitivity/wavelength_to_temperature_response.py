import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# ── Constants ────────────────────────────────────────────────────────────────
h  = 6.626e-34   # Planck constant (J·s)
c  = 3.0e8       # Speed of light (m/s)
kB = 1.381e-23   # Boltzmann constant (J/K)

# ── 1. Responsivity data ─────────────────────────────────────────────────────
CSV_FILE = "InGaAs.csv"   # <-- change this to your filename

import pandas as pd
df = pd.read_csv(CSV_FILE, header=None, names=["wavelength_nm", "responsivity"])
df = df.sort_values("wavelength_nm").dropna()

# ── Unit check: if wavelength values are in microns (i.e. all < 100),
#    convert to nm. Adjust this if your CSV is already in nm.
wavelength_nm = df["wavelength_nm"].values
if wavelength_nm.max() < 100:
    print(f"Wavelength values appear to be in µm (max={wavelength_nm.max():.2f}) — converting to nm")
    wavelength_nm = wavelength_nm * 1000

responsivity = df["responsivity"].values

# ── 2. Interpolate responsivity onto a fine wavelength grid ──────────────────
lam_min = wavelength_nm.min()
lam_max = wavelength_nm.max()
lam_nm  = np.linspace(lam_min, lam_max, 2000)          # fine grid (nm)
lam_m   = lam_nm * 1e-9                                 # convert to metres

R_interp = interp1d(wavelength_nm, responsivity,
                    kind='cubic', bounds_error=False, fill_value=0.0)
R = R_interp(lam_nm)
R = np.clip(R, 0, None)                                 # no negative responsivity

# ── 3. Planck spectral radiance ──────────────────────────────────────────────
def planck(lam_m, T):
    """Spectral radiance B(λ, T) in W·sr⁻¹·m⁻²·m⁻¹"""
    return (2*h*c**2 / lam_m**5) / (np.exp(h*c / (lam_m*kB*T)) - 1)

# ── 4. Compute signal S(T) and sensitivity dS/dT ────────────────────────────
T_range = np.linspace(600, 3500, 500)                   # adjust range as needed
S = np.zeros_like(T_range)

for i, T in enumerate(T_range):
    B = planck(lam_m, T)
    S[i] = simpson(B * R, x=lam_m)                      # weighted integral

# Normalise S to its maximum for easier plotting
S_norm = S / S.max()

# Sensitivity: dS/dT (numerical derivative)
dSdT      = np.gradient(S,      T_range)
dSdT_norm = np.gradient(S_norm, T_range)

# ── 5. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 11))
fig.suptitle("Photodiode Blackbody Temperature Sensitivity", fontsize=13)

# — Plot 1: Weighted product B(λ,T)·R(λ) + responsivity ————————————————————
ax1 = axes[0]
ax1b = ax1.twinx()

for T, col in [(800, '#4477AA'), (1200, '#66CCEE'), (1800, '#CCBB44'),
               (2400, '#EE6677'), (3200, '#AA3377')]:
    B = planck(lam_m, T)
    BR = B * R
    # Normalise only over the diode's wavelength window so the
    # Planck tail outside the band doesn't collapse everything to zero
    peak = BR.max()
    BR_norm = BR / peak if peak > 0 else BR
    # Suppress values that are truly negligible (< 0.1% of peak)
    BR_norm[BR_norm < 1e-3] = np.nan
    ax1b.plot(lam_nm, BR_norm, color=col, alpha=0.6, linewidth=1.4,
              label=f'{T} K')

ax1.plot(lam_nm, R, 'k-', linewidth=2, label='Responsivity R(λ)', zorder=5)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Responsivity (normalised)', color='k')
ax1b.set_ylabel('B(λ,T)·R(λ) (normalised)', color='grey')
ax1b.tick_params(axis='y', labelcolor='grey')
ax1b.set_ylim(bottom=0)
ax1.set_xlim(lam_min, lam_max)
ax1.set_ylim(bottom=0)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
ax1.set_title("Effective spectral contribution B(λ,T)·R(λ) with responsivity overlay")
ax1.grid(True, alpha=0.3)

# — Plot 2: Integrated signal S(T) ————————————————————————————————————————————
ax2 = axes[1]
ax2.plot(T_range, S_norm, color='steelblue', linewidth=2)
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('Signal S(T) (normalised)')
ax2.set_title("Integrated photodiode signal vs blackbody temperature")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(T_range[0], T_range[-1])

# — Plot 3: Sensitivity dS/dT ——————————————————————————————————————————————
ax3 = axes[2]
ax3.plot(T_range, dSdT_norm * 1e-3, color='crimson', linewidth=2)  # scale for readability
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('dS/dT (normalised, ×10⁻³ K⁻¹)')
ax3.set_title("Temperature sensitivity dS/dT")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(T_range[0], T_range[-1])
ax3.axhline(0, color='k', linewidth=0.8, linestyle='--')

plt.tight_layout()
plt.savefig("photodiode_temp_sensitivity.png", dpi=150, bbox_inches='tight')
plt.show()