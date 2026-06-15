import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.signal import medfilt

# ── Constants ────────────────────────────────────────────────────────────────
h  = 6.626e-34   # Planck constant (J·s)
c  = 3.0e8       # Speed of light (m/s)
kB = 1.381e-23   # Boltzmann constant (J/K)

# ── 1. Detector definitions ───────────────────────────────────────────────────
DETECTORS = [
    {"csv": "sim/photodiode_sensitivity/Si.csv",     "label": "Si",     "color": "#1DB954"},
    {"csv": "sim/photodiode_sensitivity/InGaAs.csv", "label": "InGaAs", "color": "#00B4AE"},
]

# ── 2. Load and process responsivity data ────────────────────────────────────
def load_responsivity(csv_file):
    df = pd.read_csv(csv_file, header=None, names=["wavelength", "responsivity"])
    df = df.sort_values("wavelength").dropna()
    wl_um = df["wavelength"].values
    if wl_um.max() > 100:  # values are in nm, convert to µm
        print(f"{csv_file}: wavelengths appear to be in nm — converting to µm")
        wl_um = wl_um / 1000
    resp = medfilt(df["responsivity"].values, [5])
    return wl_um, resp

# ── 3. Planck spectral radiance ──────────────────────────────────────────────
def planck(lam_m, T):
    """Spectral radiance B(λ, T) in W·sr⁻¹·m⁻²·m⁻¹"""
    return (2*h*c**2 / lam_m**5) / (np.exp(h*c / (lam_m*kB*T)) - 1)

# ── 4. Compute signal S(T) and sensitivity dS/dT for each detector ───────────
T_plot  = [600, 1200, 1800, 2400, 3000, 3600]
T_range = np.linspace(T_plot[0], T_plot[-1], 500)

for det in DETECTORS:
    wl_um, resp_raw = load_responsivity(det["csv"])
    lam_um = np.linspace(wl_um.min(), wl_um.max(), 2000)
    lam_m  = lam_um * 1e-6

    R = np.clip(interp1d(wl_um, resp_raw, kind='linear',
                         bounds_error=False, fill_value=0.0)(lam_um), 0, None)

    S = np.array([simpson(planck(lam_m, T) * R, x=lam_m) for T in T_range])

    det.update({"lam_um": lam_um, "lam_m": lam_m,
                "lam_min": wl_um.min(), "lam_max": wl_um.max(),
                "R": R, "S": S})

# ── 5. Normalise S(T) individually; group-normalise dS/dT ───────────────────
for det in DETECTORS:
    S_norm = det["S"] / det["S"].max()
    det["S_norm"]  = S_norm
    det["dSdT"]    = np.gradient(S_norm, T_range)

dSdT_ref = max(np.abs(det["dSdT"]).max() for det in DETECTORS)
for det in DETECTORS:
    det["dSdT_norm"] = det["dSdT"] / dSdT_ref

# ── 7. Wavelength grid spanning both detectors ───────────────────────────────
lam_global_um = np.linspace(min(d["lam_min"] for d in DETECTORS),
                             max(d["lam_max"] for d in DETECTORS), 2000)
lam_global_m  = lam_global_um * 1e-6

# ── 8. Plots ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
})

fig = plt.figure(figsize=(6.30, 6.3))   # A4 width minus 25 mm margins, × height
fig.suptitle("Photodiode Blackbody Temperature Sensitivity", fontsize=11)

gs     = GridSpec(2, 1, figure=fig, hspace=0.22, height_ratios=[1, 1])
gs_bot = gs[1].subgridspec(2, 1, hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs_bot[0])
ax3 = fig.add_subplot(gs_bot[1], sharex=ax2)
plt.setp(ax2.get_xticklabels(), visible=False)

# — Plot 1: Planck curves + responsivity windows ————————————————————————————
ax1b = ax1.twinx()

cmap = plt.colormaps['plasma'].resampled(len(T_plot))
for i, T in enumerate(T_plot):
    B = planck(lam_global_m, T)
    ax1b.plot(lam_global_um, B * 1e-6, color=cmap(i), alpha=0.9,
              linewidth=1.4, label=f'{T} K')

for det in DETECTORS:
    ax1.fill_between(det["lam_um"], det["R"], alpha=0.2, color=det["color"])
    ax1.plot(det["lam_um"], det["R"], color=det["color"], linewidth=1.5)
    x_mid = (det["lam_um"].min() + det["lam_um"].max()) / 2
    ax1.text(x_mid, det["R"].max() * 0.45, det["label"],
             color=det["color"], fontweight='bold',
             ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=3))

ax1.set_xlabel('Wavelength (µm)')
ax1.set_ylabel(r'Responsivity, $R(\lambda)$ [A/W]')
ax1b.set_ylabel(r'Spectral radiance, $B(\lambda, T)$' + '\n[W·sr⁻¹·m⁻²·µm⁻¹]')
ax1b.set_yscale('log')
ax1b.tick_params(axis='y', labelcolor='black')
ax1b.set_ylim(top=max(B)*20*1e-6)
ax1.set_xlim(lam_global_um.min(), lam_global_um.max())
ax1.set_ylim(bottom=0)
_, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(*ax1b.get_legend_handles_labels(), loc='lower right', ncol=2, framealpha=0.7)

# — Plot 2: Integrated signal S(T) ————————————————————————————————————————————
for det in DETECTORS:
    ax2.plot(T_range, det["S_norm"], color=det["color"], linewidth=2, label=det["label"])
ax2.set_ylabel('S(T) [norm.]')
ax2.set_xlim(T_plot[0], T_plot[-1])
ax2.set_xticks(T_plot)
ax2.text(0.97, 0.03, r'$S(T) = \int B(\lambda,T)\cdot R(\lambda)\,\mathrm{d}\lambda$',
         transform=ax2.transAxes, va='bottom', ha='right', fontsize=9)

# — Plot 3: Sensitivity dS/dT ——————————————————————————————————————————————
for det in DETECTORS:
    ax3.plot(T_range, det["dSdT_norm"], color=det["color"],
             linewidth=2, label=det["label"])
ax3.set_xlabel('Temperature [K]')
ax3.set_ylabel('dS/dT [norm.]')

ax3.legend(loc='lower right')
ax3.set_xlim(T_plot[0], T_plot[-1])
ax3.set_xticks(T_plot)


fig.subplots_adjust(left=0.12, right=0.82, top=0.93, bottom=0.08)
plt.savefig("photodiode_temp_sensitivity.png", dpi=600, bbox_inches='tight')
plt.show()
