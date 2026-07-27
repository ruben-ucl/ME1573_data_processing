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

# ── Options ──────────────────────────────────────────────────────────────────
USE_FILTERS     = False    # set False to integrate over the full responsivity band
HIDE_FILTERS    = True    # hide QUAD transmission curve (subplot 2) and post-filter responsivity panel (subplot 3)
USE_EMISSIVITY  = False     # apply temperature-dependent graybody emissivity
EMISSIVITY_REF  = 0.3      # ε at T_plot[0]
EMISSIVITY_GRAD = 0.000025 # dε/dT [K⁻¹]
FILTER_CSV      = "sim/photodiode_sensitivity/QUAD_transmission.csv"  # optical filter; set None to disable

# ── 1. Detector definitions ───────────────────────────────────────────────────
DETECTORS = [
    {"csv": "sim/photodiode_sensitivity/Si.csv",     "label": "PD1", "color": "#f48849", "filter_um": (0.700, 1.050), "gain": 3.3E6},
    {"csv": "sim/photodiode_sensitivity/InGaAs.csv", "label": "PD2", "color": "#5302a3", "filter_um": (1.090, 1.700), "gain": 4.4E6},
]

# ── 2. Load and process responsivity data ────────────────────────────────────
def load_responsivity(csv_file):
    df = pd.read_csv(csv_file, header=None, names=["wavelength", "responsivity"])
    df = df.sort_values("wavelength").dropna()
    wl_um = df["wavelength"].values
    resp = medfilt(df["responsivity"].values, [5])
    return wl_um, resp

def load_filter(csv_file):
    """Load optical filter CSV: wavelength in nm (converted to µm), transmission in % (converted to fraction)."""
    df = pd.read_csv(csv_file, header=0, names=["wavelength_nm", "transmission"],
                     na_values=["#NUM!"])
    df = df.dropna().sort_values("wavelength_nm")
    wl_um = df["wavelength_nm"].values / 1000.0
    trans = np.clip(df["transmission"].values / 100.0, 0.0, 1.0)
    return wl_um, trans

# ── 3. Planck spectral radiance ──────────────────────────────────────────────
def planck(lam_m, T):
    """Spectral radiance B(λ, T) in W·sr⁻¹·m⁻³"""
    return (2*h*c**2 / lam_m**5) / (np.exp(h*c / (lam_m*kB*T)) - 1)

# ── 4. Compute signal S(T) and sensitivity dS/dT for each detector ───────────
T_plot  = [1000, 1500, 2000, 2500, 3000, 3500]
# T_plot  = [1000, 2000, 3000, 4000, 5000]
T_range = np.linspace(T_plot[0], T_plot[-1], 500)

filter_wl_um = filter_trans = filter_interp = None
if FILTER_CSV:
    filter_wl_um, filter_trans = load_filter(FILTER_CSV)
    filter_interp = interp1d(filter_wl_um, filter_trans, kind='linear',
                             bounds_error=False, fill_value=0.0)

for det in DETECTORS:
    wl_um, resp_raw = load_responsivity(det["csv"])
    lam_um = np.linspace(wl_um.min(), wl_um.max(), 2000)
    lam_m  = lam_um * 1e-6

    R = np.clip(interp1d(wl_um, resp_raw, kind='linear',
                         bounds_error=False, fill_value=0.0)(lam_um), 0, None)

    if USE_FILTERS:
        filt_min, filt_max = det["filter_um"]
        R_filt = R * ((lam_um >= filt_min) & (lam_um <= filt_max))
    else:
        R_filt = R

    if filter_interp is not None:
        R_filt = R_filt * filter_interp(lam_um)

    def emissivity(T):
        if not USE_EMISSIVITY:
            return 1.0
        return np.clip(EMISSIVITY_REF + EMISSIVITY_GRAD * (T - T_plot[0]), 0.0, 1.0)

    S = np.array([simpson(emissivity(T) * planck(lam_m, T) * R_filt, x=lam_m) for T in T_range])
    if det["gain"] is not None:
        S = S * det["gain"]

    det.update({"lam_um": lam_um, "lam_m": lam_m,
                "lam_min": wl_um.min(), "lam_max": wl_um.max(),
                "R": R, "R_filt": R_filt, "S": S})

# ── 5. Compute dS/dT ─────────────────────────────────────────────────────────
for det in DETECTORS:
    det["dSdT"] = np.gradient(det["S"], T_range)

# ── 7. Wavelength grid spanning both detectors ───────────────────────────────
lam_global_um = np.linspace(min(d["lam_min"] for d in DETECTORS),
                             max(d["lam_max"] for d in DETECTORS), 2000)
lam_global_m  = lam_global_um * 1e-6

# ── 8. Plots ─────────────────────────────────────────────────────────────────
FS_SUPTITLE = 11   # figure suptitle
FS_LABEL    = 9   # axis labels
FS_TICK     =  8   # tick labels (x and y)
FS_LEGEND   =  8   # legend entries
FS_ANNOT    =  8   # in-axes annotation text (formula, detector labels)

plt.rcParams.update({
    'font.size':        FS_LABEL,
    'axes.labelsize':   FS_LABEL,
    'axes.titlesize':   FS_LABEL,
    'xtick.labelsize':  FS_TICK,
    'ytick.labelsize':  FS_TICK,
    'legend.fontsize':  FS_LEGEND,
})

# Reference per-row heights (from the original 3-top-row, 7.3 in layout) are held fixed;
# figure height and group spacing are then derived so every panel keeps the same size
# regardless of how many top rows are shown or how much whitespace separates the groups.
TOP_MARGIN, BOTTOM_MARGIN = 0.07, 0.08
TOP_HSPACE, BOT_HSPACE    = 0.08, 0.18
GROUP_HSPACE = 0.28   # whitespace between the top and bottom groups (was 0.175)

_ref_axes_area = 7.3 * (1 - TOP_MARGIN - BOTTOM_MARGIN)
_ref_mean      = (2.05 + 1) / 2
_ref_top_h     = _ref_axes_area * 2.05 / (2.05 + 1 + 0.175 * _ref_mean)
_ref_bot_h     = _ref_axes_area * 1    / (2.05 + 1 + 0.175 * _ref_mean)
TOP_ROW_H = _ref_top_h / (3 + TOP_HSPACE * 2)
BOT_ROW_H = _ref_bot_h / (2 + BOT_HSPACE)

n_top_rows  = 2 if HIDE_FILTERS else 3
top_group_h = TOP_ROW_H * (n_top_rows + TOP_HSPACE * (n_top_rows - 1))
bot_group_h = BOT_ROW_H * (2 + BOT_HSPACE)
group_mean  = (top_group_h + bot_group_h) / 2
axes_area   = top_group_h + bot_group_h + GROUP_HSPACE * group_mean
fig_height  = axes_area / (1 - TOP_MARGIN - BOTTOM_MARGIN)

fig = plt.figure(figsize=(6.30, fig_height))   # A4 width minus 25 mm margins, × height
fig.suptitle("Photodiode Blackbody Temperature Sensitivity", fontsize=FS_SUPTITLE)

gs     = GridSpec(2, 1, figure=fig, hspace=GROUP_HSPACE, height_ratios=[top_group_h, bot_group_h])
gs_top = gs[0].subgridspec(n_top_rows, 1, hspace=TOP_HSPACE, height_ratios=[1] * n_top_rows)
gs_bot = gs[1].subgridspec(2, 1, hspace=BOT_HSPACE)
ax1_planck = fig.add_subplot(gs_top[0])
ax1_resp   = fig.add_subplot(gs_top[1], sharex=ax1_planck)
ax_filt    = None if HIDE_FILTERS else fig.add_subplot(gs_top[2], sharex=ax1_planck)
plt.setp(ax1_planck.get_xticklabels(), visible=False)
plt.setp(ax1_resp.get_xticklabels(), visible=HIDE_FILTERS)
ax2 = fig.add_subplot(gs_bot[0])
ax3 = fig.add_subplot(gs_bot[1], sharex=ax2)
plt.setp(ax2.get_xticklabels(), visible=False)

# — Plot 1: Planck spectral radiance ————————————————————————————————————————
cmap = plt.colormaps['plasma'].resampled(len(T_plot))
for i, T in enumerate(T_plot):
    B = planck(lam_global_m, T)
    ax1_planck.plot(lam_global_um, B * 1e-6, color=cmap(i), alpha=0.9,
                    linewidth=1.4, label=f'{T} K')
ax1_planck.set_ylabel(r'$B(\lambda, T)$' + '\n[W·sr⁻¹·m⁻²·µm⁻¹]')
ax1_planck.set_yscale('log')
ax1_planck.set_xlim(lam_global_um.min(), lam_global_um.max())
ax1_planck.legend(loc='lower right', ncol=2, framealpha=0.7)

# — Plot 2: Raw responsivity ——————————————————————————————————————————————————
if filter_wl_um is not None and not HIDE_FILTERS:
    ax1_resp.fill_between(filter_wl_um, filter_trans, alpha=0.10, color='grey')
    ax1_resp.plot(filter_wl_um, filter_trans, color='grey', linewidth=1.0,
                  linestyle='--', alpha=0.8, label='QUAD transmission')

for det in DETECTORS:
    if USE_FILTERS:
        filt_min, filt_max = det["filter_um"]
        mask = (det["lam_um"] >= filt_min) & (det["lam_um"] <= filt_max)
    else:
        mask = np.ones(len(det["lam_um"]), dtype=bool)
    ax1_resp.fill_between(det["lam_um"], det["R"], where=mask, alpha=0.2, color=det["color"])
    ax1_resp.plot(det["lam_um"], det["R"], color=det["color"], linewidth=1.5, alpha=0.8)
    x_mid = (det["lam_um"].min() + det["lam_um"].max()) / 2
    ax1_resp.text(x_mid, det["R"].max() * 0.45, det["label"],
                  color=det["color"], fontweight='bold', fontsize=FS_ANNOT,
                  ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=3))
ax1_resp.set_ylabel(r'Responsivity, $R(\lambda)$' + '\n[A/W]')
ax1_resp.set_xlim(lam_global_um.min(), lam_global_um.max())
ax1_resp.set_ylim(bottom=0)
if HIDE_FILTERS:
    ax1_resp.set_xlabel('Wavelength (µm)')
else:
    h_resp, l_resp = ax1_resp.get_legend_handles_labels()
    ax1_resp.legend(h_resp, l_resp, loc='upper left', framealpha=0.7)

# — Plot 1b: Post-filtering effective responsivity ————————————————————————————
if ax_filt is not None:
    for det in DETECTORS:
        ax_filt.fill_between(det["lam_um"], det["R_filt"], alpha=0.25, color=det["color"])
        ax_filt.plot(det["lam_um"], det["R_filt"], color=det["color"], linewidth=1.5, label=det["label"], alpha=0.8)
        x_mid = (det["lam_um"].min() + det["lam_um"].max()) / 2
        r_max = det["R_filt"].max()
        if r_max > 0:
            ax_filt.text(x_mid, r_max * 0.45, det["label"],
                         color=det["color"], fontweight='bold', fontsize=FS_ANNOT,
                         ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=3))
    ax_filt.set_xlabel('Wavelength (µm)')
    ax_filt.set_ylabel(r'Filtered $R(\lambda)$' + '\n[A/W]')
    ax_filt.set_xlim(lam_global_um.min(), lam_global_um.max())
    ax_filt.set_ylim(bottom=0)

# — Plot 2: Integrated signal S(T) ————————————————————————————————————————————
has_gain = any(d["gain"] is not None for d in DETECTORS)
s_unit   = r'V·sr⁻¹·m⁻²'   if has_gain else r'A·sr⁻¹·m⁻²'
dsdt_unit = r'V·sr⁻¹·m⁻²·K⁻¹' if has_gain else r'A·sr⁻¹·m⁻²·K⁻¹'

for det in DETECTORS:
    ax2.plot(T_range, det["S"], color=det["color"], linewidth=2, label=det["label"])
ax2.set_ylabel(r'$S(T)$' + f'\n[{s_unit}]')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax2.set_xlim(T_plot[0], T_plot[-1])
ax2.set_xticks(T_plot)
formula = (r'$S(T) = G\int B(\lambda,T)\cdot R_{filt}(\lambda)\,\mathrm{d}\lambda$'
           if has_gain else
           r'$S(T) = \int B(\lambda,T)\cdot R{_filt}(\lambda)\,\mathrm{d}\lambda$')
ax2.text(0.09, 0.93, formula, transform=ax2.transAxes, va='top', ha='left', fontsize=FS_ANNOT)

# — Plot 3: Sensitivity dS/dT ——————————————————————————————————————————————
for det in DETECTORS:
    ax3.plot(T_range, det["dSdT"], color=det["color"],
             linewidth=2, label=det["label"])
ax3.set_xlabel('Temperature [K]')
ax3.set_ylabel(r'$\mathrm{d}S/\mathrm{d}T$' + f'\n[{dsdt_unit}]')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

ax3.legend(loc='upper left', bbox_to_anchor=(0.0, 0.78))
ax3.set_xlim(T_plot[0], T_plot[-1])
ax3.set_xticks(T_plot)


fig.subplots_adjust(left=0.12, right=0.82, top=1 - TOP_MARGIN, bottom=BOTTOM_MARGIN)
suffix = "_hide_filters" if HIDE_FILTERS else ""
plt.savefig(f"sim/photodiode_sensitivity/photodiode_temp_sensitivity{suffix}.png", dpi=600, bbox_inches='tight')
plt.show()
