"""
Rosenthal moving point-source temperature field for LPBF.

Sweeps scan speed from 0 to 2 m/s (0.1 m/s steps). For each speed:
  - computes melt pool length, width and depth from the liquidus isotherm
  - saves a top-down (surface, z=0) temperature field figure
  - appends a row to melt_pool_dimensions.csv

Physics: quasi-steady Rosenthal point-source solution on a semi-infinite solid,

    T(x,y,z) = T0 + (eta*P) / (2*pi*k*R) * exp(-v*(x+R)/(2*alpha))
    R = sqrt(x^2 + y^2 + z^2)

x, y, z are measured in the frame moving with the source: x > 0 is ahead of
the source (direction of travel), x < 0 is in its wake, z is depth into the
plate. alpha = k / (rho*Cp) is the thermal diffusivity.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Material properties / process parameters (AlSi10Mg) - edit as needed
# ---------------------------------------------------------------------------
K_THERMAL = 120.0       # thermal conductivity, W/(m.K)
RHO = 2670.0            # density, kg/m^3
CP = 900.0              # specific heat capacity, J/(kg.K)
T_LIQUIDUS = 868.0      # liquidus temperature, K (~595 C), defines melt pool boundary
ABSORPTIVITY = 0.18     # fraction of incident laser power absorbed

LASER_POWER = 250.0     # W
T0 = 293.0              # baseplate / ambient temperature, K

ALPHA = K_THERMAL / (RHO * CP)  # thermal diffusivity, m^2/s

# ---------------------------------------------------------------------------
# Sweep / grid / output settings
# ---------------------------------------------------------------------------
SCAN_SPEEDS = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 1)  # m/s

DOMAIN_X = 0.002        # m, plot width (scan direction)
DOMAIN_Y = 0.002        # m, plot height (transverse direction)
LASER_STANDOFF = 0.0005  # m from the right-hand edge (source at x=1.5 mm), laser scans left -> right
N_GRID = 400            # pixels per axis for the top-down plot

DEPTH_RANGE = 0.002      # m, search range for melt depth below the surface
N_DEPTH = 400

T_COLORBAR_MAX = T_LIQUIDUS  # K, fixed colourbar ceiling (values above are clipped, shown as an "extend" arrow)
CONTOUR_STEP = 50.0          # K, spacing between isotherm lines
EPS = 1e-6                   # m, regularisation to avoid the point-source singularity

OUTPUT_DIR = Path(__file__).parent / "rosenthal_output"
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "melt_pool_dimensions.csv"


def rosenthal_temperature(x, y, z, v):
    """Quasi-steady Rosenthal point-source temperature field."""
    Q = ABSORPTIVITY * LASER_POWER
    R = np.sqrt(x**2 + y**2 + z**2 + EPS**2)
    return T0 + Q / (2 * np.pi * K_THERMAL * R) * np.exp(-v * (x + R) / (2 * ALPHA))


def melt_pool_dimensions(v, x_lab, y_lab, T_surface, x_source):
    """Length/width from the surface field, depth from a centerline z-sweep.

    Length is split into the portion ahead of the laser spot (x_lab > x_source)
    and the trailing portion behind it (x_lab < x_source); the two sum to length.
    """
    molten = T_surface >= T_LIQUIDUS
    edge_touch = False

    if not molten.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, edge_touch

    cols_with_melt = np.where(molten.any(axis=0))[0]
    length = x_lab[cols_with_melt[-1]] - x_lab[cols_with_melt[0]]
    length_ahead = max(x_lab[cols_with_melt[-1]] - x_source, 0.0)
    length_behind = max(x_source - x_lab[cols_with_melt[0]], 0.0)
    if cols_with_melt[0] == 0 or cols_with_melt[-1] == len(x_lab) - 1:
        edge_touch = True

    width = 0.0
    for col in cols_with_melt:
        rows = np.where(molten[:, col])[0]
        if rows[0] == 0 or rows[-1] == len(y_lab) - 1:
            edge_touch = True
        w = y_lab[rows[-1]] - y_lab[rows[0]]
        width = max(width, w)

    z = np.linspace(0, DEPTH_RANGE, N_DEPTH)
    T_centerline = rosenthal_temperature(0.0, 0.0, z, v)
    molten_z = T_centerline >= T_LIQUIDUS
    if molten_z.any():
        depth = z[molten_z][-1]
        if molten_z[-1]:
            edge_touch = True
    else:
        depth = 0.0

    return length, length_ahead, length_behind, width, depth, edge_touch


def plot_top_down(v, T_surface, x_lab, y_lab, x_source, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [x_lab.min() * 1e3, x_lab.max() * 1e3, y_lab.min() * 1e3, y_lab.max() * 1e3]
    im = ax.imshow(T_surface, origin="lower", extent=extent, cmap="inferno",
                    vmin=T0, vmax=T_COLORBAR_MAX, aspect="equal")

    contour_levels = np.arange(T0, T_LIQUIDUS, CONTOUR_STEP)
    ax.contour(x_lab * 1e3, y_lab * 1e3, T_surface, levels=contour_levels,
               colors="white", linewidths=0.6, alpha=0.7)
    ax.contour(x_lab * 1e3, y_lab * 1e3, T_surface, levels=[T_LIQUIDUS],
               colors="black", linewidths=1.0)

    laser_x_mm = x_source * 1e3
    ax.plot(laser_x_mm, 0, "o", color="black", markersize=2)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Top-down melt pool, v = {v:.1f} m/s")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("Temperature (K)")

    fig.canvas.draw()
    tick_length_pts = mpl.rcParams["ytick.major.size"]
    cax_width_pts = cax.get_window_extent().width * 72.0 / fig.dpi
    tick_frac = tick_length_pts / cax_width_pts
    cbar.ax.plot([1.0, 1.0 + tick_frac], [T_LIQUIDUS, T_LIQUIDUS],
                 transform=cbar.ax.get_yaxis_transform(), color="black",
                 linewidth=1.0, clip_on=False)
    cbar.ax.text(1.0 + tick_frac + 0.2, T_LIQUIDUS, rf"$T_{{liq}}$ = {T_LIQUIDUS:.0f} K",
                 transform=cbar.ax.get_yaxis_transform(), va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    x_lab = np.linspace(0, DOMAIN_X, N_GRID)
    y_lab = np.linspace(-DOMAIN_Y / 2, DOMAIN_Y / 2, N_GRID)
    x_source = DOMAIN_X - LASER_STANDOFF
    Xg, Yg = np.meshgrid(x_lab - x_source, y_lab)

    rows = []
    for v in SCAN_SPEEDS:
        T_surface = rosenthal_temperature(Xg, Yg, 0.0, v)
        length, length_ahead, length_behind, width, depth, edge_touch = melt_pool_dimensions(
            v, x_lab, y_lab, T_surface, x_source)

        if edge_touch:
            print(f"[warning] v={v:.1f} m/s: melt pool touches domain edge - "
                  f"dimensions may be underestimated")

        rows.append({
            "scan_speed_m_s": v,
            "length_mm": length * 1e3,
            "length_ahead_mm": length_ahead * 1e3,
            "length_behind_mm": length_behind * 1e3,
            "width_mm": width * 1e3,
            "depth_mm": depth * 1e3,
        })

        out_path = OUTPUT_DIR / f"melt_pool_v{v:.1f}ms.png"
        plot_top_down(v, T_surface, x_lab, y_lab, x_source, out_path)
        print(f"v={v:.1f} m/s -> length={length*1e3:.3f} mm "
              f"(ahead={length_ahead*1e3:.3f} mm, behind={length_behind*1e3:.3f} mm), "
              f"width={width*1e3:.3f} mm, depth={depth*1e3:.3f} mm")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scan_speed_m_s", "length_mm", "length_ahead_mm", "length_behind_mm", "width_mm", "depth_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
