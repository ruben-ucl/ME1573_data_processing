"""
Bubble dynamics in a viscous fluid: buoyancy force, terminal velocity, and the
dimensionless numbers that characterize the rise (Reynolds, Eoetvoes, Weber)
— all plotted as functions of bubble diameter.

Assumes a spherical bubble rising (or falling) under gravity in a quiescent
Newtonian fluid. Buoyancy uses the net weight of displaced fluid vs bubble
contents (Archimedes); drag uses the Schiller-Naumann correlation, which
reduces to Stokes' law as Re -> 0, so results vary smoothly across the whole
diameter range. At each diameter, the terminal velocity is found first
(balancing buoyancy against drag); Re and We are then evaluated at that
terminal velocity, while Eo depends only on diameter and fluid properties.

Edit the constants under "Inputs" below and run:
    python bubble_dynamics.py
A figure sweeping bubble diameter is saved to OUTPUT_FIGURE.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Inputs — edit these to change the scenario
# ---------------------------------------------------------------------------
RHO_FLUID = 4087.0     # fluid density [kg/m^3]
RHO_BUBBLE = 1.2       # bubble (gas) density [kg/m^3]
MU = 4e-3              # fluid dynamic viscosity [Pa.s]
SIGMA = 1.49          # fluid surface tension [N/m] (water-air default)
G = 9.81               # gravitational acceleration [m/s^2]

D_MIN = 1e-5           # smallest bubble diameter swept [m] (10 um)
D_MAX = 1e-3           # largest bubble diameter swept [m] (5 mm)
N_POINTS = 300         # number of diameters sampled (log-spaced)
REFERENCE_DIAMETER = 1e-4  # diameter to highlight on the plots [m] (100 um)

OUTPUT_FIGURE = "bubble_dynamics.png"

# Palette (validated categorical slot 1, plus chart ink/chrome roles)
COL_LINE = "#2a78d6"       # series-1 blue, reused per-panel (each panel is a
                           # single series, so no cross-panel identity clash)
COL_PRIMARY = "#0b0b0b"    # primary ink
COL_SECONDARY = "#52514e"  # secondary ink
COL_MUTED = "#898781"      # muted ink (axis/labels)
COL_GRID = "#e1e0d9"       # hairline gridline
COL_SURFACE = "#fcfcfb"    # chart surface


def buoyancy_force(radius, rho_fluid, rho_bubble, g=9.81):
    """Net buoyant force on a sphere (N). Positive = upward (bubble rises)."""
    volume = (4.0 / 3.0) * math.pi * radius**3
    return (rho_fluid - rho_bubble) * g * volume


def drag_force(radius, velocity, mu, rho_fluid):
    """Viscous drag force (N) via the Schiller-Naumann correlation."""
    re = reynolds_number(radius, velocity, rho_fluid, mu)
    cd = 0.0 if re == 0 else (24.0 / re) * (1.0 + 0.15 * re**0.687)
    area = math.pi * radius**2
    return 0.5 * cd * rho_fluid * velocity**2 * area


def reynolds_number(radius, velocity, rho_fluid, mu):
    """Re = inertial / viscous forces, based on bubble diameter and terminal velocity."""
    diameter = 2.0 * radius
    return rho_fluid * abs(velocity) * diameter / mu


def eotvos_number(radius, rho_fluid, rho_bubble, sigma, g=9.81):
    """Eo (aka Bond number) = buoyancy / surface tension forces. Velocity-independent."""
    diameter = 2.0 * radius
    return g * abs(rho_fluid - rho_bubble) * diameter**2 / sigma


def weber_number(radius, velocity, rho_fluid, sigma):
    """We = inertial / surface tension forces, evaluated at terminal velocity."""
    diameter = 2.0 * radius
    return rho_fluid * velocity**2 * diameter / sigma


def terminal_velocity_stokes(radius, rho_fluid, rho_bubble, mu, g=9.81):
    """
    Closed-form terminal velocity (m/s) from balancing buoyancy against Stokes
    drag: v_t = 2 * r^2 * g * (rho_fluid - rho_bubble) / (9 * mu). Used only as
    the initial guess for the iterative Schiller-Naumann solve below.
    """
    return (2.0 * radius**2 * g * (rho_fluid - rho_bubble)) / (9.0 * mu)


def terminal_velocity(radius, rho_fluid, rho_bubble, mu, g=9.81, tol=1e-9, max_iter=200):
    """
    Solve for terminal velocity by balancing buoyancy against Schiller-Naumann
    drag via fixed-point iteration.
    """
    f_b = buoyancy_force(radius, rho_fluid, rho_bubble, g)
    v = terminal_velocity_stokes(radius, rho_fluid, rho_bubble, mu, g)  # initial guess

    for _ in range(max_iter):
        f_d = drag_force(radius, v, mu, rho_fluid)
        if f_d <= 0:
            break
        # F_b = F_d at terminal velocity; F_d ~ v^2, so rescale v accordingly
        v_new = v * math.sqrt(abs(f_b) / f_d)
        if abs(v_new - v) < tol:
            v = v_new
            break
        v = v_new

    return v


def analyze_bubble(radius, rho_fluid, rho_bubble, mu, sigma, g=9.81):
    """Run the full set of calculations for a single bubble radius."""
    f_b = buoyancy_force(radius, rho_fluid, rho_bubble, g)
    v_t = terminal_velocity(radius, rho_fluid, rho_bubble, mu, g)

    return {
        "buoyancy_force_N": f_b,
        "terminal_velocity_m_s": v_t,
        "reynolds_number": reynolds_number(radius, v_t, rho_fluid, mu),
        "eotvos_number": eotvos_number(radius, rho_fluid, rho_bubble, sigma, g),
        "weber_number": weber_number(radius, v_t, rho_fluid, sigma),
    }


def sweep_diameters(diameters, rho_fluid, rho_bubble, mu, sigma, g=9.81):
    """Run analyze_bubble() across an array of diameters; return arrays of results."""
    n = len(diameters)
    out = {
        "diameter_m": diameters,
        "buoyancy_force_N": np.empty(n),
        "terminal_velocity_m_s": np.empty(n),
        "reynolds_number": np.empty(n),
        "eotvos_number": np.empty(n),
        "weber_number": np.empty(n),
    }

    for i, d in enumerate(diameters):
        r = analyze_bubble(d / 2.0, rho_fluid, rho_bubble, mu, sigma, g)
        for key in ("buoyancy_force_N", "terminal_velocity_m_s", "reynolds_number",
                    "eotvos_number", "weber_number"):
            out[key][i] = r[key]

    return out


def _style_axis(ax):
    ax.set_facecolor(COL_SURFACE)
    ax.grid(True, which="both", color=COL_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COL_MUTED)
    ax.tick_params(colors=COL_MUTED, which="both")


def make_figure(sweep, reference_diameter, output_path):
    """
    Small multiples, sharing a log diameter axis. Each panel is a single
    series — the y-axis label names the quantity, so no per-panel legend or
    title is needed; one reference-diameter marker and caption apply to all.
    """
    d_mm = sweep["diameter_m"] * 1e3
    ref_mm = reference_diameter * 1e3

    panels = [
        ("buoyancy_force_N", "Buoyancy force [N]"),
        ("terminal_velocity_m_s", "Terminal velocity [m/s]"),
        ("reynolds_number", "Reynolds number [-]"),
        ("eotvos_number", "Eötvös number [-]"),
        ("weber_number", "Weber number [-]"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 14), dpi=150, sharex=True)
    fig.patch.set_facecolor(COL_SURFACE)

    for ax, (key, ylabel) in zip(axes, panels):
        _style_axis(ax)
        ax.loglog(d_mm, sweep[key], color=COL_LINE, linewidth=2, solid_capstyle="round")
        ax.set_ylabel(ylabel, color=COL_SECONDARY)
        ax.axvline(ref_mm, color=COL_MUTED, linestyle="--", linewidth=1)

    # Re = 1 marks the upper bound of Stokes' law validity (creeping flow)
    re_ax = axes[2]
    re_ax.axhline(1.0, color=COL_MUTED, linestyle=":", linewidth=1)
    re_ax.text(d_mm[-1], 1.0, "  Re = 1 (Stokes limit)", color=COL_SECONDARY,
               fontsize=9, va="bottom", ha="right")

    axes[0].annotate(
        f"d = {ref_mm:.3g} mm", xy=(ref_mm, axes[0].get_ylim()[1]), xycoords="data",
        xytext=(4, -4), textcoords="offset points",
        fontsize=9, color=COL_PRIMARY, va="top", ha="left",
    )

    axes[-1].set_xlabel("Bubble diameter [mm]", color=COL_SECONDARY)
    fig.suptitle(
        "Bubble rise dynamics vs. diameter\n"
        f"$\\rho_f$={sweep['rho_fluid']:.4g} kg/m$^3$, "
        f"$\\rho_b$={sweep['rho_bubble']:.4g} kg/m$^3$, "
        f"$\\mu$={sweep['mu']:.4g} Pa·s, $\\sigma$={sweep['sigma']:.4g} N/m",
        color=COL_PRIMARY, fontsize=12, x=0.02, ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    diameters = np.logspace(math.log10(D_MIN), math.log10(D_MAX), N_POINTS)
    sweep = sweep_diameters(diameters, RHO_FLUID, RHO_BUBBLE, MU, SIGMA, G)
    sweep["rho_fluid"], sweep["rho_bubble"], sweep["mu"], sweep["sigma"] = (
        RHO_FLUID, RHO_BUBBLE, MU, SIGMA,
    )

    ref = analyze_bubble(REFERENCE_DIAMETER / 2.0, RHO_FLUID, RHO_BUBBLE, MU, SIGMA, G)

    print(f"Reference diameter:   {REFERENCE_DIAMETER:.4g} m")
    print(f"Fluid density:        {RHO_FLUID:.4g} kg/m^3")
    print(f"Bubble density:       {RHO_BUBBLE:.4g} kg/m^3")
    print(f"Fluid viscosity:      {MU:.4g} Pa.s")
    print(f"Surface tension:      {SIGMA:.4g} N/m")
    print("-" * 40)
    print(f"Buoyancy force:       {ref['buoyancy_force_N']:.4e} N")
    print(f"Terminal velocity:    {ref['terminal_velocity_m_s']:.4e} m/s")
    print(f"Reynolds number:      {ref['reynolds_number']:.4e}")
    print(f"Eotvos number:        {ref['eotvos_number']:.4e}")
    print(f"Weber number:         {ref['weber_number']:.4e}")
    if ref["reynolds_number"] >= 1.0:
        print("Note: Re >= 1 at the reference diameter, so Stokes' law alone "
              "would not be strictly valid there; Schiller-Naumann drag was used.")

    make_figure(sweep, REFERENCE_DIAMETER, OUTPUT_FIGURE)
    print(f"Figure saved to:      {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
