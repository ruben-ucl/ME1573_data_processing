from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from itertools import combinations

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
# Solidified bead (above the substrate surface, z >= 0) - as in melt_bead_model.py
L_bead, W_bead, H_bead = 700, 250, 60
n_bead = 60

# Melt pool (below the substrate surface, z <= 0) - as in plot_mp.py. Reuses
# the bead's own L and W so the two shapes' z=0 profiles share one scale;
# d_pool is the only pool-specific parameter (its keyhole depth).
L_pool, W_pool, d_pool = L_bead, W_bead, 50
n_pool = 80

def scale(y, W):
    """Parabolic taper factor: 1 at y=0, 0 at y=+-W/2."""
    return 1 - 4 * y**2 / W**2

# ------------------------------------------------------------------
# Pool: built in its native frame (nose at +d_pool, tail tip at
# d_pool-L_pool, both relative to the deep point at 0), then mirrored
# through the origin (x -> -x) so it points the same way the bead's
# tip does. Reusing L_pool=L_bead and W_pool=W_bead means the pool
# tail's reach (L_pool-d_pool) equals the bead's reach R below, so a
# pure mirror - with no added shift - lands the tail tip exactly on
# the bead's tip once the bead is rotated into place.
# ------------------------------------------------------------------
def generate_nose(a, b, c, resolution):
    """Front quarter-ellipsoid: x in [0, a], z in [-c, 0]."""
    uu = np.linspace(-np.pi/2, np.pi/2, resolution)
    vv = np.linspace(np.pi/2, np.pi, resolution)
    x = a * np.outer(np.cos(uu), np.sin(vv))
    y = b * np.outer(np.sin(uu), np.sin(vv))
    z = c * np.outer(np.ones_like(uu), np.cos(vv))
    return x, y, z

def generate_tail(a, b, c, resolution):
    """Trailing paraboloid tail: x in [a, 0], z in [c, 0], clipped where it
    would overlap the nose (x >= 0)."""
    c1 = np.sqrt(-a / b**2)
    c2 = np.sqrt(-a / c**2)
    y = np.linspace(-b, b, resolution)
    z = np.linspace(c, 0, resolution)
    y, z = np.meshgrid(y, z)
    x = (c1 * y)**2 + (c2 * z)**2 + a
    x = np.where(x >= 10, np.nan, np.where(x >= 0, 0, x))
    return x, y, z

R = L_bead - d_pool   # bead reach: same "L - d" convention as the pool's tail reach

Xn, Yn, Zn = generate_nose(d_pool, W_pool/2, d_pool, n_pool)
Xn = -Xn

Xt, Yt, Zt = generate_tail(d_pool - L_pool, W_pool/2, -d_pool, n_pool)
Xt = -Xt

# ------------------------------------------------------------------
# Bead: built in its native frame (back at 0, tip at R, as in
# melt_bead_model.py), then rotated 180 degrees about its tip (R, 0)
# so it tapers away from the pool instead of into it. A 180 degree
# rotation about that point maps x -> 2R - x (y is unaffected, since
# the cross-section is already symmetric about y=0).
# ------------------------------------------------------------------
y_vals = np.linspace(-W_bead/2, W_bead/2, n_bead)
u_vals = np.linspace(0, 1, n_bead)
Y, U = np.meshgrid(y_vals, u_vals)
S = scale(Y, W_bead)

X2, Z2, Y2 = 2*R - U * R * S, np.zeros_like(Y), Y   # z=0 base face
X3, Z3, Y3 = 2*R - U * R * S, H_bead * S, Y          # curved top
X4, Z4, Y4 = 2*R - R * S, U * H_bead * S, Y          # curved tip (tapers to R)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111, projection='3d')

surf_kwargs = dict(rstride=2, cstride=2, linewidth=0, antialiased=True, shade=True)

ax.plot_surface(X2, Y2, Z2, color='tab:blue',   alpha=0.85, **surf_kwargs)  # bead base
ax.plot_surface(X3, Y3, Z3, color='tab:green',  alpha=0.85, **surf_kwargs)  # bead top
ax.plot_surface(X4, Y4, Z4, color='tab:red',    alpha=0.85, **surf_kwargs)  # bead front (tapered tip)
ax.plot_surface(Xn, Yn, Zn, color='tab:purple', alpha=0.9,  **surf_kwargs)  # melt pool nose
ax.plot_surface(Xt, Yt, Zt, color='tab:purple', alpha=0.9,  **surf_kwargs)  # melt pool tail

# ------------------------------------------------------------------
# Transparent grey bounding box around the full combined extents
# ------------------------------------------------------------------
box_x = [-d_pool, 2*R]
box_y = [-W_bead/2, W_bead/2]
box_z = [-d_pool, H_bead]

corners = np.array([[x, y, z] for x in box_x for y in box_y for z in box_z])
for c1, c2 in combinations(corners, 2):
    diffs = np.abs(c1 - c2)
    if np.count_nonzero(diffs) == 1:  # share two coordinates -> an edge
        ax.plot(*zip(c1, c2), color='grey', alpha=0.3, linewidth=1)

# ------------------------------------------------------------------
# Labels, limits, view
# ------------------------------------------------------------------
LABEL_SIZE = 16
TICK_SIZE = 13

ax.set_xlabel('x', fontsize=LABEL_SIZE, fontweight='bold', labelpad=45)
ax.set_ylabel('y', fontsize=LABEL_SIZE, fontweight='bold', labelpad=14)
ax.set_zlabel('z', fontsize=LABEL_SIZE, fontweight='bold', labelpad=6)
ax.set_xlim(*box_x)
ax.set_ylim(*box_y)
ax.set_zlim(*box_z)
ax.set_box_aspect((box_x[1] - box_x[0], box_y[1] - box_y[0], box_z[1] - box_z[0]))
ax.set_title('Solidified bead and melt pool volume model')

# Isometric view from the x-y minima corner
ax.view_init(elev=22, azim=-135)

# Fewer, larger tick labels
ax.set_xticks(np.linspace(*box_x, 5))
ax.set_yticks(np.linspace(*box_y, 3))
ax.set_zticks(np.linspace(*box_z, 3))
ax.tick_params(axis='x', which='major', labelsize=TICK_SIZE, pad=10)
ax.tick_params(axis='y', which='major', labelsize=TICK_SIZE, pad=4)
ax.tick_params(axis='z', which='major', labelsize=TICK_SIZE, pad=4)

# Bolder mesh/grid lines on the 3D panes
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"].update(color=(0.4, 0.4, 0.4, 1.0), linewidth=1.2)
    axis.pane.set_edgecolor((0.3, 0.3, 0.3, 1.0))
    axis.pane.set_linewidth(1.2)

plt.tight_layout()

output_dir = Path(__file__).parent / 'melt_pool_bead_model_output'
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / 'melt_pool_bead_model_isometric.png', dpi=300)

plt.show()
