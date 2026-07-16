from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from itertools import combinations

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
L, W, H = 700, 250, 60
n = 60  # grid resolution

def scale(y):
    """Parabolic taper factor: 1 at y=0, 0 at y=+-W/2."""
    return 1 - 4 * y**2 / W**2

y_vals = np.linspace(-W/2, W/2, n)
u_vals = np.linspace(0, 1, n)  # normalised parameter along x or z

Y, U = np.meshgrid(y_vals, u_vals)
S = scale(Y)  # taper factor broadcast over the grid

# ------------------------------------------------------------------
# Four bounding surfaces
# ------------------------------------------------------------------
# 1) x = 0 plane (back face): z runs 0 -> H*scale(y)
X1 = np.zeros_like(Y)
Z1 = U * H * S
Y1 = Y

# 2) z = 0 plane (base face): x runs 0 -> L*scale(y)
X2 = U * L * S
Z2 = np.zeros_like(Y)
Y2 = Y

# 3) z = H(1 - 4y^2/W^2) (curved top): x runs 0 -> L*scale(y)
X3 = U * L * S
Z3 = H * S
Y3 = Y

# 4) x = L(1 - 4y^2/W^2) (curved front): z runs 0 -> H*scale(y)
X4 = L * S
Z4 = U * H * S
Y4 = Y

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

surf_kwargs = dict(rstride=2, cstride=2, linewidth=0, antialiased=True, shade=True)

ax.plot_surface(X1, Y1, Z1, color='tab:grey',    alpha=0.0, **surf_kwargs)   # x=0
ax.plot_surface(X2, Y2, Z2, color='tab:blue',   alpha=0.85, **surf_kwargs)   # z=0
ax.plot_surface(X3, Y3, Z3, color='tab:green',  alpha=0.85, **surf_kwargs)   # top curve
ax.plot_surface(X4, Y4, Z4, color='tab:red', alpha=0.85, **surf_kwargs)   # front curve

# ------------------------------------------------------------------
# Transparent grey bounding box (the extents beyond the volume)
# ------------------------------------------------------------------
box_x = [0, L]
box_y = [-W/2, W/2]
box_z = [0, H]

# Draw box edges
corners = np.array([[x, y, z] for x in box_x for y in box_y for z in box_z])
for c1, c2 in combinations(corners, 2):
    diffs = np.abs(c1 - c2)
    if np.count_nonzero(diffs) == 1:  # they share two coordinates -> an edge
        ax.plot(*zip(c1, c2), color='grey', alpha=0.3, linewidth=1)

# Optional: faint translucent grey faces on the box for extra context
def grey_face(xx, yy, zz):
    ax.plot_surface(xx, yy, zz, color='grey', alpha=0.0, linewidth=0, shade=False)

Xb, Yb = np.meshgrid(box_x, box_y)
grey_face(Xb, Yb, np.full_like(Xb, H))                 # top of box
grey_face(Xb, Yb, np.full_like(Xb, 0))                 # bottom of box (mostly hidden by z=0 surf)
Xb2, Zb2 = np.meshgrid(box_x, box_z)
grey_face(Xb2, np.full_like(Xb2, W/2), Zb2)            # y = +W/2 end
grey_face(Xb2, np.full_like(Xb2, -W/2), Zb2)           # y = -W/2 end

# ------------------------------------------------------------------
# Labels, limits, view
# ------------------------------------------------------------------
LABEL_SIZE = 16
TICK_SIZE = 13

ax.set_xlabel('x', fontsize=LABEL_SIZE, fontweight='bold', labelpad=45)
ax.set_ylabel('y', fontsize=LABEL_SIZE, fontweight='bold', labelpad=14)
ax.set_zlabel('z', fontsize=LABEL_SIZE, fontweight='bold', labelpad=6)
ax.set_xlim(0, L)
ax.set_ylim(-W/2, W/2)
ax.set_zlim(0, H)
ax.set_box_aspect((L, W, H))
ax.set_title('Volume bounded by x=0, z=0, z=H(1-4y²/W²), x=L(1-4y²/W²)')

# Isometric view from the x-y minima corner (x=0, y=-W/2)
ax.view_init(elev=22, azim=-135)

# Fewer, larger tick labels
ax.set_xticks(np.linspace(0, L, 5))
ax.set_yticks(np.linspace(-W/2, W/2, 3))
ax.set_zticks(np.linspace(0, H, 3))
ax.tick_params(axis='x', which='major', labelsize=TICK_SIZE, pad=10)
ax.tick_params(axis='y', which='major', labelsize=TICK_SIZE, pad=4)
ax.tick_params(axis='z', which='major', labelsize=TICK_SIZE, pad=4)

# Bolder mesh/grid lines on the 3D panes
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"].update(color=(0.4, 0.4, 0.4, 1.0), linewidth=1.2)
    axis.pane.set_edgecolor((0.3, 0.3, 0.3, 1.0))
    axis.pane.set_linewidth(1.2)

plt.tight_layout()

output_dir = Path(__file__).parent / 'melt_bead_model_output'
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / 'melt_bead_model_isometric.png', dpi=300)

plt.show()