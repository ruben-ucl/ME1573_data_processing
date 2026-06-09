#!/usr/bin/env python3
"""
CWT Image Binary Classifier — Architecture Diagram
Publication-ready, targets A4 landscape.

All tuneable values are in the PARAMETERS block at the top.
"""
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ═══════════════════════════════════════════════════
#  PARAMETERS  — edit here, then rerun
# ═══════════════════════════════════════════════════
N_COLS       = 7      # boxes per row before wrapping
FIG_W        = 11.69  # figure width (A4 landscape, inches)
# FIG_H is computed from content — do not set manually
MAR_L        = 0.55   # left margin (room for section labels)
MAR_R        = 0.35   # right margin
MAR_T        = 0.45   # top margin
MAR_B        = 0.50   # bottom margin (room for legend)
COL_GAP      = 0.16   # horizontal gap between boxes
ROW_GAP      = 0.52   # vertical gap between rows (for wrap arrows)
TITLE_H      = 0.55   # height reserved for title + subtitle
BOX_H_SCALE  = 2/3    # fraction of page-fill height; reduce to shrink boxes
FS_TITLE     = 11     # title font size (pt)
FS_SUB       = 8.5    # subtitle
FS_L1        = 9.5    # layer type (bold)
FS_L2        = 8.5    # first param line
FS_L3        = 7.5    # second param line (e.g. L2)
FS_SEC       = 8.0    # section label (rotated, in margin)
FS_LGND      = 8.5    # legend

# ═══════════════════════════════════════════════════
#  COLOURS  (face, edge, text)
# ═══════════════════════════════════════════════════
COL = {
    'input':   ('#EBEBEB', '#909090', '#222222'),
    'conv':    ('#C9DEF0', '#2D6E99', '#1A3A55'),
    'pool':    ('#1B6090', '#0D4466', '#FFFFFF'),
    'flatten': ('#888888', '#555555', '#FFFFFF'),
    'dense':   ('#FAD4A0', '#A05010', '#5A2800'),
    'dropout': ('#E8D4EC', '#8B4898', '#3D1050'),
    'output':  ('#A8D8B0', '#27753A', '#133D1E'),
}

# ═══════════════════════════════════════════════════
#  LAYER DEFINITIONS  (type, name, param1, param2)
#  v115: conv_dropout=0.20, lr=0.0005
# ═══════════════════════════════════════════════════
LAYERS = [
    # Row 0 — first conv block + per-layer dropout
    ('input',   'Input',      '256 × 100 × C',           ''),
    ('conv',    'Conv2D',     '16 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    ('conv',    'Conv2D',     '16 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    ('conv',    'Conv2D',     '32 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    # Row 1 — MaxPool then second conv block
    ('pool',    'MaxPool2D',  '2 × 2',                   ''),
    ('conv',    'Conv2D',     '32 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    ('conv',    'Conv2D',     '64 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    ('conv',    'Conv2D',     '64 filters · 3×3 · ReLU', 'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.20',                 ''),
    # Row 2 — classification head (centred)
    ('pool',    'MaxPool2D',  '2 × 2',                   ''),
    ('flatten', 'Flatten',    '',                         ''),
    ('dense',   'Dense',      '128 units · ReLU',        'L2 = 0.001'),
    ('dropout', 'Dropout',    'p = 0.50',                 ''),
    ('output',  'Dense',      '1 unit · Sigmoid',        'Binary output'),
]

# ═══════════════════════════════════════════════════
#  LAYOUT COMPUTATION
# ═══════════════════════════════════════════════════
n      = len(LAYERS)
n_rows = math.ceil(n / N_COLS)
use_w  = FIG_W - MAR_L - MAR_R
BOX_W  = (use_w - (N_COLS - 1) * COL_GAP) / N_COLS

# BOX_H: scale down from what would fill A4 (8.27 in) at this row count
avail_h_a4 = 8.27 - MAR_T - TITLE_H - MAR_B
BOX_H_full = (avail_h_a4 - (n_rows - 1) * ROW_GAP) / n_rows
BOX_H      = BOX_H_full * BOX_H_SCALE

# FIG_H fits the content exactly (no forced A4 height)
FIG_H = MAR_T + TITLE_H + n_rows * BOX_H + (n_rows - 1) * ROW_GAP + MAR_B

def box_pos(i):
    row = i // N_COLS
    col = i % N_COLS
    last = n - (n_rows - 1) * N_COLS       # items in last row
    if row == n_rows - 1 and last < N_COLS:
        lw = last * BOX_W + (last - 1) * COL_GAP
        x  = MAR_L + (use_w - lw) / 2 + col * (BOX_W + COL_GAP)
    else:
        x  = MAR_L + col * (BOX_W + COL_GAP)
    y = FIG_H - MAR_T - TITLE_H - row * (BOX_H + ROW_GAP) - BOX_H
    return x, y

P = [box_pos(i) for i in range(n)]

# ═══════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Background bands ─────────────────────────────────
band_pad_x = 0.18
band_pad_y = 0.10

def band(row_start, row_end, fc):
    """Draw a shaded background rectangle spanning a range of rows."""
    y_top = P[row_start * N_COLS][1] + BOX_H + band_pad_y
    # For the last row, row_end * N_COLS may exceed n; clamp to last layer
    last_in_band = min((row_end + 1) * N_COLS - 1, n - 1)
    y_bot = P[last_in_band][1] - band_pad_y
    ax.add_patch(plt.Rectangle(
        (MAR_L - band_pad_x, y_bot),
        use_w + 2 * band_pad_x, y_top - y_bot,
        facecolor=fc, edgecolor='none', zorder=0
    ))

band(0, 1, '#EEF5FB')   # backbone rows
band(2, 2, '#FDF5EC')   # head row

# ── Section labels in left margin (rotated) ───────────
def row_centre_y(r0, r1):
    """Vertical centre between top of row r0 and bottom of row r1."""
    y_top = P[r0 * N_COLS][1] + BOX_H
    last  = min((r1 + 1) * N_COLS - 1, n - 1)
    y_bot = P[last][1]
    return (y_top + y_bot) / 2

sec_kw = dict(ha='center', va='center', fontweight='bold',
              style='italic', rotation=90, zorder=1)
ax.text(MAR_L * 0.38, row_centre_y(0, 1),
        'Convolutional Backbone', fontsize=FS_SEC, color='#2D6E99', **sec_kw)
ax.text(MAR_L * 0.38, row_centre_y(2, 2),
        'Classification Head',   fontsize=FS_SEC, color='#A05010', **sec_kw)

# ── Layer boxes ──────────────────────────────────────
def text_lines(box_x, box_y, name, p1, p2):
    """Draw up to 3 lines of text centred in a box."""
    cx = box_x + BOX_W / 2
    cy = box_y + BOX_H / 2
    has_p2 = bool(p2)
    if p1 and has_p2:
        offsets = (0.22, 0.0, -0.22)
        sizes   = (FS_L1, FS_L2, FS_L3)
        texts   = (name, p1, p2)
    elif p1:
        offsets = (0.13, -0.13)
        sizes   = (FS_L1, FS_L2)
        texts   = (name, p1)
    else:
        offsets = (0.0,)
        sizes   = (FS_L1,)
        texts   = (name,)
    # Scale offsets relative to actual BOX_H
    scale = BOX_H / 1.8
    for off, sz, txt in zip(offsets, sizes, texts):
        bold = (sz == FS_L1)
        ax.text(cx, cy + off * scale, txt,
                ha='center', va='center', fontsize=sz,
                fontweight='bold' if bold else 'normal',
                color=tc, zorder=3)

for i, (ltype, lname, lp1, lp2) in enumerate(LAYERS):
    x, y = P[i]
    fc, ec, tc = COL[ltype]
    ax.add_patch(FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle='round,pad=0.05',
        facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2
    ))
    text_lines(x, y, lname, lp1, lp2)

# ── Arrows ───────────────────────────────────────────
AKW = dict(color='#333333', lw=1.2, mutation_scale=11, zorder=4)

for i in range(n - 1):
    x0, y0 = P[i]
    x1, y1 = P[i + 1]
    same_row = (i // N_COLS == (i + 1) // N_COLS)

    if same_row:
        # Simple horizontal arrow: right edge → left edge
        ax.annotate('', xy=(x1, y1 + BOX_H / 2),
                    xytext=(x0 + BOX_W, y0 + BOX_H / 2),
                    arrowprops=dict(arrowstyle='->', **AKW,
                                    connectionstyle='arc3,rad=0'))
    else:
        # Row-wrap: 3-segment L-shaped path
        #   ┌── down from bottom-centre of last box
        #   ├── left across the gap to first box of next row
        #   └── arrow down into top-centre of first box
        xA = x0 + BOX_W / 2   # centre-x of last box in row R
        yA = y0                # bottom edge of last box (higher Y = above)
        xB = x1 + BOX_W / 2   # centre-x of first box in row R+1
        yB = y1 + BOX_H        # top edge of first box (lower Y = below)
        yM = (yA + yB) / 2    # mid-gap

        ax.plot([xA, xA], [yA, yM], '-', color='#333333', lw=1.2,
                solid_capstyle='round', zorder=4)
        ax.plot([xA, xB], [yM, yM], '-', color='#333333', lw=1.2,
                solid_capstyle='round', zorder=4)
        ax.annotate('', xy=(xB, yB), xytext=(xB, yM),
                    arrowprops=dict(arrowstyle='->', **AKW,
                                    connectionstyle='arc3,rad=0'))

# ── Title block ──────────────────────────────────────
tx = FIG_W / 2
ty = FIG_H - MAR_T * 0.40
ax.text(tx, ty,
        'CWT Image Binary Classifier — CNN Architecture  (v115)',
        ha='center', va='center', fontsize=FS_TITLE, fontweight='bold', color='#111111')
ax.text(tx, ty - 0.27,
        'Adam  ·  lr = 0.0005  ·  Batch = 32  ·  L2 = 0.001  ·  '
        'Conv dropout = 0.20  ·  5-fold CV  ·  Early stopping (patience = 10)',
        ha='center', va='center', fontsize=FS_SUB, color='#444444')

# ── Legend (horizontal, bottom centre) ───────────────
legend_labels = {
    'input':   'Input',
    'conv':    'Conv2D',
    'pool':    'MaxPooling2D',
    'flatten': 'Flatten',
    'dense':   'Dense',
    'dropout': 'Dropout',
    'output':  'Output (sigmoid)',
}
handles = [
    mpatches.Patch(facecolor=COL[k][0], edgecolor=COL[k][1],
                   linewidth=1.0, label=v)
    for k, v in legend_labels.items()
]
legend = ax.legend(handles=handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.005),
                   ncol=len(handles), fontsize=FS_LGND,
                   framealpha=0.0, handlelength=1.2,
                   handleheight=0.8, columnspacing=0.9,
                   handletextpad=0.4)

# ── Save ─────────────────────────────────────────────
base = 'D:/ME1573_data_processing/ml/network_architecture_cwt'
plt.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(base + '.png', dpi=150, bbox_inches='tight')

print(f'Saved: {base}.pdf / .png')
print(f'BOX_W = {BOX_W:.3f} in  |  BOX_H = {BOX_H:.3f} in')
print(f'n_rows = {n_rows}  |  N_COLS = {N_COLS}  |  n_layers = {n}')
