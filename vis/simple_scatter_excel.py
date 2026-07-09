import os
import sys
from pathlib import Path

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

EXCEL_PATH = Path(r'C:\Users\lbn38569\UCL Dropbox\Rubén Lambert-Garcia\Documents\Thesis\PWM track width samples.xlsx')

X_COL = 'B'            # Excel column letter for x data
Y_COL = 'C'            # Excel column letter for y data
ERR_COL = 'E'          # Excel column letter for y error bars (leave as '' to disable)

X_LABEL = 'Duty cycle'
X_UNIT = None
Y_LABEL = 'Mean track width'
Y_UNIT = 'μm'

FIG_WIDTH  = 3.15   # inches
FIG_HEIGHT = 1.8    # inches

# =============================================================================

OUT_DIR = Path(__file__).parent / Path(__file__).stem
OUT_DIR.mkdir(exist_ok=True)

cols = ''.join(filter(None, [X_COL, Y_COL, ERR_COL])) if ERR_COL else f'{X_COL}:{Y_COL}'
usecols = f'{X_COL}, {Y_COL}, {ERR_COL}' if ERR_COL else f'{X_COL}, {Y_COL}'
df = pd.read_excel(EXCEL_PATH, usecols=usecols, header=0)
df.columns = ['x', 'y'] + (['err'] if ERR_COL else [])
df = df.dropna()

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.errorbar(
    df['x'], df['y'],
    yerr=df['err'] if ERR_COL else None,
    fmt='o',
    markersize=4,
    linewidth=1,
    capsize=3,
    color='k',
)
X_LABEL_full = f'{X_LABEL} [{X_UNIT}]' if X_UNIT != None else X_LABEL
Y_LABEL_full = f'{Y_LABEL}\n[{Y_UNIT}]' if Y_UNIT != None else Y_LABEL

ax.set_xlabel(X_LABEL_full, fontsize=9)
ax.set_ylabel(Y_LABEL_full, fontsize=9)
ax.tick_params(labelsize=8)

for spine in ax.spines.values():
    spine.set_edgecolor('black')

fig.tight_layout()

stem = f'{X_LABEL}_{Y_LABEL}'.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_')
fig.savefig(OUT_DIR / f'{stem}.pdf')
fig.savefig(OUT_DIR / f'{stem}.png', dpi=300)
print(f'Saved to {OUT_DIR}')
plt.show()
