import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

font_size = 9
dpi = 300
marker_size = 6        # legend marker diameter (points)
marker_edge_width = 0.5

entries = [
    ('unstable keyhole',               'o', '#fde725'),
    ('quasi-stable keyhole',           '^', '#5ec962'),
    ('quasi-stable vapour depression', 'D', '#21918c'),
    ('keyhole flickering',             's', '#3b528b'),
    ('conduction',                     'v', '#440154'),
]

layouts = [
    ('1col', 1),
    ('1row', len(entries)),
    ('2col', 2),
    ('3col', 3),
]

variants = [
    ('colour',          lambda c: c,   True),
    ('white',           lambda c: 'w', True),
    ('colour_noborder', lambda c: c,   False),
    ('white_noborder',  lambda c: 'w', False),
]

out_dir = Path(__file__).parent / Path(__file__).stem
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': font_size})

for layout_name, ncol in layouts:
    for variant_name, get_fc, frameon in variants:
        handles = [
            mpl.lines.Line2D([], [],
                             marker=m,
                             color='none',
                             markerfacecolor=get_fc(c),
                             markeredgecolor='k',
                             markeredgewidth=marker_edge_width,
                             markersize=marker_size,
                             label=label)
            for label, m, c in entries
        ]

        fig = plt.figure()
        legend = fig.legend(handles=handles,
                            ncol=ncol,
                            fontsize=font_size,
                            fancybox=False,
                            frameon=frameon,
                            framealpha=1,
                            edgecolor='k',
                            loc='center')
        if frameon:
            legend.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])

        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox = bbox.expanded(1.05, 1.15)

        fname = f'regime_legend_{layout_name}_{variant_name}'
        fig.savefig(out_dir / f'{fname}.svg', bbox_inches=bbox)
        fig.savefig(out_dir / f'{fname}.png', dpi=dpi, bbox_inches=bbox)
        plt.close(fig)
        print(f'Saved: {fname}.[pdf/png]')
