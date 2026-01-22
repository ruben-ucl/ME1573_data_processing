"""
Analyze regime distribution from CWT dataset CSV file.

Reads trackids from a dataset CSV, looks up their melting regimes from the logbook,
and creates a clustered bar chart showing the distribution of regimes by porosity label.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tools import get_logbook, get_regime_marker_dict

def extract_trackid_from_filename(filename):
    """
    Extract trackid from CWT image filename.

    Args:
        filename: String like "0105_01_0.2-1.2ms.png"

    Returns:
        trackid: String like "0105_01"
    """
    # Split by underscore and take first two parts
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None

def abbreviate_regime(regime):
    """Abbreviate regime names for plotting."""
    regime = regime.replace('keyhole', 'KH')
    regime = regime.replace('vapour depression', 'VD')
    return regime

def lighten_color(color, amount=0.5):
    """
    Lighten a color by blending with white.

    Args:
        color: Matplotlib color string (hex)
        amount: Factor to lighten by (0=original, 1=white)

    Returns:
        Hex color string
    """
    try:
        c = mcolors.to_rgb(color)
        white = np.array([1, 1, 1])
        lightened = (1 - amount) * np.array(c) + amount * white
        return mcolors.to_hex(lightened)
    except:
        return color

def main():
    # Define paths
    csv_path = Path(r"F:\AlSi10Mg single layer ffc\CWT_labelled_windows\dataset_definitions\AlSi10Mg_CW_L1_powder_porosity_with-test_auto-split\trainval.csv")

    print(f"Reading CSV file: {csv_path.name}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Total rows in CSV: {len(df)}")

    # Get logbook data
    print("\nLoading logbook...")
    logbook = get_logbook()

    # Look up regime for each row
    print("\nLooking up regimes for all instances...")
    regimes = []
    missing_trackids = set()

    for idx, row in df.iterrows():
        trackid = extract_trackid_from_filename(row['filename'])

        if trackid:
            # Find the regime in logbook (Layer 1)
            track_rows = logbook.loc[(logbook['trackid'] == trackid) & (logbook['Layer'] == 1)]

            if not track_rows.empty:
                regime = track_rows['Melting regime'].iloc[0]
                regimes.append(regime)
            else:
                missing_trackids.add(trackid)
                regimes.append('Unknown')
        else:
            regimes.append('Unknown')

    df['regime'] = regimes
    df['porosity_label'] = df['has_porosity'].map({0: 'No pore', 1: 'Pore'})

    if missing_trackids:
        print(f"\nWarning: {len(missing_trackids)} trackids not found in logbook:")
        print(f"  {sorted(missing_trackids)}")

    # Count instances by regime and porosity
    print("\nRegime distribution by porosity:")
    regime_porosity_counts = defaultdict(lambda: {'No pore': 0, 'Pore': 0})

    for idx, row in df.iterrows():
        regime = row['regime']
        porosity_label = row['porosity_label']
        regime_porosity_counts[regime][porosity_label] += 1

    # Print summary
    total_counts = {}
    for regime in regime_porosity_counts:
        total = regime_porosity_counts[regime]['No pore'] + regime_porosity_counts[regime]['Pore']
        total_counts[regime] = total
        print(f"  {regime}:")
        print(f"    No pore: {regime_porosity_counts[regime]['No pore']}")
        print(f"    Pore: {regime_porosity_counts[regime]['Pore']}")
        print(f"    Total: {total}")

    # Get color mapping
    regime_marker_dict = get_regime_marker_dict()

    # Sort regimes by total count (descending)
    sorted_regimes = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    regime_names = [r[0] for r in sorted_regimes]

    # Prepare data for clustered bar chart
    no_pore_counts = [regime_porosity_counts[regime]['No pore'] for regime in regime_names]
    pore_counts = [regime_porosity_counts[regime]['Pore'] for regime in regime_names]

    # Get colors for each regime
    regime_colors = []
    for regime in regime_names:
        if regime in regime_marker_dict:
            regime_colors.append(regime_marker_dict[regime]['c'])
        else:
            regime_colors.append('#888888')  # Grey for unknown regimes

    # Abbreviate regime names for x-axis labels
    regime_labels = [abbreviate_regime(regime) for regime in regime_names]

    # Create clustered bar chart
    print("\nCreating clustered bar chart...")

    # Half A4 width = 105mm = 4.13 inches
    fig_width = 4.13
    fig_height = 3.0  # Adjust height as needed

    # Set font sizes globally
    plt.rcParams.update({'font.size': 9})

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set up bar positions
    x = np.arange(len(regime_names))
    width = 0.35  # Width of each bar in cluster

    # Create bars: fully saturated for "No pore", hatched for "Pore"
    bars1 = ax.bar(x - width/2, no_pore_counts, width,
                   color=regime_colors,
                   edgecolor='black', linewidth=0.5, label='No pore')
    bars2 = ax.bar(x + width/2, pore_counts, width,
                   color=regime_colors,
                   edgecolor='black', linewidth=0.5,
                   hatch='///', label='Pore')

    # Customize plot
    ax.set_xlabel('Melting regime', fontsize=9)
    ax.set_ylabel('N samples', fontsize=9)
    ax.set_title('Regime distribution in CWT dataset (trainval.csv)', fontsize=9)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Add legend with custom handles (no color in legend boxes)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, label='No pore'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='///', label='Pore')
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=True, loc='upper right')

    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure at 600 dpi
    output_path = Path('regime_distribution.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path.absolute()}")
    print("\nDone!")

if __name__ == '__main__':
    main()
