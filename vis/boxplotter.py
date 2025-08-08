from scipy import stats
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook, define_collumn_labels

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_posthoc_heatmap(posthoc_results, title="Post-hoc Statistical Comparisons", 
                                 alpha=0.05, method_name="Mann-Whitney", order=None,
                                 dpi=300):
    """
    Create an improved heatmap optimized for high-resolution output.
    
    Parameters:
    -----------
    posthoc_results : pd.DataFrame
        Square matrix of p-values from post-hoc test
    title : str
        Title for the plot
    alpha : float
        Significance threshold
    method_name : str
        Name of the statistical method used
    order : list of str, optional
        List of regime names to dictate the order they appear on axes.
        If None, uses the original order from posthoc_results.
    dpi : int, default=300
        Resolution in dots per inch for publication quality
    """
    
    # Reorder the dataframe if order is specified
    if order is not None:
        # Check that all regimes in order exist in the data
        missing_regimes = set(order) - set(posthoc_results.index)
        if missing_regimes:
            raise ValueError(f"Regimes in order not found in data: {missing_regimes}")
        
        extra_regimes = set(posthoc_results.index) - set(order)
        if extra_regimes:
            print(f"Warning: Regimes in data not specified in order: {extra_regimes}")
            # Add missing regimes to the end of the order
            order = list(order) + list(extra_regimes)
        
        # Reorder both rows and columns
        plot_data = posthoc_results.reindex(index=order, columns=order)
    else:
        plot_data = posthoc_results.copy()
    
    # Create annotation matrix with proper formatting
    annot_data = plot_data.copy().astype(str)
    
    for i in range(len(plot_data)):
        for j in range(len(plot_data.columns)):
            if i == j:
                # Diagonal - should always be 1.0 (identical groups)
                annot_data.iloc[i, j] = "1.000"
            elif i < j:
                # Upper triangle - will be masked, so empty
                annot_data.iloc[i, j] = ""
            else:
                # Lower triangle - format the p-values
                val = plot_data.iloc[i, j]
                if val < 0.001:
                    annot_data.iloc[i, j] = "<0.001"
                elif val >= 0.999:
                    annot_data.iloc[i, j] = ">0.999"
                else:
                    annot_data.iloc[i, j] = f"{val:.3f}"
    
    # Create mask for upper triangle only (NOT including diagonal)
    mask = np.triu(np.ones_like(plot_data, dtype=bool), k=1)
    
    # Set up the matplotlib figure with high DPI
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    
    # Create custom colormap
    colors = ['#08519c', '#3182bd', '#6baed6', '#c6dbef', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Create the heatmap with appropriately scaled elements for high DPI
    sns.heatmap(plot_data, 
                mask=mask,
                annot=annot_data,
                fmt='',
                cmap=cmap,
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'p-value', 'shrink': 0.7, 'aspect': 15},
                annot_kws={'fontsize': 9, 'fontweight': 'bold'},  # Smaller for high DPI
                ax=ax,
                linewidths=0.2,  # Thinner lines for high DPI
                linecolor='white')
    
    # Customize the plot with smaller fonts for high DPI
    ax.set_title(f"{title}\n({method_name} pairwise comparisons)", 
                fontsize=12, fontweight='bold', pad=8)  # Smaller size and padding
    
    # Rotate labels for better readability with smaller fonts
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # Add significance threshold line to colorbar with thinner elements
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)  # Smaller colorbar tick size
    cbar.set_label('p-value', fontsize=11)  # Smaller colorbar label size
    cbar.ax.axhline(y=alpha, color='black', linewidth=1, alpha=0.8)  # Thinner line
    cbar.ax.text(1.1, alpha, f'α = {alpha}', transform=cbar.ax.transData, 
                fontsize=9, fontweight='bold', va='center')  # Smaller font size
    
    # Add legend explaining the visualization with smaller text
    legend_text = (f"Lower triangle + diagonal show p-values\n"
                  f"Dark blue: p < {alpha} (significant)\n"
                  f"Light colors: p ≥ {alpha} (not significant)")
    
    ax.text(0.99, 0.99, legend_text, transform=ax.transAxes, 
            fontsize=8, ha='right', va='top',  # Smaller legend font size
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
                     edgecolor='gray', linewidth=0.5))  # Thinner legend box
    
    plt.tight_layout(pad=0.3)  # Smaller padding for high DPI
    
    return fig

def analyze_regime_differences(data, value_col, group_col, plot_posthoc=True, 
                                      alpha=0.05, p_adjust=None, 
                                      posthoc_method='mannwhitney', order=None):
    """
    Perform Kruskal-Wallis test and post-hoc analysis for regime comparison.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name containing the values to compare (e.g., 'pool_length')
    group_col : str
        Column name containing the group labels (e.g., 'regime')
    plot_posthoc : bool, default=True
        Whether to create heatmap of post-hoc results
    alpha : float, default=0.05
        Significance level
    p_adjust : str or None, default=None
        Multiple comparison correction method (try 'bonferroni' or 'fdr_bh')
    posthoc_method : str, default='dunn'
        Post-hoc test method ('dunn', 'steel_dwass', 'conover', 'mannwhitney')
    order : list of str : optional
        List of regime names to dictate the order they appear on axes, if plotted.
        If None, determines automatically.
    Returns:
    --------
    results : dict
        Dictionary containing all statistical results
    """
    
    results = {}
    
    # Perform Kruskal-Wallis test
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    h_stat, p_value = stats.kruskal(*groups)
    
    results['kruskal_wallis'] = {
        'H_statistic': h_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }
    
    # Summary statistics
    summary_stats = data.groupby(group_col)[value_col].agg([
        'count', 'median', 'mean', 'std', 
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75)
    ]).round(3)
    
    summary_stats.columns = ['n', 'median', 'mean', 'std', 'Q1', 'Q3']
    summary_stats['IQR'] = summary_stats['Q3'] - summary_stats['Q1']
    
    results['summary_stats'] = summary_stats
    
    # Post-hoc analysis if significant
    if p_value < alpha:
        
        if posthoc_method == 'mannwhitney':
            posthoc_results = pairwise_mannwhitney(data, value_col, group_col)
            method_display = "Mann-Whitney U"
        elif posthoc_method == 'dunn':
            posthoc_results = sp.posthoc_dunn(data, val_col=value_col, group_col=group_col, 
                                            p_adjust=p_adjust)
            method_display = "Dunn's"
        elif posthoc_method == 'steel_dwass':
            posthoc_results = sp.posthoc_steel_dwass(data, val_col=value_col, group_col=group_col)
            method_display = "Steel-Dwass"
        elif posthoc_method == 'conover':
            posthoc_results = sp.posthoc_conover(data, val_col=value_col, group_col=group_col, 
                                               p_adjust=p_adjust)
            method_display = "Conover"
        
        results['posthoc_method'] = posthoc_method
        results['posthoc_test'] = posthoc_results
        
        # Find significant pairs
        regime_names = data[group_col].unique()
        significant_pairs = []
        
        for i, regime1 in enumerate(regime_names):
            for j, regime2 in enumerate(regime_names):
                if i < j:
                    p_val = posthoc_results.loc[regime1, regime2]
                    if p_val < alpha:
                        significant_pairs.append({
                            'pair': f"{regime1} vs {regime2}",
                            'p_value': p_val
                        })
        
        results['significant_pairs'] = significant_pairs
        
        # Create improved heatmap if requested
        if plot_posthoc:
            correction_text = f" with {p_adjust} correction" if p_adjust else ""
            title = f"Statistical Significance Matrix{correction_text}"
            fig = plot_posthoc_heatmap(posthoc_results, title=title, 
                                        alpha=alpha, method_name=method_display, order=order)
            results['fig'] = fig
    
    else:
        results['posthoc_test'] = None
        results['significant_pairs'] = []
    
    return results

def pairwise_mannwhitney(data, value_col, group_col):
    """
    Perform pairwise Mann-Whitney U tests between all groups.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name containing the values
    group_col : str
        Column name containing the group labels
    
    Returns:
    --------
    p_matrix : pd.DataFrame
        Matrix of p-values for all pairwise comparisons
    """
    
    regimes = data[group_col].unique()
    n_regimes = len(regimes)
    
    # Create empty matrix
    p_matrix = np.ones((n_regimes, n_regimes))
    
    for i, regime1 in enumerate(regimes):
        for j, regime2 in enumerate(regimes):
            if i != j:
                group1 = data[data[group_col] == regime1][value_col]
                group2 = data[data[group_col] == regime2][value_col]
                
                try:
                    _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    p_matrix[i, j] = p_val
                except:
                    # Handle edge cases (identical groups, etc.)
                    p_matrix[i, j] = 1.0
    
    # Convert to DataFrame for easy reading
    p_df = pd.DataFrame(p_matrix, index=regimes, columns=regimes)
    
    return p_df

def print_statistical_summary(results, value_name="pool length"):
    """
    Print a formatted summary of statistical results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_regime_differences()
    value_name : str
        Name of the measured variable for reporting
    """
    
    kw = results['kruskal_wallis']
    
    print(f"Statistical Analysis Summary - {value_name.title()}")
    print("=" * 60)
    print(f"Kruskal-Wallis Test:")
    print(f"  H-statistic: {kw['H_statistic']:.3f}")
    
    # Handle very small p-values
    if kw['p_value'] < 1e-10:
        print(f"  p-value: < 1e-10")
    else:
        print(f"  p-value: {kw['p_value']:.2e}")
    
    print(f"  Result: {'Significant' if kw['significant'] else 'Not significant'}")
    
    print(f"\nSummary Statistics by Regime:")
    print(results['summary_stats'])
    
    if kw['significant'] and results['significant_pairs']:
        posthoc_method = results.get('posthoc_method', 'unknown')
        print(f"\nPost-hoc Analysis ({posthoc_method.title()}):")
        print(f"Significant Pairwise Differences (p < 0.05):")
        for pair in results['significant_pairs']:
            print(f"  - {pair['pair']}: p = {pair['p_value']:.4f}")
    
    print("\n" + "=" * 60)

def plot_boxplot_by_category(df, data_column, category_column, x_label=None, y_label = None, title=None, order=None, figsize=(4, 5),
        show_counts=True, show_medians=False, show_points=False, scientific=False, dpi=300):
    """
    Create a box and whisker plot from a pandas DataFrame with optional data point counts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    data_column : str
        Name of the column containing the numeric data to plot
    category_column : str
        Name of the column containing the categories for grouping
    x_label : str, optional
        Title for the x-axis
    y_label : str, optional
        Title for the y-axis
    title : str, optional
        Title for the plot
    order : list, optional
        List of category labels defining their display order on the x-axis
    figsize : tuple, optional
        Figure size as (width, height)
    show_counts : bool, optional
        Whether to display data point counts for each category
    show_medians : bool, optional
        Whether to display median value labels for each category
    show_points : bool, optional
        Whether to plot the individual points
    scientific : bool, optional
        Whether to format the y-axis values in scientific notation
    dpi : int, default=300
        Resolution in dots per inch for publication quality
    
    
    Returns:
    --------
    matplotlib.figure.Figure
        The matplotlib figure object
    """
    
    # Create the figure and axis with high DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Format y-axis value labels
    if scientific:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # Forces scientific notation
        ax.yaxis.set_major_formatter(formatter)
    
    # Create the box plot
    # palette = sns.color_palette("viridis", n_colors=df[category_column].nunique())
    palette = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    
    marker_dict = {'unstable keyhole': {'m': 'o', 'c': '#fde725'},
                   'keyhole flickering': {'m': 's', 'c': '#3b528b'},
                   'quasi-stable keyhole': {'m': '^', 'c': '#5ec962'},
                   'quasi-stable vapour depression': {'m': 'D', 'c': '#21918c'},
                   'conduction': {'m': 'v', 'c': '#440154'}
                   }
                   
    sns.boxplot(data=df, x=category_column, y=data_column, ax=ax, order=order, palette=palette,
                linewidth=0.8)  # Thinner lines for high DPI
    
    # Label axes if provided, otherwise column labels are used - smaller fonts for high DPI
    if x_label:
        ax.set_xlabel(x_label, fontsize=11)
    if y_label:
        ax.set_ylabel(y_label, fontsize=11)
    
    # Add data point counts if requested
    if show_counts:
        # Get counts for each category
        counts = df[category_column].value_counts().sort_index()
        
        # Add count labels below each box - smaller font for high DPI
        for i, (category, count) in enumerate(counts.items()):
            if order:
                i = order.index(category)
            ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
                   f'n={count}', 
                   ha='center', va='top', fontsize=9, color='gray', rotation=20)
    
    # Set title if provided - smaller font for high DPI
    if title:
        ax.set_title(title, fontsize=12)
    
    # Rotate x-axis labels if there are many categories
    if df[category_column].nunique() > 4:
        plt.xticks(rotation=20, ha='right')
    
    # Format labels - smaller fonts for high DPI
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    # Adjust layout to prevent label cutoff (extra space for count labels)
    if show_counts:
        plt.subplots_adjust(bottom=0.25)
    else:
        plt.tight_layout(pad=0.3)  # Smaller padding for high DPI
    
    # Print counts to console as well
    if show_counts:
        print("Data points per category:")
        for category, count in counts.items():
            print(f"  {category}: {count}")
            
    # Add median values on top of each box - smaller font for high DPI
    if show_medians:
        medians = df.groupby(category_column)[data_column].median()
        for i, median in enumerate(medians):
            ax.text(i, median, f'{median:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=6,  # Smaller font for high DPI
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                             edgecolor='gray', linewidth=0.5))  # Thinner box for high DPI
                    
    if show_points:
        # Overlay actual data points - smaller points for high DPI
        sns.stripplot(data=df, x=category_column, y=data_column, 
                      size=2, alpha=1, color='black', ax=ax)  # Smaller points for high DPI
    
    return fig

def filter_logbook(log):
    if True:
        # filters for welding or powder melting
        welding = log['Powder material'] == 'None'
        powder = np.invert(welding)

        # filters for CW or PWM laser mode
        cw = log['Point jump delay [us]'] == 0
        pwm = np.invert(cw)

        # filter for Layer 1 tracks only
        L1 = log['Layer'] == 1
        
        # filter for presence of KH pores
        pores = log['n_pores'] > 2
        
        # filter by layer thickness
        thin_layer = log['measured_layer_thickness [um]'] <= 80
        very_thin_layer = log['measured_layer_thickness [um]'] <= 35
        
        # filter by scan speed
        speed = log['Scan speed [mm/s]'] == 400
        
        # filter by beamtime
        ltp1 = log['Beamtime'] == 1
        ltp2 = log['Beamtime'] == 2
        ltp3 = log['Beamtime'] == 3
        
        # filter by substrate
        s0514 = log['Substrate No.'] == '0514'
        s0515 = log['Substrate No.'] == '0515'

        # filter by material
        AlSi10Mg = log['Substrate material'] == 'AlSi10Mg'
        Al7A77 = log['Substrate material'] == 'Al7A77'
        Al = log['Substrate material'] == 'Al'
        Ti64 = log['Substrate material'] == 'Ti64'
        lit = np.logical_or(Ti64, Al7A77)
        
        # filter by regime
        not_flickering = log['Melting regime'] != 'keyhole flickering'
        not_cond = log['Melting regime'] != 'conduction'

    # Apply combination of above filters to select parameter subset to plot
    # log_red = log[np.logical_or(AlSi10Mg, lit) & L1 & cw & powder]
    log_red = log[AlSi10Mg & L1 & cw]
    # print(log_red)
    # print(len(log_red))
    return log_red

def main():
    log = get_logbook()
    df = filter_logbook(log)
    col_dict = define_collumn_labels()
    
    # Select data to analyse
    data_label = 'St'
    cat_label = 'regime'
    
    # Fetch column labels and titles from dictionary
    val_col = col_dict[data_label][0]
    val_title = col_dict[data_label][1]
    cat_col = col_dict[cat_label][0]
    cat_title = col_dict[cat_label][1]
    
    # Clean up nan values
    df = df.dropna(subset=[val_col, cat_col])
    
    # Specify plotting order of regimes
    order = ['conduction', 'keyhole flickering', 'quasi-stable vapour depression', 'quasi-stable keyhole', 'unstable keyhole']

    # Run statistical comparisons between categorical groups
    results = analyze_regime_differences(df, val_col, cat_col, plot_posthoc=True, p_adjust=None, posthoc_method='mannwhitney', order=order)
    print_statistical_summary(results, val_title)
    # Save figure
    results['fig'].savefig(f'{val_title} significance by {cat_title} heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    # Generate boxplot of grouped data
    fig = plot_boxplot_by_category(df, val_col, cat_col, cat_title, val_title, order=order, show_points=False, show_counts=False, scientific=True)
    # Save figure
    fig.savefig(f'{val_title} by {cat_title} boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

if __name__ == "__main__":
    main()