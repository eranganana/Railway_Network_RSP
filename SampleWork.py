"""
Visualization and Analysis Module for Transportation Delay Data
=============================================================

This module provides functions for analyzing and visualizing transportation delay
data, with a focus on non-normal distributions, extreme value analysis (CVaR),
and route comparisons. It supports research on delay patterns and risk assessment
in transportation systems.

Author: Achini Nanayakkara
Date: [Current Date]
Affiliation: Uppsala University, Department of Data Science
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set consistent plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_delay_distribution(df: pd.DataFrame, 
                           delay_col: str = 'AvgFörsening') -> tuple:
    """
    Visualize the distribution of delay data, highlighting its non-normal 
    characteristics and extreme positive tail.
    
    This function creates a comprehensive histogram with density curve to 
    illustrate the right-skewed nature of delay distributions, which is 
    typical in transportation systems.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing delay data
    delay_col : str, optional
        Column name containing delay values (default: 'AvgFörsening')
        
    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The generated figure object
        delays : pandas.Series
            Cleaned delay data used for plotting
            
    Examples
    --------
    >>> fig, delays = plot_delay_distribution(df_month_data)
    >>> plt.savefig('delay_distribution.png', dpi=300)
    >>> plt.show()
    """
    
    # Data preprocessing: remove missing values and negative delays
    delays = df[delay_col].dropna()
    delays = delays[delays >= 0]  # Negative delays are treated as on-time
    
    # Initialize figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram with density overlay
    n, bins, patches = ax.hist(
        delays, 
        bins=50, 
        density=True, 
        alpha=0.7, 
        color='skyblue', 
        edgecolor='black', 
        label='Delay Distribution'
    )
    
    # Compute and plot Kernel Density Estimation (KDE)
    kde = gaussian_kde(delays)
    x_range = np.linspace(0, delays.max(), 1000)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density Curve')
    
    # Configure axes labels and title
    ax.set_xlabel('Delay (minutes)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Delay Distribution: Non-Normal Characteristics with Long Right Tail', 
        fontsize=14, 
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate distribution statistics
    stats_dict = {
        'mean': delays.mean(),
        'median': delays.median(),
        'std': delays.std(),
        'skewness': delays.skew(),
        'percentile_95': np.percentile(delays, 95),
        'max': delays.max(),
        'count': len(delays)
    }
    
    # Format statistics as multiline text
    stats_text = (
        "Distribution Properties:\n"
        f"• Mean: {stats_dict['mean']:.2f} min\n"
        f"• Median: {stats_dict['median']:.2f} min\n"
        f"• Std Dev: {stats_dict['std']:.2f} min\n"
        f"• Skewness: {stats_dict['skewness']:.2f}\n"
        f"• 95th Percentile: {stats_dict['percentile_95']:.2f} min\n"
        f"• Maximum: {stats_dict['max']:.2f} min\n"
        f"• Observations: {stats_dict['count']:,}"
    )
    
    # Add statistics annotation to plot
    ax.text(
        0.98, 0.98, 
        stats_text, 
        transform=ax.transAxes, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, 
        fontfamily='monospace'
    )
    
    plt.tight_layout()
    return fig, delays


def plot_route_comparison(df: pd.DataFrame, 
                         route1_name: str, 
                         route2_name: str, 
                         route_col: str = 'Avgångsplats', 
                         delay_col: str = 'AvgFörsening') -> plt.Figure:
    """
    Compare delay distributions between two routes using Kernel Density Estimation.
    
    This function performs a side-by-side comparison of delay patterns between
    two transportation routes, including statistical testing using the
    Mann-Whitney U test (non-parametric alternative to t-test for non-normal data).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing route and delay data
    route1_name : str
        Name of the first route to compare
    route2_name : str
        Name of the second route to compare
    route_col : str, optional
        Column containing route names (default: 'Avgångsplats')
    delay_col : str, optional
        Column containing delay values (default: 'AvgFörsening')
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object showing comparative density plots
        
    Examples
    --------
    >>> fig = plot_route_comparison(df, 'Route_A', 'Route_B')
    >>> fig.savefig('route_comparison.png', dpi=300)
    """
    
    # Filter data for each route
    route1_delays = df[df[route_col] == route1_name][delay_col].dropna()
    route2_delays = df[df[route_col] == route2_name][delay_col].dropna()
    
    # Remove negative delays (considered on-time arrivals)
    route1_delays = route1_delays[route1_delays >= 0]
    route2_delays = route2_delays[route2_delays >= 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot KDE distributions for both routes
    sns.kdeplot(
        route1_delays, 
        ax=ax, 
        label=f'{route1_name}', 
        fill=True, 
        alpha=0.6, 
        color='blue',
        linewidth=2
    )
    sns.kdeplot(
        route2_delays, 
        ax=ax, 
        label=f'{route2_name}', 
        fill=True, 
        alpha=0.6, 
        color='red',
        linewidth=2
    )
    
    # Configure plot appearance
    ax.set_xlabel('Delay (minutes)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Delay Distribution Comparison Between Routes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Perform Mann-Whitney U test for statistical comparison
    stat, p_value = mannwhitneyu(route1_delays, route2_delays)
    
    # Prepare comparison statistics
    comparison_text = (
        f"Statistical Comparison:\n"
        f"• {route1_name}: {len(route1_delays):,} obs, "
        f"Mean: {route1_delays.mean():.2f} min\n"
        f"• {route2_name}: {len(route2_delays):,} obs, "
        f"Mean: {route2_delays.mean():.2f} min\n"
        f"• Mann-Whitney U test p-value: {p_value:.4f}\n"
        f"• {'Significant' if p_value < 0.05 else 'Not significant'} "
        f"at α=0.05"
    )
    
    # Add statistical annotation
    ax.text(
        0.98, 0.98, 
        comparison_text, 
        transform=ax.transAxes,
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=10, 
        fontfamily='monospace'
    )
    
    plt.tight_layout()
    return fig


def plot_extreme_delay_tail(df: pd.DataFrame, 
                           delay_col: str = 'AvgFörsening', 
                           threshold_percentile: float = 90) -> plt.Figure:
    """
    Analyze the extreme tail of delay distribution using Conditional Value at Risk (CVaR).
    
    CVaR (also known as Expected Shortfall) measures the average loss in the worst
    cases beyond a specified Value at Risk (VaR) threshold. This is particularly
    useful for risk management in transportation systems.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing delay data
    delay_col : str, optional
        Column name for delay values (default: 'AvgFörsening')
    threshold_percentile : float, optional
        Percentile threshold for VaR calculation (default: 90)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing extreme tail analysis with VaR and CVaR markers
        
    Notes
    -----
    VaR (Value at Risk): Maximum loss not exceeded with a given confidence level
    CVaR (Conditional VaR): Average loss exceeding the VaR threshold
    """
    
    # Preprocess delay data
    delays = df[delay_col].dropna()
    delays = delays[delays >= 0]
    
    # Calculate VaR threshold
    var_threshold = np.percentile(delays, threshold_percentile)
    extreme_delays = delays[delays >= var_threshold]
    
    # Calculate CVaR (average of losses beyond VaR)
    cvar = extreme_delays.mean()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot full distribution
    n_full, bins_full, _ = ax.hist(
        delays, 
        bins=50, 
        density=True, 
        alpha=0.3, 
        color='blue', 
        label='All Delays'
    )
    
    # Highlight extreme tail region
    n_tail, bins_tail, _ = ax.hist(
        extreme_delays, 
        bins=20, 
        density=True, 
        alpha=0.7,
        color='red', 
        label=f'Extreme Delays (> {threshold_percentile}th %ile)'
    )
    
    # Add VaR and CVaR reference lines
    ax.axvline(
        var_threshold, 
        color='orange', 
        linestyle='--', 
        linewidth=2, 
        label=f'VaR {threshold_percentile}%: {var_threshold:.1f} min'
    )
    
    ax.axvline(
        cvar, 
        color='red', 
        linestyle='--', 
        linewidth=2,
        label=f'CVaR {threshold_percentile}%: {cvar:.1f} min'
    )
    
    # Configure plot
    ax.set_xlabel('Delay (minutes)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        f'Extreme Delay Tail Analysis - CVaR Methodology ({threshold_percentile}% Confidence)', 
        fontsize=14, 
        fontweight='bold'
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate tail statistics
    tail_stats = (
        f"Tail Risk Analysis ({threshold_percentile}th percentile):\n"
        f"• VaR threshold: {var_threshold:.1f} min\n"
        f"• CVaR (average beyond VaR): {cvar:.1f} min\n"
        f"• Extreme observations: {len(extreme_delays):,}\n"
        f"• % of total delays: {100*len(extreme_delays)/len(delays):.1f}%\n"
        f"• Maximum extreme delay: {extreme_delays.max():.1f} min"
    )
    
    # Add tail statistics annotation
    ax.text(
        0.02, 0.98, 
        tail_stats, 
        transform=ax.transAxes,
        verticalalignment='top', 
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontsize=10, 
        fontfamily='monospace'
    )
    
    plt.tight_layout()
    return fig


def plot_temporal_delay_patterns(df: pd.DataFrame, 
                                timestamp_col: str = None, 
                                delay_col: str = 'AvgFörsening') -> plt.Figure:
    """
    Analyze delay patterns across temporal dimensions (time of day, day of week).
    
    This function provides insights into when delays are most likely to occur,
    which is valuable for operational planning and resource allocation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing delay data
    timestamp_col : str, optional
        Column containing timestamp information. If None, only basic distribution
        analysis is performed.
    delay_col : str, optional
        Column containing delay values (default: 'AvgFörsening')
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing temporal patterns or delay distribution percentiles
    """
    
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Handle missing values and negative delays
    df_clean[delay_col] = df_clean[delay_col].fillna(0).clip(lower=0)
    delays = df_clean[delay_col]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram of delays
    n, bins, patches = ax.hist(
        delays, 
        bins=100, 
        density=True, 
        alpha=0.7, 
        color='green', 
        edgecolor='black',
        label='Delay Distribution'
    )
    
    # Define percentiles to highlight
    percentiles = [50, 75, 90, 95, 99]
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    percentile_labels = ['Median', '75th', '90th', '95th', '99th']
    
    # Add percentile reference lines
    for p, color, label in zip(percentiles, colors, percentile_labels):
        percentile_val = np.percentile(delays, p)
        ax.axvline(
            percentile_val, 
            color=color, 
            linestyle=':', 
            linewidth=2,
            label=f'{label} ({p}%): {percentile_val:.1f} min'
        )
    
    # Configure plot
    ax.set_xlabel('Delay (minutes)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Delay Distribution with Percentile Markers', 
        fontsize=14, 
        fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main execution function demonstrating usage of all visualization tools.
    
    This serves as both an example of how to use the module and can be run
    directly for data exploration.
    """
    print("=" * 60)
    print("Transportation Delay Analysis Module")
    print("=" * 60)
    
    try:
        # Load transportation delay data
        print("Loading data from 'Cst_to_M.csv'...")
        df_month_data = pd.read_csv("Cst_to_M.csv", index_col=False, encoding="utf-8")
        
        print(f"Data loaded successfully. Shape: {df_month_data.shape}")
        print(f"Columns: {list(df_month_data.columns)}")
        
        # Example 1: Overall delay distribution
        print("\n1. Plotting overall delay distribution...")
        fig1, delays = plot_delay_distribution(df_month_data)
        plt.savefig('results/delay_distribution_non_normal.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Saved as 'delay_distribution_non_normal.png'")
        
        # Example 2: Extreme tail analysis for risk assessment
        print("\n2. Plotting extreme delay tail (CVaR analysis)...")
        fig2 = plot_extreme_delay_tail(df_month_data, threshold_percentile=90)
        plt.savefig('results/extreme_delay_tail_cvar.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Saved as 'extreme_delay_tail_cvar.png'")
        
        # Example 3: Route comparison (if route data exists)
        if 'Avgångsplats' in df_month_data.columns:
            print("\n3. Attempting route comparison...")
            # Get top 2 most common routes for comparison
            route_counts = df_month_data['Avgångsplats'].value_counts()
            if len(route_counts) >= 2:
                top_routes = route_counts.index[:2].tolist()
                fig3 = plot_route_comparison(df_month_data, 
                                            top_routes[0], 
                                            top_routes[1])
                plt.savefig('results/route_comparison.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
                print(f"   ✓ Compared {top_routes[0]} vs {top_routes[1]}")
        
        # Example 4: Temporal patterns (if timestamp available)
        print("\n4. Generating percentile analysis...")
        fig4 = plot_temporal_delay_patterns(df_month_data)
        plt.savefig('results/delay_percentiles.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Saved as 'delay_percentiles.png'")
        
        print("\n" + "=" * 60)
        print("Analysis complete! All visualizations saved to /results/ folder")
        print("=" * 60)
        
    except FileNotFoundError:
        print("Error: Data file 'Cst_to_M.csv' not found.")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    """
    Execution entry point.
    
    When run directly, this script performs a complete analysis of
    transportation delay data using all visualization functions.
    """
    main()