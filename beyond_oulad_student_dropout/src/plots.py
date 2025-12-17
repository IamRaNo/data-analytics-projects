import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==========================================
# 1. DESIGN SYSTEM & PALETTES
# ==========================================

# A mix of Seaborn & Matplotlib colormaps for variety
# - 'tab10', 'tab20': Standard distinct colors
# - 'Set2', 'Set3', 'Pastel1': Soft, academic look
# - 'viridis', 'plasma': Modern, dark-mode friendly

PALETTE_OPTIONS = [
    "tab10", "tab20", "Set2", "Set3", "Pastel1", "Pastel2", 
    "Paired", "Accent", "rocket", "mako", "viridis", "Spectral"
]

def apply_custom_style():
    plt.style.use('fivethirtyeight')
    
    # Overrides for a "Modern" look
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.color': '#e6e6e6',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': True,
        'axes.titlelocation': 'left',
        'axes.titleweight': 'bold',
        'axes.titlesize': 18,
        'axes.labelweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.frameon': False,
    })
    print("Modern 538 Design System Applied (With Random Color Injection!)")

def _get_random_palette(n_colors=None):
    """
    Returns a random palette object or a list of colors.
    """
    choice = random.choice(PALETTE_OPTIONS)
    # If n_colors is specified, get exact list, else return palette name
    if n_colors:
        return sns.color_palette(choice, n_colors)
    return choice

def _get_random_color():
    """Returns a single random color from the active options."""
    # Pick a palette, then pick a random color from it
    pal = sns.color_palette(random.choice(PALETTE_OPTIONS))
    return random.choice(pal)

def _add_title_subtitle(ax, title, subtitle):
    """Internal helper to add consistent 538-style headers."""
    ax.text(x=0, y=1.12, s=title, fontsize=18, fontweight='bold', 
            transform=ax.transAxes, ha='left')
    ax.text(x=0, y=1.05, s=subtitle, fontsize=12, color='#555555', 
            transform=ax.transAxes, ha='left')

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

# -----------------------------------------------------------------------------
# Pie Chart
# -----------------------------------------------------------------------------
def plot_pie(column, data):
    df = (data[column]
          .value_counts(normalize=True)
          .mul(100)
          .round(2)
          .reset_index(name='percentage')
          )
    ax = plt.gca()
    # RANDOMIZER: Get a fresh set of colors for the slices
    colors = _get_random_palette(n_colors=len(df))
    
    ax.pie(
        x='percentage',
        data=df,
        labels=df[column],
        autopct="%.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 10}
    )
    
    _add_title_subtitle(ax, 
                        title=f"Breakdown of {column}", 
                        subtitle=f"Distribution of categories in percentage")

# -----------------------------------------------------------------------------
# Box Plot (Univariate)
# -----------------------------------------------------------------------------
def plot_box(column, data, line_val=0.95):
    ax = plt.gca()
    
    # RANDOMIZER: Pick one distinct color for this box
    box_color = _get_random_color()
    
    sns.boxplot(x=column, data=data, linewidth=1.5, ax=ax, width=0.5, color=box_color)
    
    # Add quantile line (Contrast color: Red always works well for alerts)
    quantile_val = data[column].quantile(line_val)
    ax.axvline(quantile_val, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(quantile_val, -0.4, f'{int(line_val*100)}th %tile', color='#fc4f30', fontweight='bold')

    ax.set_xlabel(column.replace('_', ' ').title())
    
    _add_title_subtitle(ax, 
                        title=f"Distribution of {column}", 
                        subtitle=f"Boxplot showing median and outliers")

# -----------------------------------------------------------------------------
# KDE Plot (Univariate)
# -----------------------------------------------------------------------------
def plot_kde(column, data, line_val=0.95):
    ax = plt.gca()
    
    # RANDOMIZER: Pick one distinct color
    kde_color = _get_random_color()
    
    sns.kdeplot(x=column, data=data, fill=True, linewidth=2, ax=ax, alpha=0.5, color=kde_color)
    
    # Add quantile line
    quantile_val = data[column].quantile(line_val)
    ax.axvline(quantile_val, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add the missing LABEL for the line
    # We place it slightly above the x-axis (y=0.02 in axis coordinates)
    # transform=ax.get_xaxis_transform() ensures the text stays at the bottom regardless of plot height
    ax.text(quantile_val, 0.02, f' {int(line_val*100)}th %tile', 
            color='#fc4f30', fontweight='bold', transform=ax.get_xaxis_transform())
    
    ax.set_ylabel("Density")
    ax.set_yticks([]) 
    ax.set_xlabel(column.replace('_', ' ').title())

    _add_title_subtitle(ax, 
                        title=f"Density Curve of {column}", 
                        subtitle=f"Showing the shape of the data distribution")

# -----------------------------------------------------------------------------
# Bar Plot (Target Analysis)
# -----------------------------------------------------------------------------
def percentage_in_that_class(column, data, target, orient='v'):
    df = (data.groupby(column)[target]
          .value_counts(normalize=True)
          .mul(100)
          .round(2)
          .reset_index(name='pct')
          )
    
    ax = plt.gca()
    
    # RANDOMIZER: Pick a random palette for the bars
    rand_pal = _get_random_palette()
    
    if orient == 'h':
        sns.barplot(y=column, x='pct', data=df, hue=target, 
                    edgecolor='white', ax=ax, palette=rand_pal)
        ax.set_xlabel("Percentage (%)")
    else:
        sns.barplot(x=column, y='pct', data=df, hue=target, 
                    edgecolor='white', ax=ax, palette=rand_pal)
        ax.set_ylabel("Percentage (%)")

    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', padding=3, fontsize=10)

    ax.legend(title=target, bbox_to_anchor=(1, 1))
    
    _add_title_subtitle(ax, 
                        title=f"How {target} varies by {column}", 
                        subtitle=f"Percentage split within each category")

# -----------------------------------------------------------------------------
# KDE Plot (Multivariate)
# -----------------------------------------------------------------------------
def kde_in_all_class(column, data, target):
    ax = plt.gca()
    
    # RANDOMIZER: Pick a random palette
    rand_pal = _get_random_palette()
    
    sns.kdeplot(x=column, hue=target, data=data, fill=True, 
                linewidth=2, ax=ax, alpha=0.3, palette=rand_pal)
    
    ax.set_yticks([]) 
    ax.set_xlabel(column.replace('_', ' ').title())
    
    _add_title_subtitle(ax, 
                        title=f"{column} vs. {target}", 
                        subtitle="Compare distribution shapes to find separation")

# -----------------------------------------------------------------------------
# Box Plot (Multivariate)
# -----------------------------------------------------------------------------
def box_in_all_class(column, data, target):
    ax = plt.gca()
    
    # RANDOMIZER
    rand_pal = _get_random_palette()
    
    sns.boxplot(x=column, y=target, data=data, orient='h', 
                linewidth=1.5, ax=ax, palette=rand_pal)
    
    ax.set_xlabel(column.replace('_', ' ').title())
    
    _add_title_subtitle(ax, 
                        title=f"Comparison: {column} by {target}", 
                        subtitle="Look for differences in medians and spread")

# -----------------------------------------------------------------------------
# Scatter Plot
# -----------------------------------------------------------------------------
def scatter_in_all_class(column, target, data):
    ax = plt.gca()
    
    # RANDOMIZER: Usually a single color looks cleaner for scatter unless hue is used
    scatter_color = _get_random_color()
    
    sns.scatterplot(x=column, y=target, data=data, ax=ax, s=60, 
                    alpha=0.6, edgecolor=None, color=scatter_color)
    
    _add_title_subtitle(ax, 
                        title=f"Relationship: {column} vs {target}", 
                        subtitle="Scatter plot to identify correlation")


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------
def plot_heatmap(ct, annot=True):
    ax = plt.gca()
    
    # RANDOMIZER: For heatmaps, we want a random SEQUENTIAL palette (Blues, Greens, etc)
    seq_palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGnBu", "magma", "viridis"]
    chosen_cmap = random.choice(seq_palettes)
    
    sns.heatmap(ct, annot=annot, cmap=chosen_cmap, fmt=".1f", 
                linewidths=1, linecolor='white', cbar_kws={'shrink': .8}, ax=ax)
    
    _add_title_subtitle(ax, 
                        title="Cross-Tabulation Heatmap", 
                        subtitle=f"Intensity map (Color Scheme: {chosen_cmap})")
    
# -----------------------------------------------------------------------------
# 100% Stacked Bar Plot (Category vs Category)
# -----------------------------------------------------------------------------
def plot_stacked(crosstab, orient='v'):
    ax = plt.gca()
    
    # RANDOMIZER: Get distinct colors for the stack sections
    # We ask for exactly as many colors as there are columns (segments)
    n_cols = len(crosstab.columns)
    colors = _get_random_palette(n_colors=n_cols)
    
    # Determine plot type
    kind = 'bar' if orient == 'v' else 'barh'
    
    # Plotting (Notice we added 'color=colors')
    crosstab.plot(kind=kind, stacked=True, width=0.85, 
                  edgecolor='white', linewidth=1.5, ax=ax, color=colors)
    
    # Add Labels
    for c in ax.containers:
        # Create custom labels: Only show if value > 5% to avoid clutter
        labels = [f'{v*100:.1f}%' if v > 0.05 else '' for v in c.datavalues]
        
        ax.bar_label(c, labels=labels, label_type='center', 
                     fontsize=11, color='white', fontweight='bold', padding=0)
        
    # Formatting based on orientation
    if orient == 'v':
        ax.axhline(y=0.5, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
        ax.text(ax.get_xlim()[1], 0.5, ' 50%', color='#fc4f30', 
                fontweight='bold', va='center')
        
        ax.set_ylabel("Proportion")
        ax.set_xlabel(crosstab.index.name.replace('_', ' ').title() if crosstab.index.name else "")
        plt.xticks(rotation=0)
    else:
        ax.axvline(x=0.5, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
        ax.text(0.5, ax.get_ylim()[1], ' 50%', color='#fc4f30', 
                fontweight='bold', ha='center', va='bottom')
        
        ax.set_xlabel("Proportion")
        ax.set_ylabel(crosstab.index.name.replace('_', ' ').title() if crosstab.index.name else "")

    ax.legend(title=crosstab.columns.name, bbox_to_anchor=(1.02, 1), loc='upper left')

    title_text = f"Breakdown of {crosstab.columns.name} by {crosstab.index.name}"
    _add_title_subtitle(ax, 
                        title=title_text, 
                        subtitle="100% Stacked comparison showing relative risk")
    
# -----------------------------------------------------------------------------
# Line Plot (Category vs Numbers)
# -----------------------------------------------------------------------------
def plot_risk_by_bins(data, x_col, target_col, bins=10):
    """
    Bins a numerical variable and plots the Target Probability (Risk) for each bin.
    Automatically handles YES/NO target columns by converting them to 1/0.
    """
    ax = plt.gca()
    
    # 1. Create a working copy so we don't modify the original dataframe
    df = data.copy()
    
    # 2. AUTO-FIX: Convert Target to Numeric if it is String (YES/NO)
    if df[target_col].dtype == 'object':
        # Check if values look like YES/NO
        unique_vals = df[target_col].unique()
        if 'Withdrawn' in unique_vals or 'Stayed' in unique_vals:
            df[target_col] = df[target_col].map({'Withdrawn': 1, 'Stayed': 0})
            print(f"Note: Converted '{target_col}' from Withdrawn/Stayed to 1/0 for this plot.")
    
    # 3. Create Bins (Deciles)
    try:
        df['bin'] = pd.qcut(df[x_col], q=bins, duplicates='drop')
    except ValueError:
        df['bin'] = pd.cut(df[x_col], bins=bins)
        
    # 4. Calculate Mean Target per Bin
    # Now that target_col is 1/0, .mean() works and represents Probability
    risk_data = df.groupby('bin', observed=False)[target_col].mean()
    
    # 5. Plot
    x_labels = [str(interval) for interval in risk_data.index]
    line_color = _get_random_color()
    
    sns.lineplot(x=x_labels, y=risk_data.values, marker='o', 
                 linewidth=3, color=line_color, ax=ax)
    
    ax.fill_between(x_labels, risk_data.values, color=line_color, alpha=0.1)

    # Formatting
    ax.set_ylabel("Low Review Probability")
    ax.set_xlabel(f"{x_col.replace('_', ' ').title()} (Binned)")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add average line
    avg_risk = df[target_col].mean()
    ax.axhline(avg_risk, color='gray', linestyle='--', label=f'Avg Risk ({avg_risk:.1%})')
    ax.legend()

    _add_title_subtitle(ax, 
                        title=f"Risk Curve: {x_col}", 
                        subtitle="How Low Review probability changes as value increases")