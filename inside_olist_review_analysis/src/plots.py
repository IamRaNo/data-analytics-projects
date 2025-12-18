import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.gridspec as gridspec

# ==========================================
# 1. DESIGN SYSTEM & PALETTES
# ==========================================

# A mix of Seaborn & Matplotlib colormaps for variety
# - 'tab10', 'tab20': Standard distinct colors
# - 'Set2', 'Set3', 'Pastel1': Soft, academic look
# - 'viridis', 'plasma': Modern, dark-mode friendly

PALETTE_OPTIONS = [
    "tab10", "tab20", "Set2", "Paired", "Accent", "Dark2",
    "rocket", "mako", "viridis", "Spectral", 
    "magma", "inferno", "flare", "crest"
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
        startangle=50,
        wedgeprops={'edgecolor': 'black', 'linewidth': 2},
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
        if 'low' in unique_vals or 'high' in unique_vals:
            df[target_col] = df[target_col].map({'low': 1, 'high': 0})
            print(f"Note: Converted '{target_col}' from LOW/HIGH to 1/0 for this plot.")
    
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
    
# -----------------------------------------------------------------------------
# Statistical Dashboard (The "Nice" Side-by-Side View)
# -----------------------------------------------------------------------------
def plot_describe_dashboard(data, group_col, value_col):
    """
    Visualizes all 8 statistics from .describe() in a single 2x4 grid.
    Matches the Modern 538 design system.
    """
    # 1. Compute the statistics
    desc = data.groupby(group_col)[value_col].describe().round(2).reset_index()
    
    # Ensure the group column is treated as categorical (string) for better plotting
    desc[group_col] = desc[group_col].astype(str)
    
    # 2. Identify metrics (Count, Mean, Std, Min, 25%, 50%, 75%, Max)
    metrics = [col for col in desc.columns if col != group_col]

    # 3. Initialize Figure (2 Rows x 4 Cols)
    # Using figsize=(24, 6) to give slightly more breathing room than 5
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 6))
    axes = axes.flatten() 
    
    # 4. Loop through metrics
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # RANDOMIZER: distinct palette for each metric to separate them visually
        bar_colors = _get_random_palette(n_colors=len(desc))
        
        # Plot Horizontal Bars (y=group_col)
        sns.barplot(data=desc, y=group_col, x=metric, ax=ax, 
                    width=0.6, palette=bar_colors, edgecolor='white')
        
        # Add Data Labels (Numbers on bars)
        for container in ax.containers:
            ax.bar_label(container, padding=5, fontsize=10, fontweight='bold', color='#333333')

        # 538 Styling Elements
        # We manually style titles here because _add_title_subtitle is designed for single plots
        ax.set_title(metric.upper(), fontsize=14, fontweight='bold', loc='left', color='#333333')
        
        # Clean up axes
        ax.set_xlabel('')
        # Only show Y-label on the first column (indices 0 and 4) to reduce clutter
        if i % 4 == 0:
            ax.set_ylabel(group_col.replace('_', ' ').title(), fontweight='bold')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([]) # Hide tick labels on inner plots

        # Add grid only on X axis for horizontal bars
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.grid(False, axis='y')

    # Add a Super Title for the whole dashboard
    plt.suptitle(f"Statistical Summary: {value_col} by {group_col}", 
                 fontsize=22, fontweight='bold', x=0.01, ha='left', y=1.05)


def plot_numeric(data, col, target):
    """
    Master visualization for a numerical column against a target categorical column.
    
    IMPORTANT: Initialize figure size yourself before calling.
    Example: plt.figure(figsize=(24, 14))
    """
    
    # --- SAFETY FIX: Ensure 'col' is strictly numeric ---
    data[col] = pd.to_numeric(data[col], errors='coerce')
    plot_data = data.dropna(subset=[col, target]).copy()
    
    # 1. Get the current active figure
    fig = plt.gcf()
    
    # 2. Setup the Grid with Custom Height Ratios
    # [3, 1, 1] means:
    # Row 0 (KDE/Box) is 3x taller than the stats rows.
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[3, 1, 1])
    
    # RANDOMIZER: Get a random palette for the main visuals
    main_palette = _get_random_palette() 

    # ==========================================
    # ROW 1: The Big Visuals (Tall Row)
    # ==========================================

    # 1.1 KDE Plot
    ax_kde = fig.add_subplot(gs[0, :2])
    sns.kdeplot(data=plot_data, x=col, hue=target, fill=True, 
                linewidth=2, palette=main_palette, alpha=0.3, ax=ax_kde, warn_singular=False)
    ax_kde.set_title(f"Distribution: {col} by {target}", fontsize=14, fontweight='bold')
    ax_kde.set_xlabel('')
    ax_kde.grid(True, alpha=0.3)

    # 1.2 Box Plot
    ax_box = fig.add_subplot(gs[0, 2:])
    sns.boxplot(data=plot_data, x=col, y=target, orient='h', 
                palette=main_palette, linewidth=1.5, ax=ax_box)
    ax_box.set_title(f"Outliers: {col} by {target}", fontsize=14, fontweight='bold')
    ax_box.set_xlabel('')
    ax_box.grid(True, axis='x', alpha=0.3)

    # ==========================================
    # ROWS 2 & 3: The Statistical Dashboard (Short Rows)
    # ==========================================
    
    desc = plot_data.groupby(target)[col].describe().round(2).reset_index()
    desc[target] = desc[target].astype(str)
    
    metrics = [m for m in desc.columns if m != target]
    
    for i, metric in enumerate(metrics):
        grid_row = 1 if i < 4 else 2
        grid_col = i % 4
        
        ax = fig.add_subplot(gs[grid_row, grid_col])
        
        # RANDOMIZER: Get distinct colors for the bar elements
        bar_colors = _get_random_palette(n_colors=len(desc))
        
        sns.barplot(data=desc, y=target, x=metric, ax=ax, 
                    palette=bar_colors, edgecolor='white', width=0.6)
        
        for container in ax.containers:
            ax.bar_label(container, padding=3, fontsize=10, fontweight='bold', color='#333333')
            
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold', loc='left', color='#444444')
        ax.set_xlabel('')
        
        if grid_col == 0:
            ax.set_ylabel(target.replace('_', ' ').title(), fontweight='bold')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
            
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.grid(False, axis='y')

    plt.suptitle(f"Deep Dive: {col} vs {target}", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

def plot_categorical(data, col, target, order=None):
    """
    Master visualization for a Categorical column against a Categorical target.
    
    Layout:
    - Left (Large): 100% Stacked Horizontal Bar Chart (Row Percentages)
    - Right (Small): Heatmap of Counts & Overall Percentages
    
    IMPORTANT: Initialize figure size yourself before calling.
    Example: plt.figure(figsize=(20, 8))
    """
    
    # 1. Data Preparation
    plot_data = data.dropna(subset=[col, target])
    ct_counts = pd.crosstab(plot_data[col], plot_data[target])
    ct_props = pd.crosstab(plot_data[col], plot_data[target], normalize='index')

    # --- ORDERING LOGIC ---
    if order:
        # Reindex to force the specific order provided by the user
        ct_counts = ct_counts.reindex(order)
        ct_props = ct_props.reindex(order)
    # ----------------------

    # 2. Setup Grid with Manual Margins
    fig = plt.gcf()
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.02,
                           left=0.05, right=0.95, top=0.80, bottom=0.1)
    
    # ==========================================
    # LEFT PLOT: 100% Stacked Horizontal Bar
    # ==========================================
    ax_bar = fig.add_subplot(gs[0, 0])
    
    bar_colors = _get_random_palette(n_colors=len(ct_props.columns))
    
    ct_props.plot(kind='barh', stacked=True, width=0.85, 
                  edgecolor='white', linewidth=1, ax=ax_bar, color=bar_colors)
    
    # Invert Y-axis so first category is at TOP
    ax_bar.invert_yaxis()
    
    # Labels (Row Percentages)
    for c in ax_bar.containers:
        labels = [f'{v*100:.1f}%' if v > 0.05 else '' for v in c.datavalues]
        ax_bar.bar_label(c, labels=labels, label_type='center', 
                         fontsize=11, color='white', fontweight='bold', padding=0)
    
    # Reference Line
    ax_bar.axvline(x=0.5, color='#333333', linestyle='--', alpha=0.5, linewidth=2)
    
    # Styling
    ax_bar.set_xlabel("Proportion", fontweight='bold')
    ax_bar.set_ylabel('') 
    
    # Legend
    ax_bar.legend(title=target.replace('_', ' ').title(), 
                  bbox_to_anchor=(1.0, 1.02), loc='lower right', 
                  ncols=4, frameon=False)
    
    ax_bar.set_title(f"Proportions: {col} vs {target}", fontsize=16, fontweight='bold', loc='left', y=1.02)
    
    # ==========================================
    # RIGHT PLOT: Heatmap
    # ==========================================
    ax_heat = fig.add_subplot(gs[0, 1])
    
    # --- CUSTOM LABELS (Count + % of Overall) ---
    grand_total = ct_counts.values.sum()
    # Create a matrix of strings: "Count \n (Percent%)"
    heatmap_labels = ct_counts.applymap(lambda x: f"{x}\n({x/grand_total*100:.1f}%)")
    # --------------------------------------------

    seq_palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGnBu"]
    heat_cmap = random.choice(seq_palettes)
    
    sns.heatmap(ct_counts, annot=heatmap_labels, fmt='', cmap=heat_cmap, 
                linewidths=2, linecolor='white', cbar=True, ax=ax_heat)
    
    # Styling
    ax_heat.set_ylabel('') 
    ax_heat.set_yticklabels([]) 
    ax_heat.set_xlabel(target.replace('_', ' ').title(), fontweight='bold')
    
    ax_heat.set_title("Sample Sizes (% Overall)", fontsize=16, fontweight='bold', loc='left', y=1.02)

    # ==========================================
    # MAIN TITLE
    # ==========================================
    plt.suptitle(f"Categorical Analysis: {col} by {target}", 
                 fontsize=22, fontweight='bold', y=0.95, x=0.05, ha='left')


def plot_masked_categorical(data, col, mask_value, target, order=None):
    """
    Plots a binary comparison: 'mask_value' vs 'OTHERS'.
    """
    # 1. Create Data Logic
    mask = data[col] == mask_value
    temp_df = data.copy()
    comp_col = f"{col} Comparison" 
    temp_df[comp_col] = np.where(mask, str(mask_value), 'OTHERS')
    
    # 2. Initialize Figure (Specific size for binary comparison)
    
    # 3. Call Main Function
    # We pass the NEW comparison column as the category
    plot_categorical(temp_df, comp_col, target, order=order)

