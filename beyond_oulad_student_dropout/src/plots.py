import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random

# ==========================================
# 1. DESIGN SYSTEM & CONFIGURATION
# ==========================================

PALETTE_OPTIONS = [
    "tab10", "tab20", "Set2", "Paired", "Accent", "Dark2",
    "rocket", "mako", "viridis", "Spectral", 
    "magma", "inferno", "flare", "crest"
]

def apply_custom_style():
    """Applies the Modern 538 Design System."""
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,          # Global default (can be overridden)
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
    print("Modern 538 Design System Applied.")

def _get_random_palette(n_colors=None):
    choice = random.choice(PALETTE_OPTIONS)
    if n_colors:
        return sns.color_palette(choice, n_colors)
    return choice

def _get_random_color():
    pal = sns.color_palette(random.choice(PALETTE_OPTIONS))
    return random.choice(pal)

def _add_title_subtitle(ax, title, subtitle):
    ax.text(x=0, y=1.12, s=title, fontsize=18, fontweight='bold', transform=ax.transAxes, ha='left')
    ax.text(x=0, y=1.05, s=subtitle, fontsize=12, color='#555555', transform=ax.transAxes, ha='left')

# ==========================================
# 2. UNIVARIATE PLOTS
# ==========================================

def plot_pie(column, data):
    # Safe: value_counts returns a new object
    df = data[column].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentage')
    ax = plt.gca()
    colors = _get_random_palette(n_colors=len(df))
    
    ax.pie(x='percentage', data=df, labels=df[column], autopct="%.1f%%",
           colors=colors, startangle=50,
           wedgeprops={'edgecolor': 'black', 'linewidth': 2},
           textprops={'fontsize': 10})
    
    _add_title_subtitle(ax, f"Breakdown of {column}", "Distribution of categories in percentage")

def plot_box(column, data, line_val=0.95):
    ax = plt.gca()
    sns.boxplot(x=column, data=data, linewidth=1.5, ax=ax, width=0.5, color=_get_random_color())
    
    quantile_val = data[column].quantile(line_val)
    ax.axvline(quantile_val, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(quantile_val, -0.4, f'{int(line_val*100)}th %tile', color='#fc4f30', fontweight='bold')
    
    ax.set_xlabel(column.replace('_', ' ').title())
    _add_title_subtitle(ax, f"Distribution of {column}", "Boxplot showing median and outliers")

def plot_kde(column, data, line_val=0.95):
    ax = plt.gca()
    sns.kdeplot(x=column, data=data, fill=True, linewidth=2, ax=ax, alpha=0.5, color=_get_random_color())
    
    quantile_val = data[column].quantile(line_val)
    ax.axvline(quantile_val, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(quantile_val, 0.02, f' {int(line_val*100)}th %tile', 
            color='#fc4f30', fontweight='bold', transform=ax.get_xaxis_transform())
    
    ax.set_ylabel("Density"); ax.set_yticks([])
    ax.set_xlabel(column.replace('_', ' ').title())
    _add_title_subtitle(ax, f"Density Curve of {column}", "Showing the shape of the data distribution")

# ==========================================
# 3. BIVARIATE / MULTIVARIATE PLOTS
# ==========================================

def percentage_in_that_class(column, data, target, orient='v'):
    df = data.groupby(column)[target].value_counts(normalize=True).mul(100).round(2).reset_index(name='pct')
    ax = plt.gca()
    rand_pal = _get_random_palette()
    
    if orient == 'h':
        sns.barplot(y=column, x='pct', data=df, hue=target, edgecolor='white', ax=ax, palette=rand_pal)
        ax.set_xlabel("Percentage (%)")
    else:
        sns.barplot(x=column, y='pct', data=df, hue=target, edgecolor='white', ax=ax, palette=rand_pal)
        ax.set_ylabel("Percentage (%)")

    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', padding=3, fontsize=10)
    
    ax.legend(title=target, bbox_to_anchor=(1, 1))
    _add_title_subtitle(ax, f"How {target} varies by {column}", "Percentage split within each category")

def kde_in_all_class(column, data, target):
    ax = plt.gca()
    sns.kdeplot(x=column, hue=target, data=data, fill=True, linewidth=2, ax=ax, alpha=0.3, palette=_get_random_palette())
    ax.set_yticks([]); ax.set_xlabel(column.replace('_', ' ').title())
    _add_title_subtitle(ax, f"{column} vs. {target}", "Compare distribution shapes to find separation")

def box_in_all_class(column, data, target):
    ax = plt.gca()
    sns.boxplot(x=column, y=target, data=data, orient='h', linewidth=1.5, ax=ax, palette=_get_random_palette())
    ax.set_xlabel(column.replace('_', ' ').title())
    _add_title_subtitle(ax, f"Comparison: {column} by {target}", "Look for differences in medians and spread")

def scatter_in_all_class(column, target, data):
    ax = plt.gca()
    sns.scatterplot(x=column, y=target, data=data, ax=ax, s=60, alpha=0.6, edgecolor=None, color=_get_random_color())
    _add_title_subtitle(ax, f"Relationship: {column} vs {target}", "Scatter plot to identify correlation")

def plot_heatmap(ct, annot=True):
    ax = plt.gca()
    seq_palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGnBu", "magma", "viridis"]
    sns.heatmap(ct, annot=annot, cmap=random.choice(seq_palettes), fmt=".1f", 
                linewidths=1, linecolor='white', cbar_kws={'shrink': .8}, ax=ax)
    _add_title_subtitle(ax, "Cross-Tabulation Heatmap", "Intensity map")

def plot_stacked(crosstab, orient='v'):
    ax = plt.gca()
    colors = _get_random_palette(n_colors=len(crosstab.columns))
    kind = 'bar' if orient == 'v' else 'barh'
    
    crosstab.plot(kind=kind, stacked=True, width=0.85, edgecolor='white', linewidth=1.5, ax=ax, color=colors)
    
    for c in ax.containers:
        labels = [f'{v*100:.1f}%' if v > 0.05 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=11, color='white', fontweight='bold', padding=0)
        
    if orient == 'v':
        ax.axhline(y=0.5, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=0)
    else:
        ax.axvline(x=0.5, color='#fc4f30', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_xlabel("Proportion")

    ax.legend(title=crosstab.columns.name, bbox_to_anchor=(1.02, 1), loc='upper left')
    _add_title_subtitle(ax, f"Breakdown of {crosstab.columns.name} by {crosstab.index.name}", "100% Stacked comparison")

def plot_risk_by_bins(data, x_col, target_col, bins=10):
    ax = plt.gca()
    
    # --- SAFETY FIX: Work on a copy ---
    df = data.copy()
    
    if df[target_col].dtype == 'object':
        unique = df[target_col].unique()
        if 'low' in unique or 'high' in unique:
            df[target_col] = df[target_col].map({'low': 1, 'high': 0})

    try:
        df['bin'] = pd.qcut(df[x_col], q=bins, duplicates='drop')
    except ValueError:
        df['bin'] = pd.cut(df[x_col], bins=bins)
        
    risk_data = df.groupby('bin', observed=False)[target_col].mean()
    x_labels = [str(interval) for interval in risk_data.index]
    line_color = _get_random_color()
    
    sns.lineplot(x=x_labels, y=risk_data.values, marker='o', linewidth=3, color=line_color, ax=ax)
    ax.fill_between(x_labels, risk_data.values, color=line_color, alpha=0.1)
    
    ax.axhline(df[target_col].mean(), color='gray', linestyle='--', label='Avg Risk')
    ax.set_ylabel("Low Review Probability")
    ax.set_xlabel(f"{x_col.replace('_', ' ').title()} (Binned)")
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    _add_title_subtitle(ax, f"Risk Curve: {x_col}", "Probability change by value")

# ==========================================
# 4. MASTER DASHBOARDS
# ==========================================

def plot_describe_dashboard(data, group_col, value_col):
    """
    Visualizes stats (describe) in a 2x4 grid using the CURRENT Figure.
    IMPORTANT: Initialize figure size yourself before calling.
    """
    desc = data.groupby(group_col)[value_col].describe().round(2).reset_index()
    desc[group_col] = desc[group_col].astype(str)
    metrics = [col for col in desc.columns if col != group_col]

    # Get CURRENT figure
    fig = plt.gcf()
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    for i, metric in enumerate(metrics):
        row = 0 if i < 4 else 1
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        sns.barplot(data=desc, y=group_col, x=metric, ax=ax, width=0.6, 
                    palette=_get_random_palette(n_colors=len(desc)), edgecolor='white')
        
        for c in ax.containers:
            ax.bar_label(c, padding=5, fontsize=10, fontweight='bold', color='#333333')

        ax.set_title(metric.upper(), fontsize=14, fontweight='bold', loc='left', color='#333333')
        ax.set_xlabel('')
        if col == 0:
            ax.set_ylabel(group_col.replace('_', ' ').title(), fontweight='bold')
        else:
            ax.set_ylabel(''); ax.set_yticklabels([])

        # --- GRID DISABLED ---
        ax.grid(False)

    plt.suptitle(f"Statistical Summary: {value_col} by {group_col}", fontsize=22, fontweight='bold', x=0.01, ha='left', y=1.05)


def plot_numeric(data, col, target):
    """
    Master visualization: Numerical vs Categorical (KDE, Box, & Stats).
    IMPORTANT: Initialize figure size yourself before calling.
    """
    # --- SAFETY FIX: Copy data first ---
    plot_data = data.copy()
    
    plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    plot_data = plot_data.dropna(subset=[col, target])
    
    # Get CURRENT figure
    fig = plt.gcf()
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[3, 1, 1])
    main_palette = _get_random_palette() 

    # --- Row 1: Visuals ---
    ax_kde = fig.add_subplot(gs[0, :2])
    sns.kdeplot(data=plot_data, x=col, hue=target, fill=True, linewidth=2, 
                palette=main_palette, alpha=0.3, ax=ax_kde, warn_singular=False)
    ax_kde.set_title(f"Distribution: {col} by {target}", fontsize=14, fontweight='bold'); ax_kde.set_xlabel('')
    
    # --- GRID DISABLED ---
    ax_kde.grid(False) 

    ax_box = fig.add_subplot(gs[0, 2:])
    sns.boxplot(data=plot_data, x=col, y=target, orient='h', palette=main_palette, linewidth=1.5, ax=ax_box)
    ax_box.set_title(f"Outliers: {col} by {target}", fontsize=14, fontweight='bold'); ax_box.set_xlabel('')
    
    # --- GRID DISABLED ---
    ax_box.grid(False)

    # --- Row 2 & 3: Stats ---
    desc = plot_data.groupby(target)[col].describe().round(2).reset_index()
    desc[target] = desc[target].astype(str)
    metrics = [m for m in desc.columns if m != target]
    
    for i, metric in enumerate(metrics):
        grid_row = 1 if i < 4 else 2
        grid_col = i % 4
        ax = fig.add_subplot(gs[grid_row, grid_col])
        
        sns.barplot(data=desc, y=target, x=metric, ax=ax, palette=_get_random_palette(n_colors=len(desc)), edgecolor='white', width=0.6)
        for c in ax.containers:
            ax.bar_label(c, padding=3, fontsize=10, fontweight='bold', color='#333333')
            
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold', loc='left', color='#444444')
        ax.set_xlabel('')
        if grid_col == 0:
            ax.set_ylabel(target.replace('_', ' ').title(), fontweight='bold')
        else:
            ax.set_ylabel(''); ax.set_yticklabels([])
            
        # --- GRID DISABLED ---
        ax.grid(False)

    plt.suptitle(f"Deep Dive: {col} vs {target}", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()


def plot_categorical(data, col, target, order=None):
    """
    Master visualization: Categorical vs Categorical.
    Left: 100% Stacked Bar | Right: Heatmap (Count + %)
    IMPORTANT: Initialize figure size yourself before calling.
    """
    # Safe: dropna returns a new object
    plot_data = data.dropna(subset=[col, target])
    
    ct_counts = pd.crosstab(plot_data[col], plot_data[target])
    ct_props = pd.crosstab(plot_data[col], plot_data[target], normalize='index')

    if order:
        ct_counts = ct_counts.reindex(order)
        ct_props = ct_props.reindex(order)

    # Get CURRENT figure
    fig = plt.gcf()
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.02,
                           left=0.05, right=0.95, top=0.80, bottom=0.1)
    
    # --- Left: Stacked Bar ---
    ax_bar = fig.add_subplot(gs[0, 0])
    bar_colors = _get_random_palette(n_colors=len(ct_props.columns))
    
    ct_props.plot(kind='barh', stacked=True, width=0.85, edgecolor='white', linewidth=1, ax=ax_bar, color=bar_colors)
    ax_bar.invert_yaxis()
    
    for c in ax_bar.containers:
        labels = [f'{v*100:.1f}%' if v > 0.05 else '' for v in c.datavalues]
        ax_bar.bar_label(c, labels=labels, label_type='center', fontsize=11, color='white', fontweight='bold', padding=0)
    
    ax_bar.axvline(x=0.5, color='#333333', linestyle='--', alpha=0.5, linewidth=2)
    ax_bar.set_xlabel("Proportion", fontweight='bold'); ax_bar.set_ylabel('')
    ax_bar.legend(title=target.replace('_', ' ').title(), bbox_to_anchor=(1.0, 1.02), loc='lower right', ncols=4, frameon=False)
    ax_bar.set_title(f"Proportions: {col} vs {target}", fontsize=16, fontweight='bold', loc='left', y=1.02)
    
    # --- Right: Heatmap ---
    ax_heat = fig.add_subplot(gs[0, 1])
    grand_total = ct_counts.values.sum()
    heatmap_labels = ct_counts.applymap(lambda x: f"{x}\n({x/grand_total*100:.1f}%)")
    
    seq_palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGnBu"]
    sns.heatmap(ct_counts, annot=heatmap_labels, fmt='', cmap=random.choice(seq_palettes), 
                linewidths=2, linecolor='white', cbar=True, ax=ax_heat)
    
    ax_heat.set_ylabel(''); ax_heat.set_yticklabels([])
    ax_heat.set_xlabel(target.replace('_', ' ').title(), fontweight='bold')
    ax_heat.set_title("Sample Sizes (% Overall)", fontsize=16, fontweight='bold', loc='left', y=1.02)

    plt.suptitle(f"Categorical Analysis: {col} by {target}", fontsize=22, fontweight='bold', y=0.95, x=0.05, ha='left')


def plot_masked_categorical(data, col, mask_value, target, order=None):
    """
    Wrapper for plot_categorical that compares ONE value vs 'OTHERS'.
    IMPORTANT: Initialize figure size yourself before calling.
    """
    mask = data[col] == mask_value
    
    # --- SAFETY FIX: Ensure we are working on a copy ---
    temp_df = data.copy()
    
    comp_col = f"{col} Comparison" 
    temp_df[comp_col] = np.where(mask, str(mask_value), 'OTHERS')
    
    plot_categorical(temp_df, comp_col, target, order=order)