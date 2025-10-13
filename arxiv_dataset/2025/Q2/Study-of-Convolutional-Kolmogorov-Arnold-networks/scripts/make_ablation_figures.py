import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle # Kept if needed for future, not used in current plots
from matplotlib.colors import LinearSegmentedColormap
# from scipy.spatial.distance import pdist, squareform # Not used, can be removed
from sklearn.preprocessing import StandardScaler, MinMaxScaler # MinMaxScaler only for reference, using StandardScaler for radar
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for maximum aesthetic appeal
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl") # Using a seaborn built-in palette

# Custom color palettes (can be expanded or modified)
colors_custom = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D', 
    'grid4': '#FF6B6B',
    'grid8': '#4ECDC4',
    'grid16': '#45B7D1',
    'relu': '#96CEB4',
    'no_relu': '#FFEAA7'
}

# --- Font Size Definitions ---
# For Comprehensive Plot (remains as per previous good version)
STD_TITLE_FS = 26
STD_LABEL_FS = 24
STD_TEXT_FS = 22 # For ticks, legends, annotations

# For Individual Figures (significantly larger for paper readability)
# Base XXL sizes for Radar plot, as per user's "large large large" request
XXL_TITLE_FS = 48
XXL_LABEL_FS = 42
XXL_TEXT_FS = 38

# Adjusted XXL font sizes for non-radar individual plots (+3 points from XXL)
XXL_PLUS_TITLE_FS = XXL_TITLE_FS + 3    # 51
XXL_PLUS_LABEL_FS = XXL_LABEL_FS + 3    # 45
XXL_PLUS_TEXT_FS = XXL_TEXT_FS + 3      # 41


def load_and_process_data(csv_path):
    """Load CSV and add derived columns for analysis.
    Expects 'val_acc', 'flops', 'params', 'latency', 'use_relu', 'prune_amt', 'grid_size', 'width_mult' in CSV.
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['flops', 'params', 'latency', 'val_acc', 'use_relu', 'prune_amt', 'grid_size', 'width_mult']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")

    df['flops_millions'] = df['flops'] / 1e6
    df['params_thousands'] = df['params'] / 1e3
    df['latency_ms'] = df['latency'] * 1000
    df['accuracy_pct'] = df['val_acc'] * 100 
    
    df['config'] = df.apply(lambda x: f"{'ReLU' if x['use_relu'] else 'Identity'}_{'Pruned' if x['prune_amt'] > 0 else 'Full'}", axis=1)
    df['grid_label'] = df['grid_size'].apply(lambda x: f'Grid {x}')
    df['width_label'] = df['width_mult'].apply(lambda x: f'{x}x Width')
    
    return df

# --- Helper functions for individual plot components ---

def plot_component_grid_size_impact(df, ax, title_fs, label_fs, text_fs):
    if df.empty or 'grid_size' not in df.columns or 'accuracy_pct' not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Grid Size Impact", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Grid Size Impact on Accuracy (No Data)', fontsize=title_fs, fontweight='bold', pad=20) 
        return
        
    grid_means = df.groupby('grid_size')['accuracy_pct'].agg(['mean', 'std']).reset_index()
    bar_colors = [colors_custom['grid4'], colors_custom['grid8'], colors_custom['grid16']]
    
    for i, (_, row) in enumerate(grid_means.iterrows()):
        color = bar_colors[i % len(bar_colors)] 
        ax.bar(row['grid_size'], row['mean'], yerr=row['std'], 
               color=color, alpha=0.8, capsize=12, width=2.5,
               edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Grid Size (Spline Knots)', fontsize=label_fs, fontweight='bold') 
    ax.set_ylabel('Validation Accuracy (%)', fontsize=label_fs, fontweight='bold') 
    ax.set_title('Grid Size Impact on Accuracy', fontsize=title_fs, fontweight='bold', pad=30) 
    if not grid_means.empty: 
        min_y = (grid_means['mean'] - grid_means['std']).min()
        max_y = (grid_means['mean'] + grid_means['std']).max()
        ax.set_ylim(max(0, min_y - 0.5) , min(100, max_y + 0.5)) 
    else:
        ax.set_ylim(98,100) 
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(axis='both', which='major', labelsize=text_fs, pad=10)

    for i, (_, row) in enumerate(grid_means.iterrows()):
        ax.text(row['grid_size'], row['mean'] + row['std'] + 0.05, 
                f'{row["mean"]:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=text_fs) 

def plot_component_efficiency_frontier_basic(df, ax, title_fs, label_fs, text_fs):
    if df.empty or not all(col in df.columns for col in ['params_thousands', 'flops_millions', 'accuracy_pct', 'grid_size']):
        ax.text(0.5, 0.5, "Data unavailable for Efficiency Frontier", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Efficiency Frontier (No Data)', fontsize=title_fs, fontweight='bold', pad=20) 
        return

    sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 600 + 100 
    scatter = ax.scatter(df['flops_millions'], df['accuracy_pct'], 
                         c=df['grid_size'], s=sizes, alpha=0.7,
                         cmap='viridis', edgecolors='white', linewidth=2)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.05) 
    cbar.set_label('Grid Size', fontsize=label_fs, fontweight='bold') 
    cbar.ax.tick_params(labelsize=text_fs)

    ax.set_xlabel('FLOPs (Millions)', fontsize=label_fs, fontweight='bold') 
    ax.set_ylabel('Validation Accuracy (%)', fontsize=label_fs, fontweight='bold') 
    ax.set_title('Efficiency Frontier', fontsize=title_fs, fontweight='bold', pad=30) 
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(axis='both', which='major', labelsize=text_fs, pad=10) 

    if len(df['flops_millions'].dropna()) > 1 and len(df['accuracy_pct'].dropna()) > 1 : 
        valid_data = df[['flops_millions', 'accuracy_pct']].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['flops_millions'], valid_data['accuracy_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data['flops_millions'].min(), valid_data['flops_millions'].max(), 100)
            ax.plot(x_trend, p(x_trend), "--", 
                    color=colors_custom['accent'], alpha=0.8, linewidth=3) 

def plot_component_performance_heatmap_basic(df, ax, title_fs, label_fs, text_fs):
    if df.empty or not all(col in df.columns for col in ['accuracy_pct', 'use_relu', 'prune_amt', 'grid_size']):
        ax.text(0.5, 0.5, "Data unavailable for Performance Heatmap", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Performance Heatmap (No Data)', fontsize=title_fs, fontweight='bold', pad=20) 
        return
        
    pivot_data = df.pivot_table(values='accuracy_pct', 
                                index=['use_relu', 'prune_amt'], 
                                columns='grid_size', 
                                aggfunc='mean')
    if pivot_data.empty:
        ax.text(0.5, 0.5, "Not enough data diversity for Pivot Table", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Performance Heatmap (Pivot Empty)', fontsize=title_fs, fontweight='bold', pad=20) 
        return

    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    cmap = LinearSegmentedColormap.from_list('custom_basic_heatmap', colors_list, N=100)
    cmap.set_bad(color='lightgrey') 

    valid_pivot_values_basic = pivot_data.values[~np.isnan(pivot_data.values)]
    vmin_basic = valid_pivot_values_basic.min() if len(valid_pivot_values_basic) > 0 else 0
    vmax_basic = valid_pivot_values_basic.max() if len(valid_pivot_values_basic) > 0 else 100
    if vmin_basic == vmax_basic: 
        vmin_basic -= 0.1 
        vmax_basic += 0.1

    im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin_basic, vmax=vmax_basic)
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'Grid {x}' for x in pivot_data.columns], fontweight='bold', fontsize=text_fs) 
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f'{"ReLU" if x[0] else "Identity"} {"Pruned" if x[1] > 0 else "Full"}' 
                        for x in pivot_data.index], fontweight='bold', fontsize=text_fs) 
    
    mean_heatmap_val_basic = np.nanmean(pivot_data.values) if np.sum(~np.isnan(pivot_data.values)) > 0 else (vmin_basic + vmax_basic) / 2

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                text_color = 'black' if value > mean_heatmap_val_basic else 'white' 
                ax.text(j, i, f'{value:.2f}%', ha='center', va='center', 
                        fontweight='bold', color=text_color, fontsize=text_fs) 
    
    ax.set_title('Performance Heatmap', fontsize=title_fs, fontweight='bold', pad=30) 
    cbar = plt.colorbar(im, ax=ax, label='Accuracy (%)', shrink=0.8, pad=0.05) 
    cbar.set_label('Accuracy (%)', fontsize=label_fs, fontweight='bold') 
    cbar.ax.tick_params(labelsize=text_fs) 


def plot_component_pareto_frontier_bubble(df, ax, title_fs, label_fs, text_fs):
    if df.empty or not all(col in df.columns for col in ['params_thousands', 'config', 'latency_ms', 'accuracy_pct']):
        ax.text(0.5, 0.5, "Data unavailable for Pareto Frontier", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Speed vs Accuracy Trade-off (No Data)', fontsize=title_fs, fontweight='bold', pad=20) 
        return

    bubble_sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 800 + 150 
    config_colors_map = {
        'ReLU_Full': colors_custom['grid4'], 'ReLU_Pruned': colors_custom['grid8'], 
        'Identity_Full': colors_custom['grid16'], 'Identity_Pruned': colors_custom['accent']
    }
    
    for config_name in df['config'].unique():
        mask = df['config'] == config_name
        ax.scatter(df[mask]['latency_ms'], df[mask]['accuracy_pct'], 
                   s=bubble_sizes[mask], alpha=0.7, 
                   color=config_colors_map.get(config_name, '#000000'), 
                   label=config_name, edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Latency (ms)', fontsize=label_fs, fontweight='bold') 
    ax.set_ylabel('Validation Accuracy (%)', fontsize=label_fs, fontweight='bold') 
    ax.set_title('Speed vs Accuracy Trade-off (Bubble size = Parameters)', 
                 fontsize=title_fs, fontweight='bold', pad=30) 
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.legend(loc='lower right', fontsize=text_fs, framealpha=0.9) 
    ax.tick_params(axis='both', which='major', labelsize=text_fs, pad=10) 

    if not df.empty and df['accuracy_pct'].notna().any() and df['latency_ms'].notna().any():
        best_acc = df.loc[df['accuracy_pct'].idxmax()]
        fastest = df.loc[df['latency_ms'].idxmin()]
        
        ax.annotate(f'Best Acc\n{best_acc["accuracy_pct"]:.2f}%', 
                    xy=(best_acc['latency_ms'], best_acc['accuracy_pct']),
                    xytext=(best_acc['latency_ms'] + 1, best_acc['accuracy_pct'] - 0.3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3), 
                    fontsize=text_fs, fontweight='bold',  
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)) 
        
        ax.annotate(f'Fastest\n{fastest["latency_ms"]:.2f}ms', 
                    xy=(fastest['latency_ms'], fastest['accuracy_pct']),
                    xytext=(fastest['latency_ms'] + 1, fastest['accuracy_pct'] + 0.3),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=3), 
                    fontsize=text_fs, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7)) 

def plot_component_ablation_radar_chart(df, ax, title_fs, label_fs, text_fs):
    """Plots Top Configurations Radar Chart on a given Polar Axes object with reference styling."""
    required_metrics = ['accuracy_pct', 'latency_ms', 'flops_millions', 'params_thousands']
    if df.empty or not all(col in df.columns for col in required_metrics + ['grid_size', 'width_mult']):
        ax.text(0.5, 0.5, "Data unavailable for Radar Chart", ha='center', va='center', transform=ax.transAxes, fontsize=text_fs) 
        ax.set_title('Top Configurations Radar (No Data)', fontsize=title_fs, fontweight='bold', y=1.10, color='#212121') 
        ax.set_xticks([]) 
        ax.set_yticks([])
        ax.set_facecolor('#f8f9fa') 
        return

    ax.set_facecolor('#f8f9fa') 

    categories = ['Accuracy', 'Speed', 'Efficiency', 'Compactness'] 
    
    top_configs_df = df.nlargest(3, 'accuracy_pct')
    if top_configs_df.empty:
        ax.text(0.5, 0.5, "No top configurations to display", ha='center', va='center', transform=ax.transAxes, fontsize=text_fs) 
        ax.set_title('Top Configurations Radar (No Top Configs)', fontsize=title_fs, fontweight='bold', y=1.10, color='#212121') 
        return

    metrics_data_for_radar = top_configs_df[required_metrics].copy()
    # Original normalization: Invert metrics where lower is better
    metrics_data_for_radar['latency_ms'] = 1 / (metrics_data_for_radar['latency_ms'].replace(0, 1e-9) + 1e-9) 
    metrics_data_for_radar['flops_millions'] = 1 / (metrics_data_for_radar['flops_millions'].replace(0, 1e-9) + 1e-9)
    metrics_data_for_radar['params_thousands'] = 1 / (metrics_data_for_radar['params_thousands'].replace(0, 1e-9) + 1e-9)

    # Apply StandardScaler
    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(metrics_data_for_radar.fillna(0)) # Standardized values (z-scores)
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_plot = angles + angles[:1] 
    
    # Using your original color scheme for the series
    radar_colors_list = [colors_custom['primary'], colors_custom['secondary'], colors_custom['accent']] 
    
    all_values_plotted = [] # To determine y-axis limits from scaled data

    for i in range(len(top_configs_df)):
        values = metrics_scaled[i].tolist() # These are now z-scores
        values_for_plot = values + values[:1] 
        all_values_plotted.extend(values)
        config_row = top_configs_df.iloc[i]
                
        # Styling from reference snippet
        ax.plot(angles_plot, values_for_plot, color=radar_colors_list[i % len(radar_colors_list)], 
                linewidth=3, linestyle='solid', 
                label=f'Grid {config_row["grid_size"]}, {config_row["width_mult"]}x', 
                marker='o', markersize=9, markeredgecolor='black', markeredgewidth=0.5)
        ax.fill(angles_plot, values_for_plot, color=radar_colors_list[i % len(radar_colors_list)], alpha=0.25)

    # Styling from reference snippet
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles) 
    ax.set_xticklabels(categories, fontsize=text_fs, fontweight='normal', color='#333333')
    ax.tick_params(axis='x', which='major', pad=max(40, text_fs * 0.8)) # Significantly increased padding

    # Dynamic y-axis for StandardScaler output
    if all_values_plotted:
        min_val = np.min(all_values_plotted)
        max_val = np.max(all_values_plotted)
        # Create about 5 ticks, ensuring 0 is included if data spans it
        abs_max = max(abs(min_val), abs(max_val))
        y_ticks = np.linspace(-abs_max, abs_max, 5) # Symmetric ticks around 0 if data spans it
        if min_val >= 0: # If all positive z-scores (unlikely but possible)
            y_ticks = np.linspace(0, max_val, 5)
        elif max_val <=0: # If all negative z-scores
             y_ticks = np.linspace(min_val, 0, 5)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{val:.1f}" for val in y_ticks], fontsize=max(16, int(text_fs * 0.6)), color='dimgray')
        ax.set_ylim(y_ticks.min() - 0.1 * abs(y_ticks.min()), y_ticks.max() + 0.1 * abs(y_ticks.max()))
    else: 
        ax.set_yticks(np.arange(-2, 2.1, 1)) # Default z-score like range
        ax.set_yticklabels([f"{i}" for i in np.arange(-2, 2.1, 1)], fontsize=max(16, int(text_fs * 0.6)), color='dimgray')
        ax.set_ylim(-2.5, 2.5)
    
    ax.set_rlabel_position(0) 
    ax.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7) 
    ax.set_title('Top Configurations Radar', fontsize=title_fs, color='#212121', y=1.10, fontweight='bold') 

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=text_fs, 
              frameon=True, facecolor='white', framealpha=0.85, edgecolor='darkgray') 
    
def plot_component_parameter_scaling_analysis(df, ax, title_fs, label_fs, text_fs):
    if df.empty or not all(col in df.columns for col in ['width_mult', 'grid_size', 'params_thousands']):
        ax.text(0.5, 0.5, "Data unavailable for Parameter Scaling", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Parameter Scaling Analysis (No Data)', fontsize=title_fs, fontweight='bold', pad=20) 
        return

    width_groups = df.groupby(['width_mult', 'grid_size'])['params_thousands'].mean().unstack()
    if width_groups.empty:
        ax.text(0.5, 0.5, "Not enough data diversity for Parameter Scaling", ha='center', va='center', fontsize=text_fs) 
        ax.set_title('Parameter Scaling Analysis (Pivot Empty)', fontsize=title_fs, fontweight='bold', pad=20) 
        return

    x_labels = [f'{w}x' for w in width_groups.index]
    x_pos = np.arange(len(x_labels))
    bar_width = 0.6 
    
    bottom = np.zeros(len(width_groups.index))
    grid_colors_list = [colors_custom['grid4'], colors_custom['grid8'], colors_custom['grid16']]
    
    for i, (grid_size_val, color) in enumerate(zip(width_groups.columns, grid_colors_list)):
        values = width_groups[grid_size_val].fillna(0) 
        ax.bar(x_pos, values, width=bar_width, bottom=bottom, 
               label=f'Grid {grid_size_val}', color=color, alpha=0.8,
               edgecolor='white', linewidth=1)
        bottom += values
    
    ax.set_xlabel('Width Multiplier', fontsize=label_fs, fontweight='bold') 
    ax.set_ylabel('Parameters (Thousands)', fontsize=label_fs, fontweight='bold') 
    ax.set_title('Parameter Scaling Analysis', fontsize=title_fs, fontweight='bold', pad=30) 
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontweight='bold', fontsize=text_fs) 
    ax.legend(fontsize=text_fs, framealpha=0.9)  
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(axis='both', which='major', labelsize=text_fs, pad=10) 
    
    for i, (idx, row) in enumerate(width_groups.iterrows()): 
        total = row.sum()
        if total > 0 : 
             ax.text(x_pos[i], total + 5, f'{total:.0f}K', ha='center', va='bottom', fontweight='bold', fontsize=text_fs) 

def create_stunning_plots(df, output_filepath_prefix):
    fig = plt.figure(figsize=(24, 22)) 
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], 
                          hspace=0.6, wspace=0.4) 
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_component_grid_size_impact(df, ax1, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_component_efficiency_frontier_basic(df, ax2, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_component_performance_heatmap_basic(df, ax3, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS)
    ax4 = fig.add_subplot(gs[1, :])
    plot_component_pareto_frontier_bubble(df, ax4, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS)
    ax5 = fig.add_subplot(gs[2, 0], projection='polar')
    plot_component_ablation_radar_chart(df, ax5, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS) # Uses STD fonts for comprehensive
    ax6 = fig.add_subplot(gs[2, 1:]) 
    plot_component_parameter_scaling_analysis(df, ax6, STD_TITLE_FS, STD_LABEL_FS, STD_TEXT_FS)
    
    fig.suptitle('FastKAN Ablation Study - Comprehensive Analysis', 
                 fontsize=STD_TITLE_FS, fontweight='bold', y=0.99) 
    fig.patch.set_facecolor('#fafafa')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    save_path_png = f'{output_filepath_prefix}_comprehensive.png'
    save_path_pdf = f'{output_filepath_prefix}_comprehensive.pdf'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")
    plt.close(fig)

def save_individual_component_figures(df, output_dir, base_filename_prefix):
    component_plotters = {
        "grid_impact": (plot_component_grid_size_impact, {"figsize": (20,18)}), 
        "efficiency_basic": (plot_component_efficiency_frontier_basic, {"figsize": (20,18)}), 
        "heatmap_basic": (plot_component_performance_heatmap_basic, {"figsize": (20,18)}), 
        "pareto_frontier": (plot_component_pareto_frontier_bubble, {"figsize": (22,18)}), 
        "radar_chart": (plot_component_ablation_radar_chart, {"projection": "polar", "figsize": (24,24)}),
        "parameter_scaling": (plot_component_parameter_scaling_analysis, {"figsize": (20,16)}) 
    }

    for name, (plot_func, kwargs) in component_plotters.items():
        fig_size = kwargs.get("figsize", (20, 18)) 
        projection = kwargs.get("projection")
        
        fig = plt.figure(figsize=fig_size)
        fig.patch.set_facecolor('#ffffff') 

        if projection == "polar":
            ax = fig.add_subplot(111, projection='polar')
            # ax.set_facecolor('#f8f9fa') # Set inside radar plot function
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor('#f8f9fa') 
        
        if name == "radar_chart":
             plot_func(df, ax, XXL_TITLE_FS, XXL_LABEL_FS, XXL_TEXT_FS) 
        else:
             plot_func(df, ax, XXL_PLUS_TITLE_FS, XXL_PLUS_LABEL_FS, XXL_PLUS_TEXT_FS) 
        
        plt.tight_layout(pad=3.0) 
        filepath_prefix = os.path.join(output_dir, f"{base_filename_prefix}_{name}")
        save_path_png = f'{filepath_prefix}.png'
        save_path_pdf = f'{filepath_prefix}.pdf'
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
        plt.close(fig)

def create_special_individual_plots(df, output_filepath_prefix):
    if df.empty:
        print("Warning: DataFrame is empty. Skipping special individual plots.")
        return

    # 1. SEXY EFFICIENCY PLOT (Special Version)
    fig_eff, ax_eff = plt.subplots(figsize=(26, 22)) 
    fig_eff.patch.set_facecolor('#ffffff')
    ax_eff.set_facecolor('#f8f9fa')

    if not all(col in df.columns for col in ['params_thousands', 'flops_millions', 'accuracy_pct', 'grid_size']):
        ax_eff.text(0.5,0.5, "Data missing for Special Efficiency Plot", ha='center', va='center', fontsize=XXL_PLUS_TEXT_FS) 
    else:
        sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 800 + 150 
        scatter = ax_eff.scatter(df['flops_millions'], df['accuracy_pct'], 
                                 c=df['grid_size'], s=sizes, alpha=0.8,
                                 cmap='plasma', edgecolors='white', linewidth=3) 
        
        if len(df['flops_millions'].dropna()) > 1 and len(df['accuracy_pct'].dropna()) > 1:
            valid_data = df[['flops_millions', 'accuracy_pct']].dropna()
            if len(valid_data)>1:
                z = np.polyfit(valid_data['flops_millions'], valid_data['accuracy_pct'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data['flops_millions'].min(), valid_data['flops_millions'].max(), 100)
                ax_eff.plot(x_trend, p(x_trend), "--", color='red', alpha=0.8, linewidth=4, label='Trend') 
        
        cbar = plt.colorbar(scatter, ax=ax_eff, pad=0.05) 
        cbar.set_label('Grid Size (Spline Knots)', fontsize=XXL_PLUS_LABEL_FS, fontweight='bold') 
        cbar.ax.tick_params(labelsize=XXL_PLUS_TEXT_FS) 
        
        if 'flops_millions' in df.columns and df['flops_millions'].notna().any() and (df['flops_millions'].abs() > 1e-9).any(): 
            efficiency_ratio = df['accuracy_pct'] / (df['flops_millions'].replace(0, 1e-9) + 1e-9) 
            idx_best_efficiency = efficiency_ratio.idxmax()
            best_efficiency_row = df.loc[idx_best_efficiency]
            
            ax_eff.annotate('Best Efficiency', 
                            xy=(best_efficiency_row['flops_millions'], best_efficiency_row['accuracy_pct']),
                            xytext=(best_efficiency_row['flops_millions'] * 1.2 + 0.1, best_efficiency_row['accuracy_pct'] - 0.2), 
                            arrowprops=dict(arrowstyle='->', color='gold', lw=4), 
                            fontsize=XXL_PLUS_TEXT_FS, fontweight='bold', 
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='gold', alpha=0.7)) 

    ax_eff.set_xlabel('FLOPs (Millions)', fontsize=XXL_PLUS_LABEL_FS, fontweight='bold') 
    ax_eff.set_ylabel('Validation Accuracy (%)', fontsize=XXL_PLUS_LABEL_FS, fontweight='bold') 
    ax_eff.set_title('FastKAN Efficiency Frontier \nBubble size = Parameters', 
                     fontsize=XXL_PLUS_TITLE_FS, fontweight='bold', pad=30) 
    ax_eff.grid(True, alpha=0.3)
    ax_eff.tick_params(axis='both', which='major', labelsize=XXL_PLUS_TEXT_FS, pad=15) 
    plt.tight_layout(pad=3.0) 
    save_path_png_eff = f'{output_filepath_prefix}_efficiency_frontier_special.png'
    save_path_pdf_eff = f'{output_filepath_prefix}_efficiency_frontier_special.pdf'
    plt.savefig(save_path_png_eff, dpi=300, bbox_inches='tight', facecolor=fig_eff.get_facecolor())
    plt.savefig(save_path_pdf_eff, bbox_inches='tight', facecolor=fig_eff.get_facecolor())
    print(f"Saved: {save_path_png_eff}")
    print(f"Saved: {save_path_pdf_eff}")
    plt.close(fig_eff) 
    
    # 2. GORGEOUS HEATMAP (Special Version)
    fig_hm, ax_hm = plt.subplots(figsize=(28, 24)) 
    fig_hm.patch.set_facecolor('#ffffff')
    ax_hm.set_facecolor('#f8f9fa')

    if not all(col in df.columns for col in ['accuracy_pct', 'grid_size', 'width_mult', 'use_relu', 'prune_amt']):
        ax_hm.text(0.5,0.5, "Data missing for Special Heatmap", ha='center', va='center', fontsize=XXL_PLUS_TEXT_FS) 
    else:
        pivot_full = df.pivot_table(values='accuracy_pct', 
                                    index=['grid_size', 'width_mult'], 
                                    columns=['use_relu', 'prune_amt'], 
                                    aggfunc='mean')
        if pivot_full.empty:
            ax_hm.text(0.5,0.5, "Not enough data diversity for Special Heatmap pivot", ha='center', va='center', fontsize=XXL_PLUS_TEXT_FS) 
        else:
            colors_list_hm = ['#FF416C', '#FF4B2B', '#FF8E53', '#FF6B6B', '#4ECDC4', '#45B7D1']
            cmap_hm = LinearSegmentedColormap.from_list('fastkan_special', colors_list_hm, N=256)
            cmap_hm.set_bad(color='lightgrey') 

            valid_pivot_full_values = pivot_full.values[~np.isnan(pivot_full.values)]
            vmin_special = valid_pivot_full_values.min() if len(valid_pivot_full_values) > 0 else 0
            vmax_special = valid_pivot_full_values.max() if len(valid_pivot_full_values) > 0 else 100
            if vmin_special == vmax_special: 
                vmin_special -= 0.1
                vmax_special += 0.1

            im = ax_hm.imshow(pivot_full.values, cmap=cmap_hm, aspect='auto', interpolation='nearest', vmin=vmin_special, vmax=vmax_special)
            
            ax_hm.set_xticks(range(len(pivot_full.columns)))
            ax_hm.set_xticklabels([f'{"ReLU" if x[0] else "ID"}\n{"Pruned" if x[1] > 0 else "Full"}' 
                                   for x in pivot_full.columns], fontweight='bold', fontsize=XXL_PLUS_TEXT_FS) 
            ax_hm.set_yticks(range(len(pivot_full.index)))
            ax_hm.set_yticklabels([f'Grid {x[0]}\n{x[1]}x Width' for x in pivot_full.index], fontweight='bold', fontsize=XXL_PLUS_TEXT_FS) 
            
            mean_pivot_full_val = np.nanmean(pivot_full.values) if np.sum(~np.isnan(pivot_full.values)) > 0 else (vmin_special + vmax_special) / 2

            for i in range(len(pivot_full.index)):
                for j in range(len(pivot_full.columns)):
                    value = pivot_full.values[i, j]
                    if not np.isnan(value):
                        text_color = 'black' if value > mean_pivot_full_val else 'white'
                        ax_hm.text(j, i, f'{value:.2f}%', ha='center', va='center', 
                                   fontweight='bold', color=text_color, fontsize=XXL_PLUS_TEXT_FS) 
            
            cbar_hm = plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.05) 
            cbar_hm.set_label('Validation Accuracy (%)', fontsize=XXL_PLUS_LABEL_FS, fontweight='bold') 
            cbar_hm.ax.tick_params(labelsize=XXL_PLUS_TEXT_FS) 

    ax_hm.set_title('FastKAN Performance Heatmap \nValidation Accuracy Across All Configurations', 
                     fontsize=XXL_PLUS_TITLE_FS, fontweight='bold', pad=30) 
    ax_hm.tick_params(axis='both', which='major', labelsize=XXL_PLUS_TEXT_FS, pad=15) 
    plt.tight_layout(pad=3.0) 
    save_path_png_hm = f'{output_filepath_prefix}_heatmap_special.png'
    save_path_pdf_hm = f'{output_filepath_prefix}_heatmap_special.pdf'
    plt.savefig(save_path_png_hm, dpi=300, bbox_inches='tight', facecolor=fig_hm.get_facecolor())
    plt.savefig(save_path_pdf_hm, bbox_inches='tight', facecolor=fig_hm.get_facecolor())
    print(f"Saved: {save_path_png_hm}")
    print(f"Saved: {save_path_pdf_hm}")
    plt.close(fig_hm)

# --- Main execution ---
if __name__ == "__main__":
    output_directory = "mnist_ablate_results"
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = 'mnist_ablate_results/results.csv' 
    df_for_plotting = None 

    if not os.path.exists(csv_file_path):
        print(f"Warning: CSV file not found at '{os.path.abspath(csv_file_path)}'.")
        print("Creating a dummy 'results.csv' for demonstration purposes.")
        data = {
            'grid_size': np.random.choice([4, 8, 16], 20),
            'width_mult': np.random.choice([1, 2, 4], 20),
            'use_relu': np.random.choice([True, False], 20),
            'prune_amt': np.random.choice([0, 0.1, 0.25], 20),
            'val_loss': np.random.rand(20) * 0.05, 
            'val_acc': np.random.uniform(0.98, 0.999, 20), 
            'params': np.random.randint(10000, 100000, 20),
            'flops': np.random.randint(1000000, 10000000, 20),
            'latency': np.random.rand(20) * 0.01 
        }
        dummy_df = pd.DataFrame(data)
        try:
            dummy_df.to_csv(csv_file_path, index=False)
            print(f"Dummy 'results.csv' created at '{os.path.abspath(csv_file_path)}'.")
            print("Please replace it with your actual data for meaningful results or run the script again.")
        except IOError as e:
            print(f"Error: Could not write dummy CSV file to '{os.path.abspath(csv_file_path)}': {e}")
            print("Please check directory permissions. Exiting.")
            exit()
    
    try:
        df_for_plotting = load_and_process_data(csv_file_path)
        print(f"Successfully loaded and processed data from '{os.path.abspath(csv_file_path)}'.")
    except FileNotFoundError:
        print(f"Critical Error: CSV file '{os.path.abspath(csv_file_path)}' not found.")
        print("Exiting.")
        exit()
    except ValueError as ve: 
        print(f"Error processing CSV data: {ve}")
        print("Exiting.")
        exit()
    except Exception as e: 
        print(f"An unexpected error occurred while loading or processing '{os.path.abspath(csv_file_path)}': {e}")
        print("Please check the CSV file's format and content.")
        print("Exiting.")
        exit()

    if df_for_plotting.empty:
        print(f"Warning: The DataFrame loaded from '{os.path.abspath(csv_file_path)}' is empty. "
              "Plots might be empty or incorrect. Ensure the CSV contains data.")
    
    base_filename = "fastkan_ablation"
    output_filepath_prefix_main = os.path.join(output_directory, base_filename)

    print(f"\n--- Generating Comprehensive Plot ---")
    create_stunning_plots(df_for_plotting, output_filepath_prefix_main)
    
    print(f"\n--- Generating Individual Component Figures ---")
    save_individual_component_figures(df_for_plotting, output_directory, base_filename)
    
    print(f"\n--- Generating Special Individual Plots ---")
    create_special_individual_plots(df_for_plotting, output_filepath_prefix_main)
    
    print("\n🎉 All plots created successfully!")
    print(f"📁 Files saved in: {os.path.abspath(output_directory)}")

