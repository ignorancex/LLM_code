import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import cmcrameri.cm as cmc  # Import the cmcrameri colormaps

def load_benchmark_results(csv_file):
    """Load benchmark results from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} benchmark results.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def create_summary_tables(df):
    """Create and print summary tables of the benchmark results."""
    # Print overall summary
    print("\nBenchmark Results Summary:")
    print(tabulate(df.describe(), headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by interpolation method
    print("\nSummary by Interpolation Method:")
    interp_summary = df.groupby('Interpolation').agg({
        'Speedup': ['mean', 'min', 'max']
    })
    print(tabulate(interp_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by image size
    print("\nSummary by Image Size:")
    size_summary = df.groupby('Image Size').agg({
        'Speedup': ['mean', 'min', 'max']
    })
    print(tabulate(size_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by SIP usage if it exists
    if 'SIP' in df.columns:
        print("\nSummary by SIP Distortion:")
        sip_summary = df.groupby('SIP').agg({
            'Speedup': ['mean', 'min', 'max'],
            'Reproject Time (s)': 'mean',
            'DFReproject Time (s)': 'mean'
        })
        print(tabulate(sip_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))





def plot_sip_comparison(df, name, output_file_prefix='sip_comparison'):
    """Create visualizations comparing performance with and without SIP."""
    if 'SIP' not in df.columns:
        print("SIP column not found in data. Skipping SIP comparison plots.")
        return

    if name == 'cuda':
        name = 'gpu'
    # 3. Line plot with SIP as groups
    plt.figure(figsize=(14, 10))

    interp_line_styles = ['-', '--']
    # Create a grouped line plot for each SIP + interpolation combination
    for sip in df['SIP'].unique():
        sip_data = df[df['SIP'] == sip]
        
        for interp in sip_data['Interpolation'].unique():
            filter_data = sip_data[sip_data['Interpolation'] == interp]
            
            # Convert image size to numeric for proper scaling
            filter_data['Size_Numeric'] = filter_data['Image Size'].apply(
                lambda x: int(x.split('x')[0]) * int(x.split('x')[1])
            )
            
            # Sort by numeric size
            filter_data = filter_data.sort_values('Size_Numeric')
            
            # Calculate color index for gradient
            interp_idx = list(df['Interpolation'].unique()).index(interp)
            sip_idx = list(df['SIP'].unique()).index(sip)
            color_idx = (interp_idx * len(df['SIP'].unique()) + sip_idx) / (len(df['Interpolation'].unique()) * len(df['SIP'].unique()) - 1)
            
            plt.plot(filter_data['Image Size'], filter_data['Speedup'], 
                    marker='o', linewidth=2, 
                    label=f"{interp} ({sip})",
                    color=cmc.berlin(color_idx),
                     linestyle=interp_line_styles[sip_idx])
    desired_ticks = ['256x256', '512x512', '1024x1024', '2048x2048', '4000x6000']
    plt.xticks(ticks=desired_ticks, labels=desired_ticks, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim(3, 50)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Image Size', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('Speedup Factor (dfreproject/reproject)' , fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title(f'Speedup Factor for {name.upper()}', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_line_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SIP comparison plots saved with prefix {output_file_prefix}")

def main():
    # Get the CSV file from user
    name = 'cuda'
    csv_file = f"./reproject_benchmark_detailed_results_{name}.csv"#input("Enter the path to the benchmark results CSV file: ")
    
    # Load the benchmark results
    df = load_benchmark_results(csv_file)
    if df is None:
        return
    
    # Create summary tables
    create_summary_tables(df)

    # Generate SIP comparison plots if applicable
    plot_sip_comparison(df, name)
    
    print("\nVisualization complete. All plots saved to current directory.")

if __name__ == "__main__":
    main()
