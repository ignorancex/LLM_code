import numpy as np
import time
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
import reproject
import dfreproject
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import cmcrameri.cm as cmc



def generate_test_data(size, include_sip=True):
    """
    Generate test data and WCS objects for benchmarking.

    Parameters:
    -----------
    size : tuple
        Image size (width, height)
    include_sip : bool
        Whether to include SIP distortion coefficients
    """
    # Create input array
    input_array = np.random.random(size)

    # Create input WCS
    input_wcs = WCS(naxis=2)
    input_wcs.wcs.crpix = [size[0] // 2, size[1] // 2]
    input_wcs.wcs.cdelt = np.array([-0.0002777777777778, 0.0002777777777778])
    input_wcs.wcs.crval = [269.99999999999994, 65.00000000000014]
    input_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Add SIP distortion coefficients if requested
    if include_sip:
        # Convert to TAN-SIP type
        input_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

        # Add SIP coefficients to the WCS header directly
        header = input_wcs.to_header(relax=True)

        # A_ORDER and B_ORDER specify the polynomial order (2 in this case)
        header['A_ORDER'] = 2
        header['B_ORDER'] = 2

        # Add the polynomial coefficients
        header['A_0_2'] = 2.0e-7
        header['A_1_1'] = 4.0e-7
        header['A_2_0'] = 5.0e-7
        header['A_1_0'] = 3.0e-7
        header['A_0_1'] = 1.0e-7

        header['B_0_2'] = 1.5e-7
        header['B_1_1'] = 2.5e-7
        header['B_2_0'] = 3.5e-7
        header['B_1_0'] = 1.0e-7
        header['B_0_1'] = 2.5e-7

        # Create a new WCS from this header
        input_wcs = WCS(header, relax=True)

    # Create output WCS (slightly offset)
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.crpix = [size[0] // 2 + 10, size[1] // 2 - 10]
    output_wcs.wcs.cdelt = np.array([-0.0002777777777778, 0.0002777777777778])
    output_wcs.wcs.crval = [270.05, 65.05]
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Add SIP distortion coefficients to output WCS if requested
    if include_sip:
        # Convert to TAN-SIP type
        output_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

        # Add SIP coefficients to the WCS header directly
        header = output_wcs.to_header(relax=True)

        # A_ORDER and B_ORDER specify the polynomial order (2 in this case)
        header['A_ORDER'] = 2
        header['B_ORDER'] = 2

        # Add the polynomial coefficients (slightly different from input)
        header['A_0_2'] = 1.8e-7
        header['A_1_1'] = 3.8e-7
        header['A_2_0'] = 4.7e-7
        header['A_1_0'] = 2.8e-7
        header['A_0_1'] = 1.2e-7

        header['B_0_2'] = 1.7e-7
        header['B_1_1'] = 2.3e-7
        header['B_2_0'] = 3.3e-7
        header['B_1_0'] = 1.2e-7
        header['B_0_1'] = 2.3e-7

        # Create a new WCS from this header
        output_wcs = WCS(header, relax=True)

    return input_array, input_wcs, output_wcs, size


def benchmark_reproject(input_array, input_wcs, output_wcs, output_shape, order):
    """
    Benchmark the reproject package.

    Parameters:
    -----------
    order : str
        Interpolation order: 'nearest', 'bilinear', or 'bicubic'
    """
    # Map interpolation names to reproject's order parameter
    order_map = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3
    }

    start_time = time.time()
    output_array, footprint = reproject.reproject_interp(
        (input_array, input_wcs), output_wcs, shape_out=output_shape,
        order=order_map[order])
    end_time = time.time()
    return end_time - start_time


def benchmark_dfreproject(input_array, input_wcs, output_wcs, output_shape, interp_mode, device):
    """
    Benchmark the dfreproject package.

    Parameters:
    -----------
    interp_mode : str
        Interpolation mode: 'nearest', 'bilinear', or 'bicubic'
    """
    # Create an HDU object containing both data and header
    hdu = fits.PrimaryHDU(data=input_array, header=input_wcs.to_header())

    start_time = time.time()
    output_array = dfreproject.calculate_reprojection(
        hdu, output_wcs, shape_out=output_shape, order=interp_mode, converve_flux=True, device=device)
    end_time = time.time()
    return end_time - start_time


def run_benchmarks(image_sizes, interpolation_methods, num_trials=2, device='cpu'):
    """
    Run benchmarks for both packages on all image sizes and interpolation methods.

    Parameters:
    -----------
    image_sizes : list
        List of image sizes to benchmark
    interpolation_methods : list
        List of interpolation methods to benchmark
    num_trials : int
        Number of trials for each configuration
    """
    results = []

    # Run for both with and without SIP distortion
    for sip in [True, False]:
        sip_label = "with SIP" if sip else "without SIP"

        for size in image_sizes:
            if isinstance(size, tuple):
                size_label = f"{size[0]}x{size[1]}"
                shape = size
            else:
                size_label = f"{size}x{size}"
                shape = (size, size)

            for interp in interpolation_methods:
                print(f"Benchmarking {size_label} {sip_label} with {interp} interpolation...")

                reproject_times = []
                dfreproject_times = []

                for trial in range(num_trials):
                    print(f"  Trial {trial + 1}/{num_trials}")
                    input_array, input_wcs, output_wcs, output_shape = generate_test_data(shape, include_sip=sip)

                    # Run reproject benchmark
                    reproject_time = benchmark_reproject(input_array, input_wcs, output_wcs, output_shape, interp)
                    reproject_times.append(reproject_time)

                    # Run dfreproject benchmark
                    dfreproject_time = benchmark_dfreproject(input_array, input_wcs, output_wcs, output_shape, interp, device)
                    dfreproject_times.append(dfreproject_time)

                # Calculate average times
                avg_reproject_time = np.mean(reproject_times)
                avg_dfreproject_time = np.mean(dfreproject_times)

                # Calculate speedup
                speedup = avg_reproject_time / avg_dfreproject_time if avg_dfreproject_time > 0 else float('inf')

                results.append({
                    'Image Size': size_label,
                    'Interpolation': interp,
                    'SIP': sip_label,
                    'Reproject Time (s)': avg_reproject_time,
                    'DFReproject Time (s)': avg_dfreproject_time,
                    'Speedup': speedup
                })

    return results


def display_results(results, device):
    """Display benchmark results in tables and graphs."""
    # Create DataFrame
    df = pd.DataFrame(results)

    # Print table
    print("\nBenchmark Results:")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))

    # Save detailed results to CSV
    df.to_csv(f'reproject_benchmark_detailed_results_{device}.csv', index=False)

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

    # Create summary table grouped by SIP usage
    print("\nSummary by SIP Distortion:")
    sip_summary = df.groupby('SIP').agg({
        'Speedup': ['mean', 'min', 'max'],
        'Reproject Time (s)': 'mean',
        'DFReproject Time (s)': 'mean'
    })
    print(tabulate(sip_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))

    # Create visualizations
    create_visualizations(df, device)

    return df


def create_visualizations(df, device):
    """Create various visualizations of the benchmark results."""
    # Set the style
    sns.set(style="whitegrid")

    # Create combined category for faceted plots
    df['Size_Interp'] = df['Image Size'] + '\n' + df['Interpolation']
    df['Size_Interp_SIP'] = df['Image Size'] + '\n' + df['Interpolation'] + '\n' + df['SIP']

    # 1. Bar chart comparing execution times with SIP as groups
    plt.figure(figsize=(16, 10))

    # Sort by Image Size, Interpolation, and SIP
    df_sorted = df.sort_values(['Image Size', 'Interpolation', 'SIP'])

    # Create the grouped bar chart
    g = sns.catplot(
        data=df_sorted,
        kind="bar",
        x="Size_Interp", y="Reproject Time (s)",
        hue="SIP", col="Interpolation",
        height=5, aspect=1.2, palette="viridis",
        legend=True, legend_out=True
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{col_name}")
    g.fig.suptitle('Reproject Execution Time by Interpolation Method & SIP', y=1.05)
    g.set_xlabels('Image Size')
    g.set_ylabels('Execution Time (seconds)')
    plt.tight_layout()
    plt.savefig('reproject_execution_times_by_sip.png', dpi=300, bbox_inches='tight')

    # 2. Bar chart for DFReproject with SIP as groups
    plt.figure(figsize=(16, 10))
    g = sns.catplot(
        data=df_sorted,
        kind="bar",
        x="Size_Interp", y="DFReproject Time (s)",
        hue="SIP", col="Interpolation",
        height=5, aspect=1.2, palette="mako",
        legend=True, legend_out=True
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{col_name}")
    g.fig.suptitle('DFReproject Execution Time by Interpolation Method & SIP', y=1.05)
    g.set_xlabels('Image Size')
    g.set_ylabels('Execution Time (seconds)')
    plt.tight_layout()
    plt.savefig('dfreproject_execution_times_by_sip.png', dpi=300, bbox_inches='tight')

    # 3. Heatmap of speedup factors with SIP
    plt.figure(figsize=(14, 10))

    # Create two separate heatmaps for with/without SIP
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, sip_value in enumerate(df['SIP'].unique()):
        sip_data = df[df['SIP'] == sip_value]

        # Reshape data for heatmap
        heatmap_data = sip_data.pivot_table(
            index='Interpolation',
            columns='Image Size',
            values='Speedup'
        )

        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmc.navia,
                    linewidths=.5, ax=axes[i],
                    cbar_kws={'label': 'Speedup Factor'})

        axes[i].set_title(f'Speedup Factor {sip_value}')

    plt.tight_layout()
    plt.savefig('reproject_speedup_heatmap_by_sip.png', dpi=300, bbox_inches='tight')

    # 4. Line plot showing scaling with image size, grouped by SIP
    plt.figure(figsize=(14, 10))

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

            plt.plot(filter_data['Image Size'], filter_data['Speedup'],
                     marker='o', linewidth=2,
                     label=f"{interp} ({sip})")

    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Image Size')
    plt.ylabel('Speedup Factor (reproject/dfreproject)')
    plt.title('Speedup Factor by Image Size, Interpolation Method, and SIP Distortion')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'reproject_speedup_by_size_and_sip_{device}.png', dpi=300, bbox_inches='tight')

    # 5. Box plot of speedup by SIP and interpolation
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Interpolation', y='Speedup', hue='SIP', data=df, palette='Set3')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Speedup by Interpolation Method and SIP')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reproject_speedup_boxplot.png', dpi=300, bbox_inches='tight')

    plt.close('all')


def main():
    device = 'cuda'
    # Image sizes to benchmark
    # image_sizes = [
    #     #(256, 256),
    #     #(512, 512),
    #     #(1024, 1024),
    #     (4000, 6000)
    # ]
    aspect_ratios = [1.0, 1.2, 1.5]
    num_sizes = 1

    must_have_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4000, 6000)]

    # Generate widths and heights with varying aspect ratios
    widths = np.geomspace(256, 4000, num=num_sizes).astype(int)
    widths = np.unique(widths)

    # Build initial image sizes from aspect ratios
    image_sizes = []
    for i, w in enumerate(widths):
        ar = aspect_ratios[i % len(aspect_ratios)]
        h = int(w * ar)
        image_sizes.append((w, h))

    # Combine and deduplicate with must-have sizes
    image_sizes.extend(must_have_sizes)
    image_sizes = list(set(image_sizes))  # remove duplicates
    image_sizes.sort()  # optional: sort for consistency

    # Interpolation methods to benchmark
    interpolation_methods = [
        'nearest',
        'bilinear',
        'bicubic'
    ]

    # Number of trials for each configuration
    num_trials = 2

    # Run benchmarks
    print("Starting benchmarks...")
    results = run_benchmarks(image_sizes, interpolation_methods, num_trials, device)

    # Display results
    display_results(results, device)

    print("\nBenchmark complete. Results saved to CSV and PNG files.")


if __name__ == "__main__":
    main()
