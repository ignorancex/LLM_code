#!/usr/bin/env python3
"""
Simple visualization script for Energy Correlator outputs.
This script reads the histogram data produced by write_pencs and creates plots to visualize the results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def load_histogram(filename):
    """Load histogram data from the given file."""
    # Initialize variables
    data = {}

    # Read the file
    with open(filename, 'r') as f:
        content = f.read()

    # Extract variables by running the content as Python code
    exec(content, globals(), data)

    return data


def plot_re3c(filename):
    # Create a 2D plot
    plt.figure(figsize=(10, 6))

    data = load_histogram(filename)

    hist = np.array(data['hist'][1:-1][:])
    R1 = data['theta1_edges'][1:-1]
    R2_R1 = data['theta2_over_theta1_edges']

    X, Y = np.meshgrid(R1, R2_R1)

    # Creating density plot
    pc = plt.pcolormesh(X, Y, 2*np.pi*hist.T,
                        norm=colors.Normalize(vmin=0, vmax=1),
                        cmap='magma_r', alpha=1.0,
                        rasterized=True, antialiased=True)
    pc.set_edgecolor('face')

    # Add colorbar
    cbar = plt.colorbar(pc, pad=0.1)

    plt.title('$\phi_2$-Integrated RE3C', size=24)
    plt.xlabel(r'$R_1$', size=16)
    plt.ylabel(r'$R_2/R_1$', size=16)

    plt.xlim((5e-3, 5e-1))
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(filename.replace('.py','.png'), dpi=300)
    plt.close()

def main():
    """Main function to create visualizations from histogram data."""
    # Find the output files
    output_files = [f for f in os.listdir(os.getcwd()+"/output") if
                    (f.startswith("re3c_example"))
                    and f.endswith(".py")]
    output_files = ["output/" + f for f in output_files]

    if not output_files:
        print("Error: No output files found in the output directory.")
        return 1

    print(f"Found {len(output_files)} output files.")

    # Process each file
    for filename in output_files:
        print(f"Processing file: {filename}")

        # Load the histogram data
        try:
            hist_data = load_histogram(filename)
        except Exception as e:
            print(f"Error loading histogram from {filename}: {e}")
            continue

        plot_re3c(filename)

        print(f"Created visualizations for {filename}")


    print(f"RE3C visualizations saved")

    return 0


if __name__ == "__main__":
    main()
