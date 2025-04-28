#!/usr/bin/env python3
"""
Simple visualization script for Energy Correlator outputs.
This script reads the histogram data produced by write_pencs and creates plots to visualize the results.
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt

colors = {
    2  : 'mediumseagreen',
    5  : 'cornflowerblue',
    10 : 'mediumorchid',
    50 : 'indianred',
    100: 'goldenrod'
}


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


def plot_penc(hist_data, filename, ax):
    N = float(filename.removesuffix('.py')\
            .split('nu')[1].replace('-', '.'))+1
    if (N-math.floor(N)) < 1e-8:
        N = int(N)

    # TOGGLE to plot each ENC on its own
    """
    fig1, ax1 = plt.subplots()

    # Plotting
    ax1.plot(hist_data['theta1_centers'], hist_data['hist'],
             lw='2', color=colors.get(N, 'firebrick'))
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$R_1$')
    ax1.set_ylabel(rf'$R_1$ d$\Sigma_{{{N}}}$/d$R_1$')

    # Saving
    fig1.savefig(f"{filename.replace('.py', '')}.png")
    plt.close(fig1)
    """


    # Plotting
    ax.plot(hist_data['theta1_centers'], hist_data['hist'],
             lw='2', color=colors.get(N, 'firebrick'),
             label=f'{N=}')


def main():
    """Main function to create visualizations from histogram data."""
    # Find the output files
    output_files = [f for f in os.listdir(os.getcwd()+"/output") if
                    (f.startswith("penc_example"))
                    and f.endswith(".py")]
    output_files = ["output/" + f for f in output_files]

    if not output_files:
        print("Error: No output files found in the output directory.")
        return 1

    print(f"Found {len(output_files)} output files.")

    # Plotting multiple PENCs together
    penc_fig, penc_ax = plt.subplots()
    penc_ax.set_xscale('log')
    penc_ax.set_xlabel(r'$R_1$')
    penc_ax.set_ylabel(r'$R_1$ d$\Sigma$/d$R_1$')
    penc_ax.set_xlim((5e-3, 1e0))
    penc_ax.set_ylim((0, 3.7e0))

    # Process each file
    for filename in output_files:
        print(f"Processing file: {filename}")

        # Load the histogram data
        try:
            hist_data = load_histogram(filename)
        except Exception as e:
            print(f"Error loading histogram from {filename}: {e}")
            continue

        plot_penc(hist_data, filename, penc_ax)

        print(f"Created visualizations for {filename}")


    # Preparing grouped PENC legend
    handles, labels = penc_fig.gca().get_legend_handles_labels()
    def label_to_val(label):
        return float(label.lstrip('N='))
    handles, labels = zip(*[(handles[i], labels[i])
            for i in sorted(range(len(handles)),
                key=lambda k: list(map(label_to_val,labels))[k])])
    penc_ax.legend(handles, labels, frameon=False)
    penc_fig.savefig(f"output/penc_example.png")
    plt.close()

    print(f"PENC visualizations saved")

    return 0


if __name__ == "__main__":
    main()
