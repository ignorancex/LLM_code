#!/usr/bin/env python3

# Place import files below
import generate_paper_results
import fig_01_plot_selection_criteria
import fig_03_plot_cumulative_rad_dist
import fig_04_plot_luminosity_functions
import fig_05_plot_mock_sdss_observations
import print_paper_results


def main():
    # Generate paper data
    print("0. Generating paper data")
    print("Note: this step is slow...")
    generate_paper_results.main()

    # Plot Fig. 1
    print("1. Plotting Fig. 1")
    fig_01_plot_selection_criteria.main()

    # Plot Fig. 3
    print("2. Plotting Fig. 3")
    fig_03_plot_cumulative_rad_dist.main()

    # Plot Fig. 4
    print("3. Plotting Fig. 4")
    fig_04_plot_luminosity_functions.main()

    # Plot Fig. 5
    print("4. Plotting Fig. 5")
    fig_05_plot_mock_sdss_observations.main()

    # Print paper results
    print("5. Printing relevant data to screen")
    print_paper_results.main()
    return None


if __name__ == "__main__":
    main()
