#!/usr/bin/env python3

# Place import files below
import warnings

import fig_02_key_galaxy_properties
import fig_03_gas_bh_properties
import fig_04_z0_mass_function
import fig_05_rates_of_change
import fig_06_07_gc_properties
import fig_08_fhalo
import print_paper_results


def main():
    # Warning suppression
    # RuntimeWarnings will still appear when scripts are run separately
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Running all plotting scripts")
    print("#" * 72)

    # Plot Fig. 1
    print("1. Plotting Fig. 2")
    fig_02_key_galaxy_properties.main()

    # Plot Fig. 2
    print("2. Plotting Fig. 3")
    fig_03_gas_bh_properties.main()

    # Plot Fig. 3
    print("3. Plotting Fig. 4")
    fig_04_z0_mass_function.main()

    # Plot Fig. 4
    print("4. Plotting Fig. 5")
    fig_05_rates_of_change.main()

    # Plot Figs 5 and 6
    print("5. Plotting Figs 6 & 7")
    fig_06_07_gc_properties.main()

    # Plot Fig. 7
    print("6. Plotting Fig. 8")
    fig_08_fhalo.main()

    # Print paper results
    print("7. Printing relevant data to screen")
    print()
    print_paper_results.main()

    return None


if __name__ == "__main__":
    main()
