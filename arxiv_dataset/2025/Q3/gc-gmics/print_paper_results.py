#!/usr/bin/env python3

# Place import files below
import numpy as np

from gcgmics.process_data import EvolutionData, return_print_format_lists
from gcgmics.settings import Printing, Simulations


def main():
    # Load relevant data
    property_list = Printing["print_property_list"]
    ev_data = [EvolutionData(sim) for sim in Simulations.get("Standard")["sim_list"]]
    print_labels = return_print_format_lists(property_list)
    col_sep = " " * 4

    ####################################################################
    # Print z=0 properties
    ####################################################################
    print("#" * 72)
    print("z = 0")

    # Collect all data and strings for z=0 table
    z0_table = []
    header_row = ["Property"] + print_labels
    z0_table.append(header_row)

    # Get exponents from first simulation
    first_sim = ev_data[0]
    z0_vals = [first_sim.med_spread(prop)[0][0] for prop in property_list]
    exponents = np.floor(np.log10(z0_vals))
    scale_row = [""] + [f"(10^{exp:.0f})" for exp in exponents]
    z0_table.append(scale_row)

    # Process each simulation
    for sim, name in zip(ev_data, Simulations.get("Standard")["sim_names"]):
        row_data = [name]
        for prop, exp in zip(property_list, exponents):
            med, spread = sim.med_spread(prop)
            val = med[0] / 10**exp
            upper = (spread[1, 0] - med[0]) / 10**exp
            lower = (spread[0, 0] - med[0]) / 10**exp
            row_data.append(f"{val:.1f}^+{upper:.1f}_{lower:.1f}")
        z0_table.append(row_data)

    # Calculate column widths and print
    col_widths = [
        max(len(str(row[i])) for row in z0_table) for i in range(len(z0_table[0]))
    ]
    for row in z0_table:
        formatted_row = []
        for i, item in enumerate(row):
            formatted_row.append(str(item).ljust(col_widths[i]))
        print(col_sep.join(formatted_row))

    ####################################################################
    # Print peak values
    ####################################################################
    print("#" * 72)
    print("Peak values")

    # Collect all data for peak table
    peak_table = []
    header_row = ["Property"] + print_labels
    peak_table.append(header_row)

    # Get exponents from first simulation
    first_peak_vals = []
    for prop in property_list:
        med, _ = first_sim.med_spread(prop)
        max_idx = np.nanargmax(med)
        first_peak_vals.append(med[max_idx])
    peak_exponents = np.floor(np.log10(first_peak_vals))
    scale_row = [""] + [f"(10^{exp:.0f})" for exp in peak_exponents]
    peak_table.append(scale_row)

    # Process each simulation
    for sim, name in zip(ev_data, Simulations.get("Standard")["sim_names"]):
        # Main value row
        main_row = [name]
        time_row = ["t_lb_max [Gyr]"]
        z_row = ["z_max"]

        for prop, exp in zip(property_list, peak_exponents):
            med, spread = sim.med_spread(prop)
            max_idx = np.nanargmax(med)
            val = med[max_idx] / 10**exp
            upper = (spread[1, max_idx] - med[max_idx]) / 10**exp
            lower = (spread[0, max_idx] - med[max_idx]) / 10**exp

            # Format with consistent decimal places
            main_row.append(f"{val:.1f}^+{upper:.1f}_{lower:.1f}")
            time_row.append(f"{sim.t_lb[max_idx]:.3f}")
            z_row.append(f"{sim.z[max_idx]:.3f}")

        peak_table.extend([main_row, time_row, z_row])

    # Calculate column widths
    col_widths = [
        max(len(str(row[i])) for row in peak_table) for i in range(len(peak_table[0]))
    ]
    # Print table with column separation
    for row in peak_table:
        formatted_row = []
        for i, item in enumerate(row):
            # Indent sub-rows
            if row is peak_table[0] or row is peak_table[1]:
                formatted_row.append(str(item).ljust(col_widths[i]))
            else:
                if i == 0 and row[0] in ["t_lb_max [Gyr]", "z_max"]:
                    indent = "   "
                    formatted_row.append(
                        f"{indent}{str(item).ljust(col_widths[i] - len(indent))}"
                    )
                else:
                    formatted_row.append(str(item).ljust(col_widths[i]))
        print(col_sep.join(formatted_row))

    return None


if __name__ == "__main__":
    main()
