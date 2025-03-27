import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
import toml
from matplotlib.colors import Normalize


def parse_file(filepath):
    """
    Reads a file with a specific format and parses the header (i, j indices) and associated data.

    Args:
        filepath (str): Path to the file.

    Returns:
        tuple: A dictionary of tuples (indices, data), where indices are (i, j) tuples and data is a numpy array,
               and a tuple of phase diagram bounds (v_0_range, a_s_range).
    """
    results = {}
    v_0_range, a_s_range = None, None

    with open(filepath, 'r') as file:
        for line in file:
            # Check for the header line containing the ranges
            if line.startswith("==="):
                match = re.search(r"v_0 in \[(.+?), (.+?)\], a_s in\[(.+?), (.+?)\]", line)
                if match:
                    v_0_range = (float(match.group(1)), float(match.group(2)))
                    a_s_range = (float(match.group(3)), float(match.group(4)))
            # Parse the indices and wavefunction data
            elif "|" in line:
                header, data = line.split("|")

                # Parse the indices (i, j)
                i, j = map(int, header.strip().split(","))

                # Parse the data values
                data_values = np.array(list(map(float, data.strip().split())))

                results[(i, j)] = data_values

    return results, (v_0_range, a_s_range)


def read_phase_diagram_config(config_path):
    """
    Reads the phase diagram configuration from a TOML file.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        tuple: Grids for a_s and v_0 values.
    """
    config = toml.load(config_path)

    g_min = config["g_min"]
    g_max = config["g_max"]
    n_g = config["n_g"]
    v_0_min = config["v_0_min"]
    v_0_max = config["v_0_max"]
    n_v_0 = config["n_v_0"]

    g_values = np.linspace(g_min, g_max, n_g)
    v_0_values = np.linspace(v_0_min, v_0_max, n_v_0)

    return g_values, v_0_values

def save_wavefunction_plot(data, indices, output_folder):
    """
    Saves the plot of the wavefunction to a file.

    Args:
        data (np.array): The wavefunction data.
        indices (tuple): The (i, j) indices.
        output_folder (str): Path to the output folder.
    """
    ix, iy = indices
    filename = os.path.join(output_folder, f"name_{ix}_{iy}.png")

    plt.figure(figsize=(4, 3))
    plt.plot(data)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\psi|^2$")
    plt.title(f"GS of ({ix}, {iy})")
    plt.tight_layout() 
    plt.savefig(filename, dpi=300)
    plt.close()

def compute_widths(data):
    indexes = np.arange(len(data))
    norm = np.sum(data)
    mean = np.sum(indexes * data)/norm
    variance = np.sum(data * (indexes - mean)**2)
    return np.sqrt(variance) /norm

def compute_peaks(data):
    """
    Computes the number of peaks in the wavefunction data.

    Args:
        data (np.array): The wavefunction data.

    Returns:
        int: The number of peaks.
    """
    peaks, _ = find_peaks(data)
    return len(peaks)

def generate_phase_diagram(peaks_data, g_values, v_0_values, output_file, n_g, n_v_0, z_label):
    """
    Generates and saves a phase diagram based on the number of peaks.

    Args:
        peaks_data (dict): A dictionary with keys as (i, j) and values as number of peaks.
        output_file (str): Path to save the phase diagram.
    """
    # Extract unique i and j values
    indices = np.array(list(peaks_data.keys()))
    values = np.array(list(peaks_data.values()))

    i_vals = np.unique(indices[:, 0])
    j_vals = np.unique(indices[:, 1])
    
    # Create a grid for the phase diagram
    phase_grid = np.zeros((n_g,  n_v_0))

    for (ix, iy), peaks in peaks_data.items():
        i_idx = np.where(i_vals == ix)[0][0]
        j_idx = np.where(j_vals == iy)[0][0]
        phase_grid[i_idx, j_idx] = peaks
    print(phase_grid)
    phase_grid[phase_grid == 0] = np.nan
    # Plot the phase diagram
    plt.figure(figsize=(4, 3.5))
    norm = Normalize(vmin=np.nanmin(phase_grid), vmax=np.nanmax(phase_grid))
  
    im  = plt.imshow(phase_grid.T, 
               origin="lower", 
               aspect="auto", 
               cmap="viridis",
               extent=[g_values[0], g_values[-1], v_0_values[0], v_0_values[-1]], 
               norm=norm)
    coordinates = [
        (-1.5356888459195894, 0.0003600062491648792),
        (-1.3002692122202875, 0.24040945616414477),
        (-1.2059865190112733, 0.556060218403791),
        (-1.0839267398603898, 2.9953878444682434),
        (-0.8800065208679094, 2.992806667587437),
        (-0.15484253688931982, 0.8689260266404628)
    ]
    data = np.array(coordinates)
    plt.scatter(data[:, 0], data[:, 1], color='black', s=70, marker='x')
    cbar = plt.colorbar(im)
    cbar.set_label(z_label)
    # Set custom ticks and labels
    cbar_ticks = np.linspace(np.nanmin(phase_grid), np.nanmax(phase_grid), 5)  # Custom ticks
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([rf'{tick:.3f}' for tick in cbar_ticks])
    
    # plt.colorbar(label=z_label)
    plt.xlabel(r"$a_s \quad [a_0]$")
    plt.ylabel(r"$V_0 \quad [E_r]$")
    # plt.title("Phase Diagram of Peaks")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def fill_phase_diagram():
    # Filepath and output configuration
    filepath = "results/phase_diagram.txt"
    config_file = "input/phase_diagram.toml"
    output_folder = "media/gs"
    peaks_diagram_file = os.path.join(output_folder, "pd_peaks.png")
    width_diagram_file = os.path.join(output_folder, "pd_width.png")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    
    parsed_data, _ = parse_file(filepath)
    g_values, v_0_values = read_phase_diagram_config(config_file)
    cf = toml.load(config_file)
    
    print("Parsed data from the file.")
    print("The data contains the following indices:")
    print("\n".join(str(indices) for indices in parsed_data.keys()))
    
    skipping = 9
    print(f"Plotting sampled ground states with skipping {skipping}...")
    for indices, data in parsed_data.items():
        if indices[0] % skipping != 0 or indices[1] % skipping != 0:
            continue
        save_wavefunction_plot(data, indices, output_folder)
    peaks_data = {}
    width_data = {}
    
    dx = cf["l"] / cf["n"]
    for indices, data in parsed_data.items():
        # peaks_count = compute_widths(data)
        peaks_count = compute_peaks(data)
        width = compute_widths(data) * dx
        print(f"({indices[0]:>5d}, {indices[1]:>5d}) >> n_peaks: {peaks_count:>5.3e}, width: {width:>5.3e}.")
        peaks_data[indices] = peaks_count
        width_data[indices] = width

    print("Plotting the phase diagrams...")
    # Generate and save the phase diagram
    generate_phase_diagram(peaks_data, g_values, v_0_values, peaks_diagram_file, cf["n_g"], cf["n_v_0"], z_label=r"$\mathrm{Number \; of \; peaks}$")
    generate_phase_diagram(width_data, g_values, v_0_values, width_diagram_file, cf["n_g"], cf["n_v_0"], z_label=r"$w \; [\ell_\perp]$")
    print("Done.")
    
if __name__ == "__main__":
    fill_phase_diagram()
    skipping = 10
    print("Plotting sampled ground states...")