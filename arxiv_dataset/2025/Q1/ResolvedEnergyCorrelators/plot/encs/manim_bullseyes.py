import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from histogram import HistogramData, HistogramData2D

from utils.plot_utils import adjust_lightness
from utils.postprocess import collision_stamp
from encs.plots import plot_2d_bullseye, bullseye_inset,\
    density_colormap

from plotter import enc_data_dir, manim_enc_dir

# =====================================
# Create Directories for Saving PDFs
# =====================================
bullseye_3part_dir = manim_enc_dir/"bullseye_3part"
bullseye_4part_dir = manim_enc_dir/"bullseye_4part"

# Create directories if they don't exist
os.makedirs(bullseye_3part_dir, exist_ok=True)
os.makedirs(bullseye_4part_dir, exist_ok=True)

# =====================================
# Dictionary for Storing Paths to PDFs
# =====================================
od_img_dict_3part = {}
# od_img_dict_4part = {}

# =====================================
# Functions for Generating PDFs and Saving to Folders
# =====================================
fig_params = {
    'ylim': (0, 1.0),
    'cbar_cmap': density_colormap['opendata'],
}


# Set default text and line colors to white
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'  # Axis color
plt.rcParams['axes.titlecolor'] = 'white'  # Title color


def save_bullseye_3part(theta1, hist_data, params, metadata):
    """Generate and save bullseye_3part plots."""
    t1_bin = (hist_data.edges['theta1'][0],
              hist_data.edges['theta1'][1])

    # Generate and stamp the bullseye plot
    bullseye = plot_2d_bullseye(hist_data=hist_data,
                                save=None, theta1=theta1,
                                vmin=min(5e-4/theta1**4.2, 1e3),
                                vmax=min(1.5e2/theta1**1.8, 1e8),
                                **params)
    if not bullseye:
        return

    bullseye_inset(bullseye, inner_rad=theta1,
                   inner_var='theta1')

    # Save the figure to the directory
    filename = f"re3c_{theta1}.png"
    filepath = os.path.join(bullseye_3part_dir, filename)
    bullseye.bullseye.fig.savefig(filepath,
                                  transparent=True, dpi=300)
    plt.close()

    return filepath


def save_bullseye_4part(theta1, theta2, phi2, hist_data,
                        params, metadata):
    """Generate and save bullseye_4part plots."""
    # Generate and stamp the bullseye plot
    # bin edges and centers
    bin3_edges = hist_data.edges[t3_t2_key]
    t3_edges   = theta2*bin3_edges
    bin3_centers = hist_data.centers[t3_t2_key]
    t3_centers   = theta2*bin3_centers

    hist_data.hist = np.array([
        hist_data.hist[i3] / t3
        for i3, t3 in enumerate(t3_centers)
    ])

    # Readjusting for correct defn of angles (not ratios)
    hist_data.edges[t3_t2_key] = t3_edges
    hist_data.centers[t3_t2_key] = t3_centers

    # Making plot
    try:
        hist_data.make_plot('bullseye',
                 log_norm=True, save=None,
                 radial_coordinate='theta3_over_theta2',
                 **fig_params)
    except ValueError:
        return

    ax = hist_data.bullseye.axes[0]
    ax.set_ylim(0, theta2)

    bullseye_inset(hist_data, inner_rad=theta1*theta2,
                   inset_pts=[(theta1, -phi2)])

    # Save the figure to the directory
    filename = f"re4c_{theta1}_{theta2}_{phi2}.png"
    filepath = os.path.join(bullseye_4part_dir, filename)
    hist_data.bullseye.fig.savefig(filepath,
                                   transparent=True, dpi=300)

    plt.close()

    return filepath

# =====================================
# Main Function for Plot Generation and Dict Creation
# =====================================

if __name__ == "__main__":
    # ========================================
    # Loading Data
    # ========================================
    # RE3C
    re3c_hist = HistogramData(
        file_name=enc_data_dir/
              '3particle_od_100k_150bins_nus_1-00_1-00.py'
    )

    # RE4C
    t2_t1_key = 'theta2_over_theta1'
    t3_t2_key = 'theta3_over_theta2'
    # Histogram processing
    re4c_hist = HistogramData(
        file_name=enc_data_dir/
           f"4particle_od_100k_25bins_nus_1-00_1-00_1-00.py",
        variable_order=['theta1', t2_t1_key, 'phi2',
                        t3_t2_key, 'phi3']
    )


    # ========================================
    # Generating Figures
    # ========================================
    # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
    # RE3C
    # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
    for theta1 in re3c_hist.centers['theta1']:
        filepath_3part = save_bullseye_3part(theta1, re3c_hist,
                                    fig_params, metadata={})
        # Store path in dict
        if filepath_3part:
            od_img_dict_3part[theta1] = filepath_3part

    # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
    # RE4C
    # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
    # for theta1 in re4c_hist.centers['theta1']:
    #     for t2_t1_val in re4c_hist.centers[t2_t1_key]:
    #         try:
    #             pre_bullseye_4part = re4c_hist.\
    #                                     get_sub_histogram('theta1',
    #                                                       theta1).\
    #                                     get_sub_histogram(t2_t1_key,
    #                                                       t2_t1_val)
    #         except ValueError:
    #             continue
    #         for phi2 in re4c_hist.centers['phi2']:
    #             bullseye_4part = HistogramData2D(
    #                 hist_data = pre_bullseye_4part.\
    #                                 get_sub_histogram('phi2', phi2)
    #             )

    #             filepath_4part = save_bullseye_4part(
    #                                     theta1, t2_t1_val,
    #                                     phi2, bullseye_4part,
    #                                     fig_params, metadata={})

    #             # Store path in dict
    #             key = (theta1, theta1*t2_t1_val, phi2)
    #             od_img_dict_4part[key] = filepath_4part

    # ========================================
    # Saving Dicts
    # ========================================
    # Save the dictionaries for later use in animation
    with open(os.path.join(bullseye_3part_dir,
                           'od_img_dict_3part.pkl'), 'wb') as f:
        pickle.dump(od_img_dict_3part, f)

    # with open(os.path.join(bullseye_4part_dir,
    #                        'od_img_dict_4part.pkl'), 'wb') as f:
    #     pickle.dump(od_img_dict_4part, f)
