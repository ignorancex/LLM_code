import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

import json
from unyt import unyt_array, unyt_quantity

# Load plot style
plt.style.use("mnras.mplstyle")


class CustomJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # Convert dictionaries that represent unyt quantities or arrays
        if "value" in obj and "unit" in obj:
            value = obj["value"]
            unit = obj["unit"]

            if isinstance(value, list):  # If value is a list, it's an array
                return unyt_array(value, unit)
            else:  # If value is a single number, it's a quantity
                return unyt_quantity(value, unit)

        return obj  # Return unchanged if not matching expected format


def load_json_to_dict(filename):
    with open(filename, "r") as f:
        return json.load(f, cls=CustomJSONDecoder)


# Load JSON file and decode it
group_data = load_json_to_dict("../data/fig2_maps/group_+8res_maps.json")

# Convert lists back to numpy arrays where appropriate
for key, value in group_data.items():
    if isinstance(value, list):  # Assume lists should be numpy arrays
        group_data[key] = np.array(value)

cluster_data = load_json_to_dict("../data/fig2_maps/cluster_+1res_maps.json")

# Convert lists back to numpy arrays where appropriate
for key, value in cluster_data.items():
    if isinstance(value, list):  # Assume lists should be numpy arrays
        cluster_data[key] = np.array(value)

# Number of original redshift rows
num_redshifts = 3
total_rows = num_redshifts * 2 + 2  # Extra rows for spacing ("Group" & "Cluster")

# Panel Titles & Colorbar Labels
titles = ['Projected mass ratio\n(hot/cold)', 'EMW-Temperature\n(cold)', 'EMW-Temperature\n(hot)',
          'MW-Entropy\n(cold)', 'MW-Entropy\n(hot)']

cbar_labels = [
    r'$\sum_\mathrm{hot} m_i / \sum_\mathrm{cold} m_i$',
    r'$T_{\rm emw, cold}/\mathrm{K}$',
    r'$T_{\rm emw, hot}/\mathrm{K}$',
    r'$K_{\rm mw, cold}/\mathrm{keV~cm^2}$',
    r'$K_{\rm mw, hot}/\mathrm{keV~cm^2}$'
]

# Adjust figure size and grid layout
fig = plt.figure(figsize=(15, 18))
row_heights = [0.2, 1, 1, 1, 0.2, 1, 1, 1]
gs = gridspec.GridSpec(total_rows, 5, figure=fig, wspace=0, hspace=0, height_ratios=row_heights)

# Labels for leftmost panel
phase_labels = ['A', 'B', 'C']
section_labels = {0: "Group high-res", 4: "Cluster mid-res"}  # Rows marking section labels
redshift_rows = {0: None, 1: 0, 2: 1, 3: 2, 4: None, 5: 0, 6: 1, 7: 2}

for row in range(total_rows):

    # Handle "Group" and "Cluster" labels (spacing rows)
    if row in section_labels:
        section_name = section_labels[row]
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if col == 2:  # Center "Group" / "Cluster" label
                ax.text(0.5, 0.5, section_name,
                        transform=ax.transAxes,
                        fontsize=16, fontweight='bold',
                        ha='center', va='center')

            # Titles for the top row
            if row == 0:
                ax.set_title(titles[col], fontsize="xx-large")

        continue  # Skip further processing for this row


def display_data(row, col, maps):
    ax = fig.add_subplot(gs[row, col])
    ax.set_xticks([])
    ax.set_yticks([])

    # Compute the phase index for labeling
    new_row = redshift_rows[row]
    if new_row is None:
        return

    # Add alphabetical "Phase" labels to the leftmost column
    if col == 0:
        ax.text(-0.15, 0.5, f"Phase {phase_labels[new_row]}",
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='center', va='center', rotation=90, zorder=20)


        median_mass = maps["m500"][new_row]
        exp = np.floor(np.log10(median_mass))
        mantissa = np.power(10, np.log10(median_mass) - exp)
        mass_string = f"$M_{{500}}={mantissa:.2f}\\times 10^{{{exp:.0f}}}$ M$_{{\\odot}}$"
        ax.text(0.95, 0.95, mass_string, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', color="white",
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'), path_effects=[pe.withStroke(linewidth=2, foreground="black")], zorder=20)
    elif col == 1:
        ax.text(0.95, 0.95, f"z = {maps['redshift'][new_row]:.2f}", transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', color="white",
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'), path_effects=[pe.withStroke(linewidth=2, foreground="black")], zorder=20)

    elif col == 2:
        x_text, y_text = maps["center"][new_row][:-1]
        ax.text(x_text, y_text + 1.05 * maps["r500"][new_row], r"$r_{500}$", transform=ax.transData, fontsize="xx-large", verticalalignment='bottom', horizontalalignment='center', color="green",
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'), path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=20)

    ax.add_patch(Circle((maps["center"][new_row][:-1]), maps["r500"][new_row], fc="none", ec='green', lw=2, zorder=20, path_effects=[pe.withStroke(linewidth=3, foreground="white")]))
    ax.add_patch(Circle((maps["center"][new_row][:-1]), 0.15 * maps["r500"][new_row], fc="none", ec='green', ls="--", lw=0.5, zorder=20, path_effects=[pe.withStroke(linewidth=1.25, foreground="white")]))

    return ax


row, phase_idx = 1, 0

# ========================================== #

col = 0
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["masses_hightemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmax=1.01e7), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1.0e-2, vmax=10.0), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1e-2] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=8.0, vmax=200.0), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

row, phase_idx = 2, 1

# ========================================== #

col = 0
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["masses_hightemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmax=1.01e7), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1.0e-2, vmax=100.0), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1e-2] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

row, phase_idx = 3, 2

# ========================================== #

col = 0
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["masses_hightemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="40%", height="4%", loc='lower left', borderpad=.75)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=10, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

img2 = ax.imshow(group_data["masses_hightemp"][phase_idx], norm=LogNorm(), extent=group_data["roi"][phase_idx][:-2], cmap='Greys', zorder=5)

cax = inset_axes(ax, width="40%", height="4%", loc='lower right', borderpad=.75)
cbar = plt.colorbar(img2, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(r'$\log_{10}\sum_\mathrm{hot} m_i/\mathrm{M}_{\odot}$', fontsize=9, labelpad=10, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=5e3, vmax=1e5), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["temperature_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmax=1.01e7), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_lowtemp"][phase_idx] / group_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=2.0e-1, vmax=200.0), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, group_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = group_data["entropies_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1e-2] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=80.0, vmax=800), extent=group_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

row, phase_idx = 5, 0

# ========================================== #

col = 0
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["masses_hightemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=9e3, vmax=6.0e4), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=5e5, vmax=1.01e7), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
# data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-3), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1e-2] = np.nan
palette = cm.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=9.0, vmax=1.1e2), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

row, phase_idx = 6, 1

# ========================================== #

col = 0
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["masses_hightemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e4, vmax=1e5), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(0.0))

img = ax.imshow(data, norm=LogNorm(), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1e-3, vmax=20), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
data[data < 1e-2] = np.nan
palette = cm.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

row, phase_idx = 7, 2

# ========================================== #

col = 0
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["masses_hightemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx]
data[data < 1e-2] = np.nan
palette = plt.get_cmap('RdBu_r')

img = ax.imshow(data, norm=LogNorm(vmin=1e-1, vmax=1e2), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="40%", height="4%", loc='lower left', borderpad=.75)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks([0.1, 1, 10, 100])
cbar.ax.set_xticklabels(["0.1", "1", "10", "100"], color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=10, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

img2 = ax.imshow(cluster_data["masses_hightemp"][phase_idx], norm=LogNorm(), extent=cluster_data["roi"][phase_idx][:-2], cmap='Greys', zorder=5)

cax = inset_axes(ax, width="40%", height="4%", loc='lower right', borderpad=.75)
cbar = plt.colorbar(img2, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(r'$\log_{10}\sum_\mathrm{hot} m_i/\mathrm{M}_{\odot}$', fontsize=9, labelpad=10, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 1
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=4.e3), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 2
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["temperature_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1.0] = np.nan
palette = plt.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 3
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_lowtemp"][phase_idx] / cluster_data["masses_lowtemp"][phase_idx] / 1e10
data[data < 1.e-2] = np.nan
palette = plt.get_cmap('twilight')
palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(vmin=1, vmax=1e3), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #

col = 4
ax = display_data(row, col, cluster_data)
print(f"Processing row: {row}, column: {col}, phase index: {phase_idx}")

data = cluster_data["entropies_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10
# data[data < 1e-2] = np.nan
palette = cm.get_cmap('twilight')
# palette.set_bad(color=palette(1.0))

img = ax.imshow(data, norm=LogNorm(), extent=cluster_data["roi"][phase_idx][:-2], cmap=palette, zorder=10)

cax = inset_axes(ax, width="75%", height="4%", loc='lower center', borderpad=.5)
cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.outline.set_edgecolor('black')

# Apply path effects to all spines
path_effects_style = [pe.withStroke(linewidth=1.5, foreground="white")]
for spine in cax.spines.values():
    spine.set_path_effects(path_effects_style)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.set_label(cbar_labels[col], fontsize="xx-large", labelpad=9, color='white', path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cbar.ax.tick_params(axis='x', which="both", direction='in', labelsize="large", colors='white')

# ========================================== #
plt.savefig("maps.pdf")
# plt.show()
