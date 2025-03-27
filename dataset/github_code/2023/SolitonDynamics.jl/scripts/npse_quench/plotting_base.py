import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import genlaguerre
import seaborn as sns
import matplotlib.animation as animation
import toml
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
import re

def plot_heatmap(filename="results/experiment1.csv"):
  # Set up matplotlib for LaTeX-compatible fonts
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  cf = toml.load("input/config.toml")
  l_perp = np.sqrt(1.0546e-34/(cf['m']*cf['omega_perp']))
  dx = cf['l']/cf['n']
  
  data = pd.read_csv(filename)
  time = data.iloc[0, :].values  # First column is time
  t_min = 0.0
  t_max = cf['t_f'] * 1e3
  x_min = -cf['l']/2 * l_perp * 1e6
  x_max =  -x_min
  # print("\033[91mWarn:\033[0m playing sketchy stuff with data extraction")
  psi2_values = data.iloc[1:, :].values  # Remaining columns are ψ² values
  time_points = psi2_values.shape[1]
  space_points = psi2_values.shape[0]
  time_ticks = np.linspace(t_min, t_max, time_points)
  
  atom_number = psi2_values.sum(axis=0) * dx
  # print(atom_number)
  # print(psi2_values[:, -1])
  
  fig = plt.figure(figsize=(4, 3.5))
  gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[4, 1], wspace=0.15, hspace=0.1)

  # Heatmap plot
  ax_heatmap = fig.add_subplot(gs[0, 0])
  sns.heatmap(
      psi2_values,
      cmap="viridis",
      cbar=False,  # Disable the default colorbar
      ax=ax_heatmap
  )
  
  x_zoom = 30
  lim_bottom = int(round((x_max-x_zoom)/(2*x_max) * space_points))
  lim_top = int(round((x_max+x_zoom)/(2*x_max)* space_points))
  ax_heatmap.set_yticks([0, lim_bottom, int(round(space_points/2)), lim_top, space_points - 1])  # Positions: start and end of space
  ax_heatmap.set_ylim(bottom=lim_bottom, top=lim_top)
  ax_heatmap.set_yticklabels([f"{x_min:.1f}", f"{-x_zoom:.1f}", f"{0.0:.1f}", f"{x_zoom:.1f}", f"{x_max:.1f}"])  # Labels: min and max space
  ax_heatmap.set_ylabel(r'$x \quad [\mu m]$')
  ax_heatmap.axhline((x_max+cf["d"]/2*1e6)/(2*x_max)*space_points, color='w', linestyle='--', lw=0.2)
  ax_heatmap.axhline((x_max-cf["d"]/2*1e6)/(2*x_max)*space_points, color='w', linestyle='--', lw=0.2)
  # Add colorbar to the right of the entire plot
  cbar_ax = fig.add_subplot(gs[:, 1])  # Colorbar spans both rows
  norm = plt.Normalize(vmin=np.min(psi2_values), vmax=np.max(psi2_values))
  sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
  cbar = Colorbar(cbar_ax, sm, orientation='vertical')
  cbar.set_label(r'$|\psi|^2$', rotation=90)

  # Line plot for atom number
  ax_lineplot = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
  ax_lineplot.plot(atom_number, color='blue')
  ax_lineplot.set_xlabel(r'$t \quad [\mathrm{ms}]$')
  ax_lineplot.set_ylabel(r'$N(t)/N_0$')
  ax_lineplot.set_xticks([0, time_points - 1])  # Positions: start and end of time
  ax_lineplot.set_xticklabels([f"{t_min:.1f}", f"{t_max:.1f}"])  # Labels: min and max time
  ax_heatmap.get_xaxis().set_visible(False)
  # ax_lineplot.text(f'{atom_number[:-1]}')
  ax_lineplot.text(
    0.95, 0.05,  # Position of text (relative to axes, [x, y] from bottom-left corner)
    f'$N(t_f)/N_0 = {atom_number[-1]:.2f}$',  # Format the final value
    transform=ax_lineplot.transAxes,  # Use axes coordinates
    color='black', fontsize=8, ha='right', va='bottom'
  )
  ax_lineplot.set_ylim((0.0, 1.1))
  # ax_lineplot.legend(loc="upper right")
  fig.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15)

  match = re.search(r"(\d{1,2})(?=\.csv)", filename)
  if match:
      number = match.group(1)
  else:
      print("No number found before .csv")
      number = 0
  if filename == "results/experiment1.csv":
    heatmap_filename = "media/simple.png"
  else: 
    heatmap_filename = "media/td_heatmap_"+str(number)+".png"
  plt.savefig(heatmap_filename, dpi=300)
  plt.close()
  print(f"Heatmap saved as {heatmap_filename}")


def plot_animation(filename = "results/experiment1.csv"): 
  # Set up matplotlib for LaTeX-compatible fonts
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')

  # Load the CSV file
  filename = "results/experiment1.csv"  # Replace with your CSV file path
  data = pd.read_csv(filename)
  time = data.iloc[0, :].values  # First row is time
  # print("\033[91mWarn:\033[0m playing sketchy stuff with data extraction")
  psi2_values = data.iloc[1:, :].values  # Remaining rows are ψ² values
  # print(np.shape(psi2_values))

  # Set up the plot for animation
  fig, ax = plt.subplots(figsize=(4, 3))
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$|\psi|^2$')
  # ax.set_xlim(min(time), max(time))
  ax.set_ylim(psi2_values.min(), psi2_values.max())

  # Initialize an empty line plot
  space = range(len(psi2_values[:, 1]))
  line, = ax.plot(space, psi2_values[:, 0], lw=1.5)
  
  # Function to update the line plot for each time frame
  def update_frame(i):
      x = space
      y = psi2_values[:, i]
      line.set_data(x, y)  # Update the line with new data
      ax.set_title(f"Time: {time[i]:.2f} $2\pi\omega_\perp^{-1}$")
      return [line]
    
  # Create the animation
  decimation = 2
  ani = animation.FuncAnimation(fig, update_frame, frames=range(0, len(time), decimation), interval=50 * decimation, blit=True)
  fig.tight_layout()
  # Save the animation as a GIF
  if filename == "results/experiment1.csv":
    gif_filename = "media/simple.gif"
  else:
    print("\033[91mWarn:\033[0m fancy naming of animations is still to be implemented")
    gif_filename = "media/td_line_animation.gif"
  ani.save(gif_filename, writer='pillow', dpi=100)
  print(f"Animation saved as {gif_filename}")
  plt.close()

def apply_noise_to_widths(w, l, noise_atoms, n_atoms):
  return (w*n_atoms+1/12*l**2*noise_atoms)/(n_atoms+noise_atoms)

def plot_widths(use_simulation=True, noise=0.0):
  """
  Confrontation with the experimental data
  """
  if not use_simulation: 
    data = pd.read_csv("input/widths.csv", header=None, names=["a_s", "width"]) 
  else:
    data = pd.read_csv("results/widths_final.csv", header=0, names=["a_s", "width", "width_sim", "width_rough", "particle_fraction"])
  # Extract columns
  a_s = data["a_s"]  # First column as x-axis
  width = data["width"]  # Second column as y-axis
  # Create the plot
  plt.figure(figsize=(3.6, 3))
  plt.plot(a_s, width, marker='o', linestyle='-', color='b', label='Width vs a_s')
  cf = toml.load("input/config.toml")
  n_atoms = cf["n_atoms"]
  print("applying the noise")
  noise_atoms = n_atoms * 0.23 # guess
  l = 8 # lattice sites
  width = data["width_sim"]
  print("before ", width)
  width = apply_noise_to_widths(width, l, noise_atoms, n_atoms)
  print("after ", width)
  if use_simulation:
    print("\033[91mWarn:\033[0m Adding +1")
    # plt.plot(a_s, data["width_rough"], marker='x', linestyle='--', color='r', label='Width vs a_s (sim)')
    plt.plot(a_s, width, marker='.', linestyle='-.', color='orange', label='Width vs a_s (sim)')
  plt.xlabel(r"$a_s/a_0$")
  plt.ylabel(r"$w_z$ [sites] ")
  plt.tight_layout()
  plt.savefig("media/widths.pdf", dpi=300)
  if use_simulation:
    fraction = data["particle_fraction"]  # Second column as y-axis
    plt.clf()
    plt.figure(figsize=(3.6, 3))
    plt.plot(a_s, fraction, marker='o', linestyle='-.', color='r', label='Width vs a_s (sim)')
    plt.xlabel(r"$a_s/a_0$")
    plt.ylabel(r"$N_{\mathrm{tot}}/N_0$")
    plt.tight_layout()
    plt.savefig("media/fraction.pdf", dpi=300)


def plot_final(filename="results/experiment1.csv"):
  # Set up matplotlib for LaTeX-compatible fonts
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  cf = toml.load("input/config.toml")
  l_perp = np.sqrt(1.0546e-34/(cf['m']*cf['omega_perp']))
  dx = cf['l']/cf['n']
  
  data = pd.read_csv(filename)
  time = data.iloc[0, :].values  # First column is time
  t_min = 0.0
  t_max = cf['t_f'] * 1e3
  x_min = -cf['l']/2 * l_perp * 1e6
  x_max =  -x_min
  # print("\033[91mWarn:\033[0m playing sketchy stuff with data extraction")
  psi2_values = data.iloc[1:, :].values  # Remaining columns are ψ² values
  time_points = psi2_values.shape[1]
  space_points = psi2_values.shape[0]
  space_axis = np.linspace(x_min, x_max, space_points) * l_perp / cf["d"]
  time_ticks = np.linspace(t_min, t_max, time_points)

  atom_number = psi2_values.sum(axis=0) * dx
  # print(atom_number)
  # print(psi2_values[:, -1])
  
  plt.figure(figsize=(3, 2.5))
  plt.plot(space_axis, psi2_values[:, -1], lw=0.7)
  vlines = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
  for vline in vlines:
    plt.axvline(vline, color='r', linestyle='--', lw=0.5)
  # plt.axvline(-1, color='r', linestyle='--', lw=0.5)
  # plt.axvline(+1, color='r', linestyle='--', lw=0.5)
  plt.xlabel(r'sites')
  # plt.xlim([-5, 5])
  match = re.search(r"(\d{1,2})(?=\.csv)", filename)
  if match:
      number = match.group(1)
  else:
      print("No number found before .csv")
      number = 0
  plot_filename = "media/final_"+str(number)+".png"
  plt.savefig(plot_filename, dpi=300)
  plt.tight_layout()
  plt.close()

  print(f"Final plot saved as {plot_filename}")
