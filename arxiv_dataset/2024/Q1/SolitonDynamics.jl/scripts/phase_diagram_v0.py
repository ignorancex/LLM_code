import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import toml
from gva import energy, first_derivative, second_derivative, count_zeros, count_peaks, count_minima

# Define parameter ranges
cf = toml.load("input/phase_diagram.toml")
hbar = 1.0545718e-34
l_perp = np.sqrt(hbar/(cf['m']*cf['omega_perp']))
dL0 = cf["d"] # um
Er = (hbar*np.pi/dL0)**2 / (2*cf["m"]) / (hbar* cf["omega_perp"])

n_points = 2000
g_values = np.linspace(-1.7, -0.2, n_points)
V0_values = np.linspace(0, 3, n_points)
V0_values = V0_values * Er # Er for dL/l_perp = 1.887
dL = dL0
kL = (np.pi / dL) * l_perp
print()
heatmap_data = np.zeros((len(V0_values), len(g_values)))

if False:
  # Compute number of zeros for each (g, V0) pair
  for i, V0 in enumerate(V0_values):
    print(f"row index : {i:>10}")
    for j, g in enumerate(g_values):
      heatmap_data[i, j] = count_zeros(first_derivative, second_derivative, g, V0, kL)
      # heatmap_data[i, j] = count_peaks(energy, g, V0, kL)
      # heatmap_data[i, j] = count_minima(energy, g, V0, kL)
  np.save("results/phase_diagram/variational_v0.npy", heatmap_data)

heatmap_data = np.load("results/phase_diagram/variational_v0.npy")
heatmap_data = np.clip(heatmap_data, -2, 1)
V0_values /= Er # go back to units of Er
print(f"Plotting the {heatmap_data.shape} phase diagram...")
plt.figure(figsize=(3.7, 3))
ax = sns.heatmap(heatmap_data[::-1, :], 
            # yticks = [0, n_points-1],
            yticklabels=[V0_values[0], V0_values[-1]],
            # xticks=[0, n_points-1],
            xticklabels=[g_values[0], g_values[-1]], cmap='jet')

# comparison with others
coordinates = [
    (-1.5356888459195894, 0.0003600062491648792),
    (-1.3002692122202875, 0.24040945616414477),
    (-1.2059865190112733, 0.556060218403791),
    (-1.0839267398603898, 2.9953878444682434),
    (-0.8800065208679094, 2.992806667587437),
    (-0.15484253688931982, 0.8689260266404628)
]
data = np.array(coordinates)
print(data.shape)
x_scaled = np.interp(data[:, 0], (-1.7, -0.2), (0, len(g_values) - 1))
y_scaled = np.interp(data[:, 1], (0, 3), (0, len(V0_values) - 1))
y_scaled = len(V0_values) - 1 - y_scaled  # Reverse the Y-axis for correct positioning
ax.scatter(x_scaled, y_scaled, color='black', s=70, marker='x')

ytick_positions = [0, len(V0_values) - 1]
xtick_positions = [0, len(g_values) - 1]
ytick_labels = [round(V0_values[0], 2), round(V0_values[-1], 2)][::-1]
xtick_labels = [round(g_values[0], 2), round(g_values[-1], 2)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels)
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels)

plt.xlabel(r'$g$')
plt.ylabel(r'$V_0$ [$E_r$]')
plt.tight_layout()
plt.savefig('media/variational_v0_phase_diagram.png', dpi=300)