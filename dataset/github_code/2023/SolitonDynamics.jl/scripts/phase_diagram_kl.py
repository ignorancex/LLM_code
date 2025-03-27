import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
import toml
from gva import first_derivative, second_derivative, count_zeros

cf = toml.load("input/phase_diagram.toml")
hbar = 1.0545718e-34
l_perp = np.sqrt(hbar/(cf['m']*cf['omega_perp']))
dL0 = cf["d"] # um
Er = (hbar*np.pi/dL0)**2 / (2*cf["m"]) / (hbar* cf["omega_perp"])

# Define parameter ranges
n_points = 2000
g_values = np.linspace(-1.7, -0.2, n_points)
dL_values = np.linspace(0, 10, n_points) # um
kL_values = np.pi/(dL_values/(l_perp*1e6))
V0 = 1.3 * Er # Er for dL/l_perp = 1.887
heatmap_data = np.zeros((len(kL_values), len(g_values)))

if False:
  # Compute number of zeros for each (g, V0) pair
  for i, kL in enumerate(kL_values):
    print(f"row index : {i:>10}")
    for j, g in enumerate(g_values):
      heatmap_data[i, j] = count_zeros(first_derivative, second_derivative, g, V0, kL)
  np.save("results/phase_diagram/variational_kl.npy", heatmap_data)

heatmap_data = np.load("results/phase_diagram/variational_kl.npy")

print(f"Plotting the {heatmap_data.shape} phase diagram...")
plt.figure(figsize=(3.7, 3))
ax = sns.heatmap(heatmap_data[::-1, :], 
            cmap='jet')

coordinates = [
(-1.552127659574468, 0.009453395763042494),
(-1.5569148936170212, 1.498169570107187),
(-1.047872340425532, 3.467703527737611),
(-0.9999999999999999, 2.032537469301568),
(-1.2952127659574466, 9.985568867772956),
(-0.3457446808510636, 9.96624606945305),
(-0.15425531914893598, 2.5966713718469556) ,
]
data = np.array(coordinates)
print(data.shape)
x_scaled = np.interp(data[:, 0], (-1.7, -0.2), (0, len(g_values) - 1))
# x_scaled = len(g_values) - 1 - x_scaled  # Reverse the Y-axis for correct positioning
y_scaled = np.interp(data[:, 1], (0, 10), (0, len(kL_values) - 1))
y_scaled = len(kL_values) - 1 - y_scaled  # Reverse the Y-axis for correct positioning
ax.scatter(x_scaled, y_scaled, color='black', s=70, marker='x')

ytick_positions = [0, len(dL_values) - 1][::-1]
xtick_positions = [0, len(g_values) - 1]
ytick_labels = [round(dL_values[0], 2), round(dL_values[-1], 2)]
xtick_labels = [round(g_values[0], 2), round(g_values[-1], 2)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels)
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels)

plt.xlabel(r'$g$')
plt.ylabel(r'$d_L$ [$\mu$m]')
plt.tight_layout()
plt.savefig('media/variational_kl_phase_diagram.png', dpi=300)
