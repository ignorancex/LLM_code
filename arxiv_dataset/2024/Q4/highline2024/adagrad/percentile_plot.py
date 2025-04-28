import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ode import AdaGradODE
from sgd import AdaGrad

KIND = "ls"
plt.rcParams['font.size'] = 23
# Parameters
if KIND == "ls":
    e_var = 1
    b_0 = 0.3
else:
    e_var = 0
    b_0 = 1
G = 1
if KIND == "ls":
    T = 10
else:
    T = 20
num_runs = 30  # Number of stochastic runs for AdaGrad per dimension
kind = "least squares" if KIND == "ls" else "logistic regression"
# kind = "logistic regression"  # Kind of optimization problem
if KIND == "ls":
    dimensions = [2**i for i in [8, 10, 12, 14]]
else:
    dimensions = [2**i for i in [4, 5, 6, 7]]

# Cache directory setup
cache_dir = "simulation_cache"
os.makedirs(cache_dir, exist_ok=True)

# Initialize storage for losses
all_sgd_losses = {}
all_sgd_stepsizes = {}

# Fixed eigenvalues for all simulations
# Use the largest dimension to set all eigenvalues to 1
fixed_eigs = np.ones(max(dimensions))

# Single ODE run with fixed eigenvalues
ode = AdaGradODE(kind, G=G, b_0=b_0, e_var=e_var)
X_ = np.random.normal(size=max(dimensions))
X_ = X_ / np.sqrt((X_ @ X_))
ode_losses, ode_stepsizes = ode.get_losses(fixed_eigs, T=T, X_=X_)
# Simulation loop for AdaGrad across dimensions
for d in dimensions:
    cache_path = os.path.join(
        cache_dir, f"sgd_losses_kind={kind}_d={d}_e_var={e_var}_b_0={b_0}_G={G}_T={T}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            sgd_losses, sgd_stepsizes = pickle.load(f)
    else:
        sgd_losses = []
        sgd_stepsizes = []
        for run in range(num_runs):
            X_ = np.random.normal(size=d)
            X_ = X_ / np.sqrt((X_ @ X_))
            sgd = AdaGrad(kind, G=G, b_0=b_0, e_var=e_var)
            losses, stepsizes = sgd.get_losses(fixed_eigs[:d], T=T)
            sgd_losses.append(losses)
            sgd_stepsizes.append(stepsizes)
        with open(cache_path, "wb") as f:
            both = (sgd_losses, sgd_stepsizes)
            pickle.dump(both, f)

    all_sgd_losses[d] = sgd_losses
    all_sgd_stepsizes[d] = sgd_stepsizes

fig, step_ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.plasma(np.linspace(0, 1, len(dimensions)))

loss_ax = step_ax.twinx()
for idx, d in enumerate(dimensions):
    sgd_stepsizes = np.array(all_sgd_stepsizes[d])
    sgd_lower = np.percentile(sgd_stepsizes, 10, axis=0)
    sgd_upper = np.percentile(sgd_stepsizes, 90, axis=0)
    step_ax.fill_between(np.linspace(0, T, len(sgd_upper)), sgd_lower, sgd_upper,
                         color=colors[idx], label=f"$d={d}$")

for idx, d in enumerate(dimensions):
    sgd_losses = np.array(all_sgd_losses[d])
    sgd_lower = np.percentile(sgd_losses, 10, axis=0)
    sgd_upper = np.percentile(sgd_losses, 90, axis=0)
    loss_ax.fill_between(np.linspace(0, T, len(sgd_upper)), sgd_lower, sgd_upper,
                         color=colors[idx], label=f"$d={d}$")

# done for the sake of the legend later on
loss_ax.loglog(np.linspace(0, T, len(ode_losses)), ode_losses, color='red',
               linewidth=3, label="Theory, learn. rate", linestyle='--')

loss_ax.loglog(np.linspace(0, T, len(ode_losses)), ode_losses, color='red',
               linewidth=3, label="Theory, risk")

step_ax.loglog(np.linspace(0, T, len(ode_stepsizes)), ode_stepsizes, color='red',
               linewidth=3, linestyle='--')

# Add grid lines
loss_ax.grid(True, linestyle='--', alpha=0.5)

if KIND == "ls":
    loss_ax.set_title("AdaGrad-Norm Least Squares", fontsize=25)
else:
    loss_ax.set_title("AdaGrad-Norm Logistic Regression", fontsize=25)
loss_ax.set_ylabel("Risk")
if KIND == "ls":
    loss_ax.set_xlim(left=1e-2)
else:
    loss_ax.set_xlim(left=0.1)
    # loss_ax.set_ylim(top=1)
    step_ax.set_ylim(bottom=0.92)
step_ax.set_ylabel("Learning rate")

step_ax.yaxis.tick_right()
step_ax.yaxis.set_label_position("right")
loss_ax.yaxis.tick_left()
loss_ax.yaxis.set_label_position("left")

step_ax.set_xlabel("SGD Iterations/$d$")
loss_ax.legend(loc="lower left")

plt.tight_layout()
# make the directory if it doesn't exist
os.makedirs("figures/adagrad", exist_ok=True)
plt.savefig(
    f"figures/adagrad/sgd_concentration_vs_dimension_{kind.replace(' ', '_')}.pdf", bbox_inches='tight')
