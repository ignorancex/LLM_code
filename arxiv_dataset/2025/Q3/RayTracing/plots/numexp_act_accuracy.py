import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import einops as ein
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

#plt.style.use("myplots.mplstyle")


#
parser = argparse.ArgumentParser(
    description="Plot activation vs. accuracy and number of experts vs. accuracy."
)
parser.add_argument(
    "--config",
    type=str,
    default="default_run",
    help="Path to the config file.",
)
args = parser.parse_args()
data_dir = Path("../data") / args.config
plot_dir = Path(".") / args.config
plot_dir.mkdir(parents=True, exist_ok=True)

activation_sequences = np.load(data_dir / "acts_seqs.npy")
prediction_sequences = np.load(data_dir / "all_preds.npy")
true_labels = np.load(data_dir / "labels.npy")


# PLOT 1: number of experts vs. accuracy
predicted_labels = np.argmax(prediction_sequences, axis=-1)
accuracies_per_num_expert = ein.reduce(
    (predicted_labels == true_labels[:, None]).astype(float), "n t -> t", "mean"
)
stds = ein.reduce(
    (predicted_labels == true_labels[:, None]).astype(float), "n t -> t", np.std
)

fig, ax = plt.subplots()
num_exps = len(accuracies_per_num_expert)
experts = np.arange(num_exps) + 1
ax.plot(experts, accuracies_per_num_expert, "o-", label="Avg.")
ax.fill_between(
    experts,
    accuracies_per_num_expert - 1.96 * stds / np.sqrt(len(predicted_labels)),
    accuracies_per_num_expert + 1.96 * stds / np.sqrt(len(predicted_labels)),
    alpha=0.5,
    label="95% C.I.",
)
ax.set_xlabel("N. experts")
ax.set_ylabel("Accuracy")
ax.set_xlim((1, num_exps + 1))
plt.legend()
plt.savefig(plot_dir / "exps_vs_accs.pdf")
plt.close()

# plot 2: accuracy per n. of exps activated

n_exps_activated = jax.vmap(jnp.searchsorted, in_axes=(0, None))(activation_sequences, 0.5)
preds_per_num_exps = jnp.argmax(prediction_sequences[np.arange(len(prediction_sequences)), n_exps_activated], axis=-1)
accs_per_num_exps = jax.ops.segment_sum((preds_per_num_exps==true_labels).astype(int), n_exps_activated)
unique, counts = jnp.unique_counts(n_exps_activated)
idxs = jnp.argsort(unique)
unique = unique[idxs]
counts = counts[idxs]
accs_per_num_exps = accs_per_num_exps / counts
fig , ax = plt.subplots()
ax.plot(
    unique, accs_per_num_exps
)
ax.plot(
    unique, 
)
ax.set_xlabel("Target n. of experts")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy per Target Number of Experts Activated by Each Test Example")
plt.savefig(plot_dir / "n_act_vs_accs.pdf")
cmap = plt.get_cmap('tab20')  # or 'tab20b', 'tab20c'
colors = [cmap(i) for i in range(20)]  # Get 20 distinct colors
fig, ax = plt.subplots()
markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', '*', 'h', 'H', 'x', 'D', '+', '.', ',', '1', '2', '3', '4', '8']
for i, n_exp in enumerate(unique):
    accuracy_line = np.zeros(n_exp+1)
    for j in range(n_exp+1):
        accuracy_line[j] = (jnp.argmax(prediction_sequences[n_exps_activated==n_exp, j], axis=-1) == true_labels[n_exps_activated==n_exp]).mean()
    accuracy_line=np.pad(accuracy_line, (0, len(unique)- len(accuracy_line)), mode='edge')
    color = colors[i % len(colors)]  # cycle through colors safely
    ax.plot(unique[:n_exp+1]+1, accuracy_line[:n_exp+1], label=n_exp+1, marker=f"${n_exp+1}$", color=color)
    ax.plot(unique[n_exp:]+1, accuracy_line[n_exp:], linestyle='--', color=color)
ax.plot(unique+1, accuracies_per_num_expert[:len(unique)], label="All",  color = colors[len(unique) % len(colors)], marker= markers[-1])
ax.set_xticks(unique+1)
ax.set_xlabel("Number of experts used")
ax.set_ylabel("Accuracy")
ax.set_title(f"Accuracy breakdown by target expert (MNIST)")
plt.grid()
plt.legend(title= 'target experts',loc='center left',  bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(plot_dir / "accuracy_per_target.pdf")