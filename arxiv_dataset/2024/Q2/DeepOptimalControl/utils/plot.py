import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_cmp(results, problem_id, experiment_name):
    save_path = f"output/{problem_id}_{experiment_name}/results"
    max_value = 1.0
    for method, ratio in results.items():
        max_value = max(max_value, max(ratio))
        bins = max_value ** np.linspace(0, 1, 80)

    for method, ratio in results.items():
        values, base = np.histogram(ratio, bins=bins)
        cumulative = np.cumsum(values)
        base = np.concatenate(([1],base))
        cumulative = np.concatenate(([0],cumulative))
        if method == "Supervised Learning":
            color = "tab:orange"
        elif method == "Direct Optimization":
            color = "tab:green"
        elif method == "Pre-train and Fine-tune":
            color = "tab:blue"
        else:
            color = "tab:red"
        plt.plot(base[:-1], cumulative, label=method, color=color)
    ax = plt.gca()
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Policy Cost / Optimal Cost ', fontsize=19)
    plt.ylabel('Percentage', fontsize=19)
    plt.legend(loc='best', prop={'size':19})
    ax.tick_params(axis='both', labelsize=18)
    plt.savefig(f"{save_path}/Deter_cmp.png", dpi=500, bbox_inches="tight")
    plt.close()

def plot_cmp_sto(results, problem_id, experiment_name, sigma):
    save_path = f"output/{problem_id}_{experiment_name}/results"
    max_value = 1.0
    for method, ratio in results.items():
        max_value = max(max_value, max(ratio))
        bins = max_value ** np.linspace(0, 1, 80)

    for method, ratio in results.items():
        values, base = np.histogram(ratio, bins=bins)
        cumulative = np.cumsum(values)
        base = np.concatenate(([1],base))
        cumulative = np.concatenate(([0],cumulative))
        if method == "Supervised Learning":
            color = "tab:orange"
        elif method == "Direct Optimization":
            color = "tab:green"
        elif method == "Pre-train and Fine-tune":
            color = "tab:blue"
        else:
            color = "tab:red"
        plt.plot(base[:-1], cumulative, label=method, color=color)
    ax = plt.gca()
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Policy Cost / Optimal Cost', fontsize=18)
    plt.ylabel('Percentage', fontsize=18)
    plt.title(f'$\sigma={sigma}$', fontsize=18)
    plt.legend(loc='best', prop={'size':18})
    ax.tick_params(axis='both', labelsize=18)
    plt.savefig(f"{save_path}/sto_cmp_{sigma}.png", dpi=500, bbox_inches="tight")
    plt.close()
