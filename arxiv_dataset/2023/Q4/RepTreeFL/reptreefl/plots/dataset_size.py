import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from constants import *
from utils.utils import *
from plots import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

manual_seed = 1773
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

create_directories_for_saving()

mae_all = {}
methods_per = []
methods = ['Standalone', 'FedAvg', 'FedDyn', 'FedDC', 'RepFL', 'ReplicaFL']
# no_samples = ['20', '50', '100', '200']
no_samples = ['20', '100', '200']

# Add names of the methods here
run_names = []

print(run_names)

for method in methods:
    for per in no_samples:
        methods_per.append(method + per)
        for client in range(constants.NUMBER_CLIENTS):
            mae_all[method + per + str(client)] = []

for mp, run_name in zip(methods_per, run_names):
    for fold in range(NUMBER_FOLDS):
        mae_current_fold = read_acc_from_file(fold, run_name)
        for client in range(constants.NUMBER_CLIENTS):
            mae_all[mp + str(client)].append(mae_current_fold[client])

# one plot per client
for client in range(constants.NUMBER_CLIENTS):
    data = []
    for method in methods:
        for per in no_samples:
            values = mae_all[method + per + str(client)]
            for val in values:
                val = val.tolist()
                print(val)
                data.append([method, per, val])

    # Create a DataFrame
    columns = ['Method', 'Per', 'Run']
    df = pd.DataFrame(data, columns=columns)

    print(df)

    plot_colours = ['turquoise', 'purple', 'palevioletred', 'coral', 'deepskyblue', 'cornflowerblue']
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    # sns.lineplot(
    #     data=df,
    #     x="Per", y="Run", hue="Method", style="Method", palette=plot_colours,
    #     markers=True, dashes=False
    # )
    ax = sns.barplot(data=df, x='Per', y='Run', hue='Method', errorbar='sd', palette=plot_colours, alpha=0.8)
    plt.ylim([0.65, 1.0])
    ax.legend_.remove()  # Remove the legend

    # plt.title('MAE with Standard Deviation across 5 Runs')
    plt.xlabel('Dataset Size')
    plt.ylabel('Acc')

    # plt.show()
    plt.savefig(f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}/dataset_size_client{client}.png')

    plt.close()


# with legend
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Per', y='Run', hue='Method', errorbar='sd', palette=plot_colours, alpha=0.8)

plt.ylim([0.95, 1.0])

plt.xlabel('Depth')
plt.ylabel('Acc')

plt.savefig(f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}/dataset_size_legend.png')

plt.close()