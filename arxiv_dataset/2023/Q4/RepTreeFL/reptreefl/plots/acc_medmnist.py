import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import random

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

# Add names of the methods here
run_names = []


methods = ['Standalone', 'FedAvg', 'FedDyn', 'FedDC', 'RepFL', 'ReplicaFL']
index_to_client = {'0': 'H1', '1': 'H2', '2': 'H3'}

mae_all = {}

for client in range(NUMBER_CLIENTS):
    for method in methods:
        mae_all[str(client) + method] = []

for method, run_name in zip(methods, run_names):
    for fold in range(NUMBER_FOLDS):
        print(run_name)
        mae_current_fold = read_acc_from_file(fold, run_name)
        for client in range(NUMBER_CLIENTS):
            mae_all[str(client) + method].append(mae_current_fold[client])


data = []
for client in range(NUMBER_CLIENTS):
    for method in methods:
        values = mae_all[str(client) + method]
        for val in values:
            val = val.tolist()
            print(val)
            data.append([index_to_client[str(client)], method, val])

# Create a DataFrame
columns = ['Client', 'Method', 'Run']
df = pd.DataFrame(data, columns=columns)

print(df)

# Create the bar plot using Seaborn
plot_colours = ['turquoise', 'purple', 'palevioletred', 'coral', 'deepskyblue', 'cornflowerblue']
# plot_colours = ['turquoise', 'purple', 'palevioletred', 'coral', 'mediumblue', 'cornflowerblue']
# sns.set_palette("Paired")
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Client', y='Run', hue='Method', errorbar='sd', palette=plot_colours, alpha=0.8)
# ax.legend_.remove()  # Remove the legend

# # pneumonia
# plt.ylim([0.91, 0.99])

plt.ylim([0.58, 0.9])

# plt.title('MAE with Standard Deviation across 5 Runs')
plt.xlabel('Hospital')
plt.ylabel('Accuracy')
# plt.legend(loc='upper right')
# plt.show()

plt.savefig(f'{SAVING_FOLDER_PATH}/{GLOBAL_RUN_NAME}/average_runs_acc.png')

plt.close()


# # With legend
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(data=df, x='Client', y='Run', hue='Method', errorbar='sd', palette=plot_colours, alpha=0.8)
# # ax.legend_.remove()  # Remove the legend

# plt.ylim([0.93, 1.0])

# # plt.title('MAE with Standard Deviation across 5 Runs')
# plt.xlabel('Hospital')
# plt.ylabel('Accuracy')
# # plt.legend(loc='upper right')
# # plt.show()

# plt.savefig(f'{SAVING_FOLDER_PATH}/{GLOBAL_RUN_NAME}/average_runs_acc_legend.png')

# plt.close()