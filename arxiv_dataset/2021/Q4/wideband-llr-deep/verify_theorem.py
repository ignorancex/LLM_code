#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# Meta-params
num_layers  = np.arange(2, 6, dtype=int)
input_array = [4, 6, 8]

# Plot
linewidth  = 3.7
markersize = 12
alpha      = 0.7
plt.rcParams['font.size'] = 24
plt.figure(figsize=(13, 10))
colors = ['orange', 'blue', 'green']
star_colors = ['red', 'cyan', 'lime']

# Outputs
var_theory_array = []
var_emp_array    = []
percentile_emp_array = []

# For each input size
for mod_idx, input_size in enumerate(input_array):
    hidden_size = 4 * input_size
    output_size = 1
    # Statistical params
    input_std  = np.sqrt(2 / (input_size + hidden_size))
    hidden_std = np.sqrt(2 / (hidden_size + hidden_size))
    output_std = np.sqrt(2 / (hidden_size + 1))
    
    # Params
    batch_size  = 20000
    
    # All possible inputs
    input_vector = np.unpackbits(np.arange(
        2**input_size, dtype=np.uint8)).reshape(-1, 8)[:, -input_size:]
    input_vector = 2*input_vector - 1.
    
    # For each network depth
    variance_array   = np.zeros(len(num_layers))
    percentile_array = np.zeros(len(num_layers))
    mean_array       = np.zeros(len(num_layers))
    # All logs
    var_log  = []
    mean_log = []
    
    for idx, depth in enumerate(num_layers):
        # Start from the input
        signal = np.copy(input_vector)[None, ..., None]
        # Input layer
        input_weights = input_std * \
            np.random.normal(size=(batch_size, 1, hidden_size, input_size))
        signal = np.matmul(input_weights, signal)
        signal = np.maximum(signal, 0.)
        var_log.append(np.var(signal))
        mean_log.append(np.mean(signal))
        
        # For all hidden layers
        for hidden_idx in range(depth-2):
            # Hidden layer
            hidden_weights = hidden_std * \
                np.random.normal(size=(batch_size, 1, hidden_size, hidden_size))
            signal = np.matmul(hidden_weights, signal)
            signal = np.maximum(signal, 0.)
            var_log.append(np.var(signal))
            mean_log.append(np.mean(signal))
            
        # Output layer
        output_weights = output_std * \
            np.random.normal(size=(batch_size, 1, 1, hidden_size))
        signal = np.matmul(output_weights, signal)
        # No activation
        
        # Get variance
        variance_array[idx] = np.var(signal)
        percentile_array[idx] = np.percentile(np.tanh(np.abs(signal)).flatten(),
                                              q=99.9)
        mean_array[idx]     = np.mean(signal)
        
    # Theoretical expression - only for one hidden layer
    var_theory = 8/5 * input_size / (4*input_size + 1)

    plt.plot(num_layers-1, np.sqrt(variance_array), 
              linewidth=linewidth, marker='o',
              linestyle='dashed',
              markeredgecolor='k',
              markersize=markersize,
              label=r'$\hat{\sigma}_z$ ($K$=%d)' % input_size,
              color=colors[mod_idx],
              zorder=-1)
    # Also plot effective span
    plt.plot(num_layers-1, percentile_array,
              marker='s', markersize=markersize,
              linestyle='solid',
              linewidth=linewidth,
              label=r'$\hat{P}_{99.9, z}$ ($K$=%d)' % input_size, color=colors[mod_idx],
              zorder=-1)
    
    plt.scatter(1, np.sqrt(var_theory), marker='*',
                color=star_colors[mod_idx],
                s=650, label=r'$\sigma_z$ (K=%d)' % input_size,
                edgecolors='k',
                alpha=0.5, zorder=1)
    
    # Collect stuff
    var_theory_array.append(var_theory)
    var_emp_array.append(variance_array)
    percentile_emp_array.append(percentile_array)
    
# Convert to arrays
var_emp_array = np.asarray(var_emp_array)
percentile_emp_array = np.asarray(percentile_emp_array)

plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 3, 4, 5] # Handmade
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower center', bbox_to_anchor=(0.5, 0.),
    fancybox=False, shadow=False, ncol=3, prop={'size': 22})
plt.xlabel('Num. Hidden Layers')
plt.ylabel('Magnitude')
plt.ylim([-0.03, 1.03])
plt.xticks(num_layers-1)
# plt.savefig('theory.png', dpi=300)
plt.show()

# A different style of plot
# Theoretical sigma
plt.figure(figsize=(13, 10))
plt.plot(input_array, np.sqrt(var_theory_array), 
         linewidth=linewidth, marker='*',
         linestyle='--',
         markeredgecolor='k',
         markersize=markersize*2,
         label=r'$\sigma_z$ (1 hidden layer)', color=colors[0],
         zorder=-1)
# Empirical sigma
plt.plot(input_array, np.sqrt(var_emp_array[:, 0]),
         linewidth=linewidth, marker='o',
         linestyle='dashed',
         markeredgecolor='k',
         markersize=markersize,
         label=r'$\hat{\sigma}_z$ (1 hidden layer)', color=colors[0],
         zorder=1)
plt.plot(input_array, np.sqrt(var_emp_array[:, -1]),
         linewidth=linewidth, marker='o',
         linestyle='dashed',
         markeredgecolor='k',
         markersize=markersize,
         label=r'$\hat{\sigma}_z$ (4 hidden layers)', color=colors[1],
         zorder=1)
# Empirical ranges
plt.plot(input_array, np.sqrt(percentile_emp_array[:, 0]),
         linewidth=linewidth, marker='s',
         linestyle=':',
         markeredgecolor='k',
         markersize=markersize,
         label=r'$\hat{P}_{99.9, z}$ (1 hidden layer)', color=colors[0],
         zorder=1)
plt.plot(input_array, np.sqrt(percentile_emp_array[:, -1]),
         linewidth=linewidth, marker='s',
         linestyle=':',
         markeredgecolor='k',
         markersize=markersize,
         label=r'$\hat{P}_{99.9, z}$ (4 hidden layers)', color=colors[1],
         zorder=1)
# Neutral plot for fake legend
plt.plot(np.NaN, np.NaN, '-', color='none', label=' ')

handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 5, 1, 2, 3, 4] # Handmade
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower center', bbox_to_anchor=(0.5, 0.),
    fancybox=False, shadow=False, ncol=3, prop={'size': 15})

plt.grid()
plt.ylim([-0.03, 1.03])
plt.xlabel(r'Input size $(K)$')
plt.ylabel('Magnitude')
plt.xticks(input_array)
plt.show()