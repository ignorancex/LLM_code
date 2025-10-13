import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the font sizes
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

# Set the general font size
plt.rcParams['font.size'] = TICK_SIZE

# Data dictionary with all models and their metrics
data = {
    'AlexNet KAN': {
        'Accuracy': 42.79,
        'Loss': 2.62,
        'Inference Time': 0.0074,
        'FLOPs': 1611568352,
        'Parameters': 39756776
    },
    'AlexNet': {
        'Accuracy': 56.62,
        'Loss': 2.31,
        'Inference Time': 0.0018,
        'FLOPs': 714197696,
        'Parameters': 61100840
    },
    'LeNet KAN': {
        'Accuracy': 98.81,
        'Loss': 0.036,
        'Inference Time': 0.003,
        'FLOPs': 3298728,
        'Parameters': 82128
    },
    'LeNet': {
        'Accuracy': 98.89,
        'Loss': 0.031,
        'Inference Time': 0.0007,
        'FLOPs': 429128,
        'Parameters': 61750
    },
    'Tabular CKAN': {
        'Accuracy': 45.09,
        'Loss': 0.0172,
        'Inference Time': 0.0001,
        'FLOPs': 79861586,
        'Parameters': 6265998
    },
    'Tabular CNN': {
        'Accuracy': 47.61,
        'Loss': 0.0167,
        'Inference Time': 0.00004,
        'FLOPs': 37853010,
        'Parameters': 7818482
    }
}

def normalize_metric_ratio(val1, val2, inverse=False):
    """Normalize based on ratio between two values"""
    if val1 == val2:
        return 0.5, 0.5
    
    # For metrics where lower is better, invert the values
    if inverse:
        val1, val2 = 1/val1, 1/val2
    
    # Calculate ratio and normalize to [0,1]
    total = val1 + val2
    return val1/total, val2/total

def normalize_pair_data(model1_data, model2_data, inverse_metrics):
    """Normalize data for a pair of models using ratios"""
    normalized_data = {}, {}
    
    for metric in model1_data.keys():
        val1, val2 = model1_data[metric], model2_data[metric]
        inverse = metric in inverse_metrics
        norm1, norm2 = normalize_metric_ratio(val1, val2, inverse)
        normalized_data[0][metric] = norm1
        normalized_data[1][metric] = norm2
    
    return normalized_data[0], normalized_data[1]

# Create figure with three subplots
fig = plt.figure(figsize=(20, 7))

# Model pairs to compare
model_pairs = [
    ('AlexNet KAN', 'AlexNet'),
    ('LeNet KAN', 'LeNet'),
    ('Tabular CKAN', 'Tabular CNN')
]

# Categories and which ones should be inverted (lower is better)
categories = ['Accuracy', 'Loss', 'Inference Time', 'FLOPs', 'Parameters']
inverse_metrics = {'Loss', 'Inference Time', 'FLOPs', 'Parameters'}

for idx, (model1, model2) in enumerate(model_pairs, 1):
    # Create subplot
    ax = plt.subplot(1, 3, idx, polar=True)
    
    # Normalize data for this pair
    norm_data1, norm_data2 = normalize_pair_data(data[model1], data[model2], inverse_metrics)
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Get normalized values
    values1 = [norm_data1[cat] for cat in categories]
    values2 = [norm_data2[cat] for cat in categories]
    
    # Add the first value again to close the polygon
    values1 += values1[:1]
    values2 += values2[:1]
    
    # Plot
    ax.plot(angles, values1, 'o-', linewidth=3, label=model1)
    ax.fill(angles, values1, alpha=0.1)
    ax.plot(angles, values2, 'o-', linewidth=3, label=model2)
    ax.fill(angles, values2, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=LABEL_SIZE)
    
    # Set the labels
    ax.set_rlabel_position(0)
    ax.set_rticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], size=TICK_SIZE)
    
    # Add legend with larger font
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=LEGEND_SIZE)
    
    # Add title with larger font and increased spacing
    plt.title(f'{model1} vs {model2}', size=TITLE_SIZE, y=1.15, weight='bold')
    
    # Add gridlines
    ax.grid(True)

plt.suptitle('Model Comparison Radar Plots\nHigher values indicate better performance in the specific metric', 
             size=TITLE_SIZE, weight='bold', y=1.05)

plt.tight_layout()
plt.savefig("graphs/model_comparison_radar_plots.png", dpi=300, bbox_inches="tight")
plt.show()
