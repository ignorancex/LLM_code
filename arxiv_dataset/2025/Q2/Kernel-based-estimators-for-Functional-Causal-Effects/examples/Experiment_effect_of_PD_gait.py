# -*- coding: utf-8 -*-
"""
Demo code for estiming the effect of Parkinson's disease on gait energy'
If you use this code or any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.
If you use the passive monitoring data shared as part of the study, also cite: 
Evers, L.J., Raykov, Y.P., Krijthe, J.H., Silva de Lima, A.L., Badawy, R., Claes, K., Heskes, T.M., Little, M.A., Meinders, M.J. and Bloem, B.R., 2020. Real-life gait performance as a digital biomarker for motor fluctuations: the Parkinson@ Home validation study. Journal of medical Internet research, 22(10), p.e19068.

@author: Yordan P. Raykov, Hengrui Luo, Justin Strait
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator
    
# Function to load all Excel files in the current folder
def load_excel_files(folder_path, file_names):
    data_dict = {}
    for file_name in file_names:
        file_path = os.path.join(folder_path, f"{file_name}.xlsx")  # Construct the full path
        try:
            data = pd.read_excel(file_path, header=None)
            data_dict[file_name] = data
            print(f"Loaded {file_name} with shape: {data.shape}")
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
    return data_dict

# Plot the data
def plot_data_by_group(loaded_data, tremor_group, group_colors):
    plt.figure(figsize=(12, 8))

    # Iterate through loaded_data and tremor_group
    for i, (file_name, data) in enumerate(loaded_data.items()):
        if i < len(tremor_group):  # Ensure tremor_group index matches
            group = tremor_group[i]  # Get the group number
            color = group_colors.get(group, "black")  # Default to black if group not found

            # Plot the data (assuming data is in the first column)
            curve = data.iloc[:, 0].values  # Extract the curve data
            plt.plot(curve, color=color, label=f"Group {group}" if f"Group {group}" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Add labels, legend, and title
    plt.xlabel("Time of Day (Hours)")
    plt.ylabel("Value")
    plt.title("Tremor probability conditioned on non-gait activity observed")
    plt.legend(loc="upper right")
    plt.show()
    

def enrich_loaded_data_with_metadata(loaded_data, loaded_metadata):
    """
    Enriches `loaded_data` dictionary with corresponding metadata for patients.

    Args:
        loaded_data (dict): Dictionary where keys contain patient IDs in the format 'hbvXXX'.
        loaded_metadata (dict): Dictionary where patient metadata is stored with 'Record Id' as key.

    Returns:
        dict: Updated `loaded_data` with metadata entries for corresponding patient IDs.
    """
    # Define the metadata fields to copy
    metadata_fields = ["year_of_birth", "gender", "hours_awake", "hours_off", "continuous_gait_problems", "episodic_gait_problems", "year_diagnosis"]

    # Normalize "Record Id" entries in loaded_metadata
    normalized_metadata_ids = [record_id.strip().lower() for record_id in loaded_metadata["Record Id"]]

    # Iterate over the keys in loaded_data
    for key in loaded_data.keys():
        # Extract and normalize the patient ID (e.g., 'hbvXXX') from the key
        patient_id = key.split('_')[-1].strip().lower()

        # Debugging: Print the extracted patient ID
        print(f"Processing key: {key}")
        print(f"Extracted patient_id: {patient_id}")

        # Locate the corresponding metadata entry in loaded_metadata
        if patient_id in normalized_metadata_ids:
            # Find the index of the patient ID in the normalized list
            index = normalized_metadata_ids.index(patient_id)

            # Extract the metadata fields for the patient
            metadata = {field: loaded_metadata[field][index] for field in metadata_fields if field in loaded_metadata}

            # Debugging: Ensure metadata is correctly formatted
            print(f"Metadata for {patient_id}: {metadata}")

            # Update loaded_data: ensure it is compatible with Pandas and dictionary structure
            if isinstance(loaded_data[key], dict):
                loaded_data[key].update(metadata)
            else:
                loaded_data[key] = {"data": loaded_data[key], **metadata}
        else:
            print(f"Metadata for patient ID '{patient_id}' not found in loaded_metadata.")

    return loaded_data

# Define the path to your folder
folder_path = 'data - PD@Home\patient curves gait'

if os.path.exists(folder_path):
    print(f"Contents of '{folder_path}':")
    for file_name in os.listdir(folder_path):
        print(file_name)
else:
    print(f"The folder '{folder_path}' does not exist. Please check the path.")


# List of Excel file names
file_names = [
    "mean_gait_energy_hbv002",
    "mean_gait_energy_hbv012",
    "mean_gait_energy_hbv013",
    "mean_gait_energy_hbv017",
    "mean_gait_energy_hbv018",
    "mean_gait_energy_hbv022",
    "mean_gait_energy_hbv024",
    "mean_gait_energy_hbv038",
    "mean_gait_energy_hbv043",
    "mean_gait_energy_hbv047",
    "mean_gait_energy_hbv051",
    "mean_gait_energy_hbv063",
    "mean_gait_energy_hbv072",
    "mean_gait_energy_hbv074",
    "mean_gait_energy_hbv083",
    "mean_gait_energy_hbv084",
    "mean_gait_energy_hbv087",
    "mean_gait_energy_hbv090",
    "mean_gait_energy_hbv091",
    "mean_gait_energy_hbv093",
    "mean_gait_energy_hbv099",
    "mean_gait_energy_hbv100"
]

# Tremor group label: 1 - PD participants, 0 - Non-PD controls 
PD_group = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0]


# Load the files
loaded_data = load_excel_files(folder_path, file_names)

# Inspect the loaded data
for file_name, data in loaded_data.items():
    print(f"\nFile: {file_name}")
    print(data.head())  # Display the first few rows of the sheet

group_colors = {0: "blue", 1: "red"}  # Map group numbers to colors


# Call the function
plot_data_by_group(loaded_data, PD_group, group_colors)

# Load the files
file_path = os.path.join(folder_path, "Home_based_validation_excel_export_metadata.xlsx")  # Construct the full path
loaded_metadata = pd.read_excel(file_path)


updated_loaded_data = enrich_loaded_data_with_metadata(loaded_data, loaded_metadata)

###### First method - (1) Inverse Probability Weighting Estimator (with local smoothing)  from (Abrevaya, Hsu and Lieli, 2015) ######
n_timepoints = len(updated_loaded_data['mean_gait_energy_hbv002']['data']) # all curves are same length so pick one
varphi_ipw_ate = np.zeros((n_timepoints,))
varphi_ipw_cate_error = np.zeros((n_timepoints,))

## Instantiate treatment variables
D = np.array(PD_group) # Label for PD status {PD vs Control}
Y = [ updated_loaded_data[key]['data']
    for key in updated_loaded_data
    if key.startswith('mean_gait_energy_hbv')
]
Y = np.array(Y) # shape (number of participants, n_timepoints, 1)
X = [ updated_loaded_data[key]['gender'] # Example covariates which confounds the effect
    for key in updated_loaded_data
    if key.startswith('mean_gait_energy_hbv')]
X = np.array(X)


# Initialize the logistic regression model
loo = LeaveOneOut()
n_samples = len(X)
varphi = np.zeros((n_samples, int(n_timepoints)))

# Leave-One-Out Cross-Validation loop
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]

    # Train the logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train.reshape(-1, 1), D_train)

    # Compute the propensity score for the test observation
    propensity = logistic_model.predict_proba(X_test.reshape(-1, 1))[0]

    # Compute the estimated treatment effect for the test observation
    varphi[test_index,:] = D_test * np.squeeze(Y[test_index,:],axis=2) / propensity[1] - (1 - D_test) * np.squeeze(Y[test_index,:],axis=2) / (1 - propensity[1])


# Compute the average treatment effect (ATE) and its standard deviation
ipw_ate = np.mean(varphi,axis=0)
ipw_std_ate = np.std(varphi,axis=0)

print("Estimated Average Treatment Effect (ATE):", ipw_ate)
print("Standard Deviation of ATE:", ipw_std_ate)

plt.figure(figsize=(10, 6))  
# x-axis range for visualization
x = np.linspace(0, 24, n_timepoints)  # Arbitrary range for illustration
# Plot the mean effect as a horizontal line
plt.plot(x, ipw_ate, label="Mean Effect (ATE)", color="blue")

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, ipw_ate + ipw_std_ate, 'r--', label="+1 Std Dev")
plt.plot(x, ipw_ate - ipw_std_ate, 'r--', label="-1 Std Dev")

# Add labels and legend
plt.xlabel("X-axis (example, e.g., sample index or feature)")
plt.ylabel("Estimated Effect")
plt.title("Average Treatment Effect (ATE) with Standard Deviation")
plt.legend()
plt.grid(True)
plt.show()

## Plotting the distribution of the potential outcomes 

from scipy.stats import gaussian_kde

potential_outcome1 = np.squeeze(Y[test_index,:],axis=2) / propensity[1]
potential_outcome2 = np.squeeze(Y[test_index,:],axis=2) / (1 - propensity[1])
# Kernel Density Estimates
kde1 = gaussian_kde(potential_outcome1.ravel())
kde2 = gaussian_kde(potential_outcome2.ravel())

# X-axis range for KDE
x_min = min(potential_outcome1.min(), potential_outcome2.min())
x_max = max(potential_outcome1.max(), potential_outcome2.max())
x_vals = np.linspace(x_min, x_max, 1000)

# Calculate averages
avg_potential_outcome1 = np.mean(potential_outcome1)
avg_potential_outcome2 = np.mean(potential_outcome2)


# Reduce figure width
plt.figure(figsize=(15, 8)) 

# Compute the minimum of the two KDE curves for shading
x_common = np.linspace(min(x_vals), max(x_vals), 500)  # Ensure a common x range
y_min = np.minimum(kde1(x_common), kde2(x_common))  # Take the lower density at each x

# Plot KDEs
plt.plot(x_vals, kde1(x_vals), label='PD', lw=2)
plt.plot(x_vals, kde2(x_vals), label='No-PD Control', lw=2)

# Shade the difference area with more transparency
plt.fill_between(x_common, kde1(x_common), kde2(x_common), where=(kde1(x_common) > kde2(x_common)), 
                 color='blue', alpha=0.1, label="More likely PD Gait Energy")  # Reduced alpha for more transparency
plt.fill_between(x_common, kde1(x_common), kde2(x_common), where=(kde1(x_common) < kde2(x_common)), 
                 color='orange', alpha=0.1, label="More likely Non-PD Control Gait Energy")  # Reduced alpha

# Add vertical lines for averages
plt.axvline(avg_potential_outcome1, color='blue', linestyle='--', label=r'$\mathbb{E}[Y^{(1)}]$ (PD Gait Energy)')
plt.axvline(avg_potential_outcome2, color='orange', linestyle='--', label=r'$\mathbb{E}[Y^{(0)}]$ (Non-PD Control Gait Energy)')

# Customize plot with larger fonts
plt.title("Distribution of Potential Outcomes", fontsize=26)  # Increased font size
plt.xlabel("Potential Outcomes", fontsize=26)
plt.ylabel("Density", fontsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(True)

# Adjust layout to make space below the plot
plt.subplots_adjust(bottom=0.35)

# Move legend below the plot with larger text
plt.legend(fontsize=26, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show plot
plt.show()

## Second benchmark using kernel methods 
kernel__lambd1 = 0.01
kernel_sigma=0.05

varphi_kernel = np.zeros((n_samples, int(n_timepoints)))
# Leave-One-Out Cross-Validation loop
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Step 3: Extract the best lambda value and re-use them in simulations
    best_srvf_lambd1 = kernel__lambd1
    flag_srvf_ridge_penalties = 2 # no need to re-estimate best parameter values for the same n_samples

    operator_kernel_SRVF_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=best_srvf_lambd1, kernel_sigma = kernel_sigma, use_operator_valued_kernel=True, apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False)  # Apply SRVF registration to the treated group)
    theta_srvf_ate_vector_per_t = operator_kernel_SRVF_causal_estimator.fit(X=X_train.reshape(-1, 1), D=D_train.reshape(-1, 1), Y=Y_train[:,:,0])
    predictions_ate_srvf = operator_kernel_SRVF_causal_estimator.predict(X_new = X_test.reshape(-1, 1), D_new = D_test.reshape(-1, 1)) # per covariate value
    varphi_kernel[test_index,:] = predictions_ate_srvf.flatten()

  # Compute the average treatment effect (ATE) and its standard deviation
kernel_ate = np.mean(varphi_kernel,axis=0)
kernel_std_ate = np.std(varphi_kernel,axis=0)

print("Estimated Average Treatment Effect (ATE):", ipw_ate)
print("Standard Deviation of ATE:", ipw_std_ate)


plt.figure(figsize=(10, 6))  
# x-axis range for visualization
x = np.linspace(0, 24, n_timepoints)  # Hour of day range for illustration
# Plot the mean effect as a horizontal line
plt.plot(x, kernel_ate, label="Mean Effect (ATE)", color="blue")

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, kernel_ate + kernel_std_ate, 'r--', label="+1 Std Dev")
plt.plot(x, kernel_ate - kernel_std_ate, 'r--', label="-1 Std Dev")

# Add labels and legend
plt.xlabel("X-axis (example, e.g., sample index or feature)")
plt.ylabel("Estimated Effect")
plt.title("Average Treatment Effect (ATE) with Standard Deviation")
plt.legend()
plt.grid(True)
plt.show()

# Generate time values from 00:00 to 24:00
n_timepoints = len(kernel_ate)  # Ensure alignment with your data
x = np.linspace(0, 24, n_timepoints)  # 24-hour range

# Convert x-axis to HH:MM format
time_labels = [f"{int(h):02d}:00" for h in np.linspace(0, 24, 7)]  # 00:00, 04:00, ..., 24:00

# Set figure size
plt.figure(figsize=(10, 6))

# Plot the mean effect as a solid line
plt.plot(x, kernel_ate, label="Mean Effect (ATE)", color="blue", lw=2)

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, kernel_ate + kernel_std_ate, 'r--', label="+1 Std Dev", lw=2)
plt.plot(x, kernel_ate - kernel_std_ate, 'r--', label="-1 Std Dev", lw=2)

# Fill between standard deviation bounds with transparency
plt.fill_between(x, kernel_ate - kernel_std_ate, kernel_ate + kernel_std_ate, color='red', alpha=0.1)

# Customize labels and title with larger fonts
plt.xlabel("Time (HH:MM)", fontsize=22)
plt.ylabel("Estimated Treatment Effect", fontsize=22)
plt.title("Average Treatment Effect (ATE) with Standard Deviation", fontsize=24)

# Format x-axis ticks as HH:MM
plt.xticks(np.linspace(0, 24, 7), time_labels, fontsize=18)  # Spacing 00:00, 04:00, ..., 24:00
plt.yticks(fontsize=18)

# Add grid for readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.3)

# Move legend below the plot with larger text
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show the figure
plt.show()