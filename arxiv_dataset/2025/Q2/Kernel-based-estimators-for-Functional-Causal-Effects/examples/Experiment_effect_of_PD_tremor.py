# -*- coding: utf-8 -*-
"""
Demo code for estiming the effect of Parkinson's disease category on estimated tremor duration'
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

#from utils import rbf_kernel, nadaraya_watson, fisher_rao_distance, evaluate_functional_model
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

    

folder_path = 'data - PD@Home\patient curves tremor'

# List of Excel file names
file_names = [
    "mean_conditional_tremor_given_nongait_hbv002",
    "mean_conditional_tremor_given_nongait_hbv012",
    "mean_conditional_tremor_given_nongait_hbv013",
    "mean_conditional_tremor_given_nongait_hbv017",
    "mean_conditional_tremor_given_nongait_hbv018",
    "mean_conditional_tremor_given_nongait_hbv022",
    "mean_conditional_tremor_given_nongait_hbv023",
    "mean_conditional_tremor_given_nongait_hbv024",
    "mean_conditional_tremor_given_nongait_hbv038",
    "mean_conditional_tremor_given_nongait_hbv043",
    "mean_conditional_tremor_given_nongait_hbv047",
    "mean_conditional_tremor_given_nongait_hbv051",
    "mean_conditional_tremor_given_nongait_hbv054",
    "mean_conditional_tremor_given_nongait_hbv063",
    "mean_conditional_tremor_given_nongait_hbv072",
    "mean_conditional_tremor_given_nongait_hbv074",
    "mean_conditional_tremor_given_nongait_hbv083",
    "mean_conditional_tremor_given_nongait_hbv084",
    "mean_conditional_tremor_given_nongait_hbv087",
    "mean_conditional_tremor_given_nongait_hbv090",
    "mean_conditional_tremor_given_nongait_hbv091",
    "mean_conditional_tremor_given_nongait_hbv093",
    "mean_conditional_tremor_given_nongait_hbv099",
    "mean_conditional_tremor_given_nongait_hbv100"
]

# Tremor group label: 1- PD with tremor, 2-PD with no annotated tremor, 3-Non-PD controls 
tremor_group = [2,1,1,1,1,1,1,2,1,2,2,2,2,2,3,2,3,3,3,1,3,3,3,3]


# Load the files
loaded_data = load_excel_files(folder_path, file_names)

# Inspect the loaded data
for file_name, data in loaded_data.items():
    print(f"\nFile: {file_name}")
    print(data.head())  # Display the first few rows of the sheet

group_colors = {1: "blue", 2: "green", 3: "red"}  # Map group numbers to colors

# Call the function
plot_data_by_group(loaded_data, tremor_group, group_colors)

file_path = os.path.join(folder_path, "Home_based_validation_excel_export_metadata.xlsx")  # Construct the full path
loaded_metadata = pd.read_excel(file_path)

updated_loaded_data = enrich_loaded_data_with_metadata(loaded_data, loaded_metadata)


# -------  First method - (1) Inverse Probability Weighting Estimator (with local smoothing)  from (Abrevaya, Hsu and Lieli, 2015) ------- 
n_timepoints = len(updated_loaded_data['mean_conditional_tremor_given_nongait_hbv002']['data']) # all curves are same length so pick one
varphi_ipw_ate = np.zeros((n_timepoints,))
varphi_ipw_cate_error = np.zeros((n_timepoints,))

## Instantiate treatment variables
#binary_tremor_label = np.where(tremor_group <= 2, 1, 2) # Label for tremor status {Tremor-PD, Non-Tremor-PD, non-PD-Controls}
D = np.where(np.array(tremor_group) <= 1, 1, 0) # Label for tremor status {Tremor-PD, Non-Tremor-PD, non-PD-Controls}
Y = [ updated_loaded_data[key]['data']
    for key in updated_loaded_data
    if key.startswith('mean_conditional_tremor_given_nongait_hbv')
]
Y = np.array(Y) # shape (number of participants, n_timepoints, 1)
X = [  (updated_loaded_data[key]['gender'], updated_loaded_data[key]['hours_awake']) # Example covariates which confounds the effect
    for key in updated_loaded_data
    if key.startswith('mean_conditional_tremor_given_nongait_hbv')]

X = np.array(X)
X = np.nan_to_num(X, nan=0)

from sklearn.preprocessing import StandardScaler
# Extract the columns to scale (year_of_birth and year_diagnosis)
columns_to_scale = X[:, 1:]  # Selecting only the last two columns
# Scale the columns
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(columns_to_scale)
# Replace the scaled columns back into X
X[:, 1:] = scaled_columns



# Initialize the logistic regression model
loo = LeaveOneOut()
n_samples = len(X)
varphi = np.zeros((n_samples, int(n_timepoints)))
D_test_values = np.zeros((n_samples,))
potential_daily_outcome1 = np.zeros((n_samples,))
potential_daily_outcome2 = np.zeros((n_samples,))
Y_potential_daily_1 = np.zeros((n_samples, int(n_timepoints)))
Y_potential_daily_2 = np.zeros((n_samples, int(n_timepoints)))

# Leave-One-Out Cross-Validation, evaluate out-of-sample the ATE using leave-one-subject-out
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]

    # Train the logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, D_train)

    # Compute the propensity score for the test observation
    propensity = logistic_model.predict_proba(X_test)[0][1] # probability of D = 1, which is the PD group

    # Compute the estimated treatment effect for the test observation
    for j in range(Y.shape[1]):
        if np.squeeze(Y[test_index,j]) > 0.1:
            varphi[test_index,j] = (D_test * np.squeeze(Y[test_index,j]) / propensity) - ((1 - D_test) * np.squeeze(Y[test_index,j]) / (1 - propensity))
        else:
            varphi[test_index,j] = 0
    Y_potential_daily_1[test_index,:] = D_test *np.squeeze(Y[test_index,:],axis=2) / propensity
    Y_potential_daily_2[test_index,:] = (1 - D_test)*np.squeeze(Y[test_index,:],axis=2) / (1 - propensity)
    D_test_values[test_index] = D_test

    mask1 = Y_potential_daily_1[test_index,:] > 0.001
    filtered_Y1 = np.where(mask1, Y_potential_daily_1[test_index,:], np.nan)
    # Compute the mean across axis=1, ignoring NaNs
    potential_daily_outcome1[test_index] = np.nanmean(filtered_Y1, axis=1)

    mask2 = Y_potential_daily_2[test_index,:] > 0.001
    filtered_Y2 = np.where(mask2, Y_potential_daily_2[test_index,:], np.nan)
    potential_daily_outcome2[test_index] = np.nanmean(filtered_Y2, axis=1)

potential_daily_outcome1 =  potential_daily_outcome1[~np.isnan(potential_daily_outcome1)]
potential_daily_outcome2 =  potential_daily_outcome2[~np.isnan(potential_daily_outcome2)]
# Compute single potential outcome per daily gait energy and curve potential outcome with values per time-of-day
# Potential outcomes which are Nan are replace with 0

# First, compute average daily score of gait_energy per unit of time (15 minute window block)
# Create a mask for values greater than 0.1, threshold for considering actual walking occured (conservative threshold based on energy at the shortest bouts during visits)
mask = varphi > 0.001
# Replace values below 0.1 with NaN
filtered_varphi = np.where(mask, varphi, np.nan)
# Compute the mean across axis=1, ignoring NaNs
mean_daily_values = np.nanmean(filtered_varphi, axis=1)

# Daily scores ate estimates
ipw_ate_daily = np.nanmean(mean_daily_values) # Average change in gait energy for PD vs Control
ipw_std_ate_daily = np.nanstd(mean_daily_values)

print("Estimated Average Treatment Effect (ATE) for daily tremor score:", ipw_ate_daily)
print("Standard Deviation of ATE for daily tremor score:", ipw_std_ate_daily)

## Plotting the distribution of the potential outcomes for daily scores

from scipy.stats import gaussian_kde

# Kernel Density Estimates
kde1 = gaussian_kde(potential_daily_outcome1.ravel())
kde2 = gaussian_kde(potential_daily_outcome2.ravel())

# X-axis range for KDE
x_min = min(potential_daily_outcome1.min(), potential_daily_outcome2.min())
x_max = max(potential_daily_outcome1.max(), potential_daily_outcome2.max())
x_vals = np.linspace(x_min, x_max, 1000)

# Calculate averages
avg_daily_potential_outcome1 = np.mean(potential_daily_outcome1)
avg_daily_potential_outcome2 = np.mean(potential_daily_outcome2)



# Increase figure width to accommodate legend
plt.figure(figsize=(10, 6))  # Maintain consistent figure width

# Plot KDEs
plt.plot(x_vals, kde1(x_vals), label='PD', lw=2)
plt.plot(x_vals, kde2(x_vals), label='No-PD Control', lw=2)

# Shade the difference area with transparency
plt.fill_between(x_vals, kde1(x_vals), kde2(x_vals), where=(kde1(x_vals) > kde2(x_vals)), 
                 color='blue', alpha=0.1, label="More likely PD Tremor")
plt.fill_between(x_vals, kde1(x_vals), kde2(x_vals), where=(kde1(x_vals) < kde2(x_vals)), 
                 color='orange', alpha=0.1, label="More likely No-PD Tremor")

# Add vertical lines for averages
plt.axvline(avg_daily_potential_outcome1, color='blue', linestyle='--', label=r'$\mathbb{E}[Y^{(1)}]$ (PD Tremor)')
plt.axvline(avg_daily_potential_outcome2, color='orange', linestyle='--', label=r'$\mathbb{E}[Y^{(0)}]$ (No-PD Tremor)')

# Customize plot
plt.title("Distribution of Daily Potential Outcomes", fontsize=24)
plt.xlabel("Daily Potential Outcomes", fontsize=22)
plt.ylabel("Density", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to make space below the plot
plt.subplots_adjust(bottom=0.35)

# Move legend below the plot
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Save the figure in .eps format
plt.savefig("Distribution_potential_outcomes_tremor.eps", format='eps', bbox_inches='tight')

# Show plot
plt.show()


# ------- Second method - (2) Kernel Causal Effect Estimator with elastic registration of the outcomes ------- 

# Step 1: Define the hyperparameter search space for lambd1, lambd2 and lambd3
# Define the grid of hyperparameters


## Second benchmark using kernel methods
kernel__lambd1 = 0.01
kernel__lambd2 = 0.01
kernel__lambd3 = 0.01
kernel_sigma=0.05

varphi_kernel = np.zeros((n_samples, int(n_timepoints)))
D_test_values = np.zeros(n_samples)
# Leave-One-Out Cross-Validation loop
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    operator_kernel_SRVF_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=kernel__lambd1, kernel_sigma = kernel_sigma, use_operator_valued_kernel=True, apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False)  # Apply SRVF registration to the treated group)
    theta_srvf_ate_vector_per_t = operator_kernel_SRVF_causal_estimator.fit(X=X_train, D=D_train.reshape(-1, 1), Y=Y_train[:,:,0])
    predictions_ate_srvf = operator_kernel_SRVF_causal_estimator.predict(X_new = X_test, D_new = D_test.reshape(-1, 1)) # per covariate value
    varphi_kernel[test_index,:] = predictions_ate_srvf.flatten()
    D_test_values[test_index] = D_test

  # Compute the average treatment effect (ATE) and its standard deviation
kernel_ate = np.mean(varphi_kernel,axis=0)
kernel_std_ate = np.std(varphi_kernel,axis=0)

# x-axis range for visualization
x = np.linspace(0, 24, n_timepoints)  # Arbitrary range for illustration

# Increase figure width to accommodate legend
plt.figure(figsize=(10, 6))  # Increased width from 8 to 10

# Plot the mean effect as a line
plt.plot(x, kernel_ate, label="Mean Effect (ATE)", color="blue")

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, kernel_ate + kernel_std_ate, 'r--', label="+1 Std Dev")
plt.plot(x, kernel_ate - kernel_std_ate, 'r--', label="-1 Std Dev")

# Add labels and title
plt.xlabel("Time of day (h)", fontsize=18)
plt.ylabel("Estimated Effect", fontsize=18)
plt.title("Dynamic ATE with Standard Deviation", fontsize=18)

# Customize x-axis to show time format
tick_positions = np.linspace(0, 24, 5)  # Positions at 0, 6, 12, 18, 24 hours
tick_labels = [f"{int(hour):02d}:00" for hour in tick_positions]  # Generate time labels
plt.xticks(tick_positions, tick_labels, fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)

# Move legend outside the figure
plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1.02, 0.5))

# Adjust layout to accommodate the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusted for right padding

# Show plot
plt.show()
# Set figure size for consistency
plt.figure(figsize=(10, 6))

# Plot the mean effect as a solid line
plt.plot(x, kernel_ate, label="Mean Effect (ATE)", color="blue", lw=2)

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, kernel_ate + kernel_std_ate, 'r--', label="+1 Std Dev", lw=2)
plt.plot(x, kernel_ate - kernel_std_ate, 'r--', label="-1 Std Dev", lw=2)

# Fill between standard deviation bounds with transparency
plt.fill_between(x, kernel_ate - kernel_std_ate, kernel_ate + kernel_std_ate, color='red', alpha=0.1)

# Set the y-axis range

# Customize labels and title with larger fonts
plt.xlabel("Time of Day (HH:MM)", fontsize=22)
plt.ylabel("Estimated Effect", fontsize=22)
plt.title("Dynamic ATE with Standard Deviation", fontsize=24)

# Customize x-axis to show time format
tick_positions = np.linspace(0, 24, 7)  # Positions at 00:00, 04:00, ..., 24:00
tick_labels = [f"{int(hour):02d}:00" for hour in tick_positions]  # Generate HH:MM labels
plt.xticks(tick_positions, tick_labels, fontsize=18)
plt.yticks(fontsize=18)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to make space below the plot
plt.subplots_adjust(bottom=0.3)

# Move legend below the plot
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show the figure
plt.show()


## Instantiate treatment variables

D = np.where(np.array(tremor_group) <= 2, 1, 0) # Label for tremor status {Tremor-PD, Non-Tremor-PD, non-PD-Controls}
Y = [ updated_loaded_data[key]['data']
    for key in updated_loaded_data
    if key.startswith('mean_conditional_tremor_given_nongait_hbv')
]
Y = np.array(Y) # shape (number of participants, n_timepoints, 1)
X = [  (updated_loaded_data[key]['gender'], updated_loaded_data[key]['hours_awake']) # Example covariates which confounds the effect
    for key in updated_loaded_data
    if key.startswith('mean_conditional_tremor_given_nongait_hbv')]

X = np.array(X)
X = np.nan_to_num(X, nan=0)

from sklearn.preprocessing import StandardScaler
# Extract the columns to scale (year_of_birth and year_diagnosis)
columns_to_scale = X[:, 1:]  # Selecting only the last two columns
# Scale the columns
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(columns_to_scale)
# Replace the scaled columns back into X
X[:, 1:] = scaled_columns



## Second benchmark using kernel methods
kernel__lambd1 = 0.01
kernel_sigma=0.05

varphi_kernel = np.zeros((n_samples, int(n_timepoints)))
D_test_values = np.zeros(n_samples)
# Leave-One-Out Cross-Validation loop
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    operator_kernel_SRVF_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=kernel__lambd1, kernel_sigma = kernel_sigma, use_operator_valued_kernel=True, apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False)  # Apply SRVF registration to the treated group)
    theta_srvf_ate_vector_per_t = operator_kernel_SRVF_causal_estimator.fit(X=X_train, D=D_train.reshape(-1, 1), Y=Y_train[:,:,0])
    predictions_ate_srvf = operator_kernel_SRVF_causal_estimator.predict(X_new = X_test, D_new = D_test.reshape(-1, 1)) # per covariate value
    varphi_kernel[test_index,:] = predictions_ate_srvf.flatten()
    D_test_values[test_index] = D_test

  # Compute the average treatment effect (ATE) and its standard deviation
kernel_ate = np.mean(varphi_kernel,axis=0)
kernel_std_ate = np.std(varphi_kernel,axis=0)

# x-axis range for visualization
x = np.linspace(0, 24, n_timepoints)  # Arbitrary range for illustration
# Set figure size for consistency
plt.figure(figsize=(10, 6))  # Increased width slightly for better readability

# Plot the mean effect as a solid line
plt.plot(x, kernel_ate, label="Mean Effect (ATE)", color="blue", lw=2)

# Plot plus/minus standard deviation as dashed lines
plt.plot(x, kernel_ate + kernel_std_ate, 'r--', label="+1 Std Dev", lw=2)
plt.plot(x, kernel_ate - kernel_std_ate, 'r--', label="-1 Std Dev", lw=2)

# Fill between standard deviation bounds with transparency
plt.fill_between(x, kernel_ate - kernel_std_ate, kernel_ate + kernel_std_ate, color='red', alpha=0.1)

# Set y-axis limits to avoid zooming in too much


# Customize labels and title with larger fonts
plt.xlabel("Time of Day (HH:MM)", fontsize=22)
plt.ylabel("Estimated Effect", fontsize=22)
plt.title("Dynamic ATE with Standard Deviation", fontsize=24)

# Customize x-axis to show time format
tick_positions = np.linspace(0, 24, 7)  # Positions at 00:00, 04:00, ..., 24:00
tick_labels = [f"{int(hour):02d}:00" for hour in tick_positions]  # Generate HH:MM labels
plt.xticks(tick_positions, tick_labels, fontsize=18)
plt.yticks(fontsize=18)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout for better spacing
plt.subplots_adjust(bottom=0.3)

# Move legend below the plot
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show the figure
plt.show()


# Initialize the logistic regression model
loo = LeaveOneOut()
n_samples = len(X)
varphi = np.zeros((n_samples, int(n_timepoints)))
D_test_values = np.zeros((n_samples,))
potential_daily_outcome1 = np.zeros((n_samples,))
potential_daily_outcome2 = np.zeros((n_samples,))
Y_potential_daily_1 = np.zeros((n_samples, int(n_timepoints)))
Y_potential_daily_2 = np.zeros((n_samples, int(n_timepoints)))

# Leave-One-Out Cross-Validation, evaluate out-of-sample the ATE using leave-one-subject-out
for train_index, test_index in loo.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    D_train, D_test = D[train_index], D[test_index]

    # Train the logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, D_train)

    # Compute the propensity score for the test observation
    propensity = logistic_model.predict_proba(X_test)[0][1] # probability of D = 1, which is the PD group

    # Compute the estimated treatment effect for the test observation
    for j in range(Y.shape[1]):
        if np.squeeze(Y[test_index,j]) > 0.1:
            varphi[test_index,j] = (D_test * np.squeeze(Y[test_index,j]) / propensity) - ((1 - D_test) * np.squeeze(Y[test_index,j]) / (1 - propensity))
        else:
            varphi[test_index,j] = 0
    Y_potential_daily_1[test_index,:] = D_test *np.squeeze(Y[test_index,:],axis=2) / propensity
    Y_potential_daily_2[test_index,:] = (1 - D_test)*np.squeeze(Y[test_index,:],axis=2) / (1 - propensity)
    D_test_values[test_index] = D_test

    mask1 = Y_potential_daily_1[test_index,:] > 0.001
    filtered_Y1 = np.where(mask1, Y_potential_daily_1[test_index,:], np.nan)
    # Compute the mean across axis=1, ignoring NaNs
    potential_daily_outcome1[test_index] = np.nanmean(filtered_Y1, axis=1)

    mask2 = Y_potential_daily_2[test_index,:] > 0.001
    filtered_Y2 = np.where(mask2, Y_potential_daily_2[test_index,:], np.nan)
    potential_daily_outcome2[test_index] = np.nanmean(filtered_Y2, axis=1)

potential_daily_outcome1 =  potential_daily_outcome1[~np.isnan(potential_daily_outcome1)]
potential_daily_outcome2 =  potential_daily_outcome2[~np.isnan(potential_daily_outcome2)]
# Compute single potential outcome per daily gait energy and curve potential outcome with values per time-of-day
# Potential outcomes which are Nan are replace with 0

# First, compute average daily score of gait_energy per unit of time (15 minute window block)
# Create a mask for values greater than 0.1, threshold for considering actual walking occured (conservative threshold based on energy at the shortest bouts during visits)
mask = varphi > 0.001
# Replace values below 0.1 with NaN
filtered_varphi = np.where(mask, varphi, np.nan)
# Compute the mean across axis=1, ignoring NaNs
mean_daily_values = np.nanmean(filtered_varphi, axis=1)

# Daily scores ate estimates
ipw_ate_daily = np.nanmean(mean_daily_values) # Average change in gait energy for PD vs Control
ipw_std_ate_daily = np.nanstd(mean_daily_values)

print("Estimated Average Treatment Effect (ATE) for daily tremor score:", ipw_ate_daily)
print("Standard Deviation of ATE for daily tremor score:", ipw_std_ate_daily)

## Plotting the distribution of the potential outcomes for daily scores

from scipy.stats import gaussian_kde

# Kernel Density Estimates
kde1 = gaussian_kde(potential_daily_outcome1.ravel())
kde2 = gaussian_kde(potential_daily_outcome2.ravel())

# X-axis range for KDE
x_min = min(potential_daily_outcome1.min(), potential_daily_outcome2.min())
x_max = max(potential_daily_outcome1.max(), potential_daily_outcome2.max())
x_vals = np.linspace(x_min, x_max, 1000)

# Calculate averages
avg_daily_potential_outcome1 = np.mean(potential_daily_outcome1)
avg_daily_potential_outcome2 = np.mean(potential_daily_outcome2)


# Set figure size for consistency
plt.figure(figsize=(10, 6))  # Slightly wider for better clarity

# Plot KDEs
plt.plot(x_vals, kde1(x_vals), label='PD', lw=2)
plt.plot(x_vals, kde2(x_vals), label='No-PD Control', lw=2)

# Shade the difference area with transparency
plt.fill_between(x_vals, kde1(x_vals), kde2(x_vals), where=(kde1(x_vals) > kde2(x_vals)), 
                 color='blue', alpha=0.1, label="More likely PD Tremor")
plt.fill_between(x_vals, kde1(x_vals), kde2(x_vals), where=(kde1(x_vals) < kde2(x_vals)), 
                 color='orange', alpha=0.1, label="More likely No-PD Tremor")

# Add vertical lines for averages
plt.axvline(avg_daily_potential_outcome1, color='blue', linestyle='--', label=r'$\mathbb{E}[Y^{(1)}]$ (PD Tremor)')
plt.axvline(avg_daily_potential_outcome2, color='orange', linestyle='--', label=r'$\mathbb{E}[Y^{(0)}]$ (No-PD Tremor)')

# Customize labels and title with larger fonts
plt.title("Distribution of Daily Potential Outcomes", fontsize=24)
plt.xlabel("Daily Potential Outcomes", fontsize=22)
plt.ylabel("Density", fontsize=22)

# Adjust tick label size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Add grid for readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout for better spacing
plt.subplots_adjust(bottom=0.35)

# Move legend below the plot for consistency
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show the figure
plt.show()