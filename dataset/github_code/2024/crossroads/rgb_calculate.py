import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

# Load the dataset
data = pd.read_csv("results/dalle_images_with_rgb.csv")

# Assume 'rgb_values' column already contains the RGB values stored as np arrays
data['rgb_values'] = data['rgb_values'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Calculate the global average RGB
global_average_rgb = np.mean(data['rgb_values'].tolist(), axis=0)

# Calculate deltas for each country
country_deltas = data.groupby('country')['rgb_values'].apply(lambda x: np.mean(x, axis=0) - global_average_rgb)

# Convert the result to a DataFrame for easier handling
country_deltas_df = pd.DataFrame(country_deltas.tolist(), index=country_deltas.index, columns=['Delta Red', 'Delta Green', 'Delta Blue'])

country_deltas_df['magnitude'] = np.sqrt(country_deltas_df['Delta Red']**2 + country_deltas_df['Delta Green']**2 + country_deltas_df['Delta Blue']**2)
filtered_df = country_deltas_df[country_deltas_df['magnitude'] > 0]  # Change threshold as needed

cmap = ListedColormap(['#FF6961', '#A6C9A6', '#92B0C4'])

# Plotting in 3D
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with variable color and size based on the component
# Mapping colors to their respective delta values
scatter = ax.scatter(filtered_df['Delta Red'], filtered_df['Delta Green'], filtered_df['Delta Blue'],
                     c=[cmap(0) if abs(r) >= abs(g) and abs(r) >= abs(b) else
                        cmap(1) if abs(g) > abs(r) and abs(g) >= abs(b) else
                        cmap(2)
                        for r, g, b in zip(filtered_df['Delta Red'], filtered_df['Delta Green'], filtered_df['Delta Blue'])],
                     s=300 * filtered_df['magnitude'], alpha=0.5)

# Annotate countries
for i, txt in enumerate(filtered_df.index):
    ax.text(filtered_df['Delta Red'][i], filtered_df['Delta Green'][i], filtered_df['Delta Blue'][i], txt, size=12)

ax.set_xlabel('Delta Red', fontsize=18)
ax.set_ylabel('Delta Green', fontsize=18)
ax.set_zlabel('Delta Blue', fontsize=18)
# ax.set_title('RGB Delta Values by Country')

ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='z', which='major', labelsize=20)

# remove whitespace around the figure
plt.tight_layout()

# plt.colorbar(scatter, label='Magnitude of RGB Delta', drawedges=True)
plt.savefig('rgb_deltas_3d.png')

# Calculate the global average RGB
global_average_rgb = np.mean(data['rgb_values'].tolist(), axis=0)

# Calculate deltas for each country
country_deltas = data.groupby('country')['rgb_values'].apply(lambda x: np.mean(x, axis=0) - global_average_rgb)

# Convert the result to a DataFrame for easier handling
country_deltas_df = pd.DataFrame(country_deltas.tolist(), index=country_deltas.index, columns=['Delta Red', 'Delta Green', 'Delta Blue'])

# split into 3 subfigures
std_devs = country_deltas_df.std()

fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# Colors and labels for the plots
# colors = ['red', 'green', 'blue']
colors = ['#FF6961','#A6C9A6','#92B0C4']
labels = ['Delta Red', 'Delta Green', 'Delta Blue']

# Plot each color component in a separate subplot
for i, (ax, color, label) in enumerate(zip(axes, colors, labels)):
    # Plot bars for the current color component
    country_deltas_df[label].plot(kind='bar', ax=ax, color=color, width=0.8)

    # Calculate the mean value for reference
    mean_val = country_deltas_df[label].mean()

    # Draw standard deviation lines
    ax.axhline(y=mean_val, color='black', linestyle='-', linewidth=1, label='Mean')
    ax.axhline(y=mean_val + std_devs[label], color=color, linestyle='--', linewidth=1, label='Std Dev +')
    ax.axhline(y=mean_val - std_devs[label], color=color, linestyle='--', linewidth=1, label='Std Dev -')

    # Set titles and labels
    ax.set_title(f'{label} Values by Country', fontsize=16)
    ax.set_ylabel('Delta Values', fontsize=14)
    ax.legend()

# Set x-label for the last subplot
axes[-1].set_xlabel('Country', fontsize=14)

# Rotate x-tick labels for better visibility
for ax in axes:
    ax.tick_params(axis='x', rotation=90)

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('rgb_deltas_stacked.png')

# create a figure with only blue subplot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for the blue color component
country_deltas_df['Delta Blue'].plot(kind='bar', ax=ax, color='#92B0C4', width=0.8)

# Calculate the mean value for reference
mean_val = country_deltas_df['Delta Blue'].mean()

# Draw standard deviation lines
ax.axhline(y=mean_val, color='black', linestyle='-', linewidth=1, label='Mean')
ax.axhline(y=mean_val + std_devs['Delta Blue'], color='blue', linestyle='--', linewidth=1, label='Std Dev +')
ax.axhline(y=mean_val - std_devs['Delta Blue'], color='blue', linestyle='--', linewidth=1, label='Std Dev -')

# Set title and labels
# ax.set_title('Delta Blue Values by Country', fontsize=16)
ax.set_ylabel('Delta Values', fontsize=10)
ax.set_xlabel('Country', fontsize=14)
ax.legend()

# Rotate x-tick labels for better visibility
ax.tick_params(axis='x', rotation=90)

# Save the plot as a PNG file
plt.tight_layout()

plt.savefig('rgb_deltas_blue.png')


# Define a function to select countries based on delta values
def select_countries(data, num_minimal=10, num_zero=10, num_large=10):
    # Get absolute values and sort them
    sorted_data = data.abs().sort_values()

    # Largest absolute values
    large = sorted_data.tail(num_large)

    # Closest to zero
    zero_close = sorted_data.head(num_zero)

    # Minimal: Get values that are not in the smallest or largest, but are less than the large ones
    # Exclude zero_close and large, then take the smallest remaining
    minimal = sorted_data[~sorted_data.index.isin(zero_close.index.union(large.index))].head(num_minimal)

    return pd.concat([zero_close, minimal, large]).drop_duplicates()

# Apply selection for each RGB component
selected_countries_red = select_countries(country_deltas_df['Delta Red'])
selected_countries_green = select_countries(country_deltas_df['Delta Green'])
selected_countries_blue = select_countries(country_deltas_df['Delta Blue'])

# Combine all selected countries
all_selected_countries = pd.concat([selected_countries_red, selected_countries_green, selected_countries_blue]).drop_duplicates()
y_min = -30
y_max = 30

# Plotting the selected countries for each RGB component
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 1 row, 3 columns
# colors = ['red', 'green', 'blue']
colors = ['#FF6961','#A6C9A6','#92B0C4']
labels = ['Delta Red', 'Delta Green', 'Delta Blue']
selected_dfs = [selected_countries_red, selected_countries_green, selected_countries_blue]

for ax, color, label, selected_df in zip(axes, colors, labels, selected_dfs):
    # Data for current subplot
    data_subset = country_deltas_df.loc[selected_df.index][label]
    data_subset.plot(kind='bar', ax=ax, color=color, width=0.8)

    # Calculate mean and standard deviation for the current subset
    mean_val = data_subset.mean()
    std_dev = data_subset.std()

    # Draw standard deviation lines
    # ax.axhline(y=mean_val, color='black', linestyle='-', linewidth=1, label='Mean')
    ax.axhline(y=mean_val + std_dev, color=color, linestyle='--', linewidth=1, label='Std Dev +')
    ax.axhline(y=mean_val - std_dev, color=color, linestyle='--', linewidth=1, label='Std Dev -')

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Set titles and labels
    # ax.set_title(f'{label} Selected Countries', fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Delta Values', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.legend(fontsize=14, loc='upper left')

plt.tight_layout()
plt.savefig('selected_rgb_deltas.png')


# # Function to calculate average RGB values of an image
# def calculate_average_rgb(image_path):
#     try:
#         img = Image.open(image_path)
#         np_img = np.array(img)
#         average_rgb = np.mean(np_img, axis=(0, 1))
#         return average_rgb
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return np.array([0, 0, 0])  # Return zero array if error occurs

# data['rgb_values'] = [None] * len(data)
# # Apply the function to each image path
# for i in tqdm(range(len(data))):
#     data['rgb_values'][i] = calculate_average_rgb(data['image_path'][i])
#     # print(data["image_path"][i], data['rgb_values'][i])
# # data['rgb_values'] = data['image_path'].apply(lambda x: calculate_average_rgb(x))

# data.to_csv("results/dalle_images_with_rgb.csv", index=False)
