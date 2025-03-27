import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

# Load the previously saved JSON dictionary
with open("results/dalle_objects/country_dict.json", "r") as f:
    country_dict = json.load(f)

# Define a list of terms that are variants or synonymous with person
person_variants = ["person", "persons", "man", "men", "woman", "women", "people", "humans", "human", "child", "children", "pedestrians", "pedestrian", "rider", "family", "family members", "people in traditional clothing", "human figures", "baby", "boy", "boys", "girl", "girls", "lady", "gentleman"]

# def count_people(items):
#     counts = []
#     for item in items:
#         # Check every key's value for a match with person_variants
#         for value in item.values():
#             # Since values can be strings or lists, handle both cases
#             if isinstance(value, str) and any(term in value.lower() or value.lower().split() for term in person_variants):
#                 count = item.get('count', 0)
#                 counts.append(count)
#                 break  # Found a match, no need to check further for this item
#             elif isinstance(value, list):
#                 # Check if any string in the list matches the person_variants
#                 if any(any(term in str(sub_value).lower() for term in person_variants) for sub_value in value):
#                     count = item.get('count', 0)
#                     counts.append(count)
#                     break  # Found a match, no need to check further for this item
#     return counts

def count_people(items):
    counts = []
    for item in items:
        # Check every key's value for a match with person_variants
        for value in item.values():
            # Since values can be strings or lists, handle both cases
            if isinstance(value, str) and any(term in value.lower() for term in person_variants):
                # Handle string case for 'count'
                count_str = item.get('count', '0')
                if isinstance(count_str, list):
                    count_str = count_str[0]  # Take the first element if it is a list
                try:
                    count = int(count_str)
                except ValueError:
                    if 'more than' in count_str:
                        count = int(count_str.split(' ')[-1]) + 1
                    else:
                        count = 0  # Default to 0 if conversion is not possible
                counts.append(count)
                break  # Found a match, no need to check further for this item
            elif isinstance(value, list) and any(any(term in str(sub_value).lower() for term in person_variants) for sub_value in value):
                # Handle list case for 'count'
                count_str = item.get('count', '0')
                if isinstance(count_str, list):
                    count_str = count_str[0]  # Take the first element if it is a list
                try:
                    count = int(count_str)
                except ValueError:
                    if 'more than' in count_str:
                        count = int(count_str.split(' ')[-1]) + 1
                    else:
                        count = 0  # Default to 0 if conversion is not possible
                counts.append(count)
                break  # Found a match, no need to check further for this item
    return counts

# Dictionary to hold the counts of people-like objects per country and concept
people_counts = {}

for country, concepts in country_dict.items():
    people_counts[country] = {}
    for concept, details in concepts.items():
        relevant_people_counts = count_people(details['Relevant_Items'])
        non_relevant_people_counts = count_people(details['Non_Relevant_Items'])
        # Combine both relevant and non-relevant counts
        total_people_counts = relevant_people_counts + non_relevant_people_counts
        if total_people_counts:
            people_counts[country][concept] = total_people_counts

# Summing up the counts for each country
for country, concepts in people_counts.items():
    people_counts[country] = sum(concepts.values(), [])

# Function to categorize counts into buckets
def categorize_into_buckets(counts):
    bucket_1_5 = 0
    bucket_5_10 = 0
    bucket_more_than_10 = 0
    for count in counts:
        if 1 <= count <= 5:
            bucket_1_5 += 1
        elif 5 < count <= 10:
            bucket_5_10 += 1
        elif count > 10:
            bucket_more_than_10 += 1
    return [bucket_1_5, bucket_5_10, bucket_more_than_10]

# Applying the categorization to each country
buckets_counts = {country: categorize_into_buckets(counts) for country, counts in people_counts.items()}

# Data preparation for plotting
countries = list(buckets_counts.keys())
buckets_1_5 = [bucket[0] for bucket in buckets_counts.values()]
buckets_5_10 = [bucket[1] for bucket in buckets_counts.values()]
buckets_more_than_10 = [bucket[2] for bucket in buckets_counts.values()]

# Setting up the figure and axes for the subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharex=True)

# Titles for each subplot
titles = [
    'Distribution of Counts of People by Country (1-5)',
    'Distribution of Counts of People by Country (5-10)',
    'Distribution of Counts of People by Country (More than 10)'
]

# Colors for each subplot
colors = ['#FF6961', '#A6C9A6', '#92B0C4']

# Data for each bucket
buckets_data = [buckets_1_5, buckets_5_10, buckets_more_than_10]

for ax, bucket, title, color in zip(axes, buckets_data, titles, colors):
    # Calculate mean and standard deviation for line markers
    mean = np.mean(bucket)
    std_dev = np.std(bucket)

    # Create bars
    ax.bar(countries, bucket, color=color)

    # Add lines for mean and std deviation
    ax.axhline(y=mean, color='k', linestyle='-', linewidth=1.5, label='Mean')
    ax.axhline(y=mean + std_dev, color='k', linestyle='--', linewidth=1.5, label='Std Dev +')
    ax.axhline(y=mean - std_dev, color='k', linestyle='--', linewidth=1.5, label='Std Dev -')

    # Set titles and labels
    ax.set_title(title)
    ax.set_ylabel('How many people')
    ax.legend()

# Set x-axis labels on the bottom subplot
axes[-1].set_xticks(np.arange(len(countries)))
axes[-1].set_xticklabels(countries, rotation=90, ha='center')

plt.tight_layout()
plt.savefig("people_counts_by_country.png")

bucket_data = {
    'Country': countries,
    'Bucket_1_5': buckets_1_5,
    'Bucket_5_10': buckets_5_10,
    'Bucket_More_Than_10': buckets_more_than_10
}
bucket_df = pd.DataFrame(bucket_data).set_index('Country')

# Define a function to select countries based on delta values
def select_countries(data, num_minimal=10, num_zero=10, num_large=10):
    # Get absolute values and sort them
    sorted_data = data.abs().sort_values()

    # Largest absolute values
    large = sorted_data.tail(num_large)

    # Closest to zero
    zero_close = sorted_data.head(num_zero)

    # Minimal: Get values that are not in the smallest or largest, but are less than the large ones
    minimal = sorted_data[~sorted_data.index.isin(zero_close.index.union(large.index))].head(num_minimal)

    return pd.concat([zero_close, minimal, large]).drop_duplicates()

# Apply selection for each bucket
selected_countries_1_5 = select_countries(bucket_df['Bucket_1_5'])
selected_countries_5_10 = select_countries(bucket_df['Bucket_5_10'])
selected_countries_more_than_10 = select_countries(bucket_df['Bucket_More_Than_10'])

all_data = pd.concat([selected_countries_1_5, selected_countries_5_10, selected_countries_more_than_10])
y_min = all_data.min().min()
y_max = all_data.max().max()

# Plotting the selected countries for each bucket
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 1 row, 3 columns
colors = ['#FF6961','#A6C9A6','#92B0C4']
labels = ['Bucket_1_5', 'Bucket_5_10', 'Bucket_More_Than_10']
selected_dfs = [selected_countries_1_5, selected_countries_5_10, selected_countries_more_than_10]

for ax, color, label, selected_df in zip(axes, colors, labels, selected_dfs):
    # Data for current subplot
    data_subset = bucket_df.loc[selected_df.index][label.split(' ')[0]]
    data_subset.plot(kind='bar', ax=ax, color=color, width=0.8)
    # rename the labels

    # Calculate mean and standard deviation for the current subset
    mean_val = data_subset.mean()
    std_dev = data_subset.std()

    # Draw standard deviation lines
    ax.axhline(y=mean_val + std_dev, color=color, linestyle='--', linewidth=1, label='Std Dev +')
    ax.axhline(y=mean_val - std_dev, color=color, linestyle='--', linewidth=1, label='Std Dev -')

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Set titles and labels
    ax.set_ylabel('How many people', fontsize=14)

    # set xticks size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('', fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    ax.legend(fontsize=14, loc='upper left')

plt.tight_layout()
plt.savefig('selected_people_counts_by_country.png')

# plot another figure with only the last bucket
fig, ax = plt.subplots(figsize=(10, 6))
# selected_countries_more_than_10.plot(kind='bar', ax=ax, color=colors[2],
# width=0.8)
buckets_more_than_10 = bucket_df['Bucket_More_Than_10']
# sort
buckets_more_than_10 = buckets_more_than_10.sort_values(ascending=True)
buckets_more_than_10.plot(kind='bar', ax=ax, color=colors[2], width=0.8)

# Calculate mean and standard deviation for the current subset
mean_val = buckets_more_than_10.mean() #selected_countries_more_than_10.mean()
std_dev = buckets_more_than_10.std() #selected_countries_more_than_10.std()

# Draw standard deviation lines
std_plus_line = ax.axhline(y=mean_val + std_dev, color='#92B0C4', linestyle='--', linewidth=1, label='Std Dev +')
std_minus_line = ax.axhline(y=mean_val - std_dev, color='#92B0C4', linestyle='--', linewidth=1, label='Std Dev -')

# Set y-axis limits
ax.set_ylim(y_min, y_max)

# Set titles and labels
# ax.set_title('Distribution of Counts of People by Country (More than 10)', fontsize=18)
ax.set_ylabel('How many people', fontsize=16)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('', fontsize=16)
ax.tick_params(axis='x', rotation=90)

handles, labels = ax.get_legend_handles_labels()
handles = [std_plus_line, std_minus_line]
labels = ['Std Dev +', 'Std Dev -']
ax.legend(handles, labels, fontsize=16, loc='upper left')

plt.tight_layout()

plt.savefig('all_people_counts_more_than_10.png')
