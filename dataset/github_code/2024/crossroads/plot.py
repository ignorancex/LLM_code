"""
import os
import pandas as pd
# PATH = "corrected/dalle_eval/llava/vivid/data/"

# # combine all csv files in the folder
# combined_csv = pd.concat([pd.read_csv(f"{PATH}{f}") for f in os.listdir(PATH) if f.endswith(".csv")], ignore_index=True)

# df1 = pd.read_csv("corrected/dalle_street_vivid_llava.csv")
# df2 = pd.read_csv("corrected/dalle_street_natural_llava.csv")
# combined_csv = pd.concat([df1, df2], ignore_index=True)
# export to csv
# combined_csv.to_csv(f"corrected/dalle_street_llava.csv", index=False)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import os
from tqdm import tqdm
import itertools

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

DATA = "dollar" # "dalle"
MODEL = "llava" # "llava"
PATH = f"corrected/{DATA}_street_{MODEL}.csv"

#income plot
# Load JSON mapping file
with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
subregion_to_continent = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        subregion_to_continent[subregion] = continent
        for country in data['countries_included']:
            country_to_subregion[country] = (subregion, continent)

# Load the CSV file
df = pd.read_csv(PATH)
# create a blank column for the predicted continent
df["predicted_continent"] = ""

if MODEL == "llava":
    invalids = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Antarctica", "Middle East", "ResponsibleAIPolicyViolation"]
else:
    invalids = ["Africa", "Americas", "Asia", "Europe", "Oceania", "ResponsibleAIPolicyViolation"]
df.loc[df["predicted_continent"] == "", "predicted_continent"] = df.loc[df["predicted_continent"] == "", "response"].map(subregion_to_continent)
for invalid in invalids:
    df.loc[df["response"].str.contains(invalid), "predicted_continent"] = invalid
print(df.head())

continents_to_plot = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

# Calculate if the prediction is correct
df['correct_prediction'] = df['predicted_continent'] == df['true_continent']

# Filter the dataframe to keep only the relevant continents
continents_to_plot = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
df = df[df['true_continent'].isin(continents_to_plot)]

# Calculate income quartiles for each continent separately
def calculate_income_bucket(group):
    try:
        return pd.qcut(group, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        return pd.cut(group.rank(method='first'), bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])


df['income_bucket'] = df.groupby('true_continent', group_keys=False)['income'].apply(calculate_income_bucket)

# Calculate the total and correct counts by continent and income bucket
total_counts = df.groupby(['true_continent', 'income_bucket']).size().reset_index(name='total_counts')
correct_counts = df[df['correct_prediction']].groupby(['true_continent', 'income_bucket']).size().reset_index(name='correct_counts')

# Merge the counts
merged_df = pd.merge(correct_counts, total_counts, on=['true_continent', 'income_bucket'], how='left')

# Calculate the normalized correct classification percentages
merged_df['normalized_correct_counts'] = round((merged_df['correct_counts'] / merged_df['total_counts']) * 100, 2)

# Sorting the results to ensure the plot is ordered
merged_df.sort_values(by=['true_continent', 'income_bucket'], ascending=[True, True], inplace=True)

# Plotting with seaborn barplot
plt.figure(figsize=(20, 10))
sns.set_theme(style="whitegrid")
sns.barplot(data=merged_df, x='true_continent', y='normalized_correct_counts', hue='income_bucket', palette='deep')

# Improve the aesthetics
plt.yticks(fontsize=32)
plt.xticks(fontsize=32)
# plt.title('Accuracy across normalized income quartiles', fontsize=24)
plt.xlabel('', fontsize=22)
plt.ylabel('Correct Classification Percentage', fontsize=22)
legend = plt.legend(title='Income Quartile', fontsize=24)
plt.setp(legend.get_title(),fontsize=26)
plt.tight_layout()

# Save the plot
plt.savefig(f"accuracy_by_income_quartiles_{DATA}_street_{MODEL}.png")
# Save the merged dataframe
merged_df.to_csv(f"corrected/accuracy_by_income_quartiles_{DATA}_street_{MODEL}.csv", index=False)

"""#confusion matrix subregion plot
# Load JSON mapping file
with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
subregion_to_continent = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        subregion_to_continent[subregion] = continent
        for country in data['countries_included']:
            country_to_subregion[country] = subregion

# Load the CSV file
df = pd.read_csv(PATH)

# Map true subregions from true_country
df['true_sub_region'] = df['true_country'].map(country_to_subregion)

# Replace 'ResponsibleAIPolicyViolation' with 'ResponsibleAI'
df['response'] = df['response'].replace('ResponsibleAIPolicyViolation', 'ResponsibleAI')

# Map predicted subregions
df['predicted_sub_region'] = df['response'].apply(lambda x: x if x in subregion_to_continent else "Invalid")

# Replace invalid responses in the response column
invalids = ["Invalid", "ResponsibleAI"]
df.loc[df['response'].isin(invalids), 'predicted_sub_region'] = df['response']

# Ensure there are no NaNs in true_sub_region or predicted_sub_region
df = df.dropna(subset=['true_sub_region', 'predicted_sub_region'])

# Get the list of all subregions
subregions_set = set(df['true_sub_region'].unique()).union(set(df['predicted_sub_region'].unique()))

# Convert to list and sort by continent, then alphabetically within each continent
sorted_subregions = sorted(subregions_set, key=lambda x: (subregion_to_continent.get(x, ""), x))

# Ensure 'Invalid' and 'ResponsibleAI' are at the end
all_subregions = [subregion for subregion in sorted_subregions if subregion not in invalids] + invalids

# Calculate confusion matrix
conf_matrix = confusion_matrix(df['true_sub_region'], df['predicted_sub_region'], labels=all_subregions)

# Plot confusion matrix
plt.figure(figsize=(20, 15))
h = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_subregions, yticklabels=all_subregions)
cbar = h.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted Subregion', fontsize=18)
plt.ylabel('True Subregion', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Get the current axes
ax = plt.gca()

# Find the indices for each continent
region_indices = {}
for i, subregion in enumerate(all_subregions):
    continent = subregion_to_continent.get(subregion, subregion)  # Use subregion name if continent not found
    if continent not in region_indices:
        region_indices[continent] = []
    region_indices[continent].append(i)

# Draw rectangles around each region
for region, indices in region_indices.items():
    start = min(indices)
    end = max(indices)
    width = end - start + 1
    rect = Rectangle((start, start), width, width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# Add box for 'Invalid' and 'ResponsibleAI'
n = len(all_subregions)
rect_invalid = Rectangle((n-2, n-2), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
rect_responsibleai = Rectangle((n-1, n-1), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(rect_invalid)
ax.add_patch(rect_responsibleai)

plt.tight_layout()

# Save the plot
plt.savefig(f"confusion_matrix_subregions_{DATA}_street_{MODEL}.png")
"""
"""#confusion matrix regions
# Load JSON mapping file
with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
subregion_to_continent = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        subregion_to_continent[subregion] = continent
        for country in data['countries_included']:
            country_to_subregion[country] = (subregion, continent)

# Load the CSV file
df = pd.read_csv(PATH)
# create a blank column for the predicted continent
df["predicted_continent"] = ""

if MODEL == "llava":
    if DATA == "dollar":
        invalids = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Antarctica", "Middle East", "ResponsibleAIPolicyViolation"]
    if DATA == "dalle":
        invalids = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Arctic", "Antarctica", "Middle East", "ResponsibleAIPolicyViolation"]
else:
    invalids = ["Africa", "Americas", "Asia", "Europe", "Oceania", "ResponsibleAIPolicyViolation"]
df.loc[df["predicted_continent"] == "", "predicted_continent"] = df.loc[df["predicted_continent"] == "", "response"].map(subregion_to_continent)
for invalid in invalids:
    df.loc[df["response"].str.contains(invalid), "predicted_continent"] = invalid
print(df.head(10))

if MODEL == "llava":
    if DATA == "dollar":
        continent_order = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Antarctica", "Middle East", "ResponsibleAI"]
    if DATA == "dalle":
        continent_order = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Arctic", "Antarctica", "Middle East", "ResponsibleAI"]
else:
    continent_order = ["Africa", "Americas", "Asia", "Europe", "Oceania", "ResponsibleAI"]
conf_matrix = confusion_matrix(df['true_continent'], df['predicted_continent'], labels=invalids)
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=continent_order, yticklabels=continent_order)
plt.xlabel('Predicted Continent')
plt.ylabel('True Continent')
plt.title('Confusion Matrix: True Continent vs Predicted Continent')
plt.tight_layout()

# Save the plot
plt.savefig(f"confusion_matrix_{DATA}_street_{MODEL}.png")
"""

"""#subregion plot

# Load JSON mapping file
with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        for country in data['countries_included']:
            country_to_subregion[country] = (subregion, data['continent'])

valid_subregions = set(itertools.chain(*[list(regions.keys()) for continent, regions in mappings.items()]))

# Load the CSV file
df = pd.read_csv(PATH)

# Calculate if the prediction is correct
df['correct_prediction'] = df['response'] == df['true_sub_region']

# Calculate accuracy for each country
accuracy_by_country = df.groupby('true_country')['correct_prediction'].mean() * 100
accuracy_by_country = accuracy_by_country.round(2)

# Convert to DataFrame for better display
accuracy_by_country_df = accuracy_by_country.reset_index()
accuracy_by_country_df.columns = ['true_country', 'accuracy']

# Merge with true_sub_region and true_continent
accuracy_by_country_df = pd.merge(accuracy_by_country_df, df[['true_country', 'true_sub_region', 'true_continent']].drop_duplicates(), on='true_country', how='left')

# first order by true_continent and then within each continent, order by
# true_sub_region and then within each subregion, order by accuracy
accuracy_by_country_df.sort_values(by=['true_continent', 'true_sub_region', 'accuracy'], ascending=[True, True, False], inplace=True)

# Sort by 'true_sub_region' and 'accuracy' for a consistent bar order
# accuracy_by_country_df.sort_values(by=['true_sub_region', 'accuracy'], ascending=[True, False], inplace=True)

# Assign colors to each subregion based on the unique values in 'true_sub_region'
subregion_order = accuracy_by_country_df['true_sub_region'].unique()  # Order of subregions for the legend
subregion_colors = sns.color_palette("husl", len(subregion_order))  # Generate as many colors as subregions
subregion_palette = dict(zip(subregion_order, subregion_colors))  # Create a dictionary to map subregions to colors

# Plotting with seaborn barplot
plt.figure(figsize=(20, 10))
sns.set_theme(style="whitegrid")
sns.barplot(data=accuracy_by_country_df, x='true_country', y='accuracy', hue='true_sub_region', dodge=False, palette=subregion_palette)

# Improve the aesthetics
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Accuracy (%)')
plt.legend(title='Subregion')
plt.tight_layout()

# Save the plot and the DataFrame to files
plt.savefig(f"country_wise_accuracy_by_subregion_{DATA}_street_{MODEL}.png")
accuracy_by_country_df.to_csv(f"corrected/country_wise_accuracy_by_subregion_{DATA}_street_{MODEL}.csv", index=False)
"""
"""#continent plot
# Load JSON mapping file
with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        for country in data['countries_included']:
            country_to_subregion[country] = (subregion, data['continent'])

valid_subregions = set(itertools.chain(*[list(regions.keys()) for continent, regions in mappings.items()]))

df = pd.read_csv(PATH)
# print(df.head())
# print(df.columns)

# if papua new guinea, print true_continet
# print(df[df['true_country'] == 'Papua New Guinea']['true_continent'])

# Calculate if the prediction is correct
df['correct_prediction'] = df['response'] == df['true_sub_region']

# Calculate accuracy for each country
accuracy_by_country = df.groupby('true_country')['correct_prediction'].mean() * 100
# round to 2 decimal places
accuracy_by_country = accuracy_by_country.round(2)

# Convert to DataFrame for better display
accuracy_by_country_df = accuracy_by_country.reset_index()
accuracy_by_country_df.columns = ['true_country', 'accuracy']

# Merge with true_continent
accuracy_by_country_df = pd.merge(accuracy_by_country_df, df[['true_country', 'true_continent']].drop_duplicates(), on='true_country', how='left')

# Sort by 'true_continent' and 'accuracy' for a consistent bar order
accuracy_by_country_df.sort_values(by=['true_continent', 'accuracy'], ascending=[True, False], inplace=True)

# Assign colors to each continent based on the unique values in 'true_continent'
continent_order = accuracy_by_country_df['true_continent'].unique()  # Order of continents for the legend
continent_colors = sns.color_palette("husl", len(continent_order))  # Generate as many colors as continents
continent_palette = dict(zip(continent_order, continent_colors))  # Create a dictionary to map continents to colors

# Plotting with seaborn barplot
plt.figure(figsize=(20, 10))
sns.set_theme(style="whitegrid")
sns.barplot(data=accuracy_by_country_df, x='true_country', y='accuracy', hue='true_continent', dodge=False, palette=continent_palette)

# Improve the aesthetics
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Accuracy (%)')
plt.legend(title='Continent')
plt.tight_layout()
plt.savefig(f"country_wise_accuracy_{DATA}_street_{MODEL}.png")
accuracy_by_country_df.to_csv(f"corrected/country_wise_accuracy_{DATA}_street_{MODEL}.csv", index=False)
"""
