"""
import os
import pandas as pd
PATH = "corrected/marvl/llava/data/"

# combine all csv files in the folder
combined_csv = pd.concat([pd.read_csv(f"{PATH}{f}") for f in os.listdir(PATH) if f.endswith(".csv")], ignore_index=True)

# export to csv
combined_csv.to_csv(f"corrected/marvl_llava.csv", index=False)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.patches import Rectangle
import json
import os
from tqdm import tqdm
import itertools

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

MODEL = "llava" # "gpt"
PATH = f"corrected/marvl_{MODEL}.csv" # "corrected/marvl_gpt.csv"

"""# accuracy by country
df = pd.read_csv(PATH)
# Calculate if the prediction is correct
df['correct_prediction'] = df['response'] == df['true_country']
# Calculate accuracy for each country
accuracy_by_country = df.groupby('split')['correct_prediction'].mean() * 100
accuracy_by_country = accuracy_by_country.round(2)
accuracy_by_country_df = accuracy_by_country.reset_index()
accuracy_by_country_df.columns = ['lang', 'accuracy']
print(accuracy_by_country_df)

# bar plot
plt.figure(figsize=(5,4))
sns.set_theme(style="whitegrid")
sns.barplot(data=accuracy_by_country_df, x='lang', y='accuracy')

# Improve the aesthetics
plt.xticks(rotation=90)
plt.xlabel('Language')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(f"marvl_accuracy_{MODEL}.png")
accuracy_by_country_df.to_csv(f"corrected/marvl_accuracy_{MODEL}.csv", index=False)
"""
"""# confusion matrix
df = pd.read_csv(PATH)
labels = list(df["true_country"].unique()) + ["Africa", "Asia", "Europe", "Mediterranean", "Middle East", "Northern", "South", "Oceania", "ResponsibleAIPolicyViolation"] #["Oceania", "ResponsibleAIPolicyViolation"]
cm = confusion_matrix(df['true_country'], df['response'])
conf_matrix = confusion_matrix(df['true_country'], df['response'], labels=labels)
print(df.head())
# Get the list of all true_country and response
true_country = set(df['true_country'].unique())
response = set(df['response'].unique())
print("True Country:", true_country)
print("Response:", response)

plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Subregion')
plt.ylabel('True Subregion')
# plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"marvl_confusion_matrix_{MODEL}.png")
"""
df = pd.read_csv(PATH)

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

# Replace 'ResponsibleAIPolicyViolation' with 'ResponsibleAI'
df['response'] = df['response'].replace('ResponsibleAIPolicyViolation', 'ResponsibleAI')

# Map predicted subregions
df['predicted_sub_region'] = df['response'].apply(lambda x: x if x in subregion_to_continent else "Invalid")

# Replace invalid responses in the response column
invalids = ["Invalid", "ResponsibleAI"]
df.loc[df['response'].isin(invalids), 'predicted_sub_region'] = df['response']

# Get the list of all true_country and response
true_country = set(df['true_country'].unique())
response = set(df['response'].unique())
print("True Country:", true_country)
print("Response:", response)

# Create a list of subregions including invalid entries
subregions_set = set(df['true_country'].unique()).union(set(df['predicted_sub_region'].unique()))

# Convert to list and sort by continent, then alphabetically within each continent
sorted_subregions = sorted(subregions_set, key=lambda x: (subregion_to_continent.get(x, ""), x))

# Ensure 'Invalid' and 'ResponsibleAI' are at the end
all_subregions = [subregion for subregion in sorted_subregions if subregion not in invalids] + invalids

# Calculate confusion matrix
conf_matrix = confusion_matrix(df['true_country'], df['predicted_sub_region'], labels=all_subregions)

# Plot confusion matrix
plt.figure(figsize=(20, 15))
sns.set_theme(style="whitegrid")
h = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_subregions, yticklabels=all_subregions, annot_kws={"size": 20})
plt.xlabel('Predicted Subregion', fontsize=18)
plt.ylabel('True Subregion', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)


cbar = h.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted Subregion', fontsize=18)
plt.ylabel('True Subregion', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Draw rectangles around each region
ax = plt.gca()
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
    rect = plt.Rectangle((start, start), width, width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# Add box for 'Invalid' and 'ResponsibleAI'
n = len(all_subregions)
rect_invalid = plt.Rectangle((n-2, n-2), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
rect_responsibleai = plt.Rectangle((n-1, n-1), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(rect_invalid)
ax.add_patch(rect_responsibleai)

# calculate accuracy
correct = 0
total = 0
for i in range(conf_matrix.shape[0]):
    correct += conf_matrix[i][i]
    total += sum(conf_matrix[i])

accuracy = correct / total
print(f"Accuracy: {accuracy}")


plt.tight_layout()
plt.savefig(f"marvl_confusion_matrix_{MODEL}.png")
