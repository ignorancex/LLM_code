import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

with open('unsd_geoscheme.json') as f:
    mappings = json.load(f)

# Flatten the JSON mapping for easier lookup
country_to_subregion = {}
for continent, regions in mappings.items():
    for subregion, data in regions.items():
        for country in data['countries_included']:
            country_to_subregion[country] = (subregion, data['continent'])

continent_mapping = {
    "af": "Africa",
    "am": "Americas",
    "as": "Asia",
    "eu": "Europe",
    "oc": "Oceania",
}

BASE1 = f"results/human-study/study1/"
path1 = [BASE1 + f for f in os.listdir(BASE1) if f.endswith(".csv")]
dfs1 = [pd.read_csv(p) for p in path1]
df1 = pd.concat(dfs1, axis=0)

BASE2 = f"results/human-study/study2/"
path2 = [BASE2 + f for f in os.listdir(BASE2) if f.endswith(".csv")]
dfs2 = [pd.read_csv(p) for p in path2]
df2 = pd.concat(dfs2, axis=0)

def extract_taxonomy(taxonomy_str):
    if pd.isna(taxonomy_str):
        # Return an empty list for NaN values
        return []

    try:
        # Ensure the input is a string
        taxonomy_str = str(taxonomy_str)

        # Parse the JSON string to a Python object
        taxonomy_obj = json.loads(taxonomy_str)

        # Extract the nested list from the parsed JSON
        if isinstance(taxonomy_obj, list) and len(taxonomy_obj) > 0:
            return taxonomy_obj[0].get("taxonomy", [])
        else:
            return []
    except (json.JSONDecodeError, TypeError):
        # Return an empty list if parsing fails
        return []

def create_pairs(label, transcription):
    try:
        # Handle cases where label or transcription might be NaN
        if pd.isna(label) or pd.isna(transcription):
            return []

        # Ensure the input is a string
        label_str = str(label)
        transcription_str = str(transcription)

        # Parse the JSON string to a list of dictionaries
        label_list = json.loads(label_str)
        transcription_list = json.loads(transcription_str)

        # Extract the rectanglelabels and text
        rectanglelabels = [item["rectanglelabels"][0] for item in label_list]
        texts = [item["text"][0] for item in transcription_list]

        # Create pairs
        pairs = list(zip(rectanglelabels, texts))
        return pairs

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        return []

def calculate_country_level_accuracy(row):
    return row["country"] in row["pred"]

def calculate_subregion_level_accuracy(row):
    return row["subregion"] in row["pred"]

def calculate_continent_level_accuracy(row):
    return row["continent"] in row["pred"]

def calculate_union_accuracy(row):
    return (row["country"] in row["pred"]) or (row["subregion"] in row["pred"]) or (row["continent"] in row["pred"])

def calculate_intersection_accuracy(row):
    return (row["country"] in row["pred"]) and (row["subregion"] in row["pred"]) and (row["continent"] in row["pred"])


df1["artifacts"] = df1.apply(lambda row: create_pairs(row["label"], row["transcription"]), axis=1)
df2["artifacts"] = df2.apply(lambda row: create_pairs(row["label"], row["transcription"]), axis=1)
# extract the second element of the tuple in the artifacts column
df1["artifacts"] = df1["artifacts"].apply(lambda x: [item[1] for item in x])
df2["artifacts"] = df2["artifacts"].apply(lambda x: [item[1] for item in x])

# combine df1 and df2 with only the country, artifacts column
df_artifacts = pd.concat([df1[["country", "artifacts"]], df2[["country", "artifacts"]]], axis=0)
# sort by country name
df_artifacts = df_artifacts.sort_values(by="country")
# combine all artifacts of a country into a single list
df_artifacts = df_artifacts.groupby("country")["artifacts"].sum().reset_index()

# save to csv
df_artifacts.to_csv("results/human-study/artifacts.csv", index=False)

# for each row, sample 3 items randomly , but ensure that they don't have
# question mark

print(df_artifacts.head())


# Accuracy metrics calculate
df1["subregion"] = df1["country"].apply(lambda x: country_to_subregion.get(x, ("", ""))[0])
df1["continent"] = df1["country"].apply(lambda x: country_to_subregion.get(x, ("", ""))[1])
df1["continent"] = df1["continent"].map(continent_mapping)

df1['pred'] = df1['taxonomy'].apply(extract_taxonomy)
df1['pred'] = df1['pred'].apply(lambda x: [item for sublist in x for item in sublist])

df1["country_level_accuracy"] = df1.apply(calculate_country_level_accuracy, axis=1)
df1["subregion_level_accuracy"] = df1.apply(calculate_subregion_level_accuracy, axis=1)
df1["continent_level_accuracy"] = df1.apply(calculate_continent_level_accuracy, axis=1)
df1["union_accuracy"] = df1.apply(calculate_union_accuracy, axis=1)
df1["intersection_accuracy"] = df1.apply(calculate_intersection_accuracy, axis=1)


print(df1[["type", "country", "subregion", "continent", "pred", "artifacts",
        "country_level_accuracy", "subregion_level_accuracy",
        "continent_level_accuracy", "union_accuracy", "intersection_accuracy"]].head())

# Calculate the accuracy metrics
country_level_accuracy = round(df1["country_level_accuracy"].mean()*100, 2)
subregion_level_accuracy = round(df1["subregion_level_accuracy"].mean()*100, 2)
continent_level_accuracy = round(df1["continent_level_accuracy"].mean()*100, 2)
union_accuracy = round(df1["union_accuracy"].mean()*100, 2)
intersection_accuracy = round(df1["intersection_accuracy"].mean()*100, 2)

# find accuracy metrics for each annotator
annotator_country_accuracy = df1.groupby("annotator")["country_level_accuracy"].mean()
annotator_subregion_accuracy = df1.groupby("annotator")["subregion_level_accuracy"].mean()
annotator_continent_accuracy = df1.groupby("annotator")["continent_level_accuracy"].mean()
annotator_union_accuracy = df1.groupby("annotator")["union_accuracy"].mean()
annotator_intersection_accuracy = df1.groupby("annotator")["intersection_accuracy"].mean()

print("Annotator Accuracy Metrics")
print(f"Country Level Accuracy: {annotator_country_accuracy}")
print(f"Subregion Level Accuracy: {annotator_subregion_accuracy}")
print(f"Continent Level Accuracy: {annotator_continent_accuracy}")
print(f"Union Accuracy: {annotator_union_accuracy}")
print(f"Intersection Accuracy: {annotator_intersection_accuracy}")


print(f"Country Level Accuracy: {country_level_accuracy}")
print(f"Subregion Level Accuracy: {subregion_level_accuracy}")
print(f"Continent Level Accuracy: {continent_level_accuracy}")
print(f"Union Accuracy: {union_accuracy}")
print(f"Intersection Accuracy: {intersection_accuracy}")

"""# Plot the accuracy metrics
plt.figure(figsize=(10, 6))
b = sns.barplot(x=["Country", "Subregion", "Continent", "Union", "Intersection"],
                y=[country_level_accuracy, subregion_level_accuracy, continent_level_accuracy, union_accuracy, intersection_accuracy],
                palette="viridis")
b.tick_params(labelsize=10)
plt.title("Human Study Accuracy Metrics", fontsize=20)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("human_study_accuracy_metrics.png", dpi=300)
"""
"""# Plot the overall appropriateness
# Calculate overall appropriateness counts
total_counts = df2["appropriateness"].value_counts()

# Check the actual labels to understand what's in total_counts.index
print("Original Labels:", total_counts.index.tolist())

# Create a mapping dictionary to replace labels
label_mapping = {
    "Neigher Agree nor Disgree": "Neither Agree\nnor Disagree",  # Corrected label
    "Strongly Agree": "Strongly\nAgree",
    "Agree": "Agree",
    "Disagree": "Disagree",
    "Strongly Disagree": "Strongly\nDisagree"
}

# Apply the mapping to replace labels
mapped_labels = [label_mapping.get(label, label) for label in total_counts.index]

# Plotting
plt.figure(figsize=(10, 6))
b = sns.barplot(x=mapped_labels, y=total_counts.values, palette="viridis")

print("new labels:", mapped_labels)

b.tick_params(labelsize=10)
plt.title("Overall Appropriateness", fontsize=20)
plt.ylabel("Count", fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("human_study_overall_appropriateness.png", dpi=300)
"""
