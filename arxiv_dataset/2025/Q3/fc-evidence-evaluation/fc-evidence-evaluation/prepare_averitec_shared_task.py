"""
Extracted samples from systems predictions to manually evaluate Averitec shared task systems.
"""
import json
import random

import pandas as pd


random.seed(10)

PATH_AVERITEC_SHARED_TASK_TESTDATA = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/data_test.json"
PATH_SUBMISSIONS_OVERVIEW = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/0_submissions_averitec.xlsx"
PATH_AVERITEC_SHARED_TASK_TESTDATA_GOLD = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_shared_task_test.json"
PATH_TEMPL_SYSTEM_PER_TEAM_EMAIL = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/{}_{}_email.json"
PATH_TEMPL_SYSTEM_PER_TEAM_EVALAI = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/{}_{}_evalai.json"
GOLD_IDS = [333, 446, 844, 1161, 1641]


with open(PATH_AVERITEC_SHARED_TASK_TESTDATA, "r") as file:
    data_test = json.load(file)

# load submission overview
df = pd.read_excel(PATH_SUBMISSIONS_OVERVIEW, header=0)

# randomly select indexes between 0 and 2214
claim_indices = random.sample(range(2215), 560)

# check if approx. half of samples have index <1000 and remaining ones index > 1000
count_first_subset = len([x for x in claim_indices if x < 1000])
assert 200 < count_first_subset < 300, "Extracted indexes not equally distributed across entire data"

# extract different categories
low_score = df[df['AVeriTeC Score'] < 0.16]
mid_low_score = df[(df['AVeriTeC Score'] >= 0.16) & (df['AVeriTeC Score'] < 0.22)]
mid_high_score = df[(df['AVeriTeC Score'] >= 0.22) & (df['AVeriTeC Score'] < 0.435)]
high_score = df[df['AVeriTeC Score'] >= 0.435]

# iterate over each category e.g. low_score sub dataframe
# load the json file for each row
# select 5 claims from this file
# stop iteration after 50 claims have been selected

# Categories list
categories = [low_score, mid_low_score, mid_high_score, high_score]
number_samples_per_category = 140

# Initialize variables to keep track of selected claims
annotation_samples = []
total_claims = 0

# load gold data
with open(PATH_AVERITEC_SHARED_TASK_TESTDATA_GOLD, "r") as file:
    gold_data = json.load(file)

# Iterate over each category
for category in categories:
    total_claims_p_category = 0
    while True:
        for index, row in category.iterrows():
            # Load the JSON file for the current row
            rank = row['Rank']
            team_name = row['Participant team'].replace(" ", "_")
            json_filename = PATH_TEMPL_SYSTEM_PER_TEAM_EMAIL.format(rank, team_name)
            try:
                source = f"{rank}_{team_name}_email.json"
                with open(json_filename, "r") as json_file:
                    claims_data = json.load(json_file)
                print("File loaded: {}".format(json_filename))
            except FileNotFoundError:
                print("File not found: {}".format(json_filename))
                json_filename = PATH_TEMPL_SYSTEM_PER_TEAM_EVALAI.format(rank, team_name)
                source = f"{rank}_{team_name}_evalai.json"
                with open(json_filename, "r") as json_file:
                    claims_data = json.load(json_file)
                print("File loaded: {}".format(json_filename))

            # Select 5 claims from the JSON data based on indices
            selected_claims_from_json = []
            for i in claim_indices[total_claims:total_claims + 5]:
                entry = claims_data[i].copy()
                entry['source'] = f"{rank}_{team_name}_evalai.json"
                entry['reference_evidence'] = gold_data[i]["questions"]
                entry["predicted_evidence"] = entry.pop("evidence")
                if "id" in entry:
                    entry["claim_id"] = entry.pop("id")
                selected_claims_from_json.append(entry)
            # selected_claims_from_json = [claims_data[i] for i in claim_indices[total_claims:total_claims+5]]

            # Add selected claims to the list
            annotation_samples.extend(selected_claims_from_json)
            total_claims += len(selected_claims_from_json)
            total_claims_p_category += len(selected_claims_from_json)
            if total_claims_p_category >= number_samples_per_category:
                break
        if total_claims_p_category >= number_samples_per_category:
            break

print(len(annotation_samples))

# remove gold eval samples and save separately
gold_samples = []
annotation_samples_final = []
for entry in annotation_samples:
    if entry["claim_id"] in GOLD_IDS:
        gold_samples.append(entry)
    else:
        annotation_samples_final.append(entry)

print(len(annotation_samples_final))
print(len(gold_samples))

# save w/ source
with open("/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/manual_eval_selection_w_source.json", "w") as file:
    json.dump(annotation_samples_final, file, indent=4)
# save w/out source
with open("/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/manual_eval_selection.json", "w") as file:
    for entry in annotation_samples_final:
        entry.pop("source")
    json.dump(annotation_samples_final, file, indent=4)
# save gold samples
with open("/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/gold_eval_selection.json", "w") as file:
    json.dump(gold_samples, file, indent=4)
