import json
import random

import pandas as pd


random.seed(10)

PATH_MANUAL_ANNOTATION_SAMPLES = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/manual_eval_selection.json"
PATH_SUBMISSIONS_OVERVIEW = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/0_submissions_averitec.xlsx"
PATH_OUTPUT = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/annotation_subsets/{}_{}_manual_evaluation_data.json"
PATH_GOLD_SAMPLES = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/gold_eval_selection.json"

# load for annotations
with open(PATH_MANUAL_ANNOTATION_SAMPLES, "r") as file:
    annotation_samples = json.load(file)

# load submission overview
submissions_overview = pd.read_excel(PATH_SUBMISSIONS_OVERVIEW, header=0)

# load gold samples
with open(PATH_GOLD_SAMPLES, "r") as file:
    gold_samples = json.load(file)

# for each team
# add five gold samples
# save in csv file

print("{} annotation samples!".format(len(annotation_samples)))

for _, row in submissions_overview.iterrows():
    team_name = row['Participant team'].replace(" ", "_")
    rank = row['Rank']
    team_output_path = PATH_OUTPUT.format(rank, team_name)
    annotation_subset = []

    # select 25 samples for annotation and remove from list
    for _ in range(25):
        # Generate a random index
        random_index = random.randint(0, len(annotation_samples) - 1)
        # Remove and return the element at the random index
        annotation_subset.append(annotation_samples.pop(random_index))

    # add gold samples
    annotation_subset.extend(gold_samples)
    # shuffle
    random.shuffle(annotation_subset)
    # Save DataFrame to CSV
    print("Saving sample of size {} to path {}.".format(len(annotation_subset), team_output_path))
    with open(team_output_path, "w") as file:
        json.dump(annotation_subset, file, indent=4)

print("Done!")
print("{} annotation samples left!".format(len(annotation_samples)))
