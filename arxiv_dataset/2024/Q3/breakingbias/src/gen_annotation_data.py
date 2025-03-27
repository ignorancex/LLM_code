import json

import pandas as pd

# Load the templates.json
with open("../data/templates.json", "r") as f:
    templates = json.load(f)

# Extract the desired information from the templates
data_list = []

for template_key, template_data in templates.items():
    # Extract positive_contact, negative_contact and key_principle
    positive_contact = template_data["positive_contact"]
    negative_contact = template_data["negative_contact"]
    key_principle = template_data["key_principle"]

    # Append to the data list
    data_list.append((positive_contact, key_principle))
    data_list.append((negative_contact, key_principle))

# Convert the data list to a DataFrame
df = pd.DataFrame(data_list, columns=["Template", "key_principle"])

# Save the DataFrame as a CSV
df.to_csv("../data/annotate.csv", index=False)
