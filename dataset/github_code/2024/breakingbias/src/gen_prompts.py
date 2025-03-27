import json
import re

import pandas as pd

holistic_bias_file = "../data/holistic_bias.json"
templates_file = "../data2/templates/type3_templates.json"

with open(holistic_bias_file, "r") as f:
    holistic_bias = json.load(f)

with open(templates_file, "r") as f:
    templates = json.load(f)


def replace_descriptor(bucket, prompt, descriptor):
    if bucket == "after_the_noun":
        # This regex looks for [descriptor] followed by a word
        match = re.search(r"\[descriptor\] (\w+)", prompt)

        # If a match is found
        if match:
            # Get the word (noun) following [descriptor]
            noun = match.group(1)

            # Replace the pattern [descriptor] <noun> with <noun> [descriptor]
            prompt = prompt.replace(f"[descriptor] {noun}", f"{noun} [descriptor]")

    # Replace [descriptor] with the actual descriptor
    return prompt.replace("[descriptor]", descriptor)


# Initial setup
df_rows = []

# Iterate over each template
for template_key, template_data in templates.items():
    # Extract the common data from the template
    id_val = template_data["id"]
    key_principle = template_data["key_principle"]
    scenario = template_data["scenario"]
    action_type = template_data["action_type"]

    # Iterate over each group in the holistic bias
    for group_key, group_data in holistic_bias.items():
        axis_val = group_data["axis"]
        bucket_val = group_data["bucket"]

        # For each descriptor, replace the placeholder with the descriptor and create a row
        for descriptor in group_data["descriptors"]:
            base_prompt = replace_descriptor(
                bucket_val, template_data["base_prompt"], descriptor
            )
            positive_contact = replace_descriptor(
                bucket_val, template_data["positive_contact"], descriptor
            )
            negative_contact = replace_descriptor(
                bucket_val, template_data["negative_contact"], descriptor
            )

            df_row = {
                "id": id_val,
                "key_principle": key_principle,
                "scenario": scenario,
                "action_type": action_type,
                "axis": axis_val,
                "bucket": bucket_val,
                "base_prompt": base_prompt,
                "positive_contact": positive_contact,
                "negative_contact": negative_contact,
            }

            df_rows.append(df_row)

# Create a DataFrame
df = pd.DataFrame(df_rows)

# Save the DataFrame to a CSV
df.to_csv("../data2/type3_dataset.csv", index=False)
