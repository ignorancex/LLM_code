import pandas as pd
import numpy as np
from collections import Counter
import re
import pandas as pd
from collections import Counter
import json
import pprint

df = pd.read_csv("results/dalle_objects/objects_proc.csv")

# [id, country, concept, response, flag]
# the response column is a string that represents a list of strings.
# but it may not be a valid list (e.g. missing the square brackets or missing
# commas or missing quotes aroudn the strings)

# parse the response column into a list of strings by ignoring any errors and
# including everything else.


# Function to parse the response column into a list of strings
def parse_response(response):
    # Use a regular expression to find all substrings that look like items
    items = re.findall(r'\"(.*?)\"|\b\S+\b', response)
    return items

# Function to count items in the parsed response list
def count_items(responses):
    item_counter = Counter()
    for response in responses:
        items = parse_response(response)
        item_counter.update(items)
    return dict(item_counter)

def clean_empty_strings(item_counts):
    return {item: count for item, count in item_counts.items() if item}

# Group the data by 'country' and 'concept'
grouped_data = df.groupby(['country', 'concept'])['response'].apply(list).reset_index()

# Apply the count_items function to each group
grouped_data['item_counts'] = grouped_data['response'].apply(count_items)

# Drop the 'response' column as it's no longer needed
grouped_data = grouped_data.drop(columns=['response'])

# Clean up the item counts by removing empty strings
grouped_data['item_counts'] = grouped_data['item_counts'].apply(clean_empty_strings)

# remove rows with empty item counts
grouped_data = grouped_data[grouped_data['item_counts'].apply(bool)]

# print(grouped_data.head(10))

# Save the results to a CSV file
grouped_data.to_csv("corrected/objects_proc.csv", index=False)

def filter_items_with_counts_greater_than_one(item_counts):
    return {item: count for item, count in item_counts.items() if count > 1}

# Apply the filter function to each group
grouped_data['items_with_count_gt_1'] = grouped_data['item_counts'].apply(filter_items_with_counts_greater_than_one)
filtered_data = grouped_data[grouped_data['items_with_count_gt_1'].apply(lambda x: len(x) > 0)]

# drop the 'item_counts' column as it's no longer needed
filtered_data = filtered_data.drop(columns=['item_counts'])

filtered_data.to_csv("corrected/objects_proc_filtered.csv", index=False)

# Create a dict of country to items, dont need counts of items
# eg. {"Austria" : ["brown table", "brown front door"], "Australia": ["red car",
# "blue car"]}
country_dict = {}
for index, row in filtered_data.iterrows():
    country = row['country']
    items = row['items_with_count_gt_1']
    if country not in country_dict:
        country_dict[country] = {}
    entries = list(items.keys())
    # make all entries lowercase
    entries = [entry.lower() for entry in entries]
    country_dict[country].update({row['concept']: entries})

d = {}
for country, concepts in country_dict.items():
    d[country] = []
    for concept, items in concepts.items():
        for item in items:
            d[country].append(item)

pprint.pp(d)



# Save the country_dict to a JSON file
with open("corrected/country_obj_dict_adj.json", "w") as f:
    f.write(json.dumps(d, indent=4))

# create another dict but from the unfiltered data
country_dict_2 = {}
for index, row in grouped_data.iterrows():
    country = row['country']
    items = row['item_counts']
    if country not in country_dict_2:
        country_dict_2[country] = {}
    entries = list(items.keys())
    # make all entries lowercase
    entries = [entry.lower() for entry in entries]
    country_dict_2[country].update({row['concept']: entries})

d = {}
for country, concepts in country_dict_2.items():
    d[country] = []
    for concept, items in concepts.items():
        for item in items:
            d[country].append(item)

pprint.pp(d)

# Save the country_dict to a JSON file
with open("corrected/country_obj_dict_adj_unfiltered.json", "w") as f:
    f.write(json.dumps(d, indent=4))
