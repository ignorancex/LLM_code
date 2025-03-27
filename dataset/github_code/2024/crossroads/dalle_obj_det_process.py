import os
import pandas as pd
import json
import re

path = f"results/dalle_objects/objects.csv"

df = pd.read_csv(path)

# Define function to parse relevant objects
def parse_relevant_objects(response):
    try:
        clean_response = re.sub(r'```json|```', '', response).strip()
        relevant_objects_str = re.search(r'"relevant_objects": \[(.*?)\],\s*"other_objects"', clean_response, re.DOTALL)
        if relevant_objects_str:
            relevant_objects_json = f'[{relevant_objects_str.group(1)}]'
            return json.loads(relevant_objects_json)
    except json.JSONDecodeError:
        return []
    return []

# Define function to parse non-relevant objects
def parse_non_relevant_objects(response):
    other_objects = []
    try:
        clean_response = re.sub(r'```json|```', '', response).strip()
        other_objects_str = re.search(r'"other_objects": \[(.*)', clean_response, re.DOTALL)
        if other_objects_str:
            other_objects_raw = f'[{other_objects_str.group(1)}]'
            pattern = re.compile(r'\{[^{}]*\}')
            matches = pattern.findall(other_objects_raw)
            for match in matches:
                try:
                    obj = json.loads(match)
                    other_objects.append(obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return []
    return other_objects

# Apply the functions to create new columns
df['Relevant_Items'] = df['response'].apply(parse_relevant_objects)
df['Non_Relevant_Items'] = df['response'].apply(parse_non_relevant_objects)

print(df.head())
df.to_csv("results/dalle_objects/objects_parsed.csv", index=False)

# Initialize the dictionary
country_dict = {}

for index, row in df.iterrows():
    country = row['true_country']
    concept = row['concept']
    if country not in country_dict:
        country_dict[country] = {}
    if concept not in country_dict[country]:
        country_dict[country][concept] = {'Relevant_Items': [], 'Non_Relevant_Items': []}
    country_dict[country][concept]['Relevant_Items'].extend(row['Relevant_Items'])
    country_dict[country][concept]['Non_Relevant_Items'].extend(row['Non_Relevant_Items'])

# Optionally, print or save the dictionary
# print(country_dict)
with open("results/dalle_objects/country_dict.json", "w") as f:
    json.dump(country_dict, f, indent=4)
