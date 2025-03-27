# Dataset 3: MARVL GPT
# Folder: results/marvl
# 5 csv files in the folder, each with the following columns:
# id,model,split,response,true_country,concept
# eg. data entries:
# marvl/marvl-images/id/images/1-Burung_gereja/1-0.jpg,gpt-4-turbo-vision-preview,id,Western Europe,id,1-Burung_gereja
# marvl/marvl-images/id/images/1-Burung_gereja/1-1.jpg,gpt-4-turbo-vision-preview,id,Western Europe,id,1-Burung_gereja
# marvl/marvl-images/id/images/1-Burung_gereja/1-10.jpg,gpt-4-turbo-vision-preview,id,Western Europe,id,1-Burung_gereja
# marvl/marvl-images/id/images/1-Burung_gereja/1-11.jpg,gpt-4-turbo-vision-preview,id,Western Europe,id,1-Burung_gereja


import json
import pandas as pd
import os
from tqdm import tqdm
import itertools

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
valid_continents = set(mappings.keys())

lang_to_subregion = {
    "id": "South-eastern Asia", # Indonesia
    "sw": "Eastern Africa",  # "Tanzania", "Kenya", "Rwanda"
    "ta": "Southern Asia", # India, Sri Lanka
    "tr": "Western Asia", # Turkey
    "zh": "Eastern Asia" # China
}

normalization_dict = {
    "North Africa": "Northern Africa",
    "Sahel": "Northern Africa",
    "East Africa": "Eastern Africa",
    "East Africa": "Eastern Africa",
    "Middle Africa": "Middle Africa",
    "Central Africa": "Middle Africa",
    "South Africa": "Southern Africa",
    "West Africa": "Western Africa",
    "Central Asia": "Central Asia",
    "East Asia": "Eastern Asia",
    "East asia": "Eastern Asia",
    "China": "Eastern Asia",
    "South East Asia": "South-eastern Asia",
    "South east asia": "South-eastern Asia",
    "South-east asia": "South-eastern Asia",
    "South-East Asia": "South-eastern Asia",
    "Southeast Asia": "South-eastern Asia",
    "Southeast asia": "South-eastern Asia",
    "Southeast Asia": "South-eastern Asia",
    "Southeastern Asia" : "South-eastern Asia",
    "South-Eastern Asia": "South-eastern Asia",
    "South Asia": "Southern Asia",
    "South asia": "Southern Asia",
    "West Asia": "Western Asia",
    "Eastern Europe": "Eastern Europe",
    "Eastern europe": "Eastern Europe",
    "Northern Europe": "Northern Europe",
    "Southern Europe": "Southern Europe",
    "South-Eastern Europe": "Southern Europe",
    "Spain": "Southern Europe",
    "Western Europe": "Western Europe",
    "North America": "Northern America",
    "Central America": "Central America",
    "Central america": "Central America",
    "South America": "South America",
    "South america": "South America",
    "Latin America": "South America",
    "Latin America And The Caribbean": "South America",
    "Southern America": "South America",
    "Andean States": "South America",
    "Southern Cone": "South America",
    "Southern United States": "Northern America",
    "Southwest": "Northern America",
    "Caribbean": "Caribbean",
    "Melanesia" : "Oceania",
    "Polynesia" : "Oceania",
    "Micronesia" : "Oceania",
    "Australia And New Zealand" : "Oceania",
    "Australasia" : "Oceania",
    "Pacific": "Oceania",
    "South Pacific": "Oceania",
}

def standardize_response(response):
    if isinstance(response, str):
        # Check for uninformative response patterns
        uninformative_keywords = ["cannot provide", "can't assist", "unable to", "insufficient data", "impossible to determine", "no image", "sorry", "not applicable", "no applicable subregion", "indeterminable from image", "image data not available", "i cannot assist with this request", "insufficient information", "content not available", "cannot determine"]
        if any(keyword in response.lower() for keyword in uninformative_keywords) or len(response) > 50:
            return "ResponsibleAIPolicyViolation"
    response = response.strip().rstrip(".").title()
    return normalization_dict.get(response, response)

all_invalid_responses = set()
for lang in ["id", "sw", "ta", "tr", "zh"]:
    INPUT_PATH = f"results/marvl/llava/{lang}.csv" # f"results/marvl/{lang}.csv"
    DATA_OUTPUT_PATH = f"corrected/marvl/llava/data/{lang}.csv" # f"corrected/marvl/data/{lang}.csv"
    df = pd.read_csv(INPUT_PATH)
    df['response'] = df['response'].apply(standardize_response)
    df['response'] = df['response'].replace("Responsibleaipolicyviolation", "ResponsibleAIPolicyViolation")
    df['true_country'] = df['true_country'].map(lang_to_subregion)
    df['response'] = df.apply(lambda row: row['true_country'] if row['response'] == "Sub-Saharan Africa" and row['true_country'] in ["Eastern Africa", "Middle Africa", "Southern Africa", "Western Africa"] else row['response'], axis=1)
    df['response'] = df['response'].replace("Sub-Saharan Africa", "Eastern Africa") # this is anyway an incorrect response, just map it to Eastern Africa for now, only one such example in the entire data

    middle_east_countries = ["Bahrain", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Quatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"]
    mediterranean_countries = ["Albania", "Algeria", "Andorra", "Bosnia and Herzegovina", "Bulgaria", "Burundi", "Croatia", "Cyprus", "Egypt", "Eritrea", "Ethiopia", "France", "Greece", "Iraq", "Israel", "Italy", "Jordan", "Kenya", "Kosovo", "Lebanon", "Libya", "Malta", "Montenegro", "Morocco", "Northern Cyprus", "North Macedonia", "Palestine", "Portugal", "Rwanda", "San Marino", "Serbia", "Slovenia", "South Sudan", "Spain", "Sudan", "Switzerland", "Syria", "Tanzania", "Tunisia", "Turkey", "Uganda", "United Kingdom", "Vatican City"]
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['true_country'] in middle_east_countries else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['true_country'] in mediterranean_countries else row['response'], axis=1)

    # if response is "Tropical", replace it with true_country
    df['response'] = df.apply(lambda row: row['true_country'] if row['response'] == "Tropical" else row['response'], axis=1)

    if lang == "ta":
        # if response is "South", make it "Southern Asia"
        df['response'] = df['response'].replace("South", "Southern Asia")

    if os.path.exists(DATA_OUTPUT_PATH):
        os.remove(DATA_OUTPUT_PATH)
    df.to_csv(DATA_OUTPUT_PATH, index=False)

    # filter out rows with uninformative responses
    invalid_responses = df[~df['response'].isin(valid_subregions)]['response'].unique().tolist()
    print(invalid_responses)
    all_invalid_responses.update(invalid_responses)

print(all_invalid_responses)
# lang = "id" # ["id", "sw", "ta", "tr", "zh"]
# INPUT_PATH = f"results/marvl/{lang}.csv"
# DATA_OUTPUT_PATH = f"corrected/marvl/data/{lang}.csv"
# df = pd.read_csv(INPUT_PATH)
# df['response'] = df['response'].apply(standardize_response)
# df['response'] = df['response'].replace("Responsibleaipolicyviolation", "ResponsibleAIPolicyViolation")
# df['true_country'] = df['true_country'].map(lang_to_subregion)

# # print(df.head())
# # print(df["response"].unique())

# a = df["response"].unique()
# # check which of these are not in valid_subregions
# invalid_subregions = set(a) - valid_subregions
# print(invalid_subregions)
# print(df.head())
