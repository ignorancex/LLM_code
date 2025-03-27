# categories = ["car", "cups_mugs_glasses", "family_snapshots", "front_door", "home", "kitchen", "plate_of_food", "social_drink", "wall_decoration", "wardrobe"]
# one csv file for each category in a folder.

# Dataset 1: DALLE Street GPT
# Folder: results/dalle_eval/natural, results/dalle_eval/vivid
# 10 csv files in the folder, each with the following columns:
# id,model,split,response,true_country,concept,type
# eg. data entries:
# results/dalle_natural/car/Austria/Austria_1.jpg,gpt-4-turbo-vision-preview,natural_car,Western Europe,Austria,car,natural
# results/dalle_natural/car/Austria/Austria_3.jpg,gpt-4-turbo-vision-preview,natural_car,Eastern Europe,Austria,car,natural
# results/dalle_natural/car/Austria/Austria_4.jpg,gpt-4-turbo-vision-preview,natural_car,Western Europe,Austria,car,natural

# Dataset 2: Dollar Street GPT
# Folder: results/gpt/azure
# 10 csv files in the folder, each with the following columns:
# id,model,split,response,true_country,true_region,true_place,income
# eg. data entries:
# 5d4be626cf0b3a0f3f34324f,gpt-4-turbo-vision-preview,car,Southern Africa,South Africa,af,pypers,90.0
# 5d4bdf39cf0b3a0f3f337517,gpt-4-turbo-vision-preview,car,"I'm sorry, I cannot provide assistance as there is no image for me to analyze and determine a geographical subregion.",Serbia,eu,jambor,228.0
# 5d4bdf42cf0b3a0f3f337609,gpt-4-turbo-vision-preview,car,Southern Europe,Serbia,eu,jambor,228.0
# 5d4bdf18cf0b3a0f3f3371a9,gpt-4-turbo-vision-preview,car,Melanesia,Indonesia,as,akim,234.0

# Dataset 3: Dollar Street LLaVa
# Folder: results/llava
# 10 csv files in the folder, each with the following columns:
# id,model,split,response,true_country,true_region,true_place,income
# eg. data entries:
# 5d4be626cf0b3a0f3f34324f,llava-v1.6-vicuna-7b-hf,car,Africa,South Africa,af,pypers,90.0
# 5d4bdf39cf0b3a0f3f337517,llava-v1.6-vicuna-7b-hf,car,North america,Serbia,eu,jambor,228.0
# 5d4bdf42cf0b3a0f3f337609,llava-v1.6-vicuna-7b-hf,car,North america,Serbia,eu,jambor,228.0

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
# print(valid_subregions)

continent_mapping = {
    "af": "Africa",
    "am": "Americas",
    "as": "Asia",
    "eu": "Europe",
    "oc": "Oceania",
    "an": "Antarctica"
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
    "Caucasus": "Eastern Europe",
    "Russia": "Eastern Europe",
    "Northern Europe": "Northern Europe",
    "Scandinavia": "Northern Europe",
    "Southern Europe": "Southern Europe",
    "Alps" : "Southern Europe",
    "South-Eastern Europe": "Southern Europe",
    "Spain": "Southern Europe",
    "Western Europe": "Western Europe",
    "North America": "Northern America",
    "Canada": "Northern America",
    "Mexico": "Northern America",
    "Central America": "Central America",
    "Central america": "Central America",
    "Mesoamerica": "Central America",
    "South America": "South America",
    "South america": "South America",
    "Latin America": "South America",
    "Latin America And The Caribbean": "South America",
    "Southern America": "South America",
    "Andean States": "South America",
    "Southern United States": "Northern America",
    "Southwest": "Northern America",
    "Caribbean": "Caribbean",
    "Melanesia" : "Oceania",
    "Polynesia" : "Oceania",
    "Micronesia" : "Oceania",
    "Australia And New Zealand" : "Oceania",
    "Pacific": "Oceania",
}


# Function to get subregion and continent for a country
def get_region_and_continent(country):
    return country_to_subregion.get(country, (None, None))

def standardize_response(response):
    if isinstance(response, str):
        # Check for uninformative response patterns
        uninformative_keywords = ["cannot provide", "can't assist", "unable to", "insufficient data", "impossible to determine", "no image", "sorry", "not applicable", "no applicable subregion", "indeterminable from image", "image data not available", "i cannot assist with this request", "insufficient information", "content not available", "cannot determine"]
        if any(keyword in response.lower() for keyword in uninformative_keywords) or len(response) > 50:
            return "ResponsibleAIPolicyViolation"
    response = response.strip().rstrip(".").title()
    return normalization_dict.get(response, response)

categories = ["car", "cups_mugs_glasses", "family_snapshots", "front_door", "home", "kitchen", "plate_of_food", "social_drink", "wall_decoration", "wardrobe"]
models = ["dalle_eval/llava/natural", "dalle_eval/llava/vivid"] #["llava", "gpt/azure", "dalle_eval/natural", "dalle_eval/vivid"]

invalids = set()
for CATEGORY, MODEL in tqdm(itertools.product(categories, models), total=len(categories) * len(models)):
    INPUT_PATH = f"results/{MODEL}/{CATEGORY}.csv"
    DATA_OUTPUT_PATH = f"corrected/{MODEL}/data/{CATEGORY}.csv"
    # EVAL_OUTPUT_PATH = f"corrected/{MODEL}/results/{CATEGORY}.json"

    # Load the CSV file
    df = pd.read_csv(INPUT_PATH)
    df[['true_sub_region', 'true_continent']] = df.apply(lambda row: pd.Series(get_region_and_continent(row['true_country'])), axis=1)
    # if "dalle_eval" not in MODEL:
    #     df['true_continent'] = df['true_region'].map(continent_mapping)
    # else:
    df['true_continent'] = df['true_continent'].map(continent_mapping)
    df['response'] = df['response'].apply(standardize_response)
    df['response'] = df['response'].str.rstrip(".")
    df['response'] = df['response'].replace("Responsibleaipolicyviolation", "ResponsibleAIPolicyViolation")
    df['response'] = df.apply(lambda row: "Eastern Europe" if row['true_country'] == "Czech Republic" and row['response'] == "Central Europe" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: "Western Europe" if row['true_country'] == "Austria" and row['response'] == "Central Europe" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['response'] == "Sub-Saharan Africa" and row['true_sub_region'] in ["Eastern Africa", "Middle Africa", "Southern Africa", "Western Africa"] else row['response'], axis=1)
    df['response'] = df.apply(lambda row: "Africa" if row['response'] == "Sub-Saharan Africa" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['response'] == "Europe" and row['true_continent'] == "Europe" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['response'] == "Africa" and row['true_continent'] == "Africa" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['response'] == "Asia" and row['true_continent'] == "Asia" else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['response'] == "Oceania" and row['true_continent'] == "Oceania" else row['response'], axis=1)

    # https://en.wikipedia.org/wiki/List_of_Middle_Eastern_countries_by_population
    # https://en.wikipedia.org/wiki/List_of_Mediterranean_countries
    middle_east_countries = ["Bahrain", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Quatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"]
    mediterranean_countries = ["Albania", "Algeria", "Andorra", "Bosnia and Herzegovina", "Bulgaria", "Burundi", "Croatia", "Cyprus", "Egypt", "Eritrea", "Ethiopia", "France", "Greece", "Iraq", "Israel", "Italy", "Jordan", "Kenya", "Kosovo", "Lebanon", "Libya", "Malta", "Montenegro", "Morocco", "Northern Cyprus", "North Macedonia", "Palestine", "Portugal", "Rwanda", "San Marino", "Serbia", "Slovenia", "South Sudan", "Spain", "Sudan", "Switzerland", "Syria", "Tanzania", "Tunisia", "Turkey", "Uganda", "United Kingdom", "Vatican City"]
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['true_country'] in middle_east_countries else row['response'], axis=1)
    df['response'] = df.apply(lambda row: row['true_sub_region'] if row['true_country'] in mediterranean_countries else row['response'], axis=1)
    # if response is "South", change it to "Southern America"
    df['response'] = df.apply(lambda row: "South America" if row['response'] == "South" else row['response'], axis=1)
    if os.path.exists(DATA_OUTPUT_PATH):
        os.remove(DATA_OUTPUT_PATH)
    # Save the modified DataFrame
    df.to_csv(DATA_OUTPUT_PATH, index=False)

    # filter out rows with uninformative responses
    df = df[df['response'] != "ResponsibleAIPolicyViolation"]
    invalid_responses = df[~df['response'].isin(valid_subregions)]['response'].unique().tolist()
    print(f"{MODEL}/{CATEGORY}: {invalid_responses} invalid responses")
    invalids.update(invalid_responses)

    # at this point of the code, any invalid response is an incorrect response
    # all other responses are either correct or ResponsibleAIPolicyViolation


print(sorted(invalids))

print("-"*80)
