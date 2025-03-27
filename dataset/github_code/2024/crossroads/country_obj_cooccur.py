import json
import pandas as pd
import numpy as np
from collections import Counter

MODE = "adj" # "no_adj" or "adj"
PATH1 = f"corrected/country_obj_dict_{MODE}_unfiltered.json" #f"corrected/country_obj_dict_{MODE}.json"
PATH2 = f"corrected/country_obj_{MODE}_co_occurence_unfiltered.csv" #f"corrected/country_obj_{MODE}_co_occurence.csv"
PATH3 = f"corrected/country_obj_{MODE}_tfidf_unfiltered.json" #f"corrected/country_obj_{MODE}_tfidf.json"
PATH4 = f"corrected/country_obj_{MODE}_tfidf_T_unfiltered.json" #f"corrected/country_obj_{MODE}_tfidf_T.json"
PATH5 = f"corrected/country_obj_{MODE}_tfidf_unfiltered.csv" #f"corrected/country_obj_{MODE}_tfidf.csv"

with open(PATH1, "r") as f:
    country_obj_dict = json.load(f)

# make a dataframe with countries as rows and items as columns, and values as
# counts
countries = list(country_obj_dict.keys())
items = []
for country in countries:
    items += country_obj_dict[country]
items = list(set(items))
df = pd.DataFrame(0, index=countries, columns=items)

# fill in the dataframe with counts
for country in countries:
    for item in country_obj_dict[country]:
        df.at[country, item] += 1

# fill NaN values with 0
df = df.fillna(0)
# print(df)
df.to_csv(PATH2)

# this dataframe now contains term frequencies for each item

# number of countries that an item occurs in (document frequency)
document_frequency = df.astype(bool).sum(axis=0)

# idf = how common/rare an item is across all countries
idf = np.log(len(countries) / document_frequency)

# tf-idf = term frequency * inverse document frequency
# A high TF-IDF score indicates an item is frequently mentioned in a particular country but rare across all countries
tfidf = df.multiply(idf, axis=1)

d = {}
for country in countries:
    # iterate through the dataframe, and get all the non-zero tf-idf scores for
    # each country into a list.
    d[country] = tfidf.loc[country][tfidf.loc[country] > 0].to_dict()
    # sort the tf-idf scores in descending order
    d[country] = dict(sorted(d[country].items(), key=lambda x: x[1], reverse=True))


# save the tf-idf scores for each country into a json file
with open(PATH3 , "w") as f:
    json.dump(d, f, indent=4)

d_T = {}
for item in items:
    d_T[item] = tfidf[item].to_dict()
    d_T[item] = dict(sorted(d_T[item].items(), key=lambda x: x[1], reverse=True))
    # remove zero values
    d_T[item] = {k: v for k, v in d_T[item].items() if v > 0}

with open(PATH4, "w") as f:
    json.dump(d_T, f, indent=4)

tfidf.to_csv(PATH5)
