import json

PATH1 = "corrected/country_obj_dict_adj_unfiltered.json" #"corrected/country_obj_dict_adj.json"
PATH2 = "corrected/country_obj_dict_no_adj_unfiltered.json" #"corrected/country_obj_dict_no_adj.json"

with open(PATH1, "r") as f:
    country_obj_dict = json.load(f)

# for each country(key) in the dictionary, there is a list of items (values)
# for each item in the list, extract the last word from the item

for country, items in country_obj_dict.items():
    for i, item in enumerate(items):
        items[i] = item.split()[-1]
        # if this item contains "_", change that to " "
        items[i] = items[i].replace("_", " ")

# remove duplicates from the list of items for each key
for country, items in country_obj_dict.items():
    country_obj_dict[country] = list(set(items))

with open(PATH2, "w") as f:
    json.dump(country_obj_dict, f, indent=4)
