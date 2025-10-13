import os 
import json
import shutil
import tqdm

#####################################################
# change the following paths
img_folder_path = '../figures'
all_MC_path = '../data.json'
#####################################################

with open(os.path.join(all_MC_path), "r") as f:
    all_MC = json.load(f)

for item in all_MC:
    item["image_path"] = os.path.join(img_folder_path, item['_id'] + ".jpg")

with open(os.path.join(all_MC_path), "w") as f:
    json.dump(all_MC, f, indent=1)