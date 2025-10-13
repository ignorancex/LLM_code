import os
import json

# 图像文件夹路径
base_path = "../med_dataset/ISIC2016/ISBI2016_ISIC_Part1_Training_Data"

# label_id JSON文件路径
label_id_path = "../atlantis/labels_ID.json"

# 读取label_id JSON文件
with open(label_id_path, 'r') as f:
    label_ids = json.load(f)

# 大类别分类
categories = {
    "nature": [
        "cliff", "cypress_tree", "delta", "fjord", "flood", "glacier",
        "hot_spring", "lake", "mangrove", "marsh", "puddle", "rapids",
        "river", "sea", "shoreline", "snow", "waterfall", "wetland"
    ],
    "artificial": [
        "breakwater", "bridge", "canal", "culvert", "dam", "ditch", "levee",
        "lighthouse", "offshore_platform", "pier", "pipeline", "pool",
        "reservoir", "ship", "spillway", "water_tower", "water_well"
    ]
}


def process_images(folder_path, category_labels, nature_file, artificial_file):
    for category_name in os.listdir(folder_path):
        category_key = category_name.replace(" ", "_").replace("-", "_").lower()
        print(f"Processing category: {category_name} ({category_key})")  # Debug print
        if category_key in category_labels:
            category_id = str(category_labels[category_key]).zfill(2)
            category_path = os.path.join(folder_path, category_name)
            for image_name in os.listdir(category_path):
                image_name_wo_ext = os.path.splitext(image_name)[0]
                label = f"{image_name_wo_ext}__{category_id}\n"
                if category_key in categories['nature']:
                    nature_file.write(label)
                elif category_key in categories['artificial']:
                    artificial_file.write(label)
        else:
            print(f"Category {category_name} not found in label_ids")


for folder_name in ["train", "val", "test"]:
    folder_path = os.path.join(base_path, folder_name)
    nature_file_path = os.path.join(base_path, f"{folder_name}_nature.txt")
    artificial_file_path = os.path.join(base_path, f"{folder_name}_artificial.txt")

    with open(nature_file_path, 'w') as nature_file, open(artificial_file_path, 'w') as artificial_file:
        print(f"Processing {folder_name} folder...")  # Debug print
        process_images(folder_path, label_ids, nature_file, artificial_file)

print("处理完成。")


