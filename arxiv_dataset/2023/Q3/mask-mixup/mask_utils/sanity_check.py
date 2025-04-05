import json
from copy import deepcopy
with open('./mmdetection/wrong_train.txt', 'r') as f:
    wrong_list = f.readlines()
f.close()
wrong_list = [item.split('\n')[0] for item in wrong_list]


with open('./MERGE_OCEAN/Annotations/merge_ocean_train.json', 'r') as f:
    annotations = json.load(f)
f.close()

print(annotations.keys())
filter_annotations = dict()
filter_annotations['categories'] = annotations['categories']
filter_annotations['images'] = []
filter_annotations['annotations'] = []
for image_info in annotations['images']:
    if image_info['file_name'] not in wrong_list:
        img_id = image_info['id']
        filter_annotations['images'].append(deepcopy(image_info))

        for ann in annotations['annotations']:
            if ann['image_id'] == img_id:
                filter_annotations['annotations'].append(deepcopy(ann))

with open('./MERGE_OCEAN/Annotations/filter_ocean_train.json', 'w') as f:
    json.dump(filter_annotations, f)
f.close()
