import os

import json


data = json.load(open('data/DDSM_2k_yolo_v5/annotations/val.json'))

# print(data['annotations'][0], data['images'][0])


def find_annotation_by_id(idx, annotations):
    return list(filter(lambda x: x['image_id'] == idx, annotations))


def find_last_annotation_id(annotations):
    return max(x['id'] for x in annotations)

def convert_bbox_to_polygon(bbox):
    print(bbox)
    if bbox == []:
        bbox = [0,0,0,0]
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    
    polygon = [x,y,(x+w),y,(x+w),(y+h),x,(y+h)]
    print(x, y, w, h, polygon)
    return([polygon])

last_annotation_id = find_last_annotation_id(data['annotations'])

print(find_annotation_by_id(0, data['annotations']))

for i in data['images']:
    # print(i)
    bbox = [0,0, i['width'], i['height']]
    last_annotation_id +=1
    print(bbox, i)
    data['annotations'].append({
        "id": last_annotation_id,
        "image_id": i['id'],
        "category_id": 0,
        "bbox": bbox,
        "area": i['width']*i['height'],
        "segmentation": convert_bbox_to_polygon(bbox),
        "iscrowd": 0
    })

json.dump(data, open("data/DDSM_2k_yolo_v5/annotations/val_newest.json", "w"))