import json
from pprint import pprint
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
def main():
    # file_path = "detection/data/GBCU-Shared/test.json"
    file_path = "detection/data/ddsm/annotations/val.json"
    f = open(file_path)
    data = json.load(f)
    for line in data["annotations"]:
        segmentation = convert_bbox_to_polygon(line["bbox"])
        line["segmentation"] = segmentation
    with open("detection/data/ddsm/annotations/val_new.json", 'w') as f:
        f.write(json.dumps(data))
    print('DONE')
main()