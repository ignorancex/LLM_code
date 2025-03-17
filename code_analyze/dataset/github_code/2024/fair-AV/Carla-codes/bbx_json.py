from PIL import Image
import numpy as np
from scipy.ndimage import label
import json
import time
import argparse
import os
from tqdm import tqdm

def main():
    target_color = [220, 20, 60, 255]
    start = time.time()
    
    root = "Carla_data/Images"
    pathlist = os.listdir(root) 
    # reverse path list
    # pathlist.reverse()
    print(pathlist)
    for path in pathlist:
        print("Generating annotations for ", path)
        gt_data = {}
        gt_data["info"] = {
        "year": 2024,
        "version": 1,
        "description": "Carla",
        "contributor": "LENS Lab"
    },
        gt_data['categories'] = [
            {
            "id": 1,
            "name": "person"
        }
        ]
        gt_data['annotations'] = []
        gt_data['images'] = []
        ann_id = 0
        for num in tqdm(range(0,1000)):
            start = time.time()
            # check if both rgb and seg_ground exist
            if(not os.path.exists(f"{root}/{path}/{num}_rgb.png")): continue
            if(not os.path.exists(f"{root}/{path}/{num}_seg_ground.png")): continue
            image = np.array(Image.open(f"{root}/{path}/{num}_seg_ground.png"))
            bblist = find_bounding_boxes_specific_color(image,target_color)
            if(len(bblist)==0): continue
            for bb in bblist:
                ann_id+=1
                gt_data['annotations'].append({
                    "id": ann_id,
                    "image_id" : num,
                    "category_id" : 1,
                    "bbox" : bb,
                    "area" : bb[2] * bb[3],
                    'iscrowd' : 0,
                })

            gt_data['images'].append({
                "id" : num,
                "width" : 1600,
                "height" : 1200,
                "file_name" : f"{num}_rgb.png"
            })
            
        json_save_folder = f"Carla_data/Preds/{path}"
        os.makedirs(json_save_folder, exist_ok=True)
        
        with open(f"{json_save_folder}/annotations.json", "w") as json_file:
            json.dump(gt_data, json_file)
        print(f"{time.time()-start} sec - experiment {path}" )
        

def parse_path():
    argparser = argparse.ArgumentParser(
        description="gets path")
    argparser.add_argument(
        '--path',
        metavar='P',
        default='',
        type = str,
        help='path')
    
    args = argparser.parse_args()
    
    return args.path

def find_bounding_boxes_specific_color(image_array, target_color):
    """
    This function finds bounding boxes in an image where a specific color indicates an object of interest.
    """
    # List to store bounding boxes
    bounding_boxes = []
    # Create an array that is True where the color matches the target color within the specified tolerance
    color_match = np.all(np.abs(image_array - target_color) == 0, axis=-1)
    # Label the connected components that match the color
    labeled_image, num_features = label(color_match, )
    # Iterate over each feature (object of interest)
    for feature in range(1, num_features + 1):
        # Find the coordinates of the current feature
        rows, cols = np.where(labeled_image == feature)
        if rows.size > 0 and cols.size > 0:
            # Get the bounding box coordinates for the current feature
            y_min, y_max = int(rows.min()), int(rows.max())
            x_min, x_max = int(cols.min()), int(cols.max())
            if((x_max - x_min) * (y_max - y_min) > 25): bounding_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return bounding_boxes
        
if __name__ == '__main__':
    main()
