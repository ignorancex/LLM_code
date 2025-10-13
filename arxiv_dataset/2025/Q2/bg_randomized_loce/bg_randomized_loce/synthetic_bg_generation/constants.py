BG_DIR = "./data/backgrounds"

BG_DIFFUSION = ["cloudscape", "space", "jungle", "desert", "arctic", "volcanic", "ocean", "abstract_patterns"]
BG_RANDOM = ["uniform_noise"]
BG_ARTIFICIAL = BG_DIFFUSION + BG_RANDOM
BG_ALL = BG_ARTIFICIAL + ["original"]

SEEDS = range(1)              

WORKING_DIR = "./experiment_outputs/background_randomization/"


VOC_CATEGORIES = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


# PASCAL VOC 2012 trainval concept mask coverage statistics
"""
 1 aeroplane      : 128 of 178 images have mask coverage within the range 4%-64% (min/max overall: 0.02%/50.99%).
 2 bicycle        : 77  of 143 images have mask coverage within the range 4%-64% (min/max overall: 0.01%/67.20%).
 3 bird           : 132 of 208 images have mask coverage within the range 4%-64% (min/max overall: 0.05%/68.02%).
 4 boat           : 103 of 150 images have mask coverage within the range 4%-64% (min/max overall: 0.01%/42.56%).
 5 bottle         : 72  of 183 images have mask coverage within the range 4%-64% (min/max overall: 0.01%/77.90%).
 6 bus            : 131 of 152 images have mask coverage within the range 4%-64% (min/max overall: 0.15%/90.53%).
 7 car            : 151 of 255 images have mask coverage within the range 4%-64% (min/max overall: 0.02%/78.36%).
 8 cat            : 225 of 250 images have mask coverage within the range 4%-64% (min/max overall: 0.69%/94.34%).
 9 chair          : 155 of 271 images have mask coverage within the range 4%-64% (min/max overall: 0.00%/99.12%).
10 cow            : 114 of 135 images have mask coverage within the range 4%-64% (min/max overall: 0.43%/79.91%).
11 diningtable    : 127 of 157 images have mask coverage within the range 4%-64% (min/max overall: 0.31%/96.71%).
12 dog            : 217 of 249 images have mask coverage within the range 4%-64% (min/max overall: 0.45%/89.17%).
13 horse          : 128 of 147 images have mask coverage within the range 4%-64% (min/max overall: 0.06%/83.96%).
14 motorbike      : 133 of 157 images have mask coverage within the range 4%-64% (min/max overall: 0.20%/71.64%).
15 person         : 597 of 888 images have mask coverage within the range 4%-64% (min/max overall: 0.00%/77.70%).
16 pottedplant    : 77  of 167 images have mask coverage within the range 4%-64% (min/max overall: 0.00%/92.06%).
17 sheep          : 87  of 120 images have mask coverage within the range 4%-64% (min/max overall: 0.12%/75.26%).
18 sofa           : 163 of 183 images have mask coverage within the range 4%-64% (min/max overall: 2.24%/99.41%).
19 train          : 155 of 167 images have mask coverage within the range 4%-64% (min/max overall: 0.83%/97.79%).
20 tvmonitor      : 109 of 157 images have mask coverage within the range 4%-64% (min/max overall: 0.15%/74.68%).
"""
