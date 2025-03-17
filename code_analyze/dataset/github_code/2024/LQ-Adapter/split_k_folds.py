import json

import os
import numpy as np
from sklearn.model_selection import KFold

from pprint import pprint

set1 = json.load(open("detection/data/GBCU-Shared/annotations/train_new.json"))
set2 = json.load(open("detection/data/GBCU-Shared/annotations/test_new.json"))

# print(set1)

categories = set1['categories']
assert categories == set2['categories']


annotations1 = set1['annotations']
annotations2 = set2['annotations']

print(len(annotations1), len(annotations2))
print(len(annotations1)/(len(annotations1)+ len(annotations2)))
print(len(annotations2)/(len(annotations1)+ len(annotations2)))


annotations = annotations1+annotations2
print(len(annotations))


def get_annotation_from_image(idx, annotations):
    return list(filter(lambda x: x['image_id'] == idx, annotations))

def findElements(lst1, lst2):
    return list(map(lst1.__getitem__, lst2))


images1 = set1['images']
images2 = set2['images']

images = images1+images2
print(len(images))

print(images[0])
kf = KFold(n_splits=10)

print(kf.get_n_splits(images))

for i, idx in enumerate(kf.split(images)):
    # print(i, idx)
    print(len(idx[0]), len(idx[1]))
    print(images[0], annotations[0])
    pprint(get_annotation_from_image(images[0]['id'], annotations))
    # print(findElements(images, list(idx[0])))
    
    # print(images[100], images[101], findElements(images, [100,101]))
    # print(images[list(idx[0])])
    