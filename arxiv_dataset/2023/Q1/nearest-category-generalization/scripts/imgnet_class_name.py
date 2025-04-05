import os.path
from glob import glob

import json
import joblib

res = joblib.load("./data/imagenet/imagenet_cls2idx.pkl")
jr = json.load(open("./data/imagenet/imagenet_class_index.json", "r"))

for k, v in jr.items():
    assert res[v[0]] == int(k), (k, v, res[v[0]])

for path in glob("./data/tiny-imagenet-200/train/*"):
    cls_name = path.split("/")[-1]
    print(jr[str(res[cls_name])][1])
