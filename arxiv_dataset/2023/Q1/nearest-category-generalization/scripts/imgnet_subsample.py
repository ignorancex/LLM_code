from glob import glob

import os
import numpy as np
import joblib

random_seed = 0
np.random.seed(random_seed)

samples_per_cls = 50
#directory = "./data/imagenet/train/"
directory = "./data/imagenet/val/"

results = {
    'img_paths': [],
    'classes': [],
    'idx': [],
}

class_to_idx = joblib.load("./data/imagenet/imagenet_cls2idx.pkl")

now = 0
for target_class in sorted(class_to_idx.keys()):
    class_index = class_to_idx[target_class]
    target_dir = os.path.join(directory, target_class)
    if not os.path.isdir(target_dir):
        continue
    #for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
    fnames = glob(os.path.join(target_dir, "*.JPEG"))
    ind = np.random.choice(
            np.arange(len(fnames)), size=samples_per_cls, replace=False)
    for i in ind:
        results['img_paths'].append(fnames[i])
        results['classes'].append(target_class)
        results['idx'].append(now + i)
    now += len(fnames)

#joblib.dump(results, f"./data/imagenet/imagenet_subsample_{samples_per_cls}_{random_seed}_trn.pkl")
joblib.dump(results, f"./data/imagenet/imagenet_subsample_{samples_per_cls}_{random_seed}_tst.pkl")
