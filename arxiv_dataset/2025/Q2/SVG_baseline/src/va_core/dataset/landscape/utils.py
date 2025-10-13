import os
import numpy as np

_BASEDIR = os.path.dirname(__file__)
_SPLIT_SEED = 803

_METADATA_URLS = {
    "all": os.path.join(_BASEDIR, "flist.txt"),
    "train": os.path.join(_BASEDIR, "flist_train.txt"),
    "test": os.path.join(_BASEDIR, "flist_test.txt"),
}

def _split_dataset_to_subsets(train_size = 0.9):
    assert os.path.exists(_METADATA_URLS["all"]), \
                f"file {_METADATA_URLS['all']} does not exists."
    
    if os.path.exists(_METADATA_URLS["train"]) \
        and os.path.exists(_METADATA_URLS["test"]):
        return
            
    with open(_METADATA_URLS["all"], "r") as f:
        all_meta = f.readlines()
    
    np.random.seed(_SPLIT_SEED)
    np.random.shuffle(all_meta)
    split_index = int(len(all_meta) * train_size)
    train_meta, test_meta = all_meta[:split_index], all_meta[split_index:]

    # "\n" is already included at each line
    with open(_METADATA_URLS["train"], "w") as f:
        f.write("".join(train_meta))
    
    with open(_METADATA_URLS["test"], "w") as f:
        f.write("".join(test_meta))