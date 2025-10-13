import os

_BASEDIR = os.path.dirname(__file__)

_METADATA_URLS = {
    "train": os.path.join(_BASEDIR, "flist_train.txt"),
    "test": os.path.join(_BASEDIR, "flist_val.txt"),
}
