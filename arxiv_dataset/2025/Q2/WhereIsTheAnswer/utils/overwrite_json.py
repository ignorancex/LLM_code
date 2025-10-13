import json
import sys
import os
import io


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_ds_config(config_path):
    config = jload(config_path)
    return config


ds_config = read_ds_config(sys.argv[1])

nvme_path = os.environ["SGE_LOCALDIR"]
ds_config["zero_optimization"]["offload_optimizer"] = {
    "device": "nvme",
    "nvme_path": nvme_path,
}
ds_config["zero_optimization"]["offload_param"] = {
    "device": "nvme",
    "nvme_path": nvme_path,
}
ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 0

with open(sys.argv[2], "w") as f:
    json.dump(ds_config, f)
