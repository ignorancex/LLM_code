from argparse import Namespace

from .logger import logger
from yacs.config import CfgNode as _CN
from copy import deepcopy


class CN(_CN):

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.recursive_cfg_update()

    def recursive_cfg_update(self):

        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        # self.freeze()
        return self

    def dump(self, *args, **kwargs):

        def change_back(cfg: CN) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return _CN(cfg).dump(*args, **kwargs)


_C = CN(new_allowed=True)

_C.DATA_PRESET = CN(new_allowed=True)

_C.DATASET = CN(new_allowed=True)
_C.DATASET.TRAIN = CN(new_allowed=True)
_C.FIT = CN(new_allowed=True)
_C.FIT.MANUAL_SEED = 1
_C.FIT.CONV_REPEATABLE = True
_C.FIT.BATCH_SIZE = 4
_C.FIT.EPOCH = 100
_C.FIT.OPTIMIZER = "Adam"
_C.FIT.LR = 0.001
_C.FIT.SCHEDULER = "StepLR"
_C.FIT.LR_DECAY_GAMMA = 0.1
_C.FIT.LR_DECAY_STEP = [70]
_C.FIT.FIND_UNUSED_PARAMETERS = False

_C.FIT.GRAD_CLIP_ENABLED = True
_C.FIT.GRAD_CLIP = CN(new_allowed=True)
_C.FIT.GRAD_CLIP.TYPE = 2
_C.FIT.GRAD_CLIP.NORM = 0.001

_C.MODEL = CN(new_allowed=True)


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, arg: Namespace = None, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    return cfg
