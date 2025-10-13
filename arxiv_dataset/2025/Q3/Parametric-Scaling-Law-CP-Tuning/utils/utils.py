import os
import random
import numpy as np
import torch
from tqdm import tqdm


__all__ = ["Registry"]


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )

        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )

        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


def set_seed(seed):
    if seed != 0:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def split_logits_labels(model, dataloader, device):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)
        logits_list = torch.cat(logits_list).to(device)
        labels_list = torch.cat(labels_list).to(device)
    return logits_list, labels_list




def build_score(conformal):
    if conformal == "aps":
        from score.aps import APS
        return APS()

    elif conformal == "raps":
        from score.raps import RAPS
        return RAPS()

    elif conformal == "saps":
        from score.saps import SAPS
        return SAPS()
    
    elif conformal == "thr":
        from score.thr import THR
        return THR()
    


