# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn


def set_submodule(module: nn.Module, submodule_name: str, submodule: nn.Module) -> None:
    """
    Set a submodule of a given nn.Module with the specified key (module[submodule_name] = submodule).
    """
    subkeys = submodule_name.split(".")
    if len(subkeys) == 1:
        setattr(module, subkeys[0], submodule)
    else:
        if subkeys[0].isnumeric():
            set_submodule(module[int(subkeys[0])], ".".join(subkeys[1:]), submodule)
        else:
            if not hasattr(module, subkeys[0]):
                set_submodule(module, subkeys[0], nn.Module())
            set_submodule(module.get_submodule(subkeys[0]), ".".join(subkeys[1:]), submodule)
