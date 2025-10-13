from typing import Union
import importlib
import pkgutil
from os.path import join
import pydoc

import difference_weighting


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr


def get_longi_network_from_plans(arch_class_name, arch_backbone_class_name, arch_kwargs, arch_kwargs_req_import, 
                            input_channels, output_channels, allow_init=True, deep_supervision: Union[bool, None] = None):
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = recursive_find_python_class(join(difference_weighting.__path__[0], "architectures"), arch_class_name,
                                           "difference_weighting.architectures")
    if nw_class is None:
        raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        backbone_class_name = arch_backbone_class_name,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network