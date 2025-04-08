from torch.nn import Module

from .LeNet_ELU_Autoencoder import LeNet_ELU_Autoencoder
from .LeNet_ELU_SVDD import LeNet_ELU_SVDD

def get(model_config, checkpoint = None) -> Module:
    cls = globals()[model_config['Type']]
    if checkpoint:
        return cls.load_from_checkpoint(checkpoint, **model_config)
    return cls(getModel=get, **model_config)

def get_SVDD(model_config, checkpoint = None) -> Module:
    cls = globals()['LeNet_ELU_SVDD']
    return cls(getModel=get, **model_config)
