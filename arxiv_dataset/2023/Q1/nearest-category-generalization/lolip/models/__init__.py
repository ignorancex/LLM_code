import os

from autovar.base import RegisteringChoiceType, VariableClass, register_var
import numpy as np


DEBUG = int(os.getenv("DEBUG", 0))

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var(argument=r'(?P<dataaug>[a-zA-Z0-9]+-)?(?P<loss>[a-zA-Z0-9\._]+)-vtor2-(?P<arch>[a-zA-Z0-9_]+)(?P<train_type>-[a-zA-Z0-9\.]+)?')
    @staticmethod
    def vari_torch_model_v2(auto_var, inter_var, dataaug, loss, arch, train_type, trnX, trny, n_channels, device, multigpu=False, **kwargs):
        from .vari_torch_model import VariTorchModel

        dataaug = dataaug[:-1] if dataaug else None

        if trny is not None:
            n_features = trnX.shape[1:]
        else:
            n_features = kwargs['n_features']
        if trny is not None:
            n_classes = len(np.unique(trny))
        else:
            n_classes = kwargs['n_classes']

        if kwargs is not None:
            params = kwargs
        else:
            params = {}
        params['eps'] = auto_var.get_var("eps")
        params['norm'] = auto_var.get_var("norm")
        params['loss_name'] = loss
        params['n_features'] = n_features
        params['n_classes'] = n_classes
        params['architecture'] = arch
        params['multigpu'] = multigpu
        params['n_channels'] = n_channels
        params['dataaug'] = dataaug
        if train_type is not None:
            params['train_type'] = train_type[1:]
        else:
            params['train_type'] = None

        params['learning_rate'] = auto_var.get_var("learning_rate")
        params['epochs'] = auto_var.get_var("epochs")
        params['momentum'] = auto_var.get_var("momentum")
        params['optimizer'] = auto_var.get_var("optimizer")
        params['batch_size'] = auto_var.get_var("batch_size")
        params['weight_decay'] = auto_var.get_var("weight_decay")
        params['device'] = device

        model = VariTorchModel(
            lbl_enc=inter_var['lbl_enc'],
            **params,
        )
        return model