from functools import partial

import numpy as np
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var(argument=r'pgd(?P<nb_iter>_[0-9]+)?')
    @staticmethod
    def pgd(auto_var, inter_var, model, n_classes, nb_iter=None, clip_min=None,
            clip_max=None, device=None, eps=None, norm=None, batch_size=None):
        from .torch.projected_gradient_descent import ProjectedGradientDescent

        if norm is None:
            norm = auto_var.get_var("norm")

        if eps is None:
            eps = auto_var.get_var("eps")

        if nb_iter is None:
            nb_iter = 10
        else:
            nb_iter = int(nb_iter[1:])

        if batch_size is None:
            batch_size = 128

        return ProjectedGradientDescent(
            model_fn=model.model,
            norm=norm,
            clip_min=clip_min,
            clip_max=clip_max,
            eps=eps,
            eps_iter=eps*2/nb_iter,
            nb_iter=nb_iter,
            batch_size=batch_size,
        )

    @register_var(argument=r'cwl2')
    @staticmethod
    def cwl2(auto_var, model, n_classes, clip_min=None, clip_max=None):
        from .torch.cw import CWL2Attack
        return CWL2Attack(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            eps=auto_var.get_var("eps"),
            clip_min=clip_min,
            clip_max=clip_max,
            batch_size=100,
        )
