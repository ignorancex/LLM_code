import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class LRIZZ(nn.Module):
    def __init__(self, l=0.5):
        """ LRIZZ loss
        
        :param l: l hyperparameter
        """
        super().__init__()
        self._l = l
    
    def forward(self, predictionsA, predictionsB, targets):
        diff = (predictionsB['prediction'] - predictionsA['prediction'])[:,0]
        loss_ineq = torch.square(F.relu(self._l - targets*diff)) * (targets != 0)
        loss_eq = torch.square(diff) * (targets == 0)
        return (loss_ineq + loss_eq).mean()


class ReconstructLRIZZ(nn.Module):
    def __init__(self, l=0.5, alpha=1.0, beta=0.1):
        """ LRIZZ wiht reconstruction loss
        
        :param l: l hyperparameter
        :param alpha: weight for LRIZZ term
        :param beta: weight for reconstruction term
        """
        super().__init__()
        self.lrizz = LRIZZ(l=l)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, predictionsA, predictionsB, targets):
        lrizz_loss = self.lrizz(predictionsA, predictionsB, targets)
        recons_loss = predictionsA['reconstruction'].mean() + predictionsB['reconstruction'].mean()
        return self.alpha * lrizz_loss + self.beta * recons_loss


def get_args(arguments):
    """ Get arguments from an argument string """
    str_args = arguments.split('/')
    args = {}
    for s in str_args:
        key, val = s.split(':')
        if val.isdigit():
            val = int(val)
        elif val.replace('.','',1).isdigit() and val.count('.') < 2:
            val = float(val)
        else:
            val = str(val) # this is redundant
        args[key] = val
    return args


def get_loss(name, **kwargs):
    """ Gets a loss from a string """
    if 'loss_args' in kwargs:
        loss_args = kwargs['loss_args']
        kwargs = {k: v for k, v in kwargs.items() if k != 'loss_args'}
        if loss_args is not None:
            parsed_args = get_args(loss_args)
            full_args = {**kwargs, **parsed_args}
        else:
            full_args = kwargs
    else:
        full_args = kwargs

    logging.info("Loss: {}".format(name))
    logging.info("Loss arguments: {}".format(full_args))
    
    if name == "lrizz":
        return LRIZZ(**full_args)
    elif name == "reconstructlrizz":
        return ReconstructLRIZZ(**full_args)
    else:
        raise ValueError("Invalid loss")
        

class HDR(nn.Module):
    def __init__(self, threshold=0.25):
        """ Human Disagreement Rate
        
        :param threshold: threshold for disagreement
        """
        super().__init__()
        self._threshold = threshold
        self._err = None
        self._tot = None
        
    def reset(self):
        """ Resets loss """
        self._err = None
        self._tot = None
        
    def compute(self):
        """ Compute the HDR using cached values """
        if self._err is None or self._tot is None:
            raise ValueError("Cannot compute HDR")
        else:
            return self._err / self._tot
        
    def forward(self, predictionsA, predictionsB, targets):
        """ Calculate HDR on given data
        
        The HDR for the specific data will returned, and the object will be modified
        to have the computed results added to cache (to allow computation of full dataset
        metric using compute()).
        """
        diff = (predictionsB['prediction'] - predictionsA['prediction'])[:,0]
        predicted_labels = torch.zeros_like(targets)
        predicted_labels[diff >  self._threshold] = 1
        predicted_labels[diff < -self._threshold] = -1
        
        err = (predicted_labels != targets).sum()
        tot = torch.ones_like(targets).sum()
        
        self._err = err if self._err is None else self._err + err
        self._tot = tot if self._tot is None else self._tot + tot
        return err / tot
