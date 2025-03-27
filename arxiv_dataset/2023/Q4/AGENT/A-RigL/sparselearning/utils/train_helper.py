import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn, optim, Tensor
from torch.optim import Optimizer
import copy
#from sgd_k import SGD

from sparselearning.utils.model_serialization import load_state_dict
from sparselearning.utils.warmup_scheduler import WarmUpLR

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *

#import torch
#from torch import Tensor
from torch.optim.optimizer import Optimizer
#from typing import List, Optional
#import copy

class SGD_snap(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SGD_snap, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    def get_param_groups(self):
            return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch. 
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]

    @torch.no_grad()
    def step(self, params, tau, mul, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, new_group, u_group in zip(self.param_groups, params, self.u):
        #for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
            #for p in group['params']:
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update 
                params_with_grad.append(p)
                #d_p_list.append(p.grad)
                d_p_list.append(p.grad.data + tau * (u.grad.data / mul - q.grad.data))
                #d_p_list.append(p.grad.data + tau * (u.grad.data - q.grad.data))
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any([grad.is_sparse for grad in grads])

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    alpha = lr if maximize else -lr
    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=alpha)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=alpha)

#from .optimizer import Optimizer, required
#from typing import List, Optional


def get_optimizer_snap(model: "nn.Module", **kwargs) -> "Tuple[optim]":
    """
    Get model optimizer

    :param model: Pytorch model
    :type model: nn.Module
    :return: Optimizer, LR Scheduler(s)
    :rtype: Tuple[optim, Tuple[lr_scheduler]]
    """
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]

    if weight_decay:
        logging.info("Excluding bias and batchnorm layers from weight decay.")
        parameters = _add_weight_decay(model, weight_decay)
        weight_decay = 0
    else:
        parameters = model.parameters()
    optimizer = SGD_snap(
        parameters
    )

    return optimizer


def get_optimizer(model: "nn.Module", **kwargs) -> "Tuple[optim, Tuple[lr_scheduler]]":
    """
    Get model optimizer

    :param model: Pytorch model
    :type model: nn.Module
    :return: Optimizer, LR Scheduler(s)
    :rtype: Tuple[optim, Tuple[lr_scheduler]]
    """
    name = kwargs["name"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    decay_frequency = kwargs["decay_frequency"]
    decay_factor = kwargs["decay_factor"]

    if name == "SGD":
        # Pytorch weight decay erroneously includes
        # biases and batchnorms
        if weight_decay:
            logging.info("Excluding bias and batchnorm layers from weight decay.")
            parameters = _add_weight_decay(model, weight_decay)
            weight_decay = 0
        else:
            parameters = model.parameters()
        optimizer = SGD(
            parameters,
            lr=lr,
            momentum=kwargs["momentum"],
            weight_decay=weight_decay,
            nesterov=kwargs["use_nesterov"],
        )
    elif name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer.")

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, decay_frequency, gamma=decay_factor
    )

    warmup_steps = kwargs.get("warmup_steps", 0)
    warmup_scheduler = WarmUpLR(optimizer, warmup_steps) if warmup_steps else None

    return optimizer, (lr_scheduler, warmup_scheduler)


def _add_weight_decay(model, weight_decay=1e-5, skip_list=())-> "Tuple[Dict[str, float],Dict[str, float]]":
    """
    Excludes batchnorm and bias from weight decay

    :param model: Pytorch model
    :type model: nn.Module
    :param weight_decay: L2 Weight decay to use
    :type weight_decay: float
    :param skip_list: names of layers to skip
    :type skip_list: Tuple[str]
    :return: Two dictionaries, with layers to apply weight decay to.
    :rtype: Tuple[Dict[str, float],Dict[str, float]]
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Bias, BN have shape 1
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return (
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    )


def save_weights(
    model: "nn.Module",
    optimizer: "optim",
    mask: "Masking",
    val_loss: float,
    step: int,
    epoch: int,
    ckpt_dir: str,
    is_min: bool = True,
):
    """
    Save progress.

    :param model: Pytorch model
    :type model: nn.Module
    :param optimizer: model optimizer
    :type optimizer: torch.optim.Optimizer
    :param mask: Masking instance
    :type mask: sparselearning.core.Masking
    :param val_loss: Current validation loss
    :type val_loss: float
    :param step: Current step
    :type step: int
    :param epoch: Current epoch
    :type epoch: int
    :param ckpt_dir: Checkpoint directory
    :type ckpt_dir: Path
    :param is_min: Whether current model achieves least val loss
    :type is_min: bool
    """
    logging.info(f"Epoch {epoch} saving weights")

    state_dict = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    if mask:
        state_dict["mask"] = mask.state_dict()

    model_path = Path(ckpt_dir) / f"epoch_{epoch}.pth"

    torch.save(state_dict, model_path)

    if is_min:
        model_path = Path(ckpt_dir) / "best_model.pth"
        torch.save(state_dict, model_path)


def load_weights(
    model: "nn.Module",
    optimizer: "optim",
    mask: "Masking",
    ckpt_dir: str,
    resume: bool = True,
) -> "Tuple[nn.Module, optim, Masking, int, int, float]":
    """
    Load model, optimizers, mask from a checkpoint file (.pth).

    :param model: Pytorch model
    :type model: nn.Module
    :param optimizer: model optimizer
    :type optimizer: torch.optim.Optimizer
    :param mask: Masking instance
    :type mask: sparselearning.core.Masking
    :param ckpt_dir: Checkpoint directory
    :type ckpt_dir: Path
    :param resume: resume or not, if not do nothing
    :type resume: bool
    :return: model, optimizer, mask, step, epoch, best_val_loss
    :rtype: Tuple[nn.Module, optim, Masking, int, int, float]
    """
    # Defaults
    step = 0
    epoch = 0
    best_val_loss = 1e6

    if not resume:
        logging.info(f"Not resuming, training from scratch.")
        return model, optimizer, mask, step, epoch, best_val_loss

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pth_files = list(ckpt_dir.glob("epoch_*.pth"))

    if not pth_files:
        logging.info(f"No checkpoint found at {ckpt_dir.resolve()}.")
        return model, optimizer, mask, step, epoch, best_val_loss

    # Extract latest epoch
    latest_epoch = max([int(re.findall("\d+", file.name)[-1]) for file in pth_files])

    # Extract latest model
    model_path = list(ckpt_dir.glob(f"*_{latest_epoch}.pth"))[0]

    logging.info(f"Loading checkpoint from {model_path}.")

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    load_state_dict(model, ckpt["model"])

    if mask and "mask" in ckpt:
        mask.load_state_dict(ckpt["mask"])
        mask.to_module_device_()

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", "not stored")

    logging.info(f"Model has val loss of {val_loss}.")

    # Extract best loss
    best_model_path = ckpt_dir / "best_model.pth"
    if best_model_path:
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        best_val_loss = ckpt.get("val_loss", "not stored")
        logging.info(
            f"Best model has val loss of {best_val_loss} at epoch {ckpt.get('epoch',1)-1}."
        )

    return model, optimizer, mask, step, epoch, best_val_loss


def load_weights_epoch(
    model: "nn.Module",
    optimizer: "optim",
    mask: "Masking",
    ckpt_dir: str,
    epoch: int,
) -> "Tuple[nn.Module, optim, Masking, int, int, float]":
    """
    Load model, optimizers, mask from a checkpoint file (.pth).

    :param model: Pytorch model
    :type model: nn.Module
    :param optimizer: model optimizer
    :type optimizer: torch.optim.Optimizer
    :param mask: Masking instance
    :type mask: sparselearning.core.Masking
    :param ckpt_dir: Checkpoint directory
    :type ckpt_dir: Path
    :param resume: resume or not, if not do nothing
    :type resume: bool
    :return: model, optimizer, mask, step, epoch, best_val_loss
    :rtype: Tuple[nn.Module, optim, Masking, int, int, float]
    """
    # Defaults
    """
    step = 0
    epoch = 0
    best_val_loss = 1e6
    """

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pth_files = list(ckpt_dir.glob("epoch_*.pth"))

    # Extract latest epoch
    #latest_epoch = max([int(re.findall("\d+", file.name)[-1]) for file in pth_files])
    latest_epoch = epoch

    # Extract latest model
    model_path = list(ckpt_dir.glob(f"*_{latest_epoch}.pth"))[0]

    logging.info(f"Loading checkpoint from {model_path}.")

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    load_state_dict(model, ckpt["model"])

    if mask and "mask" in ckpt:
        mask.load_state_dict(ckpt["mask"])
        mask.to_module_device_()

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", "not stored")

    logging.info(f"Model has val loss of {val_loss}.")

    return model, optimizer, mask, step, epoch

