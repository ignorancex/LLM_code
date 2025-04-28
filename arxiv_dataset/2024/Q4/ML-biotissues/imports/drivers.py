import numpy as np
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt
from torch import Tensor
from typing import Callable
import tqdm
from fibernn import mechanics as mech
from fibernn import derivatives
from fibernn.io import *


def compute_batch_loss(eng_p: Tensor, eng_t: Tensor, str_t: Tensor, stiff_t: Tensor,
                       loss_func: Callable[..., Tensor], model: nn.Module, F: Tensor,
                       right_CG: Tensor) -> tuple:
    """
    Computes the batch loss for neural network training.

    :param Tensor eng_p: predicted energy
    :param Tensor eng_t: target energy
    :param Tensor str_t: target stress
    :param Tensor stiff_t: target stiffness
    :param Callable[..., Tensor] loss_func: loss function... f(predict, target)
    :param nn.Module model: pytorch neural network model
    :param Tensor F: deformation gradient
    :param Tensor right_CG: right cauchy green strain tensor
    :param energy_only bool: compute the error in energy only, by default False
    :return tuple: tuple of computed loss for the batch (energy, stress, stiffness loss)
    """
    loss_h0 = loss_func(torch.squeeze(eng_p), eng_t)
    str_p, stiff_p = derivatives.compute_cauchy_stress_stiffness(model, F, right_CG)
    loss_h1 = loss_func(str_p, str_t)
    loss_h2 = loss_func(stiff_p, stiff_t)  # loss_func(stiff_p, stiff_t)
    return (loss_h0, loss_h1, loss_h2)


def model_persistance(model: nn.Module, data: dict[str, data.DataLoader],
                      tag: str = 'nn', workdir: str = './output/'):
    dummy_input = torch.ones(1, 3, 3)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(f'{workdir}{tag}_export.pt')
    model.eval()
    for dat in data.values():
        rcg, defg, eng, stress, stiff = dataset_to_array(dat)
        np.savez(f'{workdir}{tag}_true_data.npz', energy=tensor2ndarr(eng),
                 stress=tensor2ndarr(stress), stiffness=tensor2ndarr(stiff),
                 rcg=tensor2ndarr(rcg), def_grad=tensor2ndarr(defg))
        eng_p = model(rcg)
        stress_p, stiff_p = derivatives.compute_cauchy_stress_stiffness(model, defg, rcg)
        np.savez(f'{workdir}{tag}_predict_data.npz', energy=tensor2ndarr(eng_p),
                 stress=tensor2ndarr(stress_p), stiffness=tensor2ndarr(stiff_p),
                 rcg=tensor2ndarr(rcg), def_grad=tensor2ndarr(defg))
    print('Model saved to disc.')
    # if plot:
    #     plots(f'./output/{tag}', tag=tag)


def TrainLoop(dataloader: data.DataLoader, model: torch.nn.Module,
              loss_func: Callable[..., Tensor], optimizer: torch.optim.Optimizer,
              clamp_params: bool = False, max_loss_level: int = 2, create_graph: bool = False) -> tuple:
    model.train()  # set the mode to train
    h0_loss, h1_loss, h2_loss = 0, 0, 0
    # iterate over the batches
    for bid, (strain, def_grad, eng, stress, stiff) in enumerate(dataloader):
        X: Tensor = strain.to('cpu')
        X.requires_grad = True
        eng_p: Tensor = model(X)  # energy predicted by the model
        h0, h1, h2 = compute_batch_loss(eng_p, eng, stress, stiff, loss_func,
                                        model, def_grad, X)
        h0_loss += h0
        h1_loss += h1
        h2_loss += h2

        match max_loss_level:
            case 0:
                batch_loss = h0
            case 1:
                batch_loss = h0 + h1
            case 2:
                batch_loss = h0 + h1 + h2
            case _:
                raise ValueError('Invalid value of [max_loss_level], options: [0, 1, 2]')

        # backpropagtion
        optimizer.zero_grad()
        if create_graph:
            batch_loss.backward(create_graph=True)
        else:
            batch_loss.backward()
        optimizer.step()
        if clamp_params is True:
            for p in model.parameters():
                p.data.clamp_(0)
    return h0_loss / len(dataloader), h1_loss / len(dataloader), h2_loss / len(dataloader)


def TestLoop(dataloader: data.DataLoader, model: torch.nn.Module,
             loss_func: Callable[..., Tensor]) -> tuple:
    model.eval()  # set the mode to train
    h0_loss, h1_loss, h2_loss = 0, 0, 0
    for _, (strain, def_grad, eng, stress, stiff) in enumerate(dataloader):
        X: Tensor = strain.to('cpu')
        X.requires_grad = True
        eng_p: Tensor = model(X)  # output of the model
        h0, h1, h2 = compute_batch_loss(eng_p, eng, stress, stiff, loss_func,
                                        model, def_grad, X)
        h0_loss += h0
        h1_loss += h1
        h2_loss += h2

    return h0_loss / len(dataloader), h1_loss / len(dataloader), h2_loss / len(dataloader)
