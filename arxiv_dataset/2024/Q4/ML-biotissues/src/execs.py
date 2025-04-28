import sys
sys.path.append('../imports/')
import torch_optimizer as optim_h2

from fibernn.mechanics import *
from nn_functionals import *
from fibernn.io import Adios2NetworkDataset
from fibernn import mechanics, io
from typing import Callable
from torch import Tensor
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import torch
import sys
from loss_functions import NMSE_loss, NMSE_Abs_loss
from drivers import TrainLoop, TestLoop, model_persistance
from torch.utils import tensorboard as tb


def def_exec(nnet: nn.Module, loss_func: Callable[..., Tensor],
             batch_size: int, learning_rate: float, use_h2_adam: bool = False,
             weight_decay: float = 0, seed: int = 4, use_tensorboard: bool = False,
             epoch_max: int = 100, tag: str = 'NNET', max_loss_level: int = 2,
             dataset: str = 'F_large_output_vf.bp'):
    print(f'--- Initializing [TAG: {tag}] ---')
    torch.manual_seed(seed)
    # device = 'cpu'
    dlib = io.DataLib(f'../data/{dataset}', seed=seed)
    if use_tensorboard:
        tb_writer = tb.writer.SummaryWriter(f'../runs/{tag}', flush_secs=15)
        tb_writer.add_graph(nnet, torch.zeros((10, 3, 3)))
    train_dl = data.DataLoader(dlib.train, batch_size=batch_size)
    test_dl = data.DataLoader(dlib.test, batch_size=batch_size)
    # network and optimizer
    # nnet = FullyConnectNN(hl_spec=[10, 10, 10, 10, 10, 10])
    if use_h2_adam is False:
        optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
    else:
        # see: https://github.com/amirgholami/adahessian
        optimizer = optim_h2.Adahessian(nnet.parameters(), lr=learning_rate,
                                                    weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    train_params = sum(p.numel() for p in nnet.parameters() if p.requires_grad)

    print(f'Data and network initialized with {train_params} train params.')
    print('--- Training ---')
    logf = open(f'../output/loss_log_{tag}.csv', 'w')
    logf.write('# EPOCH, TRAIN H0, H1, H2, VALIDATE H0, H1, H2\n')
    # loss_func = NMSEloss  # NMSEloss  # loss function definition
    for i in range(epoch_max):
        h0t_loss, h1t_loss, h2t_loss = TrainLoop(train_dl, nnet, loss_func, optimizer,
                                                 max_loss_level=max_loss_level,
                                                 create_graph=use_h2_adam)
        h0v_loss, h1v_loss, h2v_loss = TestLoop(test_dl, nnet, loss_func)

        match max_loss_level:
            case 0:
                scheduler.step(h0t_loss)
            case 1:
                scheduler.step(h0t_loss + h1t_loss)
            case 2:
                scheduler.step(h0t_loss + h1t_loss + h2t_loss)
            case _:
                raise ValueError('Invalid value of [max_loss_level], options: [0, 1, 2]')

        logf.write(f'{i}, {h0t_loss}, {h1t_loss}, {h2t_loss}, {h0v_loss}, {h1v_loss}, {h2v_loss}\n')
        logf.flush()
        print(f'--- EPOCH {i} --- [MAX LOSS LEVEL: {max_loss_level}]')
        print(f'TRAIN:: H0: {h0t_loss: 1.4f}, H1: {h1t_loss: 1.4f}, H2: {h2t_loss: 1.4f}')
        print(f'TEST:: H0: {h0v_loss: 1.4f}, H1: {h1v_loss: 1.4f}, H2: {h2v_loss: 1.4f}')
        if i % 10 == 0 and i > 0 or i == epoch_max - 1:
            model_persistance(nnet, data={'test': dlib.test}, tag=tag, workdir='../output/')
            print(f'>>> DATA written at epoch: {i}')

        if use_tensorboard:
            tb_writer.add_scalars('loss', {'h0t': h0t_loss, 'h1t': h1t_loss, 'h2t': h2t_loss,
                                           'h0v': h0v_loss, 'h1v': h1v_loss, 'h2v': h2v_loss},
                                  global_step=i)
    logf.close()
    print('All Done!')
