from collections import defaultdict
from typing import Dict, Optional, Union

import torch as th
import torch.nn as nn
import torch.optim as optim

import hion as hn
from hion.distribution import StateDistribution
from torch.distributions import Uniform

"""
Author: Josue N Rivera
"""

class NN(nn.Module):
    def __init__(self, in_n, out_n,
                       width:int = 20,
                       depth:int = 2,
                       blocks:int = 1,
                       activation = nn.SiLU(),
                       dtype:th.dtype = th.double) -> None:
        super().__init__()

        self.entry = nn.Sequential(nn.Linear(in_n, width, dtype=dtype), activation)

        self.exit = nn.Sequential(nn.Linear(width, out_n, dtype=dtype))

        self.blocks = nn.ModuleList([
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width, dtype=dtype), activation) for _ in range(max(0, depth))]) for __ in range(blocks)]
        )

    def forward(self, x:th.DoubleTensor) -> th.DoubleTensor:
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        return self.exit(x)

class PINC_Van_Estimator(nn.Module):

    def __init__(self, dynamics):
        super().__init__()
        self.net = NN(1+2+1, 1, 20, 4, dtype=th.double)
        self.dynamics = dynamics

    def forward(self, t, x0, u):

        state = x0[:, 0:1] + x0[:, 1:2]*t + self.net(th.concat([t, x0, u], dim=1))*t**2 # As polynomial to enforce initial condition via taylor operator
        return self.dynamics.first_state_representation(t, state)

def print_progress(progress:Union[float, int], loss:Optional[Dict[str, float]] = None):

    def dict_to_print(loss)->str:
        stats = []
        for key, value in loss.items():
            if isinstance(value, dict):
                stats.append('[' + dict_to_print(value) + ']')
            else:
                stats.append(f'{key}: {value:.5f}')

        return ' '.join(stats)

    print(f'[{progress + 1:5d}] ' + dict_to_print(loss))

class VanDerPolPINCStateDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-3, dtype=th.double, device=device), th.tensor(3, dtype=th.double, device=device)),
                            Uniform(th.tensor(-3, dtype=th.double, device=device), th.tensor(3, dtype=th.double, device=device))
                        ], device=device)

if __name__ == "__main__":

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    terminal_time = 0.5

    th.autograd.set_detect_anomaly(True)

    time_distribution = hn.distribution.ConstantValueDistribution(terminal_time, device=device, dtype=th.double)

    distribution:hn.distribution.StateDistribution = VanDerPolPINCStateDistribution(device=device)
    control_dist = Uniform(th.tensor(-3, dtype=th.double, device=device), th.tensor(3, dtype=th.double, device=device))
    dynamics:hn.dynamics.Dynamics = hn.dynamics.VanDerPolDynamics(device=device)

    estimator = PINC_Van_Estimator(dynamics=dynamics).to(device=device)
    estimator.train()

    """ Start Training """

    """ To train from empty model"""
    best_log = {
        "loss": 1e+10,
        "idx": 0,
        "state": None
    }

    """ To Finetune """
    # best_log = th.load('.rug_PINC-van-estimater4.pth', weights_only=False)
    # estimator.load_state_dict(best_log['state'])

    optimizer:optim.Optimizer = optim.Adam(estimator.parameters(), lr=5e-6)

    
    running_loss = defaultdict(lambda : 0.0)
    running_loss_log = defaultdict(lambda : [])

    all_loss_log = defaultdict(lambda : [])

    def update_running_loss(cum, net, cycle, reset=True):
        for key, value in cum.items():
            if isinstance(value, dict):
                update_running_loss(value, net[key], cycle, reset=reset)
            else:
                net[key].append(value/cycle)
                if reset:
                    cum[key] = 0.0

    def to_dict(d):
        d = dict(d)
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = to_dict(value)
        return d

    progress_rate:int = 20

    n = 1000
    sampling_size:dict = {'boundary': 1000, 'transient': 20000}

    bd_scale = float(sampling_size['boundary'])/(2*sampling_size['boundary'] + sampling_size['transient'])
    trans_scale = float(sampling_size['transient'])/(2*sampling_size['boundary'] + sampling_size['transient'])

    for epoch in range(n):

        time_sample_batch = time_distribution.sample((sampling_size['boundary'], 1))
        observed_sample_batch, __ = distribution.sample(sampling_size['boundary'])

        ## Initial boundary points
        init_time_sample = th.zeros_like(time_sample_batch).to(device)
        init_time_sample.requires_grad_()
        init_observed_sample = observed_sample_batch.to(device)
        init_control_sample = control_dist.sample((sampling_size['boundary'], 1)).to(device)

        ## Transient points
        transient_idxs = th.randint(0, sampling_size['boundary'], (sampling_size['transient'],)) # Shuffle run to select from
        tran_time_sample = time_sample_batch[transient_idxs] * (th.rand((sampling_size['transient'], 1))).to(device)
        tran_time_sample.requires_grad_()
        tran_observed_sample = observed_sample_batch[transient_idxs]
        tran_control_sample = init_control_sample[transient_idxs]


        """ ---------------- Training ---------------- """

        # try:
        # Infer control, state, costate

        first_state_init = estimator(init_time_sample, init_observed_sample, init_control_sample)

        first_state_tran = estimator(tran_time_sample, tran_observed_sample, tran_control_sample)

        losses = {}
        losses["dynamics"] = hn.dynamics_loss(init_time_sample, first_state_init, init_control_sample, dynamics)*bd_scale
        losses["dynamics"] = losses["dynamics"]+hn.dynamics_loss(tran_time_sample, first_state_tran, tran_control_sample, dynamics)*trans_scale

        losses["initial-boundary"] = hn.boundary_loss(first_state_init, init_observed_sample)

        loss:th.Tensor = sum(losses.values())

        # Update parameters 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """ ---------------- Log and checkpoint ---------------- """

        for key in losses.keys():
            item_loss = losses[key].item()
            running_loss[key] += item_loss
            all_loss_log[key].append(item_loss)
        
        net_item = loss.item()
        running_loss['net'] += net_item
        all_loss_log['net'].append(net_item)

        ## Save best seen model so far (not exactly the best since it is after update but close)
        if net_item < best_log['loss']:
            best_log = {
                "loss": net_item,
                "idx": epoch,
                "state": estimator.state_dict()
            }

        """ ---------------- Print results ---------------- """
        cycle = epoch%progress_rate + 1
        if cycle == progress_rate or epoch == n-1 or epoch == 0:

            update_running_loss(running_loss, running_loss_log, cycle, reset=cycle==progress_rate)

            print_progress(epoch, loss = {
                                        "net": running_loss_log["net"][-1],
                                        "init-bd": running_loss_log["initial-boundary"][-1],
                                        "dynamics": running_loss_log["dynamics"][-1]
                                        })

    th.save(best_log, 'PINC-van-estimater4.pth')

    print('Finished Training')