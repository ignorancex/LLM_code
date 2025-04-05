
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import pad_col

def log_softplus(input, threshold=-15.):
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def nll_continuous_time_loss(phi: Tensor, idx_durations: Tensor, 
                       events: Tensor, input_times: Tensor, spacing: Tensor = None) -> Tensor:

    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)

    # remove time zero
    idx_durations = idx_durations-1
    phi = phi[:, 1:]
    ###################

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    input_times = input_times[keep, :]

    # log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    
    haz = F.softplus(phi)
    haz = haz.mul(torch.diff(input_times))
    
    haz = pad_col(haz, where='start')

    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
    loss = - log_h_e.sub(sum_haz)
    return loss.mean()



def nll_continuous_time_loss_trapezoid(phi: Tensor, idx_durations: Tensor, 
                       events: Tensor, input_times: Tensor) -> Tensor:

    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    input_times = input_times[keep, :]
    n_sample = phi.shape[0]
    n_time = phi.shape[1]


    log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    # log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)

    haz_ = F.softplus(phi)


    filters = torch.tensor([[[0.5, 0.5]]]).to(phi.device)
    haz = F.conv1d(haz_.reshape(n_sample, 1, n_time), filters).reshape(n_sample, n_time-1)
    haz = haz.mul(torch.diff(input_times))

    # pad for cumsum
    # haz = pad_col(haz, where='start')
    haz = pad_col(haz, where='start')

    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
    loss = - log_h_e.sub(sum_haz)
    return loss.mean()

def nll_continuous_time_multi_loss_trapezoid(phis: Tensor, idx_durations_all: Tensor, 
                       events_all: Tensor, input_times_all: Tensor, 
                       all_risk = torch.tensor([])) -> Tensor:
    all_loss = 0
    idx_durations_all = idx_durations_all.long()
    for risk in all_risk:
        phi = phis[:, :, risk]

        if events_all.dtype is torch.bool:
            events_all = events_all.float()
        idx_durations_all = idx_durations_all.view(-1, 1)
        events_all = events_all.view(-1)
        # first_frac_all = first_frac_all.view(-1)


        keep = idx_durations_all.view(-1) >= 0
        phi = phi[keep, :]
        idx_durations = idx_durations_all[keep, :]
        events = events_all[keep]
        input_times = input_times_all[keep, :]
        n_sample = phi.shape[0]
        n_time = phi.shape[1]

        log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul((events==(risk+1)).float())
        # log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
        haz = F.softplus(phi)
        
        filters = torch.tensor([[[1.,1.]]]).to(phis.device)
        haz = F.conv1d(haz.reshape(n_sample, 1, n_time), filters).reshape(n_sample, n_time-1)
        haz = haz.mul(torch.diff(input_times)).div(2)

        haz = pad_col(haz, where='start')

        sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
        loss = - log_h_e.sub(sum_haz)


        all_loss +=loss.mean()

    return all_loss


if __name__ == "__main__":
    phi = torch.randn(3, 10,2)
    idx_durations = torch.tensor([[1],[1],[2]])
    events = torch.tensor([0,1,2])
    not_censor = (events.sub(1) != -1).float()
    n_risk =phi.shape[2]
    print(phi)
    h = phi.gather(1, idx_durations.reshape(-1,1,1).repeat(1,1,2)).view(-1, n_risk).abs()
    log_h = h.log()
    print(h)
    print(log_h)
    print(not_censor)
    print(log_h.mul(not_censor.view(-1,1)))
    # nll_continuous_time_multi_loss(phi, idx_durations, events, torch.tensor([0.5,0.5,0.5]), torch.tensor([1.]))
