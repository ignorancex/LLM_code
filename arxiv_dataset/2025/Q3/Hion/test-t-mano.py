from typing import Optional, Tuple, Union
import hion as hn
import hion.controller as hnc
import torch as th
from logger import progress_bar

"""
Author: Josue N Rivera

python test-t-mano.py --checkpoint[-c] "checkpoint/pendulum_*" --x0-zero=[false] --delta [default:1e-2] --sim_length [default:25.0] --period [default:1.0] 
"""

def simulate(x0: th.DoubleTensor,
             time:th.DoubleTensor,
             dynamics:hn.dynamics.Dynamics,
             controller:hnc.Controller, 
             references:Optional[th.DoubleTensor] = None,
             device:Optional[th.device] = None,
             delta_t:Union[float, th.Tensor] = 1e-3,
             update_period:Optional[Union[float, th.Tensor]] = None) -> Tuple[th.DoubleTensor, th.DoubleTensor, th.DoubleTensor, th.DoubleTensor]:
    
    if update_period is None: update_period = delta_t

    device = x0.device if device is None else device
    n = len(time)

    actual_states = th.zeros(x0.shape[0], n+1, dynamics.first_order_state_n).to(device=device, dtype=th.double)
    observed_states = th.zeros(x0.shape[0], n, dynamics.first_order_state_n).to(device=device, dtype=th.double)
    estimated_states = th.zeros(x0.shape[0], n, dynamics.first_order_state_n).to(device=device, dtype=th.double)
    controls = th.zeros(x0.shape[0], n, dynamics.first_order_control_n).to(device=device, dtype=th.double)
    co_states = th.zeros(x0.shape[0], n, dynamics.first_order_state_n).to(device=device, dtype=th.double)
    nn_clocks = th.zeros(x0.shape[0], n, 1).to(device=device, dtype=th.double)
    observed_state_clocks = th.zeros(x0.shape[0], n, 1).to(device=device, dtype=th.double)

    actual_state = actual_states[:, 0, :] = x0
    observed_state = actual_state
    previos_clock = observed_state_clock = nn_clock = nn_clocks[:, 0, :]

    time = time.expand((x0.shape[0], -1))

    previos_reference = references[:, 0, :]
    
    update_epoch = int(n/100-1)

    progress_bar(0.0)
    for idx_t in range(n):
        reference = references[:, idx_t, :]

        # determine whether observed state should be updated
        ## update with estimated state if only difference in reference state seen
        diff_in_reference = th.norm(reference - previos_reference, dim=1) > 0.1*delta_t

        if any(diff_in_reference):
            previos_reference[diff_in_reference] = reference[diff_in_reference]

            temp_estimate_state = estimated_state[diff_in_reference].detach()
            temp_control = control[diff_in_reference].detach()
            next_estimated_state = temp_estimate_state + delta_t*dynamics.f(
                                                  time = time[diff_in_reference, idx_t],
                                                  first_order_state = temp_estimate_state,
                                                  first_order_control = temp_control)
            
            observed_state[diff_in_reference] = next_estimated_state

        ## update with actual state if update period arrived
        update_period_arrived = observed_state_clock.view(-1) >= update_period
        observed_state[update_period_arrived] = actual_state[update_period_arrived]
        observed_state_clock[update_period_arrived] = observed_state_clock[update_period_arrived]*0.0

        # update nn clock
        clock_update = diff_in_reference | update_period_arrived
        nn_clock[clock_update] = nn_clock[clock_update]*0.0

        # compute control, prediction and update physics
        
        nn_clock.requires_grad_(True)
        estimated_state, control, co_state = controller(nn_clock, observed_state, reference, return_costate=True)
        nn_clock.requires_grad_(False)

        actual_state = actual_state + delta_t*dynamics.f(
                                                  time = time[:, idx_t],
                                                  first_order_state = actual_state.detach(),
                                                  first_order_control = control.detach())
        
        # skip to introduce a jump in the plot
        ## skip if refererence is updated greatly or the update period arrived and it is significant
        skip = (update_period_arrived&(previos_clock.view(-1) > 0.1)) | diff_in_reference&((nn_clock - delta_t).view(-1) > 0.1)

        controls[skip, idx_t, :] = control[skip]*th.nan
        estimated_states[skip, idx_t, :] = estimated_state[skip]*th.nan
        observed_states[skip, idx_t, :] = observed_state[skip]*th.nan
        co_states[skip, idx_t, :] = co_state[skip]*th.nan

        controls[~skip, idx_t, :] = control[~skip]
        estimated_states[~skip, idx_t, :] = estimated_state[~skip]
        observed_states[~skip, idx_t, :] = observed_state[~skip]
        co_states[~skip, idx_t, :] = co_state[~skip]

        actual_states[:, idx_t+1, :] = actual_state
        
        previos_clock = observed_state_clock
        nn_clocks[:, idx_t, :] = nn_clock
        nn_clock = nn_clock + delta_t
        observed_state_clocks[:, idx_t, :] = observed_state_clock
        observed_state_clock = observed_state_clock + delta_t

        if idx_t % update_epoch == 0 or idx_t + 1 == n:
            progress_bar(float(idx_t+1)/n)

    return actual_states, controls, estimated_states, observed_states, co_states, nn_clocks, observed_state_clocks


if __name__ == '__main__':

    # Need to update with torchdiff
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    path = "logs/trash/"
    # saved_file = max([os.path.join(path, basename) for basename in os.listdir(path)], key=os.path.getctime) 
    saved_file = "logs/trash/van der pol - tracker(controller)_2024-12-24_06-08.checkpoint.pth"
    print(saved_file)

    checkpoint = th.load(saved_file)
    config = checkpoint["__configuration__"]

    n = 3

    delta_t = th.tensor(1e-3, dtype=th.double, device=device) # [seconds]
    sim_length = th.tensor(20.0, dtype=th.double, device=device) # [seconds]
    update_period = th.tensor([delta_t, 0.5, config['distribution']['terminal time']], dtype=th.double, device=device) #th.linspace(delta_t, config['distribution']['terminal time']//2, n, dtype=th.double, device=device) # [seconds]
    # update_period = th.ones(n, dtype=th.double, device=device)*config['distribution']['args']['terminal_time']

    mask_small = update_period <= delta_t
    update_period[mask_small] = update_period[update_period <= delta_t] + delta_t
    
    time = th.arange(0.0, sim_length+delta_t, delta_t).to(device=device)

    
    dynamics:hn.dynamics.Dynamics = getattr(hn.dynamics, config['dynamics']['name'])(**config['dynamics']['args'], device=device)
    distribution:hn.distribution.StateDistribution = getattr(hn.distribution, config['distribution']['state']['name'])(**config['distribution']['state']['args'], device=device)

    controller:hnc.Controller = getattr(hnc, config['controller']['name'])(**config['controller']['args'], distribution=distribution, dynamics=dynamics).to(device=device)
    controller.load_state_dict(checkpoint['__log__']['best model']['state'])
    controller.eval()

    x0 = th.ones((n, dynamics.first_order_state_n), dtype=th.double, device=device)*th.tensor([1, 0], device=device)
    references = th.zeros((n, len(time), distribution.reference_n), dtype=th.double, device=device)
    references[:, len(time)//4:, 0] = -0.5
    references[:, len(time)//2:, 0] = 0.75
    references[:, 3*len(time)//4:, 0] = 1.0

    actual_states, controls, estimated_states, observed_states, co_states, nn_clocks, observed_state_clock = simulate(x0 = x0, 
                                                                        time = time,
                                                                        dynamics = dynamics,
                                                                        controller = controller,
                                                                        references = references,
                                                                        delta_t = delta_t,
                                                                        update_period = update_period,
                                                                        device = device)
    
    stats = {
        "config":{
            "delta_t": delta_t,
            "sim_length": sim_length,
            "update_period": update_period,
            "training config": config
        },
        "simulation": {
            "initial states": x0,
            "reference": references,
            "actual states": actual_states,
            "estimated states": estimated_states,
            "controls": controls,
            "observed states": observed_states,
            "costates": co_states,
            "nn clock": nn_clocks,
            "observed state clock": observed_state_clock
        },
        "plot": {
            "variables name": {
                "initial states": [f'x{i}' for i in range(dynamics.first_order_state_n)]
            }
        }
    }
    th.save(stats, "logs/stats/simpleploblem.stats.pth")

