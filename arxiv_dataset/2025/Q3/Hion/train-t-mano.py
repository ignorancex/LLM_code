from collections import defaultdict
import os
import json, argparse

import torch as th
import torch.optim as optim

import hion as hn
import hion.fn as fn
from hion.types import LagrangianFunc, ConfigDict
from logger import ControllerLogger

"""
Author: Josue N Rivera

python train-t-mano.py --config[-c] [default:"config.linear.json"] --seed [default:-1] --finetune [default:""]
"""

def train(config:ConfigDict,
          checkpoint = None,
          device:th.device = th.device('cuda' if th.cuda.is_available() else 'cpu')) -> None:

    th.autograd.set_detect_anomaly(True)
    for key in config['lagrangian']['args'].keys():
        config['lagrangian']['args'][key] = th.tensor(config['lagrangian']['args'][key], dtype=th.double, device=device)

    
    terminal_time = config['distribution']['terminal time']
    time_distribution = hn.distribution.ConstantDistribution(terminal_time)

    distribution:hn.distribution.StateDistribution = getattr(hn.distribution, config['distribution']['state']['name'])(**config['distribution']['state']['args'], device=device)
    dynamics:hn.dynamics.Dynamics = getattr(hn.dynamics, config['dynamics']['name'])(**config['dynamics']['args'])

    if checkpoint is None:
        controller:hn.controller.TMano = getattr(hn.controller, config['controller']['name'])(**config['controller']['args'], distribution=distribution, dynamics=dynamics)
    else:
        checkpoint['__']
        controller:hn.controller.TMano = getattr(hn.controller, config['controller']['name'])(**config['controller']['args'], distribution=distribution, dynamics=dynamics)

        controller.load_state_dict(checkpoint['__log__']['best model']['state'])
        # controller.load_state_dict(checkpoint['__training__']['controller'])
    
    controller = controller.to(device=device)
    controller.train()

    optimizer:optim.Optimizer = getattr(optim, config['training']['optimizer']['name'])(controller.parameters(), **config['training']['optimizer']['args'])


    logger = ControllerLogger(controller=controller,
                    optimizer=optimizer,
                    configuration=config)
    
    logger.log("seed", th.seed())
    
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

    running_loss_log_ref = logger.start_ref_log("running losses log")
    logger.update_ref_log(running_loss_log_ref, to_dict(running_loss_log))

    all_loss_log_ref = logger.start_ref_log("all losses log")
    logger.update_ref_log(all_loss_log_ref, to_dict(all_loss_log))

    progress_rate:int = config['checkpoint']['progress rate']
    l_mult:dict = config['training']['losses']

    n = config['training']['number of epoch']
    sampling_size:dict = config['distribution']['sample']
    terminal_time = config['distribution']['terminal time']

    bd_scale = float(sampling_size['boundary'])/(2*sampling_size['boundary'] + sampling_size['transient'])
    trans_scale = float(sampling_size['transient'])/(2*sampling_size['boundary'] + sampling_size['transient'])

    """ Start Training """
    best_log = {
        "loss": 1e+10,
        "idx": 0,
        "state": None
    }
    best_log_ref = logger.start_ref_log("best model")
    logger.update_ref_log(best_log_ref, best_log)

    for epoch in range(n):

        time_sample_batch = time_distribution.sample((sampling_size['boundary'], 1))
        observed_sample_batch, reference_sample_batch = distribution.sample(sampling_size['boundary'])

        ## Initial boundary points
        init_time_sample = th.zeros_like(time_sample_batch).to(device)+fn.EPSILON
        init_time_sample.requires_grad_()
        init_observed_sample = observed_sample_batch.to(device)
        init_reference_sample = reference_sample_batch if reference_sample_batch is None else reference_sample_batch.to(device)

        ## Terminal boundary points
        term_time_sample = time_sample_batch.to(device)+fn.EPSILON
        term_time_sample.requires_grad_()
        term_observed_sample = init_observed_sample
        term_reference_sample = reference_sample_batch if reference_sample_batch is None else reference_sample_batch.to(device)

        ## Transient points
        transient_idxs = th.randint(0, sampling_size['boundary'], (sampling_size['transient'],)) # Shuffle run to select from
        tran_time_sample = time_sample_batch[transient_idxs] * (th.rand((sampling_size['transient'], 1))).to(device)+fn.EPSILON
        tran_time_sample.requires_grad_()
        tran_observed_sample = term_observed_sample[transient_idxs]
        tran_reference_sample = term_reference_sample if term_reference_sample is None else term_reference_sample[transient_idxs]

        ## dldu
        dldu_init:LagrangianFunc = getattr(fn.dldu,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = init_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = init_time_sample.detach()-fn.EPSILON)

        dldu_term:LagrangianFunc = getattr(fn.dldu,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = term_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = term_time_sample.detach()-fn.EPSILON)

        dldu_tran:LagrangianFunc = getattr(fn.dldu,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = tran_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = tran_time_sample.detach()-fn.EPSILON)
        
        ## dldx
        dldx_init:LagrangianFunc = getattr(fn.dldx,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = init_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = init_time_sample.detach()-fn.EPSILON)

        dldx_term:LagrangianFunc = getattr(fn.dldx,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = term_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = term_time_sample.detach()-fn.EPSILON)

        dldx_tran:LagrangianFunc = getattr(fn.dldx,config['lagrangian']['name'])(
                                    **config['lagrangian']['args'],
                                    reference_state = tran_reference_sample,
                                    reference_mask = distribution.reference_mask,
                                    time = tran_time_sample.detach()-fn.EPSILON)

        """ ---------------- Training ---------------- """

        # try:
        # Infer control, state, costate

        first_state_init, first_control_init, first_costate_init = controller(init_time_sample, init_observed_sample, init_reference_sample, return_costate=True)

        first_state_term, first_control_term, first_costate_term = controller(term_time_sample, term_observed_sample, term_reference_sample, return_costate=True)

        first_state_tran, first_control_tran, first_costate_tran = controller(tran_time_sample, tran_observed_sample, tran_reference_sample, return_costate=True)

        losses = {}
        if l_mult["dynamics"] > 0.0:
            losses["dynamics"] = hn.dynamics_loss(init_time_sample, first_state_init, first_control_init, dynamics)*bd_scale
            losses["dynamics"] = losses["dynamics"]+hn.dynamics_loss(term_time_sample, first_state_term, first_control_term, dynamics)*bd_scale
            losses["dynamics"] = losses["dynamics"]+hn.dynamics_loss(tran_time_sample, first_state_tran, first_control_tran, dynamics)*trans_scale
            losses["dynamics"] = losses["dynamics"]*l_mult["dynamics"]
        else:
            losses["dynamics"] = th.tensor(0.0, device=device)

        if l_mult["initial-boundary"] > 0.0:
            losses["initial-boundary"] = hn.boundary_loss(first_state_init, init_observed_sample)
            losses["initial-boundary"] = losses["initial-boundary"]*l_mult["initial-boundary"]
        else:
            losses["initial-boundary"] = th.tensor(0.0, device=device)

        if l_mult["terminal-boundary"] > 0.0:
            losses["terminal-boundary"] = hn.boundary_reference_loss(first_state_term, term_reference_sample, distribution.reference_mask)
            losses["terminal-boundary"] = losses["terminal-boundary"]*l_mult["terminal-boundary"]
        else:
            losses["terminal-boundary"] = th.tensor(0.0, device=device)

        if l_mult["costate"] > 0.0:
            losses["costate"] = hn.costate_loss(init_time_sample, first_state_init, first_control_init, first_costate_init, dldx_init, dynamics)*bd_scale
            losses["costate"] = losses["costate"] + hn.costate_loss(term_time_sample, first_state_term, first_control_term, first_costate_term, dldx_term, dynamics)*bd_scale
            losses["costate"] = losses["costate"] + hn.costate_loss(tran_time_sample, first_state_tran, first_control_tran, first_costate_tran, dldx_tran, dynamics)*trans_scale
            losses["costate"] = losses["costate"]*l_mult["costate"]
        else:
            losses["costate"] = th.tensor(0.0, device=device)

        if l_mult["hamiltonian"] > 0.0:
            losses["hamiltonian"] = hn.hamiltonian_loss(init_time_sample, first_state_init, first_control_init, first_costate_init, dldu_init, dynamics)
            losses["hamiltonian"] = losses["hamiltonian"] + hn.hamiltonian_loss(term_time_sample, first_state_term, first_control_term, first_costate_term, dldu_term, dynamics)*bd_scale
            losses["hamiltonian"] = losses["hamiltonian"] + hn.hamiltonian_loss(tran_time_sample, first_state_tran, first_control_tran, first_costate_tran, dldu_tran, dynamics)*trans_scale
            losses["hamiltonian"] = losses["hamiltonian"]*l_mult["hamiltonian"]
        else:
            losses["hamiltonian"] = th.tensor(0.0, device=device)

        if l_mult["costate-terminal"] > 0.0:
            losses["costate-terminal"] = hn.costate_terminal_loss(first_costate_term, distribution.reference_mask)
            losses["costate-terminal"] *= l_mult["costate-terminal"]
        else:
            losses["costate-terminal"] = th.tensor(0.0, device=device)

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

        logger.update_ref_log(all_loss_log_ref, to_dict(all_loss_log))

        ## Save best seen model so far (not exactly the best since it is after update but close)
        if net_item < best_log['loss']:
            best_log = {
                "loss": net_item,
                "idx": epoch,
                "state": controller.state_dict()
            }

            logger.update_ref_log(best_log_ref, best_log)

        """ ---------------- Print results ---------------- """
        cycle = epoch%progress_rate + 1
        if cycle == progress_rate or epoch == n-1 or epoch == 0:

            update_running_loss(running_loss, running_loss_log, cycle, reset=cycle==progress_rate)
            logger.update_ref_log(running_loss_log_ref, to_dict(running_loss_log))

            logger.print_progress(epoch, loss = {
                                                "net": running_loss_log["net"][-1],
                                                "init-bd": running_loss_log["initial-boundary"][-1],
                                                "term-bd": running_loss_log["terminal-boundary"][-1],
                                                "hmn": running_loss_log["hamiltonian"][-1],
                                                "cos": running_loss_log["costate"][-1],
                                                "cs-bd": running_loss_log["costate-terminal"][-1]
                                                })


    logger.print('Finished Training')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Hion Paper - Automated Training',
                    description='Trains a controller for the systems presented in the paper given a configuration file',
                    epilog='Thank you for using my model. Feel free to star the project on Github (https://github.com/wzjoriv/Hion) and support its development')
    
    parser.add_argument('-c', '--config', type=str, default='config.linear.json', help="path to system configuration file")
    parser.add_argument('-f', '--finetune', type=str, default='', help="path to checkpoint for finetuning")
    parser.add_argument('-s', '--seed', type=int, default=-1, help="to manually set a fixed seed")
    
    args = parser.parse_args()

    if args.seed >= 0:
        th.manual_seed(args.seed)

    checkpoint = None if not len(args.finetune) else th.load(args.finetune, weights_only=False)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    with open(os.path.join('configs', args.config)) as f:
        config = json.load(f)

    train(config=config, checkpoint=checkpoint, device=device)
