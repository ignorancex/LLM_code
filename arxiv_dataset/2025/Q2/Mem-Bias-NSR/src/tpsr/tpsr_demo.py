import time
import json
import os

import torch
import numpy as np
import sympy as sp

from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from rl_env import RLEnv
from default_pi import NesymresHeuristic


from ControllableNesymres.architectures.model import Model
from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id
from ControllableNesymres.dclasses import FitParams, BFGSParams
from functools import partial
from sympy import lambdify
from reward import compute_reward_nesymres
import omegaconf
import hydra
from pathlib import Path





def main_nesymres(metadata,cfg,samples,X,y):
    ## Set up BFGS load rom the hydra config yaml
    cfg.inference.bfgs.activated = True
    cfg.inference.bfgs.n_restarts = 10
    cfg.inference.n_jobs = -1
    cfg.inference.beam_size = cfg.testing.beam_size

    bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

    params_fit = FitParams(word2id=metadata.word2id, 
                                id2word={int(k): v for k,v in metadata.id2word.items()}, 
                                una_ops=metadata.una_ops, 
                                bin_ops=metadata.bin_ops, 
                                total_variables=list(metadata.total_variables),  
                                total_coefficients=list(metadata.total_coefficients),
                                rewrite_functions=list(metadata.rewrite_functions),
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

    weights_path = Path(hydra.utils.to_absolute_path("ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt"))

    ## Load architecture, set into eval mode, and pass the config parameters
    model = Model.load_from_checkpoint(weights_path, cfg=cfg)
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()

    fitfunc = partial(model.fitfunc,cfg_params=params_fit)

    cond = {"symbolic_conditioning": torch.tensor([],device="cpu").float(), "numerical_conditioning": torch.tensor([],device="cpu").float()}
    #output_ref = fitfunc(X,y,cond) 

    #print(output_ref["best_pred"])

    ### MCTS 
    rl_env = RLEnv(
        samples = samples,
        model = model,
        cfg_params=params_fit)


    ## Get self.encoded in the model to use for Sequence generation from given states
    model.to_encode(X,y, params_fit)

    dp = NesymresHeuristic(
        rl_env=rl_env,
        model=model,
        k=cfg.tpsr_params.width,
        num_beams=cfg.tpsr_params.num_beams,
        horizon=cfg.tpsr_params.horizon,
        device=cfg.tpsr_params.device,
        use_seq_cache=not cfg.tpsr_params.no_seq_cache,
        use_prefix_cache=not cfg.tpsr_params.no_prefix_cache,
        length_penalty = cfg.tpsr_params.beam_length_penalty,
        cfg_params = params_fit,
        train_value_mode=cfg.tpsr_params.train_value,
        debug=cfg.tpsr_params.debug)

    # for fair comparison, loading models and tokenizers are not included in computation time
    start = time.time()

    agent = UCT(
        action_space=[],
        gamma=1., 
        ucb_constant=1.,
        horizon=cfg.tpsr_params.horizon,
        rollouts=cfg.tpsr_params.rollout,
        dp=dp,
        width=cfg.tpsr_params.width,
        reuse_tree=True
    )

    agent.display()

    if cfg.tpsr_params.sample_only:
        horizon = 1
    else:
        horizon = 200

    done = False
    s = rl_env.state
    for t in range(horizon):
        if len(s) >= cfg.tpsr_params.horizon:
            print(f'Cannot process programs longer than {cfg.tpsr_params.horizon}. Stop here.')
            break

        if done:
            break

        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)

        if cfg.tpsr_params.debug:
            # print the current tree
            print('tree:')
            print_tree(agent.root, params_fit.id2word)

            print('took action:')
            print(repr(params_fit.id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)

    time_elapsed = time.time() - start
    #print("NeSymReS Equation Skeleton: ", output_ref)
    print("time elapsed: ", time_elapsed)
    print("samples times: ", dp.sample_times)
    print("generated ids: ", s)

    loss_bfgs_mcts , reward_mcts , pred_str = compute_reward_nesymres(model.X, model.y, s, params_fit)

    print("TPSR+NeSymReS Equation: ", pred_str)
    print("TPSR+NeSymReS Loss: ", loss_bfgs_mcts)
    print("TPSR+NeSymReS Reward: ", reward_mcts)




@hydra.main(config_path="../../scripts",config_name="config")
def main(cfg):
    np.random.seed(cfg.testing.seed)
    torch.manual_seed(cfg.testing.seed)
    torch.cuda.manual_seed(cfg.testing.seed)
    cfg.tpsr_params.debug = True
    cfg.tpsr_params.device = "cuda" if torch.cuda.is_available() else "cpu"

    
    metadata = load_metadata_hdf5(Path(hydra.utils.to_absolute_path(cfg.train_path)))    
    #metadata = retrofit_word2id(metadata, cfg)

    
    #Example of Equation-Data:
    number_of_points = 500
    n_variables = 2
    max_supp = cfg.dataset.fun_support["max"] 
    min_supp = cfg.dataset.fun_support["min"]
    X = torch.rand(number_of_points,len(list(metadata.total_variables)))*(max_supp-min_supp)+min_supp
    X[:,n_variables:] = 0
    target_eq = "(sin(x_1) + 3.36)/x_1**2" 
    X_dict = {x:X[:,idx].cpu() for idx, x in enumerate(metadata.total_variables)} 
    y = lambdify(",".join(metadata.total_variables), target_eq)(**X_dict)
    samples = {'x_to_fit':0, 'y_to_fit':0}
    samples['x_to_fit'] = [X]
    samples['y_to_fit'] = [y]
    
    #Main
    main_nesymres(metadata,cfg,samples,X,y)
    

if __name__ == '__main__':
    main()
