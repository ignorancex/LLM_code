from pathlib import Path
from ControllableNesymres.utils import return_fitfunc
from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id, load_eq
import numpy as np
import pandas as pd
from ControllableNesymres.architectures.data import compute_properties, create_negatives,\
                                                    prepare_negative_pool, sympify_equation,\
                                                    return_costants, description2tokens, prepare_pointers, create_positives_and_constants, \
                                                    tokenize, is_token_constant, get_robust_random_data, return_support_limits,sample_support,sample_images, \
                                                    remove_rationals, return_all_positive_substrees, replace_constants_with_symbol
import base64
from ControllableNesymres.dataset.generator import Generator, UnknownSymPyOperator
from ControllableNesymres.dataset.data_utils import sample_symbolic_constants
from ControllableNesymres.dclasses import FitParams, BFGSParams
from ControllableNesymres.architectures.model import Model
from functools import partial
import sympy
import torch
from torch.distributions.uniform import Uniform
import random
import time
import hydra
import os
import shutil
import warnings
import pickle
import json
import math
import omegaconf
import matplotlib.pyplot as plt
import copy
from skopt import gp_minimize
from skopt.space import Real, Integer


from tpsr.dyna_gym.agents.uct import UCT
from tpsr.dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from tpsr.rl_env import RLEnv
from tpsr.default_pi import NesymresHeuristic
from tpsr.reward import compute_reward_nesymres




MAX_ATTEMPTS = 5












def tpsr(metadata,cfg,X,y,model_path,conditioning,cond_str_tokens):
    ## Set up BFGS load rom the hydra config yaml
    cfg.tpsr_params.debug = True
    cfg.tpsr_params.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = Model.load_from_checkpoint(model_path, cfg=cfg)
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()

    ## Load architecture, set into eval mode, and pass the config parameters

    samples = {'x_to_fit':0, 'y_to_fit':0}
    samples['x_to_fit'] = [X]
    samples['y_to_fit'] = [y]

    ### MCTS 
    rl_env = RLEnv(
        samples = samples,
        model = model,
        cfg_params=params_fit,
        cfg = cfg,
        cond = conditioning)


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
        alg='uct',
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

    return pred_str, dp.sample_times

















def return_data_and_model_path(cfg):
    dataset_dict = {
        "train_nc" : Path(hydra.utils.to_absolute_path("test_datasets/train_nc")),
        "train_wc" : Path(hydra.utils.to_absolute_path("test_datasets/train_wc")),
        "ofv_nc" : Path(hydra.utils.to_absolute_path("test_datasets/only_five_variables_nc")),
        "ofv_wc" : Path(hydra.utils.to_absolute_path("test_datasets/only_five_variables_wc")),
        "aif" : Path(hydra.utils.to_absolute_path("test_datasets/aifeymann_processed")),
        "not_included" : Path(hydra.utils.to_absolute_path("test_datasets/original_test_sets/not_included.json")),
        "baseline" : Path(hydra.utils.to_absolute_path("test_datasets/original_test_sets/baseline.json")),
    }
    
    model_dict = {
        "nsrwh" : Path(hydra.utils.to_absolute_path("ControllableNeuralSymbolicRegressionWeights/nsrwh_200000000_epoch=149.ckpt")),
        "nsr" : Path(hydra.utils.to_absolute_path("ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt")),
        "nopow_finetuned" : Path(hydra.utils.to_absolute_path("weights/SmallNSRwH_100000_log_-epoch=249.ckpt")),
        "nopow_original" : Path(hydra.utils.to_absolute_path("weights/SmallNSR_00000_log_-epoch=999.ckpt")),
        "nopow_prepend_positives" : Path(hydra.utils.to_absolute_path("weights/SmallNSRgvs_100000_log_-epoch=999.ckpt")),
        "nsr_original" : Path(hydra.utils.to_absolute_path("weights/NormalNSR_10000000_log_-epoch=999.ckpt")),
        "prepend_positives": Path(hydra.utils.to_absolute_path("weights/NormalNSRgvs_10000000_log_-epoch=999.ckpt")),
    }

    data_path = dataset_dict[cfg.testing.test_set]
    model_path = model_dict[cfg.testing.model]

    return data_path, model_path






def change_config(cfg):
    if not ("nsrwh" in cfg.testing.model or cfg.testing.model == "nopow_finetuned"):
        cfg.architecture.conditioning = False
        cfg.dataset.conditioning.mode = "None"
    if cfg.testing.test_set == "aif" and cfg.testing.right > 91:
        cfg.testing.right = 91  
    if "prepend" in cfg.testing.model:
        cfg.architecture.length_eq = 200 
        cfg.architecture.number_possible_tokens = 200 
        cfg.dataset.conditioning.mode = "all"
        cfg.prepend_conditioning_during_inference = True

    cfg.dataset.conditioning.positive.prob = 1
    cfg.dataset.conditioning.positive.min_percent = 0.5
    cfg.dataset.conditioning.positive.max_percent = 0.5
    cfg.dataset.conditioning.positive.prob_pointers = 0.8
    cfg.dataset.conditioning.negative.prob = 1
    cfg.dataset.conditioning.negative.min_percent = 0.5
    cfg.dataset.conditioning.negative.max_percent = 0.5
    cfg.dataset.conditioning.negative.k = 500

    cfg.inference.bfgs.activated = True
    cfg.inference.bfgs.n_restarts = 10
    cfg.inference.n_jobs = -1
    cfg.inference.beam_size = cfg.testing.beam_size




def prepare_negative_pool_for_nopow(cfg):
    with open(hydra.utils.to_absolute_path(cfg.path_to_candidate_nopow)) as f:
        eq_candidates = json.load(f)

    eq_candidates = [eq.replace(" ", "") for eq in eq_candidates]
    eqs_candidate = sorted(eq_candidates, key=len, reverse=False)
    return eqs_candidate




def return_prior_knowledge(eq_sympy_infix_with_constants, correct_properties, metadata, cfg):
    eq_sympy_prefix_with_constants = Generator.sympy_to_prefix(eq_sympy_infix_with_constants, enable_float=True)
    costants, eq_sympy_prefix_with_c= return_costants(eq_sympy_prefix_with_constants)
    
    given_appearing_branches, _, pointer_to_cost = create_positives_and_constants(eq_sympy_prefix_with_constants, metadata, cfg)
    if "nopow" in cfg.testing.model:
        negative_pool = prepare_negative_pool_for_nopow(cfg)
    else:
        negative_pool =  prepare_negative_pool(cfg)
    negative_candidates = create_negatives(eq_sympy_prefix_with_c, negative_pool, all_positives_examples=correct_properties["all_positives_examples"], metadata=metadata, cfg=cfg)
    good_negative_candidates = []
    for candidate in negative_candidates:
        try:
            tokenize(candidate, metadata.word2id)
        except:
            continue
        good_negative_candidates.append(candidate)
    given_absent_branches = good_negative_candidates

    description = {"positive_prefix_examples": [], "negative_prefix_examples": []}

    if cfg.testing.experiment_mode == "complexity" or cfg.testing.experiment_mode == "all":
        given_complexity = int(correct_properties["complexity"].split("=")[1])
        if (given_complexity <= 0):
            print("complexity below 0")
            return None
        description["complexity"] = correct_properties["complexity"]
        print("correct complexity: ", correct_properties["complexity"].split("=")[1])
        print("given complexity: ", correct_properties["complexity"].split("=")[1])

    if cfg.testing.experiment_mode == "symmetry" or cfg.testing.experiment_mode == "all":
        try:
            description["symmetry"] = correct_properties["symmetry"]
            print("correct symmetry: ", correct_properties["symmetry"])
        except:
            return None

    if cfg.testing.experiment_mode == "positive" or cfg.testing.experiment_mode == "all" or cfg.testing.experiment_mode == "positive_without_constants":
        print("correct appearing branches: ", correct_properties["all_positives_examples"])
        if cfg.testing.experiment_mode == "positive_without_constants":
            description["positive_prefix_examples"] = []
            for positive in given_appearing_branches:
                for i,word in enumerate(positive):
                    if len(word) > 7:
                        positive[i] = 'c'
                description["positive_prefix_examples"].append(positive)
        else:
            description["positive_prefix_examples"] = given_appearing_branches
        print("given appearing branches: ", description["positive_prefix_examples"])
        if cfg.testing.experiment_mode == "positive_without_constants":
            description["cost_to_pointer"] = {}
        else:
            description["cost_to_pointer"] = pointer_to_cost
        print("constant_to_pointer", description["cost_to_pointer"])
        
    
    if cfg.testing.experiment_mode == "negative" or cfg.testing.experiment_mode == "all":
        description["negative_prefix_examples"] = given_absent_branches
        print("given absent branches", given_absent_branches)

    return description, costants






def return_candidates_from_negative_pool(variables, metadata, cfg):
    negative_pool = prepare_negative_pool(cfg)
    if "nopow" in cfg.testing.test_set:
        negative_pool = prepare_negative_pool_for_nopow(cfg)
    sampled_negative_equations = random.choices(list(negative_pool), k=20)
    disjoint_variables =  set(metadata.config["variables"]) - set(variables)
    
    candidates = []
    for entry in sampled_negative_equations:
        _, dummy_consts = sample_symbolic_constants(entry, cfg=None)
        eq_string = str((entry.format(**dummy_consts)))
        try:
            eq_sympy_infix_without_constants = sympify_equation(eq_string)
            eq_sympy_infix_without_constants = remove_rationals(eq_sympy_infix_without_constants)
            eq_sympy_infix = replace_constants_with_symbol(eq_sympy_infix_without_constants)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        except:
            continue
            
        all_subtrees = return_all_positive_substrees(eq_sympy_prefix, metadata=metadata, ignore_obvious=True, remove_gt=True)
        for cond in all_subtrees:
            if all(x not in disjoint_variables for x in cond) and len(cond) <= cfg.testing.max_branch_length:
                candidates.append(tuple(cond))
    
    big_set = [list(x) for x in set(candidates)]
    weights = [(1/len(x))**cfg.testing.random_sampling_param for x in big_set] # We want to sample more the shorter ones (if sampling_type > 0)
    random_candidates = random.choices(big_set, weights=weights, k=10)
    return random_candidates




    




def return_next_positives(sorted_positives, variables, t, metadata, cfg):
    if t == 0:
        return []

    max_length = math.ceil(cfg.testing.max_length_l0 + cfg.testing.max_length_alpha*t)
    max_length = min(20, max_length)
    max_length = max(1, max_length)
    total_length = random.randint(0,max_length)
    random_candidates = return_candidates_from_negative_pool(variables, metadata, cfg)
    
    candidates = []
    for i,random_candidate in enumerate(random_candidates):
        if i < cfg.testing.num_random_candidates:
            candidates.append(random_candidate)
        
    for i,positive in enumerate(sorted_positives):
        if positive[1][0] > cfg.testing.R2_border and i < cfg.testing.max_positive_candidates:
            candidates.append(list(positive[0]))
    

    print("Prompts Should Be At Least This Length: ",total_length)

    total_length_count = 0
    counter = 0
    chosen_candidates = []
    while (total_length_count < total_length and counter < 50):
        #print(counter)
        counter += 1
        chosen_candidate = random.choices(candidates, k=1)[0]
        if chosen_candidate in chosen_candidates:
            continue
        total_length_count += len(chosen_candidate)
        chosen_candidates.append(chosen_candidate)

    return chosen_candidates








def x_is_subset_of_y(x,y):
    dummy = copy.deepcopy(y)
    try:
        for elem in x:
            dummy.remove(elem)
    except ValueError:
        return False
    return True





def return_all_positives(eq_string, metadata, cfg):
    try:
        result_properties = compute_properties(eq_string, compute_symmetry=False, metadata=metadata, cfg=cfg, is_streamlit=False)
        all_positives = []
        positives = result_properties["all_positives_examples"]
        for positive in positives:
            for i,word in enumerate(positive):
                if len(word) > 7:
                    positive[i] = 'c'
            all_positives.append(tuple(positive))

        return [list(x) for x in set(all_positives)]
    except:
        return []



def is_next_positives_good(next_positives, history, metadata, cfg):
    flag = 0
    for prediction, _, given_positives in history:
        if x_is_subset_of_y(given_positives, next_positives) and x_is_subset_of_y(next_positives, given_positives):
            return False
        if x_is_subset_of_y(given_positives, next_positives):   # if you give [] and get sin x_1, then giving [sin] is meaningless
            try:
                all_positives = return_all_positives(prediction, metadata, cfg)
                if x_is_subset_of_y(next_positives, given_positives + all_positives):
                    return False 
            except:
                pass
    return True

        
        
def update_positives(positives, prediction, R2, metadata, cfg):
    all_positives = return_all_positives(prediction, metadata, cfg)

    for positive in all_positives:
        if len(positive) > cfg.testing.max_branch_length:
            continue
        if tuple(positive) in positives:
            old_R2 = positives[tuple(positive)][0]
            old_num = positives[tuple(positive)][1]
            new_R2 = (old_R2*old_num + max(R2,0))/(old_num + 1)
            positives[tuple(positive)] = (new_R2, old_num + 1)
        else:
            positives[tuple(positive)] = (max(R2,0), 1)
    return positives








def add_positives_list(positives, description, costants, metadata, cfg):
    new_description = copy.deepcopy(description)
    if cfg.testing.num_loops > 1:
        new_description["positive_prefix_examples"] = new_description["positive_prefix_examples"] + positives

    try:
        cond_tokens, cond_str_tokens = description2tokens(new_description, metadata.word2id , cfg)
    except:
        print("could not convert description to tokens")
        return None
    cond_tokens = torch.tensor(cond_tokens).long()
    numberical_conditioning = costants
    conditioning = {"symbolic_conditioning": cond_tokens, "numerical_conditioning": torch.tensor(numberical_conditioning,device="cpu").float()}

    return (conditioning, cond_str_tokens), new_description["positive_prefix_examples"]







def return_f_pred(prediction, variables):
    if prediction == "illegal parsing infix" or prediction == "illegal bfgs" or prediction == "illegal pointers" or prediction == "illegal parse" or prediction == None:
        return None
    pred_sympy_infix_with_constants = sympify_equation(prediction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f_pred = sympy.lambdify(variables, pred_sympy_infix_with_constants)
        except:
            return None
    return f_pred



def plot_prediction(support_limits, history, f_pred, data_points, cfg):
    low = support_limits[0].low.item()
    high = support_limits[0].high.item()

    support_tensor = torch.arange(low, high, (high - low)/5000)
    support = {"x_1": support_tensor}
    variables = ['x_1']

    for i in range(5):
        is_valid_pred, data_points_pred = sample_images(f_pred, support, variables, cfg)
        if is_valid_pred:
            break
        if i == 4 and not is_valid_pred:
            return
    
    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(10,7.5))
    data_points = data_points.squeeze(0)
    data_points_pred = data_points_pred.squeeze(0)
    
    ax.scatter(data_points_pred[0], data_points_pred[-1], color = 'tab:red', s=5)
    ax.scatter(data_points[0], data_points[-1], color = 'black')
    if history is None:
        fig.savefig(Path(hydra.utils.to_absolute_path("images/plot.png")))
    else:
        fig.savefig(Path(hydra.utils.to_absolute_path(f"images/plot{str(len(history))}.png")))



# R2 values may change slightly for same expressions, but that is the intended behaviour since support points are chosen randomly every time.
def return_R2(f, f_pred, variables, support_limits, metadata, cfg):
    if f_pred is None:
        return None
    cnt = 0
    while cnt < MAX_ATTEMPTS:
        new_support = sample_support(support_limits, variables, cfg.dataset.max_number_of_points*5,  metadata.total_variables, cfg)
        is_valid_pred, data_points_pred = sample_images(f_pred, new_support, variables, cfg) 
        if is_valid_pred:
            break
        cnt += 1
    if not is_valid_pred:
        return None
    
    new_support = {}
    for j in range(data_points_pred.shape[1] -  1):
        new_support[f"x_{j + 1}"] = data_points_pred[0, j, :]
    is_valid_test, data_points_test = sample_images(f, new_support, variables, cfg)
    if not is_valid_test:
        return None
    
    new_support = {}
    for j in range(data_points_test.shape[1] -  1):
        new_support[f"x_{j + 1}"] = data_points_test[0, j, :]
    is_valid_pred, data_points_pred = sample_images(f_pred, new_support, variables, cfg) 
    if is_valid_pred and data_points_pred.shape[2] >= cfg.testing.number_of_points:
        data_points_test = data_points_test.squeeze(0)
        data_points_pred = data_points_pred.squeeze(0)
        data_points_pred = data_points_pred[:, :cfg.testing.number_of_points]
        data_points_test = data_points_test[:, :cfg.testing.number_of_points]
        y_pred = data_points_pred[-1,:]
        y_test = data_points_test[-1,:]
        mean_y = torch.mean(y_test)
        numerator = torch.sum((y_test - y_pred)**2)
        denominator = torch.sum((y_test - mean_y)**2)
        r2 = 1 - (numerator / denominator)
        R2 = r2.item()
        return R2
    else:
        return None




def return_is_correct(f_pred, variables, X, y_original, cfg):
    repeat_number = cfg.dataset.max_number_of_points*5//cfg.testing.number_of_points
    X_support = {}
    for j in range(X.shape[1]):
        X_support[f"x_{j + 1}"] = X[:,j].repeat(repeat_number)
    is_valid_pred, data_points_pred = sample_images(f_pred, X_support, variables, cfg)

    if not is_valid_pred:
        return None
    
    y_pred = data_points_pred[0,-1,:]
    if len(y_pred)//repeat_number != len(y_original):
        return None
    
    if np.sum(np.isclose(y_pred[:cfg.testing.number_of_points],y_original))*100 >= len(y_original)*99:
        return True
    else:
        return False
    





def are_conditionings_satisfied(result_properties, correct_properties, given_appearing_branches, given_absent_branches):
    are_conditionings_satisfied_list = [0,0,0,0]
    try:
        if ("complexity" in result_properties):
            print("result complexity: ",result_properties["complexity"].split("=")[1])
            if correct_properties["complexity"] == result_properties["complexity"]:
                are_conditionings_satisfied_list[0] = 1
        else:
            print("complexity not in result_properties")
    except:
        pass

    try:
        if ("symmetry" in result_properties):
            print("results symmetry: ",result_properties["symmetry"])
            flag = 0
            for x in correct_properties["symmetry"]:
                if x not in result_properties["symmetry"]:
                    flag += 1
            if flag == 0:
                are_conditionings_satisfied_list[1] = 1
        else:
            print("symmetry not in result_properties")
    except:
        pass

    try:
        if ("all_positives_examples" in result_properties):
            result_properties_convert = result_properties["all_positives_examples"]
            given_properties_convert = given_appearing_branches
            for x,branch in enumerate(result_properties_convert):
                for y,token in enumerate(branch):
                    if len(token) > 7:
                        result_properties_convert[x][y] = 'c'
            for x,branch in enumerate(given_properties_convert):
                for y,token in enumerate(branch):
                    if len(token) > 7:
                        given_properties_convert[x][y] = 'c'
            print("results_positives: ",result_properties_convert)
            flag = 0
            for x in given_properties_convert:
                if x not in result_properties_convert:
                    flag += 1
            if flag == 0:
                are_conditionings_satisfied_list[2] = 1
        else:
            print("all_positive_examples not in result_properties")
    except:
        pass
    
    try:
        if ("all_positives_examples" in result_properties):
            flag = 0
            for x in given_absent_branches:
                if x in result_properties["all_positives_examples"]:
                    flag += 1
            if flag == 0:
                are_conditionings_satisfied_list[3] = 1
        else:
            print("all_positive_examples not in result_properties")
    except:
        pass

    return are_conditionings_satisfied_list





def experiment(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    change_config(cfg)
    #initialization#############################################################
    
    data_path, model_path = return_data_and_model_path(cfg)
    metadata = load_metadata_hdf5(Path(hydra.utils.to_absolute_path(cfg.train_path)))    
    metadata = retrofit_word2id(metadata, cfg)

    if cfg.testing.test_set in ["train_nc", "train_wc", "ofv_nc", "ofv_wc", "aif"]:
        metadata_dataset = load_metadata_hdf5(data_path)
        eqs_per_hdf = metadata_dataset.eqs_per_hdf
    else:
        with open(data_path, "r") as f:
            dataset_list = json.load(f)
            cfg.testing.right = min(cfg.testing.right, len(dataset_list) - 1)
    
    
    torch.manual_seed(cfg.testing.seed)
    np.random.seed(cfg.testing.seed)
    random.seed(cfg.testing.seed)


    result_list = []

    start = time.time()
    

    #fitting each equation######################################################

    print("begin fitting")
    
    for i in range(cfg.testing.left,cfg.testing.right + 1):
        result_dict = {
            "index" : i,
            "equation" : None,
            #"support_limits" : None,
            "given_conditionings" : None,
            "prediction" : None,
            "R2" : None,
            "is_correct" : None,
            "is_conditioning_satisfied": None,
            #"generated_symmetry" : None,
            #"generated_positives" : None,
            #"generated_negatives" : None,
            #"result_properties" : None,
            #"correct_properties" : None,
            #"positive_list_list" : None,
            "new_predictions_list" : None,
            "stored_predictions_list" : None
        }

        cfg.result_options.save_name = f"{cfg.testing.test_set}_{i}"

        if cfg.testing.test_set in ["train_nc", "train_wc", "ofv_nc", "ofv_wc", "aif"]:
            eq = load_eq(data_path, i, eqs_per_hdf)
            consts, _ = sample_symbolic_constants(eq, cfg.dataset.constants)
            eq_string = eq.expr.format(**consts)
        else:
            eq_string = dataset_list[i]["eq_string"]
        now = time.time()
        print()
        print("time = ",now - start,end = ",  ")
        print(i, " th equation")
        print("expression to predict: ",eq_string)
        eq_sympy_infix_with_constants = sympify_equation(eq_string)

        #generate points############################################################

        tmp = list(eq_sympy_infix_with_constants.free_symbols)
        variables = sorted([str(x) for x in tmp])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = sympy.lambdify(variables, eq_sympy_infix_with_constants)

        cnt = 0
        while cnt < MAX_ATTEMPTS:
            if cfg.testing.test_set == "aif" and cfg.dataset.fun_support.min_len == 1:
                support_limits = return_support_limits(cfg, metadata, support=eq.support)
            elif not cfg.testing.test_set in ["train_nc", "train_wc", "ofv_nc", "ofv_wc", "aif"] and "low" in dataset_list[i] and "high" in dataset_list[i]:
                support_limits = []
                for j in range(cfg.architecture.dim_input - 1):
                    support_limits.append(Uniform(dataset_list[i]["low"], dataset_list[i]["high"]))
            else:
                support_limits = return_support_limits(cfg, metadata, support=None)
            support = sample_support(support_limits, variables, cfg.dataset.max_number_of_points*5,  metadata.total_variables, cfg)
            is_valid, data_points = sample_images(f, support, variables, cfg)
            if is_valid:
                break
            cnt += 1
        if not is_valid:
            print("could not generate valid data points")
            continue

        torch.manual_seed(2*cfg.testing.seed)
        data_points = data_points[:, :, torch.randperm(data_points.shape[2])]
        data_points = data_points[:, :, :cfg.testing.number_of_points]
        X = data_points[0,:-1,:].T
        y = data_points[0,-1,:]
        y_original = y


        for j, value in enumerate(y):
            y[j] = value*(1 + np.random.normal(0,abs(value),1)*cfg.testing.noise_applied)

        

        #generate prior knowledge#####################################################

        correct_properties = compute_properties(eq_string, compute_symmetry=True, metadata=metadata, cfg=cfg, is_streamlit=False)
        if correct_properties is None:
            continue

        prior_knowledge = return_prior_knowledge(eq_sympy_infix_with_constants, correct_properties, metadata, cfg)
        if prior_knowledge is None:
            continue
        description, costants = prior_knowledge

        #fitting####################################################################    

        fitfunc = return_fitfunc(cfg, metadata, model_path, device="cpu")

        
        #positives = {("sin","x_1"):1, ("cos",):1}
        positives = {}
        history = []
        counter = 0
        max_R2 = -100000
        best_output = None
        best_output_t = 0
        print_partial_expression_pool_flag = True
        sample_times = 0

        while len(history) < cfg.testing.num_loops and counter < 500:
            counter += 1
            
            sorted_positives = sorted(positives.items(), key=lambda item: item[1][0], reverse=True)
            if print_partial_expression_pool_flag:
                print()
                print("Partial Expression Pool-------------------------------------------------------------------")
                print()
                for positive in sorted_positives:
                    print(positive)
                print()
                print("------------------------------------------------------------------------------------------")
            print()

            next_positives = return_next_positives(sorted_positives, variables, len(history), metadata, cfg)
            print("Consider Prompting the Model with :", next_positives)

            add_positives_list_result = add_positives_list(next_positives, description, costants, metadata, cfg)
            if add_positives_list_result is None:
                continue
            cond, next_positives = add_positives_list_result
            conditioning, cond_str_tokens = cond

            if is_next_positives_good(next_positives, history, metadata, cfg) == False:
                print("The Prompts Were not Good, Trying Again")
                print_partial_expression_pool_flag = False
                continue
            print_partial_expression_pool_flag = True
            print()
            print("Now Prompting the Model with: ", next_positives)
            print()
            
            if cfg.testing.tpsr == False:
                with torch.no_grad():
                    new_output = fitfunc(X, y, conditioning, cond_str_tokens, is_batch=False)
                    sample_times += cfg.testing.beam_size
            else:
                new_output = {}
                print("conditioning",conditioning)
                print("cond_str_tokens",cond_str_tokens)
                new_output["best_pred"], sample_time = tpsr(metadata,cfg,X,y,model_path,conditioning,cond_str_tokens)
                sample_times += sample_time*cfg.tpsr_params.num_beams

            print("prediction: ", new_output["best_pred"])
        
            f_pred = return_f_pred(new_output["best_pred"], variables)

            R2 = return_R2(f, f_pred, variables, support_limits, metadata, cfg)
            if R2 is None:
                print("R2: Could not compute")
            else:
                print("R2: ", R2)
                if R2 > max_R2:
                    max_R2 = R2
                    best_output = new_output
                    best_output_t = len(history)
                if cfg.result_options.plot == True and not f_pred is None and len(variables) == 1:
                    plot_prediction(support_limits, history, f_pred, data_points, cfg)
                print()
                print("Moving on to Next Iteration (Iteration No.", len(history)+1, ")")
                print()


            if R2 == 1.0:
                break

            history.append((new_output["best_pred"], R2, next_positives))
            if R2 != None:
                positives = update_positives(positives, new_output["best_pred"], R2, metadata, cfg)

            

        print()
        print("History of Generated Expressions<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print()
        for prediction, R2, _ in history:
            print(R2, prediction)
        print()
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print()






        #compute correctness########################################################

        try:
            best_prediction = best_output["best_pred"]
        except:
            best_prediction = None

        print("correct answer:", eq_string)
        print("best_prediction:", best_prediction)
        f_pred = return_f_pred(best_prediction, variables)

        if cfg.result_options.plot == True and not f_pred is None and len(variables) == 1:
            plot_prediction(support_limits, None, f_pred, data_points, cfg)

        R2 = return_R2(f, f_pred, variables, support_limits, metadata, cfg)
        if R2 is None:
            print("R2: Could not compute")
        else:
            print("R2: ", R2)
        is_correct = return_is_correct(f_pred, variables, X, y_original, cfg)

        print("is_correct: ",is_correct)





        #check if conditionings are satisfied#######################################

        try:
            result_properties = compute_properties(best_prediction, compute_symmetry=True, metadata=metadata, cfg=cfg, is_streamlit=False)
        except:
            result_properties = []


        are_conditionings_satisfied_list = are_conditionings_satisfied(result_properties, correct_properties, description["positive_prefix_examples"], description["negative_prefix_examples"])

        history_to_save = []
        for prediction, r2, _ in history:
            history_to_save.append((prediction,r2))


        now = time.time()

        result_dict = {
            "index" : i,
            "equation" : eq_string,
            #"support_limits" : [[support_limit.low.item(), support_limit.high.item()] for support_limit in support_limits],
            "given_conditionings" : description,
            "prediction" : best_prediction,
            "R2" : R2,
            "is_correct" : is_correct,
            "is_conditioning_satisfied": are_conditionings_satisfied_list,
            "best_output_t" : best_output_t,
            "time" : now - start,
            "sample_times" : sample_times
            #"generated_symmetry" : generated_symmetry,
            #"generated_positives" : generated_positives,
            #"generated_negatives" : generated_negatives,
            #"result_properties" : result_properties,
            #"correct_properties" : correct_properties,
            #"positive_list_list" : positives_list_list,
            #"history" : history_to_save
            
        }

        result_list.append(result_dict)

        
        
        
    if cfg.result_options.save_results:
        print("-----------------SAVING RESULTS------------------")
        try:
            if cfg.testing.noise_applied == 0.0 and cfg.testing.save_file_name == "":
                with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops.json")), "r") as f:
                    results = json.load(f)
            elif cfg.testing.save_file_name != "":
                with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops_{cfg.testing.save_file_name}.json")), "r") as f:
                    results = json.load(f)
            else:
                with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops_{cfg.testing.noise_applied}noise.json")), "r") as f:
                    results = json.load(f)
        except FileNotFoundError:
            try:
                os.makedirs(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}")))
            except FileExistsError:
                pass
            results = []

        for result_dict in result_list:
            new_result_dict = result_dict
            settings_dict = dict(cfg.testing)
            for key in settings_dict:
                new_result_dict[key] = settings_dict[key]
            results.append(new_result_dict)

        if cfg.testing.noise_applied == 0.0 and cfg.testing.save_file_name == "":
            with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops.json")), "w") as f:
                json.dump(results, f, indent=4)
        elif cfg.testing.save_file_name != "":
            with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops_{cfg.testing.save_file_name}.json")), "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.experiment_mode}_{cfg.testing.num_loops}loops_{cfg.testing.noise_applied}noise.json")), "w") as f:
                json.dump(results, f, indent=4)

    now = time.time()
    print()
    print()
    print()
    print("total time = ",now - start)


    return result_list


        



def process_R2_to_obtain_score(R2):
    if R2 <= 0:
        return 0
    elif R2 >= 0.99999:
        return 5
    else:
        return -math.log10(1 - R2)



def get_R2_sum(hyperparams,cfg):
    random_sampling_param, max_positive_candidates, R2_border, num_random_candidates,  max_length_l0, max_length_alpha, max_branch_length = hyperparams
    cfg.testing.random_sampling_param = random_sampling_param
    cfg.testing.max_positive_candidates = int(max_positive_candidates)
    cfg.testing.R2_border = R2_border
    cfg.testing.num_random_candidates = int(num_random_candidates)
    cfg.testing.max_length_l0 = max_length_l0
    cfg.testing.max_length_alpha = max_length_alpha
    cfg.testing.max_branch_length = int(max_branch_length)

    result_list = experiment(cfg)
    R2_sum = 0

    for result_dict in result_list:
        try:
            with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.left}to{cfg.testing.right}_hyperparamtuning.json")), "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            try:
                os.makedirs(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}")))
            except FileExistsError:
                pass
            results = []

        results_new = []
        results_new.append(result_dict["index"])
        results_new.append(float(random_sampling_param))
        results_new.append(int(max_positive_candidates))
        results_new.append(float(R2_border))
        results_new.append(float(num_random_candidates))
        results_new.append(float(max_length_l0))
        results_new.append(float(max_length_alpha))
        results_new.append(int(max_branch_length))
        results_new.append(result_dict["R2"])
        results_new.append(result_dict["equation"])
        results_new.append(result_dict["prediction"])
        results_new.append(result_dict["best_output_t"])
        results.append(results_new)

        R2_sum += process_R2_to_obtain_score(result_dict["R2"])

        with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.left}to{cfg.testing.right}_hyperparamtuning.json")), "w") as f:
            json.dump(results, f, indent=4)


    
    return -R2_sum











@hydra.main(config_name="config")
def main(cfg):
    if cfg.testing.hyperparam_tuning == False:
        experiment(cfg)
    else: # optimize hyperparams with bayesean optimization
        pass
        space = [
            Real(0, 4, name='random_sampling_param'),
            Integer(5, 50, name='max_positive_candidates'),
            Real(0, 0.99, name='R2_border'),
            Integer(1, 10, name='num_random_candidates'),
            Real(1, 20, name='max_length_l0'),
            Real(0, 1, name='max_length_alpha'),
            Integer(3, 10, name='max_branch_length'),
        ]

        # Run Bayesian Optimization using gp_minimize with the Expected Improvement acquisition function
        result = gp_minimize(
            func=partial(get_R2_sum,cfg=cfg),   
            dimensions=space,     
            acq_func="EI",  
            n_calls=cfg.testing.hyperparam_tuning_n_calls, 
            random_state=cfg.testing.seed
        )

        try:
            with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.left}to{cfg.testing.right}_hyperparamtuning.json")), "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            try:
                os.makedirs(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}")))
            except FileExistsError:
                pass
            results = []

        for i,x_iter in enumerate(result.x_iters):
            print(x_iter, -result.func_vals[i])
            results.append((str(x_iter), float(-result.func_vals[i])))
        results.append(("result.fun",float(result.fun)))
        results.append(("results.x",str(result.x)))
        print(result.fun)
        print(result.x)

        with open(Path(hydra.utils.to_absolute_path(f"experiments/results/{cfg.testing.model}/{cfg.testing.test_set}_{cfg.testing.left}to{cfg.testing.right}_hyperparamtuning.json")), "w") as f:
            json.dump(results, f, indent=4)

    



if __name__ == "__main__":
    main()
