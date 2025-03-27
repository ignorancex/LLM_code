import torch
import torch.nn as nn
import operator as op
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from rl_models.enc_dec import further_sel_mask_key, pred_probs_key, pred_Q_key, pred_v_key, prev_prog_key, col_id_key, select_num_feat_key, col_Q_key, col_probs_key, op_key, op_id_key, col_key, outbound_key, min_Q_val, mask_atom_representation_for_op0, mask_atom_representation1 #, down_weight_removed_feats
from rl_models.enc_dec import forward_main0_opt, create_deep_set_net_for_programs, TokenNetwork3, atom_to_vector_ls0_main
# from mortalty_prediction.full_experiments.trainer import process_curr_atoms0
from rl_models.rl_algorithm import process_curr_atoms0

from rl_models.rl_algorithm import ReplayMemory
from tabular_data_utils.tabular_dataset import tabular_Dataset
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from collections import namedtuple, deque
from utils_treatment import * #evaluate_explanation_dff, evaluate_treatment_effect_core, transform_outcome_by_rescale_back, split_treatment_control_gt_outcome, obtain_predictions2, obtain_individual_predictions2, transform_treatment_ids, perturb_samples
from treatment_prediction.baseline_methods.TransTEE.TransTEE import TransTEE, MonoTransTEE, make_transtee_backbone
from treatment_prediction.baseline_methods.TransTEE.DRNet import Drnet, Vcnet, make_drnet_backbone
from treatment_prediction.baseline_methods.TargetReg import DisCri
import torch.nn.functional as F
from treatment_prediction.baseline_methods.TransTEE.utils_TransTEE.trans_ci import Embeddings
from treatment_prediction.baseline_methods.TransTEE.utils_TransTEE.transformers import TransformerEncoder, TransformerEncoderLayer
from treatment_prediction.baseline_methods.baseline import *
Transition = namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))
import pandas as pd
import math
from scipy.integrate import romb
import random
from sklearn.linear_model import LinearRegression
from argparse import Namespace
from torch_geometric.data import Batch
from torch_geometric.data import Data
from scipy.optimize import minimize
from scipy.integrate import romb
from sklearn.tree import DecisionTreeRegressor

from treatment_prediction.tabular.evaluations.treatment_dose_evaluations import get_patient_outcome, get_true_dose_response_curve


import pickle
import copy
from posthoc_explanations.lime import Lime
from baseline_methods.dragonnet import *
from baseline_methods.self_interpretable_models.attention import AttentionExplanationModel
from baseline_methods.self_interpretable_models.ENRL import ENRL

from baseline_methods.anchor_reg.anchor_reg import anchor_tabular
from baseline_methods.lore_explainer_reg.lorem import LOREM_reg
from baseline_methods.lore_explainer_reg.util import record2str, neuclidean

from baseline_methods.self_interpretable_models.prototype import ProtoVAE_tab
from tvae.tvae import DisentangledVAE

min_Q_val = -1.01

col_key = "col"

col_id_key = "col_id"

col_Q_key = "col_Q"

pred_Q_key = "pred_Q"

op_Q_key = "op_Q"

treatment_Q_key = "treatment_Q"

treatment_prob_key = "treatment_probs"

col_probs_key = "col_probs"

pred_probs_key = "pred_probs"

op_probs_key = "op_probs"

pred_v_key = "pred_v"

op_id_key = "op_id"

treatment_id_key = "treatment_id"
        
op_key = "op"

topk=3
def compute_eval_metrics(meta_info, test_dataset, num_treatments, device, do_prediction, train=False):
        mises = []
        ites = []
        dosage_policy_errors = []
        policy_errors = []
        pred_best = []
        pred_vals = []
        true_best = []

        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        # for patient in test_patients:
        # for step, batch in enumerate(test_loader):
        with torch.no_grad():
            for idx, origin_X, A, Y, count_Y, D, patient, all_other_pats_ls in tqdm(test_dataset):
                if train and len(pred_best) > 10:
                    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)
                for treatment_idx in range(num_treatments):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    X = torch.from_numpy(test_data['x']).float()
                    X_pd_full = X
                    origin_X = X
                    X = X.to(device)
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                    A = torch.from_numpy(test_data["t"]).float().to(device)
                    test_data['d'] = treatment_strengths
                    D = torch.from_numpy(test_data["d"]).float().to(device)

                    origin_all_other_pats_ls= [all_other_pats_ls.clone() for _ in range(num_integration_samples)]
                    pred_dose_response = do_prediction(X, A, D, X_pd_full, origin_all_other_pats_ls, origin_X).numpy()
                    # pred_dose_response = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                    # pred_dose_response = pred_dose_response * (
                    #         dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                    #                         dataset['metadata']['y_min']

                    true_outcomes = [get_patient_outcome(patient, meta_info, treatment_idx, d) for d in
                                        treatment_strengths]
                    
                    # if len(pred_best) < num_treatments and train == False:
                    #     #print(true_outcomes)
                    #     print([item[0] for item in pred_dose_response])
                    mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
                    inter_r = np.array(true_outcomes) - pred_dose_response.squeeze()
                    ite = np.mean(inter_r ** 2)
                    mises.append(mise)
                    ites.append(ite)

                    best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

                    def pred_dose_response_curve(dosage):
                        test_data = dict()
                        test_data['x'] = np.expand_dims(patient, axis=0)
                        test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                        test_data['d'] = np.expand_dims(dosage, axis=0)
                        X = torch.from_numpy(test_data['x']).float()
                        X_pd_full = X
                        origin_X = X
                        X = X.to(device)
                        A = torch.from_numpy(test_data["t"]).float().to(device)
                        D = torch.from_numpy(test_data["d"]).float().to(device)
                        
                        ret_val =do_prediction(X, A, D, X_pd_full, [all_other_pats_ls], origin_X).numpy()
                        # ret_val = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                        # ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                        #             dataset['metadata']['y_min']
                        return ret_val

                    true_dose_response_curve = get_true_dose_response_curve(meta_info, patient, treatment_idx)

                    min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                            x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

                    max_pred_opt_y = - min_pred_opt.fun
                    max_pred_dosage = min_pred_opt.x
                    max_pred_y = true_dose_response_curve(max_pred_dosage)

                    min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                            x0=[0.5], method="SLSQP", bounds=[(0, 1)])
                    max_true_y = - min_true_opt.fun
                    max_true_dosage = min_true_opt.x

                    dosage_policy_error = (max_true_y - max_pred_y) ** 2
                    dosage_policy_errors.append(dosage_policy_error)

                    pred_best.append(max_pred_opt_y)
                    pred_vals.append(max_pred_y)
                    true_best.append(max_true_y)
                    

                selected_t_pred = np.argmax(pred_vals[-num_treatments:])
                selected_val = pred_best[-num_treatments:][selected_t_pred]
                selected_t_optimal = np.argmax(true_best[-num_treatments:])
                optimal_val = true_best[-num_treatments:][selected_t_optimal]
                policy_error = (optimal_val - selected_val) ** 2
                policy_errors.append(policy_error)

        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)

def retrieve_gt_explanations(data_folder, dataset_id):
    sub_data_folder = os.path.join(data_folder, "ihdp", str(dataset_id))
    meta_data_file = os.path.join(sub_data_folder, "test_synthetic_meta_info")
    with open(meta_data_file, "rb") as f:
        meta_data_mappings = pickle.load(f)
    sample_id_feat_id_mappings = {sample_id: meta_data_mappings[sample_id]["feat_ids"] for sample_id in range(len(meta_data_mappings))}
    sample_id_feat_name_mappings = {sample_id: meta_data_mappings[sample_id]["feat_names"] for sample_id in range(len(meta_data_mappings))}
    rand_coeff_mappings = {sample_id: meta_data_mappings[sample_id]["coeff"] for sample_id in range(len(meta_data_mappings))}
    return sample_id_feat_id_mappings, sample_id_feat_name_mappings, rand_coeff_mappings


def get_treatment_graphs(treatment_ids: list, id_to_graph_dict: dict):
    return [id_to_graph_dict[i] for i in treatment_ids]

def create_pt_geometric_dataset(
    units: np.ndarray, treatment_graphs: list, outcomes=None
):
    unit_tensor = torch.FloatTensor(units)
    data_list = []
    is_multi_relational = "edge_types" in treatment_graphs[0]
    for i in range(len(treatment_graphs)):
        c_size, features, edge_index, one_hot_encoding = (
            treatment_graphs[i]["c_size"],
            treatment_graphs[i]["node_features"],
            treatment_graphs[i]["edges"],
            treatment_graphs[i]["one_hot_encoding"],
        )
        one_hot_encoding = torch.FloatTensor(one_hot_encoding)
        edge_index = torch.LongTensor(edge_index)
        x = torch.Tensor(np.array(features))
        if len(edge_index.shape) == 2:
            edge_index = edge_index.transpose(1, 0)
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            covariates=torch.unsqueeze(unit_tensor[i], 0),
            one_hot_encoding=one_hot_encoding,
        )
        if outcomes is not None:
            graph_data.y = torch.Tensor([outcomes[i]])
        if is_multi_relational:
            graph_data.edge_types = torch.LongTensor(
                [treatment_graphs[i]["edge_types"]]
            ).squeeze()
        graph_data.__setitem__("c_size", torch.IntTensor([c_size]))
        data_list.append(graph_data)
    return data_list


def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=50):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_subset_modules_by_prefix(state_dict, prefix):
    return {k.split(prefix+".")[1]:v for k,v in state_dict.items() if k.startswith(prefix)}

class dragonet_model_rl(torch.nn.Module):
    def init_without_feat_groups(self, lang,  program_max_len, hidden_size, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=1, continue_act=False):
        self.topk_act=topk_act
        self.lang = lang
        self.program_max_len=program_max_len
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        self.continue_act = continue_act        

        self.feat_range_mappings = feat_range_mappings
        for i,v in self.lang.syntax.items():
            if i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))
            else:
                if i in self.lang.syntax["cat_feat"]:
                    self.grammar_num_to_token_val[i] = []
                    self.grammar_token_val_to_num[i] = []
                else:
                    self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                    self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
                
        self.op_start_pos = 0
        self.num_start_pos = 2
        self.cat_start_pos = 2 + len(self.lang.syntax["num_feat"])

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                for option in list(options_dict.keys()):        
                    # if self.op_start_pos < 0:
                    #     self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:                    
                if decision in self.lang.syntax["num_feat"]:
                    # if self.num_start_pos < 0:
                    #     self.num_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
                else:
                    # if self.cat_start_pos < 0:
                    #     self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        self.num_feat_len  = num_feat_count#len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        self.cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = self.num_feat_len+self.cat_feat_len
        self.all_input_feat_len = latent_size #self.num_feat_len+category_sum_count
        self.num_feats = num_features
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        # full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        full_input_size = self.all_input_feat_len + hidden_size# self.ATOM_VEC_LENGTH
        self.full_input_size = full_input_size
        self.feat_group_names = None
        self.feat_bound_point_ls = None
        # self.removed_feat_ls = []
        self.prefer_smaller_range = False
        self.do_medical = False
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
        # self.feat_to_num_mappings = dict()
        # self.op_to_num_mappings = dict()
        # # feat_idx = 0
        # # for col in self.lang.syntax["num_feat"]:
        # for feat_idx in range(len(self.column_ls)):
        #     self.feat_to_num_mappings[self.column_ls[feat_idx]] = feat_idx
        #     # feat_idx += 1
        # op_idx = 0
        
        # # self.op_list = list(self.lang.syntax["num_op"].keys())
        # self.op_list=[operator.__le__, operator.__ge__]
        
        # for op_idx in range(len(self.op_list)):
        #     self.op_to_num_mappings[self.op_list[op_idx]] = op_idx
        #     # op_idx += 1
        self.op_num = len(self.lang.syntax["num_op"])

        for i,v in self.lang.syntax.items():
            if i == "num_op":
                continue
            # if i in self.lang.syntax["num_feat"]:
            #     continue
            
            # if not i == "num_feat":
            #     # net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        # self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.num_feats)
            # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.num_feats)
            # self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        self.to(self.device)
    
    def __init__(self, args, model_configs, backbone_model_config, lang,  program_max_len, hidden_size, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = 1, continue_act=False, discretize_feat_value_count=20, use_mlm=True,removed_feat_ls= None, num_treatments=2, cont_treatment = False):
        super(dragonet_model_rl, self).__init__()
        input_size = num_feat_count + category_sum_count
        if args.backbone is not None and not args.backbone == "none":
            if args.backbone.lower() == "transtee":
                print("start loading transtee backbone")
                if not args.cont_treatment and args.has_dose:
                    cov_dim = backbone_model_config["cov_dim"]
                else:
                    cov_dim = input_size
                
                params = {'num_features': input_size, 'num_treatments': args.num_treatments,
                'h_dim': hidden_size, 'cov_dim':cov_dim}
                self.model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                self.shared_hidden_dim = backbone_model_config["hidden_size"]
                self.backbone_full_model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                self.backbone_model = self.backbone_full_model.encoding_features

            elif args.backbone.lower() == "drnet":
                print("start loading drnet backbone")
                cfg_density = [(input_size, 100, 1, 'relu'), (100, hidden_size, 1, 'relu')]
                num_grid = 10
                cfg = [[hidden_size, hidden_size, 1, 'relu'], [hidden_size, 1, 1, 'id']]
                isenhance = 11
                self.backbone_full_model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=backbone_model_config["h"], num_t=args.num_treatments, has_dose=args.has_dose, cont_treatment=args.cont_treatment)
                self.backbone_model = self.backbone_full_model.encoding_features
                
                # cfg_density = [(num_feat_count + category_sum_count, 100, 1, 'relu'), (100, hidden_size, 1, 'relu')]
                # self.backbone_model =  nn.Sequential(*make_drnet_backbone(cfg_density))
            # self.backbone_model_name = "transtee"
            else:
                self.backbone_full_model = dragonet_model(input_size, backbone_model_config["hidden_size"])
                self.backbone_model = self.backbone_full_model.encoding_features
                # self.backbone_model = make_dragonet_backbone2(num_feat_count + category_sum_count, hidden_size)#
            
            if args.cached_backbone is not None and os.path.exists(args.cached_backbone):
                cached_backbone_state_dict = torch.load(args.cached_backbone)
                self.backbone_full_model.load_state_dict(cached_backbone_state_dict)
                if args.fix_backbone:
                    for p in self.backbone_full_model.parameters():
                        p.requires_grad = False
        else:
            self.backbone_full_model = None
            self.backbone_model = None
            hidden_size = input_size
        self.backbone = args.backbone
        self.fix_backbone = args.fix_backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discretize_feat_value_count = discretize_feat_value_count
        self.id_attr, self.outcome_attr, self.treatment_attr = id_attr, outcome_attr, treatment_attr
        self.use_mlm = use_mlm
        self.removed_feat_ls = removed_feat_ls
        if self.removed_feat_ls is None:
            self.removed_feat_ls = []
        self.num_treatments = num_treatments
        self.cont_treatment = cont_treatment
        
        
        # params = {'num_features': num_feat_count + category_sum_count, 'num_treatments': args.num_treatments,
        #     'h_dim': model_configs["hidden_size"], 'cov_dim':model_configs["cov_dim"]}
        if args.method_two:
            args_copy = copy.deepcopy(args)
            args_copy.method = "TransTEE"
            # cohort_model_configs = {"hidden_size": int(model_configs["hidden_size"]), "cov_dim": 100}
            _, cohort_model_configs = load_configs(args_copy,root_dir=args.root_dir)
            # args_copy.epochs = 20
            self.cohort_trainer = baseline_trainer(args_copy, lang.dataset, num_feat_count + category_sum_count, cohort_model_configs, self.device)
            self.cohort_trainer.model.load_state_dict(torch.load(args.cohort_model_path))
            self.cohort_model = self.cohort_trainer.model
            for p in self.cohort_model.parameters():
                p.requires_grad = False
        self.init_without_feat_groups(lang,  program_max_len, hidden_size, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=topk_act, continue_act=continue_act)
        
        self = self.to(self.device)
        self.first_prog_embed = torch.tensor([0]*self.ATOM_VEC_LENGTH, device=self.device, dtype=torch.float)#torch.randn(self.policy_net.ATOM_VEC_LENGTH, requires_grad=True)
        

    def forward_ls0(self, features,X_pd_full, program, outbound_mask_ls, atom, epsilon=0, eval=False, init=False, train=False):
        # features,_,_ = features
        # features = features.to(self.device)
        pat_count = features.shape[0]
        
        
        total_feat_prog_array_len = self.full_input_size#program[0].shape[-1]

        # selecting feature
        
        concat_program_tensor = torch.cat(program,dim=-1)

        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([features.shape[0], total_feat_prog_array_len])
            if torch.cuda.is_available():
                hx = hx.cuda()
            hx[:,0:features[0].shape[0]] = features
            hx[:, features[0].shape[0]:] = self.program_net(torch.stack(program, dim=1).to(self.device)).squeeze(1)
            # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=self.device)
            hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,features[0].shape[0]:] = self.program_net(torch.cat(program, dim=1).to(self.device))
            # hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        
        return forward_main0_opt(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=init, train=train)

    def predict_treatment(self, X_embedding, atom_ls, eval=False, treatment_topk=5):
        treatment_pred_logits=self.treatment_pred_net(X_embedding)
        selected_treatment_ids = torch.topk(treatment_pred_logits, treatment_topk, dim=-1)[1]
        treatment_probs = torch.sigmoid(treatment_pred_logits)
        treatment_pred_Q_val = torch.tanh(treatment_pred_logits)
        if not eval:
            atom_ls[treatment_Q_key] = treatment_pred_Q_val.detach().cpu()
            atom_ls[treatment_id_key] = selected_treatment_ids.detach().cpu()
            atom_ls[treatment_prob_key] = treatment_probs.detach().cpu()
        else:
            atom_ls[treatment_Q_key] = treatment_pred_Q_val
            atom_ls[treatment_id_key] = selected_treatment_ids
            atom_ls[treatment_prob_key] = treatment_probs
        return atom_ls

    def forward_single_step(self, trainer, X_embedding, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom, epsilon=0, eval=False, init=False, train=False, all_treatment_ids=None, treatment_graph_sim_mat=None, compute_ate=False, treatment_topk=10, tr_str_two=False, method_two=False, test=False, method_three=False):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = [self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)]

            atom_ls = self.forward_ls0(X_embedding,X_pd_full, init_program, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, train=train)
        else:
            atom_ls = self.forward_ls0(X_embedding,X_pd_full, program, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, train=train)
        
        sim_treatment_ids = None
        sim_treatment_probs = None

        if not eval and program_str is not None:
        
            next_program, next_program_str, next_all_other_pats_ls, next_program_col_ls, next_outbound_mask_ls = process_curr_atoms0(trainer, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=[col_id_key, further_sel_mask_key])
            
            # next_all_other_pats_ls, transformed_expr_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
            

            treatment_pred, outcome_pred, _ = obtain_predictions2(self, X_pd_full, A, D, trainer.lang, next_all_other_pats_ls, A, D, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, all_treatment_ids=all_treatment_ids, sim_treatment_ids=sim_treatment_ids, sim_treatment_probs=sim_treatment_probs, tr_str_two=tr_str_two, treatment_graph_sim_mat=treatment_graph_sim_mat, method_two=method_two, method_three=method_three)
            
            if compute_ate:
                pos_outcome, neg_outcome, non_empty = obtain_predictions2(self, X_pd_full, A, D, trainer.lang, next_all_other_pats_ls, A, D, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, all_treatment_ids=all_treatment_ids, sim_treatment_ids=sim_treatment_ids, compute_ate=compute_ate, sim_treatment_probs=sim_treatment_probs, tr_str_two=tr_str_two, treatment_graph_sim_mat=treatment_graph_sim_mat, method_two=method_two, method_three=method_three)
            
            ind_treatment_pred, ind_outcome_pred = obtain_individual_predictions2(self, X_pd_full, A, D, trainer.lang, next_all_other_pats_ls, A, D, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, all_treatment_ids=all_treatment_ids, sim_treatment_ids=sim_treatment_ids, sim_treatment_probs=sim_treatment_probs, tr_str_two=tr_str_two, treatment_graph_sim_mat=treatment_graph_sim_mat, method_two=method_two, method_three=method_three)
            
            if test:
                torch.set_grad_enabled(False)
            
            if compute_ate:
                return treatment_pred, (outcome_pred, pos_outcome, neg_outcome, non_empty), next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
            else:
                return treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
        else:
            return atom_ls
   
    def forward(self, trainer, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=False, train=False, all_treatment_ids=None, treatment_graph_sim_mat=None, compute_ate=False, graph_data_batch=None, X_encode=None, tr_str_two=False, method_two=False, test=False, method_three=False, return_encoding=False):
        if X_encode is None:
            if self.backbone_model is None:
                X_encode = X
            else:
                if graph_data_batch is None:
                    X_encode = self.backbone_model(X)
                else:
                    if self.backbone_model_name == "transtee":
                        X_encode = self.backbone_model(graph_data_batch)
                    else:
                        X_encode = self.backbone_model(X)
            
        if len(X_encode.shape) == 3:
            X_encode_input = torch.mean(X_encode, 1)
        else:
            X_encode_input = X_encode

        pred = self.forward_single_step(trainer, X_encode_input, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, all_treatment_ids=all_treatment_ids, treatment_graph_sim_mat=treatment_graph_sim_mat, compute_ate=compute_ate, tr_str_two=tr_str_two, method_two=method_two, test=test, method_three=method_three)
        if not return_encoding:
            return pred
        else:
            return pred, X_encode

    
    def atom_to_vector_ls0(self, atom_ls):
        return atom_to_vector_ls0_main(self, atom_ls)
    

class DQN_all_tabular:
    def __init__(self, id_attr, outcome_attr, treatment_attr, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0, numeric_count=None, category_count=None, category_sum_count = None, has_embeddings=False, use_mlm=True, topk_act=1, model_config=None, backbone_model_config = None, feat_group_names = None, removed_feat_ls=None, prefer_smaller_range = False, prefer_smaller_range_coeff=0.5, method_two=False, args = None, discretize_feat_value_count=20):
        self.mem_sample_size = mem_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.topk_act = topk_act
        torch.manual_seed(seed) 
        self.use_mlm = use_mlm           
        if "hidden_size" not in model_config:
            model_config["hidden_size"] = 0
        if "latent_size" not in model_config:
            model_config["latent_size"] = 0
        self.policy_net = dragonet_model_rl(args, model_config, backbone_model_config, lang,  program_max_len, model_config["hidden_size"], model_config["latent_size"], dropout_p, numeric_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = topk_act, continue_act=False, discretize_feat_value_count=discretize_feat_value_count, use_mlm=use_mlm, removed_feat_ls= removed_feat_ls, num_treatments=args.num_treatments, cont_treatment = args.cont_treatment)
        # RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)
        
        self.target_net = dragonet_model_rl(args, model_config, backbone_model_config, lang,  program_max_len, model_config["hidden_size"], model_config["latent_size"], dropout_p, numeric_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = topk_act, continue_act=False, discretize_feat_value_count=discretize_feat_value_count, use_mlm=use_mlm, removed_feat_ls= removed_feat_ls, num_treatments=args.num_treatments, cont_treatment = args.cont_treatment)
        # self.target_net = RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        # for p in self.policy_net.backbone_model.distilbert.parameters():
        #     p.requires_grad = False

        self.memory = ReplayMemory(replay_memory_capacity)
        self.outcome_regression = True
        self.regression_ratio = args.regression_ratio
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), learning_rate)
        self.tr_str_two = args.tr_str_two
        self.first_prog_embed = torch.tensor([0]*self.policy_net.ATOM_VEC_LENGTH, device=self.device, dtype=torch.float)#torch.randn(self.policy_net.ATOM_VEC_LENGTH, requires_grad=True)

    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.policy_net.atom_to_vector(atom)

    def atom_to_vector_ls(self, atom:dict):
        return self.policy_net.atom_to_vector_ls(atom)

    def atom_to_vector_ls0(self, atom):
        return self.policy_net.atom_to_vector_ls0(atom)

    def vector_ls_to_str_ls0(self, atom):
        return self.policy_net.vector_ls_to_str0(atom)

    def vector_to_atom(self, vec):
        return self.policy_net.vector_to_atom(vec)

    #turns network Grammar Networks predictions and turns them into an atom
    def prediction_to_atom(self, pred:dict):
        return self.policy_net.prediction_to_atom(pred)

    def random_atom(self, program):
        #TODO
        if len(program) == 0:
            pred = self.policy_net.random_atom(program = [torch.tensor([0]*self.policy_net.ATOM_VEC_LENGTH, device=self.device, dtype=torch.float)])
        else:
            pred = self.policy_net.random_atom(program = program)
        return self.policy_net.prediction_to_atom(pred)

    def predict_atom(self, features, X_pd, program, epsilon):
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], ["formula"], epsilon)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, ["formula"], epsilon)
        return self.policy_net.prediction_to_atom(pred), pred
    
    def predict_atom_ls(self, features, X_pd_ls, program, outbound_mask_ls, epsilon):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)
            pred = self.policy_net.forward_ls0(features, X_pd_ls, [init_program], outbound_mask_ls, ["formula"], epsilon, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features, X_pd_ls, program, outbound_mask_ls, ["formula"], epsilon)
        # return self.policy_net.prediction_to_atom_ls(pred), pred
        return pred
    
    def predict_next_state_with_tensor_info(self, features, data, program):
        if len(program) == 0:
            pred = self.target_net(features, data, [self.first_prog_embed], ["formula"], 0)
        else:
            #program.sort()
            pred = self.target_net(features, data, program, ["formula"], 0)
        max_tensors = dict()
        for token, token_val in pred.items():
            if not self.policy_net.get_prefix(token) in self.lang.syntax["num_feat"]:
                max_tensors[token] = torch.max(token_val).reshape((1,1))
            else:
                if token.endswith("_ub"):
                    # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
                    max_tensors[self.policy_net.get_prefix(token)] = torch.max(token_val[1]).reshape(1,1)
        
        # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        
        return self.target_net.prediction_to_atom(pred), max_tensors
    
    def predict_next_state_with_tensor_info_ls(self, features, data, program):
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data),1)
            pred = self.target_net.forward_ls(features, data, [init_program], ["formula"], 0, replay=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls(features, data, program, ["formula"], 0, replay=True)
        max_tensors = dict()
        for token, token_val in pred.items():
            # if not token in self.lang.syntax["num_feat"]:
            if not type(token) is tuple:
                max_tensors[token] = torch.max(token_val, dim=1)[0].reshape((len(data),1))
            else:
                if not "pred_score" in max_tensors:
                    max_tensors["pred_score"] = [torch.zeros(len(data), device = self.device), torch.zeros(len(data), device = self.device)]
                pred_val = pred[token]
                for token_key in token:
                    
                    # token_key = token_key[0]
                    probs = pred_val[1][token_key]
                    # ub_probs = pred_val[1][token_key][1]
                    sample_ids = token_val[2][token_key].view(-1)
                    sample_cln_id_ls = token_val[3][token_key]
                    val = probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[0].view(-1)]
                    if token_key.endswith("_lb"):
                        max_tensors["pred_score"][0][sample_ids] = val
                    elif token_key.endswith("_ub"):
                        max_tensors["pred_score"][1][sample_ids] = val
                    del val
                    # val = ub_probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[1].view(-1)]      
                    # max_tensors[token][1][sample_ids] = val
                    # del val
                # print()
                # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
        
        # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        return_pred = self.target_net.prediction_to_atom_ls(pred)
        del pred
        return return_pred, max_tensors
    
    
    def predict_next_state_with_tensor_info_ls0(self, features, data, state):
        program, outbound_mask_ls = state
        graph_data = None
        X = features[0].to(self.device)
        # else:
        #     X, graph_data = features
        #     X = X.to(self.device)
        #     graph_data = graph_data.clone().to(self.device)
        
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            # pred = self.target_net.forward(features, data, [init_program], outbound_mask_ls, ["formula"], 0, eval=False, init=True)
            
            pred = self.target_net.forward(self.trainer, X, None, None,  data, [init_program], None, None, None, None, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=True, graph_data_batch=graph_data)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward(self.trainer, X, None, None, data, program, None, None, None, None, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=False, graph_data_batch=graph_data)
            # pred = self.target_net.forward(features, data, program, outbound_mask_ls, ["formula"], 0, eval=False)
            
        max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)
        selected_num_feat_tensor_bool = pred[select_num_feat_key]
        if op_Q_key in pred:

            max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors + max_op_tensors

            max_tensors = max_tensors/3
            
            #     # col_treatment_tensor_indices = pred[treatment_id_key]
            #     # treatment_tensors = torch.mean(pred[treatment_Q_key][col_treatment_tensor_indices], dim=-1)
            #     if not self.tr_str_two:
            #         treatment_tensors = self.obtain_average_treatment_Q_values(pred[treatment_Q_key], pred[treatment_id_key])
            #     else:
            #         treatment_tensors = self.obtain_average_treatment_Q_values2(pred[treatment_Q_key], pred[treatment_id_key])
            #     max_tensors = (max_tensors*3 + treatment_tensors)/4
        else:
            # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors

            max_tensors = max_tensors/2
            
            #     # col_treatment_tensor_indices = pred[treatment_id_key]
            #     # treatment_tensors = torch.mean(pred[treatment_Q_key][col_treatment_tensor_indices], dim=-1)
            #     if not self.tr_str_two:
            #         treatment_tensors = self.obtain_average_treatment_Q_values(pred[treatment_Q_key], pred[treatment_id_key])
            #     else:
            #         treatment_tensors = self.obtain_average_treatment_Q_values2(pred[treatment_Q_key], pred[treatment_id_key])
            #     max_tensors = (max_tensors*2 + treatment_tensors)/3
            
        max_tensors = max_tensors*selected_num_feat_tensor_bool + max_col_tensors*(1-selected_num_feat_tensor_bool)


        return max_tensors.to(self.device)
    
    # def predict_next_state_with_tensor_info_ls0_medical(self, features, data, state):
    #     program = state
        
    #     if len(state) == 0:
    #         init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
    #         pred = self.target_net.forward_ls0(features, data, [init_program], ["formula"], 0, eval=False, init=True)
    #         del init_program
    #     else:
    #         #program.sort()
    #         pred = self.target_net.forward_ls0(features, data, program, ["formula"], 0, eval=False)
            
    #     # max_tensors,_ = pred[pred_Q_key].max(dim=-1)

    #     # max_tensors = torch.mean(max_tensors, dim=-1)

    #     max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

    #     return max_col_tensors.to(self.device)

    def get_state_action_prediction_tensors(self, features, X_pd, program, atom_ls):
        atom, origin_atom = atom_ls
        queue = list(atom.keys())
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], queue, 0, eval=True, existing_atom=origin_atom)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, queue, 0, eval=True, existing_atom=origin_atom)

        tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        for token, token_val in atom.items():
            if token == "num_op" or token.endswith("_prob"):
                continue

            if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
                # if not token.endswith("_prob"):
                    tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
            else:
                # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
                tensor_indeces[token] = torch.argmax(atom[token][1]).item()
            # else:
            #     tensor_indeces[token] = 0
        atom_prediction_tensors = {}
        for token, tensor_idx in tensor_indeces.items():
            if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
                atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
            else:
                if token.endswith("_ub"):
                    atom_prediction_tensors[self.policy_net.get_prefix(token)] = pred[token][1][tensor_idx].view(-1)
                # atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
        # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        return atom_prediction_tensors

    def obtain_average_treatment_Q_values(self, all_treatment_Q, treatment_ids):
        selected_treatment_Q_mean = 0
        for k in range(treatment_ids.shape[1]):
            selected_treatment_Q_mean += all_treatment_Q[torch.arange(len(all_treatment_Q)),treatment_ids[:,k]]

        selected_treatment_Q_mean = selected_treatment_Q_mean/treatment_ids.shape[1]
        return selected_treatment_Q_mean.view(-1,1)
    
    def obtain_average_treatment_Q_values2(self, all_treatment_Q, treatment_ids):
        selected_treatment_Q_mean = all_treatment_Q.mean(dim=-1)
        return selected_treatment_Q_mean.view(-1,1)

    def get_state_action_prediction_tensors_ls0(self, features, X_pd, state, atom):
        # atom = atom_pair[0]
        # origin_atom = atom_pair[1]
        
        program, outbound_mask_ls = state
        
        graph_data = None
        
        X = features[0].to(self.device)
        A = features[1].to(self.device)
        Y = features[2].to(self.device)
        D = features[3]
        if D is not None:
            D = D.to(self.device)
        # else:
        #     X, graph_data = features
        #     X = X.to(self.device)
        #     graph_data = graph_data.clone().to(self.device)
        
        # if atom[col_id_key].max() == 116:
        #     print()
        
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
            # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            # pred = self.policy_net.forward(features,X_pd, [init_program], outbound_mask_ls, atom, 0, eval=True, init=True)
            pred, X_encoding = self.policy_net.forward(self.trainer, X, None, None, X_pd, [init_program], None, None, None, None, outbound_mask_ls, atom=atom, epsilon=0, eval=True, init=True, graph_data_batch=graph_data, return_encoding=True)
            del init_program
        else:
            #program.sort()
            pred, X_encoding = self.policy_net.forward(self.trainer, X, None, None, X_pd, program, None, None, None, None, outbound_mask_ls, atom=atom, epsilon=0, eval=True, init=False, graph_data_batch=graph_data, return_encoding=True)
            # pred = self.policy_net.forward(features,X_pd, program, outbound_mask_ls, atom, 0, eval=True)
            # pred = self.policy_net.forward_ls(features, X_pd, state, queue, 0, eval=True, replay=True, existing_atom=origin_atom)

        # tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        # atom_prediction_tensors = {}
        tensor_indeces = atom[pred_Q_key].argmax(-1)
        
        x_idx = torch.tensor(list(range(len(X_pd))))
        
        atom_prediction_tensors_ls = []
        for k in range(tensor_indeces.shape[-1]):
            atom_prediction_tensors_ls.append(pred[pred_Q_key][x_idx, k, tensor_indeces[:,k]])
        atom_prediction_tensors = torch.stack(atom_prediction_tensors_ls, dim=1) #atom_prediction_tensors/tensor_indeces.shape[-1]

        # col_tensor_indices = atom[col_Q_key].argmax(-1)
        # _,col_tensor_indices = torch.topk(atom[col_Q_key], k = self.topk_act, dim=-1)
        
        _,col_tensor_indices = torch.topk(atom[col_Q_key].view(len(atom[col_Q_key]),-1), k=self.topk_act, dim=-1)


        col_prediction_Q_tensor_ls = []
        
        for k in range(self.topk_act):
            col_prediction_Q_tensor_ls.append(pred[col_Q_key].view(len(pred[col_Q_key]), -1)[x_idx, col_tensor_indices[:,k]])
        
        col_prediction_Q_tensor = torch.stack(col_prediction_Q_tensor_ls, dim=1)
        # col_prediction_Q_tensor_ls = []
        # for k in range(col_tensor_indices.shape[-1]):
        #     col_prediction_Q_tensor_ls += pred[col_Q_key][x_idx, col_tensor_indices[:,k]]
        # col_prediction_Q_tensor = pred[col_Q_key][x_idx, col_tensor_indices]
        # col_prediction_Q_tensor = col_prediction_Q_tensor/col_tensor_indices.shape[-1]
        
        selected_num_feat_tensor_bool = atom[select_num_feat_key].to(self.device)
        
        if op_Q_key in atom:
            op_tensor_indices = atom[op_Q_key].argmax(-1)

            op_prediction_Q_tensor_ls = []
            for k in range(op_tensor_indices.shape[-1]):
                op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
            op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
            op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

            assert torch.sum(atom_prediction_tensors**selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) + torch.sum(op_prediction_Q_tensor == min_Q_val) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3
            
            #     # col_treatment_tensor_indices = atom[treatment_id_key]
            #     # treatment_tensors = torch.mean(pred[treatment_Q_key][col_treatment_tensor_indices], dim=-1)
            #     if not self.tr_str_two:
            #         treatment_tensors = self.obtain_average_treatment_Q_values(pred[treatment_Q_key], atom[treatment_id_key])
            #     else:
            #         treatment_tensors = self.obtain_average_treatment_Q_values2(pred[treatment_Q_key], atom[treatment_id_key])

            #     atom_prediction_tensors = (atom_prediction_tensors*3 + treatment_tensors)/4
            
        else:
            assert torch.sum(atom_prediction_tensors*selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3
            #     # col_treatment_tensor_indices = atom[treatment_id_key]
            #     # treatment_tensors = torch.mean(pred[treatment_Q_key][col_treatment_tensor_indices], dim=-1)
            #     if not self.tr_str_two:
            #         treatment_tensors = self.obtain_average_treatment_Q_values(pred[treatment_Q_key], atom[treatment_id_key])
            #     else:
            #         treatment_tensors = self.obtain_average_treatment_Q_values2(pred[treatment_Q_key], atom[treatment_id_key])
            #     atom_prediction_tensors = (atom_prediction_tensors*2 + treatment_tensors)/3
            
        atom_prediction_tensors = atom_prediction_tensors*selected_num_feat_tensor_bool + col_prediction_Q_tensor*(1-selected_num_feat_tensor_bool)

        del selected_num_feat_tensor_bool, col_prediction_Q_tensor, pred, X

        # if atom_prediction_tensors.shape[0] < 4:
        #     print()
        # loss = torch.sum(atom_prediction_tensors)
        # self.optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        #     del graph_data
        return atom_prediction_tensors, X_encoding, A, Y, D
    
    #takes an atom, and the maximal tensors used to produce it, and returns a Q value
    def get_atom_Q_value(self, atom:dict, atom_prediction_tensors: dict):
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.tensor([[1]], dtype=torch.float,device=self.device)
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                op = 1#atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                op = 1#atom_prediction_tensors["cat_op"]
            constant = atom_prediction_tensors[feat_name]
        # Q = formula*feat*op*constant[0]*constant[1]
        Q = constant
        return Q[0]
    
    def get_atom_Q_value2(self, atom:dict, atom_prediction_tensors: dict):
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.tensor([[1]], dtype=torch.float,device=self.device)
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                op = 1#atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                op = 1#atom_prediction_tensors["cat_op"]
            constant = atom_prediction_tensors[feat_name]
        # Q = formula*feat*op*constant[0]*constant[1]
        # Q = constant[0]*constant[1]
        # return Q[0]
        return torch.cat([constant[0].view(-1), constant[1].view(-1)])

    def get_atom_Q_value_ls(self, atom:dict, atom_prediction_tensors: dict):
        op=1
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.FloatTensor([[1]])
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                # op = atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                # op = atom_prediction_tensors["cat_op"]
            # constant = atom_prediction_tensors[tuple([tuple([item]) for item in list(feat_name.keys())])]
            constant = atom_prediction_tensors["pred_score"]

        Q = constant[1].view(-1)
        return Q
    
    def observe_transition(self, transition: Transition):
        self.memory.push(transition)

 
    def optimize_model(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=self.device)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.tensor([[transition.reward] for transition in batch], device=self.device, requires_grad=True, dtype=torch.float)

        #get Q value for (s,a)
        state_action_pred = [(a[0],self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 1], device=self.device, dtype=torch.float)
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        return loss.detach()
    
    def optimize_model2(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=self.device)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.tensor([transition.reward for transition in batch], device=self.device, requires_grad=True, dtype=torch.float)

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value2(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 2], device=self.device, dtype=torch.float)
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value2(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values[:,1:2].repeat(1,2), expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        return loss.detach()
    
    def optimize_model_ls0(self):
        if len(self.memory) < self.mem_sample_size: return 0.0, 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.mem_sample_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=self.device)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        # reward_batch = torch.stack([torch.from_numpy(transition.reward) for transition in batch]).to(self.device)
        reward_batch = torch.stack([transition.reward.mean(-1) for transition in batch]).to(self.device)

        #get Q value for (s,a)
        # if not self.do_medical:
        state_action_pred = [self.get_state_action_prediction_tensors_ls0(f,d, p,a) for f,d, p,a  in state_action_batch]

        state_action_values = torch.stack([t[0] for t in state_action_pred])
        X_encoding_tensors = torch.cat([t[1] for t in state_action_pred])
        treatment_tensors = torch.cat([t[2] for t in state_action_pred])
        outcome_tensors = torch.cat([t[3] for t in state_action_pred])
        dose_tensors = None
        if state_action_pred[0][4] is not None:
            dose_tensors = torch.cat([t[4] for t in state_action_pred])

        # extra_loss = torch.stack([l for t,l in state_action_pred])
        state_action_values = state_action_values.to(self.device)
        
        #get Q value for (s', max_a')
        # if not self.do_medical:
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        # else:
        #     next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0_medical(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.mem_sample_size, self.batch_size, self.topk_act], dtype=torch.float, device=self.device)
        if len(next_state_pred_non_final) > 0:
            # next_state_values_non_final = torch.stack([self.get_atom_Q_value_ls(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values_non_final = torch.stack(next_state_pred_non_final)
            next_state_values[non_final_mask] = next_state_values_non_final
            del next_state_values_non_final
        next_state_values = next_state_values.to(self.device)
        # Prepare the loss function
        expected_state_action_values = (next_state_values.view(-1) * self.gamma) + reward_batch.view(-1)
        # Compute the loss
        loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))# + 0.5*extra_loss.mean()
        # x, t, d=None, test=False, hidden=None
        if self.policy_net.backbone_full_model is not None:
            pred_outcome = self.policy_net.backbone_full_model(torch.zeros_like(X_encoding_tensors), treatment_tensors, dose_tensors, test=False, hidden=X_encoding_tensors)[1]
            if self.outcome_regression:
                reg_loss = F.mse_loss(pred_outcome.view(-1,1), outcome_tensors.view(-1,1))
            else:
                reg_loss = F.binary_cross_entropy_with_logits(pred_outcome, outcome_tensors)
        else:
            reg_loss = 0

        if not self.policy_net.fix_backbone:
            loss = loss + 0.1*reg_loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        return_loss = loss.detach().item()
        del loss
        return return_loss, self.calculate_grad_norm().item()
    def calculate_grad_norm(self):
        return torch.tensor([torch.norm(list(self.policy_net.parameters())[k].grad) for k in range(len(list(self.policy_net.parameters()))) if list(self.policy_net.parameters())[k].grad is not None]).mean()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())




class dql_algorithm:
    def __init__(self, train_dataset, valid_dataset, test_dataset, id_attr, outcome_attr, treatment_attr, lang, learning_rate, gamma, dropout_p, feat_range_mappings, program_max_len, replay_memory_capacity, rl_config, model_config, backbone_model_config, numeric_count, category_count, category_sum_count, args, topk_act=1,
                 batch_size = 64,
                 modeldir = None, log_folder = None):

        # self.model = CausalQNet_rl.from_pretrained(
        #     'distilbert-base-uncased',
        #     num_labels = 2,
        #     output_attentions=False,
        #     output_hidden_states=False)
        self.epochs = args.epochs
        self.topk_act = topk_act
        self.epsilon = rl_config["epsilon"]
        self.epsilon_falloff = rl_config["epsilon_falloff"]
        self.lang = lang
        self.train_dataset, self.valid_dataset, self.test_dataset = train_dataset, valid_dataset, test_dataset
        # else:
        #     self.train_dataset, self.valid_dataset, self.in_sample_test_data, self.out_sample_test_data = train_dataset, valid_dataset, test_dataset[0], test_dataset[1]

        self.feat_range_mappings = feat_range_mappings
        self.dqn = DQN_all_tabular(id_attr, outcome_attr, treatment_attr, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=rl_config["mem_sample_size"], seed=args.seed, numeric_count=numeric_count, category_count=category_count, category_sum_count = category_sum_count, topk_act=topk_act, model_config=model_config, backbone_model_config=backbone_model_config, args = args, discretize_feat_value_count=rl_config["discretize_feat_value_count"])
        self.dqn.trainer = self
        self.program_max_len = program_max_len
        self.target_update = rl_config["target_update"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     self.dq = self.model.cuda()

        self.batch_size = batch_size
        self.modeldir = modeldir
        self.do_medical = False
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self.num_treatments = args.num_treatments
        self.cont_treatment = args.cont_treatment
        self.lr = args.lr
        self.reward_coeff = rl_config["reward_coeff"]
        self.treatment_coeff = self.reward_coeff
        #     self.min_test_assignments, self.max_test_assignments = args.min_test_assignments, args.max_test_assignments
            
        self.tr_str_two = args.tr_str_two
        self.cat_and_cont_treatment = args.cat_and_cont_treatment
        self.dataset_name = args.dataset_name
        self.is_log = args.is_log
        self.data_folder, self.dataset_id = args.data_folder, args.dataset_id
        self.method_two=args.method_two
        self.method_three = args.method_three
        self.has_dose = args.has_dose
        self.no_tr = args.no_tr
        self.no_hyper_adj = args.no_hyper_adj
        self.regression_ratio = args.regression_ratio
        outcome_regression = True
        self.classification = not outcome_regression
        self.args = args
        # self.gpu_db = args.gpu_db
        # self.memory = ReplayMemory(replay_memory_capacity)

    def obtain_binary_treatment_reward(self, treatment_pred, outcome_pred, A, Y, epoch):
        # reward1 = ((treatment_pred > 0.5).view(-1).type(torch.float) == A.view(-1).type(torch.float)).view(-1).type(torch.float)
        treatment_pred[treatment_pred != treatment_pred] = -1
        outcome_pred[treatment_pred != treatment_pred] = -1
        reward1 = (treatment_pred >= 0).type(torch.float)*(treatment_pred*(A == 1).type(torch.float) + (1-treatment_pred)*(A == 0).type(torch.float))
        # reward2 = outcome_pred.view(-1)*(Y == 1).view(-1).type(torch.float) + (1-outcome_pred.view(-1))*(Y == 0).view(-1).type(torch.float)
        # ihdp: 0.01
        
        reward2 = (treatment_pred >= 0).type(torch.float)*(torch.exp(-self.reward_coeff*(outcome_pred - Y)**2))
        # reward1 = (treatment_pred >= 0).type(torch.float)*(torch.exp(-self.reward_coeff*(outcome_pred - Y)**2))
        # self.reward_coeff = reward_coeff
        # reward1 = (treatment_pred >= 0).type(torch.float).view(-1)*(torch.exp(-0.5*(outcome_pred.view(-1) - Y.view(-1))**2))
        if not self.no_tr:
            if epoch < 50:
                return reward1, reward2
            else:
                return reward2, reward2
        else:
            return reward2, reward2

    def obtain_multi_treatment_reward(self, treatment_pred, outcome_pred, A, Y, epoch):
        treatment_pred[treatment_pred != treatment_pred] = -1
        outcome_pred[outcome_pred != outcome_pred] = -1
        # reward1 = ((treatment_pred > 0.5).view(-1).type(torch.float) == A.view(-1).type(torch.float)).view(-1).type(torch.float)
        selected_treatment_pred = treatment_pred[torch.arange(treatment_pred.shape[0]), :, A.view(-1).long()]
        
        reward1 = (selected_treatment_pred >= 0).type(torch.float)*selected_treatment_pred.type(torch.float)
        # reward2 = outcome_pred.view(-1)*(Y == 1).view(-1).type(torch.float) + (1-outcome_pred.view(-1))*(Y == 0).view(-1).type(torch.float)
        reward2 = (selected_treatment_pred >= 0).type(torch.float)*(torch.exp(-0.5*(outcome_pred - Y)**2))
        # reward1 = (selected_treatment_pred >= 0).type(torch.float)*(torch.exp(-0.5*(outcome_pred - Y)**2))
        if epoch < 20:
            return reward1, reward2
        else:
            return reward2, reward2
    
    def obtain_cont_treatment_reward(self, treatment_pred, outcome_pred, A, Y, epoch):
        # reward1 = ((treatment_pred > 0.5).view(-1).type(torch.float) == A.view(-1).type(torch.float)).view(-1).type(torch.float)
        # selected_treatment_pred = treatment_pred[torch.arange(treatment_pred.shape[0]), A.view(-1).long()]
        
        # reward1 = (selected_treatment_pred >= 0).type(torch.float).view(-1)*selected_treatment_pred.view(-1).type(torch.float)
        treatment_gap = (torch.exp(-self.treatment_coeff*(treatment_pred - A.view(-1, 1))**2))
        treatment_gap[treatment_gap != treatment_gap] = 0
        reward1 = (treatment_pred == treatment_pred).type(torch.float)*treatment_gap
        # reward2 = outcome_pred.view(-1)*(Y == 1).view(-1).type(torch.float) + (1-outcome_pred.view(-1))*(Y == 0).view(-1).type(torch.float)
        outcome_gap = (torch.exp(-self.reward_coeff*(outcome_pred - Y.view(-1, 1))**2))
        outcome_gap[outcome_gap != outcome_gap] = 0
        reward2 = (treatment_pred == treatment_pred).type(torch.float)*(outcome_pred == outcome_pred).type(torch.float)*outcome_gap
        # reward1 = (treatment_pred == treatment_pred).type(torch.float)*(outcome_pred == outcome_pred).type(torch.float)*outcome_gap
        # return reward1, reward2
        if epoch < 50:
            return reward1, reward2
        else:
            return reward2, reward2

    def obtain_reward(self, treatment_pred, outcome_pred, A, Y, epoch):
        if not self.cont_treatment:
        
            if self.num_treatments == 2:
            # # reward1 = ((treatment_pred > 0.5).view(-1).type(torch.float) == A.view(-1).type(torch.float)).view(-1).type(torch.float)
            # reward1 = (treatment_pred >= 0).type(torch.float).view(-1)*(treatment_pred.view(-1)*(A == 1).view(-1).type(torch.float) + (1-treatment_pred.view(-1))*(A == 0).view(-1).type(torch.float))
            # # reward2 = outcome_pred.view(-1)*(Y == 1).view(-1).type(torch.float) + (1-outcome_pred.view(-1))*(Y == 0).view(-1).type(torch.float)
            # reward2 = (treatment_pred >= 0).type(torch.float).view(-1)*(torch.exp(-0.5*(outcome_pred.view(-1) - Y.view(-1))**2))
            # reward1 = (treatment_pred >= 0).type(torch.float).view(-1)*(torch.exp(-0.5*(outcome_pred.view(-1) - Y.view(-1))**2))

                return self.obtain_binary_treatment_reward(treatment_pred, outcome_pred, A, Y, epoch)
            elif self.num_treatments > 2:
                return self.obtain_multi_treatment_reward(treatment_pred, outcome_pred, A, Y, epoch)
        else:
            return self.obtain_cont_treatment_reward(treatment_pred, outcome_pred, A, Y, epoch)
   
    
    def copy_data_in_database(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(all_other_pats_ls[idx].copy())
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    def copy_data_in_database2(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(torch.clone(all_other_pats_ls[idx]))
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    
    def observe_transition(self, transition: Transition):
        self.dqn.memory.push(transition)
    
    def produce_other_data_ls(self, other_data, sample_count):
        origin_all_other_pats_ls = []
        for _ in range(sample_count):
            origin_all_other_pats_ls.append(torch.ones(len(other_data)).bool())
        return origin_all_other_pats_ls

    def do_prediction(self, X, A, D, X_pd_full, origin_all_other_pats_ls, origin_X):
        program = []
        outbound_mask_ls = []
        # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        program_str = []
        program_col_ls = []
        # if self.gpu_db:
        origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]
        all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
        # if self.gpu_db:
        stopping_conds = torch.ones_like(A).bool().to(self.device)
        # else:   
        #     stopping_conds = torch.ones_like(A.cpu()).bool()
        
        outcome_pred_curr_batch=torch.zeros(len(A))
        # if self.gpu_db:
        outcome_pred_curr_batch = outcome_pred_curr_batch.to(self.device)
        
        for arr_idx in range(self.program_max_len):
            init = (len(program) == 0)
            done = (arr_idx == self.program_max_len - 1)
            # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
            (treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, _, _), X_encoding = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, compute_ate=False, return_encoding=True)

            program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls = next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls
            if self.backbone_full_model is not None:
                reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D)
            else:
                reg_pred, reg_full_pred = torch.zeros_like(outcome_pred), 0
            if not self.cont_treatment and self.num_treatments == 2:    
                stopping_conds = torch.logical_and(stopping_conds, treatment_pred != -1)
                # outcome_pred, pos_outcome, neg_outcome, _ = outcome_pred    
                outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds].view(-1) + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                # reg_pos_outcome_curr_batch[treatment_pred != -1] = reg_full_pred[treatment_pred != -1, 0]
                # reg_neg_outcome_curr_batch[treatment_pred != -1] = reg_full_pred[treatment_pred != -1, 1]
            else:
                if not self.cont_treatment and self.num_treatments > 2:
                    selected_treatment_pred = treatment_pred[torch.arange(treatment_pred.shape[0]), A.view(-1).cpu().long()]
                    stopping_conds = torch.logical_and(stopping_conds.view(-1), (selected_treatment_pred != -1).view(-1))
                    outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds].view(-1) + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                else:
                    stopping_conds = torch.logical_and(stopping_conds, treatment_pred == treatment_pred)
                    outcome_pred_curr_batch[treatment_pred == treatment_pred] = (outcome_pred[treatment_pred == treatment_pred] + self.regression_ratio*reg_pred[treatment_pred == treatment_pred])/(1 + self.regression_ratio)

        
        return outcome_pred_curr_batch



    def compute_eval_metrics(self, meta_info, test_dataset, num_treatments, train=False):
        mises = []
        ites = []
        dosage_policy_errors = []
        policy_errors = []
        pred_best = []
        pred_vals = []
        true_best = []

        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        # for patient in test_patients:
        # for step, batch in enumerate(test_loader):
        for idx, origin_X, A, Y, count_Y, D, patient, all_other_pats_ls in tqdm(test_dataset):
            if train and len(pred_best) > 10:
                return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)
            for treatment_idx in range(num_treatments):
                test_data = dict()
                test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                X = torch.from_numpy(test_data['x'])
                X_pd_full = X
                origin_X = X
                X = X.to(self.device)
                test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                A = torch.from_numpy(test_data["t"]).to(self.device)
                test_data['d'] = treatment_strengths
                D = torch.from_numpy(test_data["d"]).to(self.device)

                origin_all_other_pats_ls= [all_other_pats_ls.clone() for _ in range(num_integration_samples)]
                pred_dose_response =self.do_prediction(X, A, D, X_pd_full, origin_all_other_pats_ls, origin_X).cpu().numpy()
                # pred_dose_response = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                # pred_dose_response = pred_dose_response * (
                #         dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                #                         dataset['metadata']['y_min']

                true_outcomes = [get_patient_outcome(patient, meta_info, treatment_idx, d) for d in
                                    treatment_strengths]
                
                # if len(pred_best) < num_treatments and train == False:
                #     #print(true_outcomes)
                #     print([item[0] for item in pred_dose_response])
                mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
                inter_r = np.array(true_outcomes) - pred_dose_response.squeeze()
                ite = np.mean(inter_r ** 2)
                mises.append(mise)
                ites.append(ite)

                best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

                def pred_dose_response_curve(dosage):
                    test_data = dict()
                    test_data['x'] = np.expand_dims(patient, axis=0)
                    test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                    test_data['d'] = np.expand_dims(dosage, axis=0)
                    X = torch.from_numpy(test_data['x'])
                    X_pd_full = X
                    origin_X = X
                    X = X.to(self.device)
                    A = torch.from_numpy(test_data["t"]).to(self.device)
                    D = torch.from_numpy(test_data["d"]).to(self.device)
                    
                    ret_val =self.do_prediction(X, A, D, X_pd_full, [all_other_pats_ls], origin_X)
                    # ret_val = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                    # ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                    #             dataset['metadata']['y_min']
                    return ret_val.item()

                true_dose_response_curve = get_true_dose_response_curve(meta_info, patient, treatment_idx)

                min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                        x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

                max_pred_opt_y = - min_pred_opt.fun
                max_pred_dosage = min_pred_opt.x
                max_pred_y = true_dose_response_curve(max_pred_dosage)

                min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                        x0=[0.5], method="SLSQP", bounds=[(0, 1)])
                max_true_y = - min_true_opt.fun
                max_true_dosage = min_true_opt.x

                dosage_policy_error = (max_true_y - max_pred_y) ** 2
                dosage_policy_errors.append(dosage_policy_error)

                pred_best.append(max_pred_opt_y)
                pred_vals.append(max_pred_y)
                true_best.append(max_true_y)
                

            selected_t_pred = np.argmax(pred_vals[-num_treatments:])
            selected_val = pred_best[-num_treatments:][selected_t_pred]
            selected_t_optimal = np.argmax(true_best[-num_treatments:])
            optimal_val = true_best[-num_treatments:][selected_t_optimal]
            policy_error = (optimal_val - selected_val) ** 2
            policy_errors.append(policy_error)

        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)

    def do_prediction_mini_batch(self, X, A, D, X_pd_full, all_other_pats_ls, origin_X, compute_ate):
        program = []
        outbound_mask_ls = []
        # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        program_str = []
        program_col_ls = []
        all_atom_ls = []
        for arr_idx in range(self.program_max_len):
            init = (len(program) == 0)
            done = (arr_idx == self.program_max_len - 1)
            # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
            treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, _, _ = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, compute_ate=compute_ate, method_three=self.method_three, method_two=self.method_two)
            all_atom_ls.append(atom_ls)
            program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls = next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls
            
        return program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls, treatment_pred, outcome_pred, all_atom_ls
    
    def eval_stability(self, test_loader, origin_explanation_str_ls, origin_explanation_col_ls, perturb_times=5):
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False

            all_explanation_perturbation_ls = []
            all_explanation_perturbation_col_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                curr_origin_explanations = [origin_explanation_str_ls[idx.item()] for idx in idx_ls]
                curr_origin_explanation_cols = [origin_explanation_col_ls[idx.item()] for idx in idx_ls]
                curr_perturbation_ls = []
                curr_col_perturbation_ls = []
                
                for _ in range(perturb_times):
                    pert_origin_X = perturb_samples(origin_X, self.train_dataset)
                    pert_X = self.train_dataset.convert_feats_to_transformed_feats(pert_origin_X)
                
                
                
                    pert_X =pert_X.to(self.device)
                    Y = Y.to(self.device)
                    A = A.to(self.device)
                    if count_Y is not None:
                        compute_ate = True
                    
                    if D is not None:
                        D = D.to(self.device)
                
                    all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                
                
                
                    X_pd_full = pert_origin_X#pd.concat(X_pd_ls)
                    # all_transformed_expr_ls = []
                    # all_treatment_pred = []
                    # all_outcome_pred = []
                    
                    # prev_reward = torch.zeros([len(A), 2])
                    
                    program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls, treatment_pred, outcome_pred, atom_ls = self.do_prediction_mini_batch(pert_X, A, D, X_pd_full, all_other_pats_ls, pert_origin_X, compute_ate)
                
                    for k in range(len(program_col_ls)):
                        curr_origin_cols = curr_origin_explanation_cols[k][0]
                        curr_cols = program_col_ls[k][0]
                        corr = len(set(curr_origin_cols).intersection(set(curr_cols)))*1.0/len(set(curr_origin_cols).union(set(curr_cols)))
                        curr_col_perturbation_ls.append(corr)
                    curr_perturbation_ls.append(evaluate_explanation_dff(curr_origin_explanations, program_str))
                    
                all_explanation_perturbation_ls.append(torch.stack(curr_perturbation_ls))
                all_explanation_perturbation_col_ls.append(torch.tensor(curr_col_perturbation_ls))
            all_explanation_perturbation_tensor = torch.cat(all_explanation_perturbation_ls, dim=-1)
            all_explanation_perturbation_col_tensor = torch.cat(all_explanation_perturbation_col_ls, dim=-1)
            mean_explanation_perturbation = torch.mean(all_explanation_perturbation_tensor)
            mean_explanation_perturbation_cols = torch.mean(all_explanation_perturbation_col_tensor)
            print("mean explanation perturbation similarity::", mean_explanation_perturbation.item())
            print("mean explanation perturbation col similarity::", mean_explanation_perturbation_cols.item())
    
    
    

    def eval_sufficiency(self, test_loader, predicted_y, origin_explanation_str_ls, fp=0.2):
        all_exp_ls = transform_explanation_str_to_exp(test_loader.dataset, origin_explanation_str_ls)
        
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            all_matched_ratio_ls = []
            for sample_id in range(len(test_loader.dataset)):
                idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = test_loader.dataset[sample_id]
                curr_exp_ls = all_exp_ls[sample_id]
                all_matched_features_boolean =  eval_booleans(curr_exp_ls, test_loader.dataset.origin_features)
                # all_matched_features_boolean = curr_exp[1](test_loader.dataset.features[curr_exp[0]], curr_exp[2])
                all_matched_pred_labels = predicted_y[all_matched_features_boolean]
                if self.has_dose:
                    all_dose_array = test_loader.dataset.dose_array[all_matched_features_boolean]
                    topk_ids = torch.topk(torch.abs(all_dose_array - D).view(-1), k=topk, largest=False)[1]
                    all_matched_pred_labels = all_matched_pred_labels[topk_ids]
                elif self.cont_treatment:
                    all_treatment_array = test_loader.dataset.treatment_array[all_matched_features_boolean]
                    topk_ids = torch.topk(torch.abs(all_treatment_array - A).view(-1), k=min(topk, len(all_treatment_array)), largest=False)[1]
                    all_matched_pred_labels = all_matched_pred_labels[topk_ids]
                matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                matched_ratio = matched_sample_count*1.0/(len(all_matched_pred_labels) - 1+1e-6)
                all_matched_ratio_ls.append(matched_ratio)
            
            sufficiency_score = np.array(all_matched_ratio_ls).mean()
            
            
            print("sufficiency score::", sufficiency_score)
    
    def eval_aopc(self, test_loader, subset_count = 10):
        
        full_base_X = self.train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_relative_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_select_feat_ids = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
            
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
            
            
            
                X_pd_full =  origin_X#pd.concat(X_pd_ls)
                
                program, program_str, program_col_ls, _, outbound_mask_ls, treatment_pred, outcome_pred, all_atom_ls = self.do_prediction_mini_batch(X, A, D, X_pd_full, all_other_pats_ls, origin_X, compute_ate)
                
                curr_perturbation_ls = []
                
                outcome_pred, pos_outcome_pred, neg_outcome_pred = outcome_pred                                
                
                # for selected_idx_set in selected_idx_set_ls:
                selected_col_ids = torch.stack([all_atom_ls[k][col_id_key] for k in range(len(all_atom_ls))], dim=2)
                all_select_feat_ids.append(selected_col_ids)
                if test_loader.dataset.y_scaler is not None:
                    outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, outcome_pred)
                    pos_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, pos_outcome_pred)
                    neg_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, neg_outcome_pred)
                
                for k in range(len(all_atom_ls)):
                    curr_base_X_ls = []
                    curr_gt_outcome_ls = []
                    for sub_idx in range(len(idx_ls)):                        
                    
                        curr_selected_col_ids = selected_col_ids[sub_idx,:, 0:k+1].view(-1).to(self.device)
                        curr_X = X[sub_idx]
                        # X, full_base_X, fid_ls
                        # if len(set(curr_selected_col_ids.tolist())) < k + 1:
                            # set(curr_selected_col_ids
                        if len(set(curr_selected_col_ids.tolist())) < k + 1:
                            not_selected_col_ids = [k for k in range(curr_X.shape[-1]) if k not in set(curr_selected_col_ids.tolist())]
                            random.shuffle(not_selected_col_ids)
                            curr_selected_col_ids = torch.cat([curr_selected_col_ids, torch.tensor(not_selected_col_ids[:k+1 - len(set(curr_selected_col_ids.tolist()))]).to(self.device)], dim=0)

                        curr_base_X = construct_base_x(curr_X, full_base_X, curr_selected_col_ids)
                        curr_base_X_ls.append(curr_base_X)
                        curr_gt_outcome_ls.append(Y[sub_idx].view(-1).cpu())
                    curr_base_X_tensor = torch.stack(curr_base_X_ls, dim=0)
                    curr_gt_outcome_tensor = torch.cat(curr_gt_outcome_ls, dim=0)
                    
                    all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                    _, _, curr_program_col_ls, _, _, curr_treatment_pred, curr_outcome_pred, curr_all_atom_ls = self.do_prediction_mini_batch(curr_base_X_tensor, A, D, curr_base_X_tensor, all_other_pats_ls, curr_base_X_tensor, compute_ate)
                    
                    curr_outcome_pred, curr_pos_outcome_pred, curr_neg_outcome_pred = curr_outcome_pred
                    
                    if test_loader.dataset.y_scaler is not None:
                        curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)
                        curr_pos_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_pos_outcome_pred)
                        curr_neg_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_neg_outcome_pred)
                        curr_gt_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_gt_outcome_tensor)
                    

                    outcome_pred_diff = torch.abs(outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1))
                    relative_diff = (torch.abs(outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1)))/(torch.abs(outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1)) + torch.abs(curr_outcome_pred.view(-1) - curr_gt_outcome_tensor.view(-1)) + 1e-5)
                    all_relative_diff_ls.append(relative_diff)
                    ate_diff = torch.abs((curr_pos_outcome_pred - curr_neg_outcome_pred) - (pos_outcome_pred - neg_outcome_pred))
                    all_outcome_diff_ls.append(outcome_pred_diff)
                    all_ate_diff_ls.append(ate_diff)
                        # for 
                                                    
            mean_ate_diff = torch.mean(torch.stack(all_ate_diff_ls, dim=0))
            mean_outcome_diff = torch.mean(torch.cat(all_outcome_diff_ls, dim=0))        
            mean_relative_diff = torch.mean(torch.cat(all_relative_diff_ls))

            
            print("mean outcome difference::", mean_outcome_diff.item())
            print("mean eta difference::", mean_ate_diff.item())
            print("mean relative difference::", mean_relative_diff.item())

        torch.save(torch.cat(all_select_feat_ids), os.path.join(self.log_folder, "selected_feat_ours"))
    
    def eval_faithfulness(self, test_loader, subset_count = 10):
        
        full_base_X = self.train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        sample_id_gt_feat_id_mappings, sample_id_gt_feat_name_mappings, rand_coeff_mappings = retrieve_gt_explanations(self.data_folder, self.dataset_id)
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_relative_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_overlap_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
            
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
            
            
            
                X_pd_full =  origin_X#pd.concat(X_pd_ls)
                
                program, program_str, program_col_ls, _, outbound_mask_ls, treatment_pred, outcome_pred, all_atom_ls = self.do_prediction_mini_batch(X, A, D, X_pd_full, all_other_pats_ls, origin_X, compute_ate)
                
                curr_perturbation_ls = []
                
                
                gt_explanation_str_ls = [sample_id_gt_feat_name_mappings[idx.item()] for idx in idx_ls]
                
                gt_explanation_col_ids_ls = [sample_id_gt_feat_id_mappings[idx.item()] for idx in idx_ls]
                
                
                
                outcome_pred, pos_outcome_pred, neg_outcome_pred = outcome_pred                                
                
                # for selected_idx_set in selected_idx_set_ls:
                selected_col_ids = torch.stack([all_atom_ls[k][col_id_key] for k in range(len(all_atom_ls))], dim=2)
                pred_explanation_col_ids_ls = [selected_col_ids[k].view(-1).tolist() for k in range(len(selected_col_ids))]
                overlap_ls = [len(set(gt_explanation_col_ids_ls[k]).intersection(set(pred_explanation_col_ids_ls[k])))*1.0/len(set(gt_explanation_col_ids_ls[k])) for k in range(len(gt_explanation_col_ids_ls))]
                
                all_overlap_ls.extend(overlap_ls)
                
            mean_overlap = np.array(all_overlap_ls).mean()
            
            print("mean overlap::", mean_overlap)
            
            print()
    
    def eval_emptiness(self, test_loader, return_explanations=False):
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            all_treatment_ls = []
            all_outcome_ls = []
            all_gt_treatment_ls = []
            all_gt_outcome_ls = []
            all_gt_count_outcome_ls = []
            all_pos_outcome_ls = []
            all_neg_outcome_ls = []
            all_non_empty_ls = []
            compute_ate = False
            all_program_ls = []
            all_program_str_ls = []
            all_program_col_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                X =X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
            
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                
                program = []
                outbound_mask_ls = []
                # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_str = []
                program_col_ls = []
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = origin_X#pd.concat(X_pd_ls)
                # all_transformed_expr_ls = []
                # all_treatment_pred = []
                # all_outcome_pred = []
                
                # prev_reward = torch.zeros([len(A), 2])
                
                outcome_pred_curr_batch=torch.zeros(len(A))
                reg_outcome_pred_curr_batch = torch.zeros(len(A))
                non_empty_curr_batch = torch.zeros(len(A), dtype=torch.long)
                if compute_ate:    
                    pos_outcome_curr_batch=torch.zeros(len(A))
                    neg_outcome_curr_batch=torch.zeros(len(A))
                    reg_pos_outcome_curr_batch=torch.zeros(len(A))
                    reg_neg_outcome_curr_batch=torch.zeros(len(A))
                
                for arr_idx in range(self.program_max_len):
                    init = (len(program) == 0)
                    done = (arr_idx == self.program_max_len - 1)
                    # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
                    (treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, _, _), X_encoding = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, compute_ate=compute_ate, method_two=self.method_two, test=True, method_three=self.method_three, return_encoding=True)
            
                    program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls = next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls
                
                    reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D)
                
                    stopping_conds = (treatment_pred != -1)

                    if compute_ate:
                        outcome_pred, pos_outcome, neg_outcome, non_empty = outcome_pred
                        pos_outcome_curr_batch[stopping_conds] = (pos_outcome[stopping_conds] + self.regression_ratio*reg_full_pred[stopping_conds, 1].view(-1))/(1 + self.regression_ratio)
                        neg_outcome_curr_batch[stopping_conds] = (neg_outcome[stopping_conds] + self.regression_ratio*reg_full_pred[stopping_conds, 0].view(-1))/(1 + self.regression_ratio)
                        non_empty_curr_batch[stopping_conds] = non_empty[stopping_conds]
                    outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds] + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                
                
                all_treatment_ls.append(treatment_pred)
                if compute_ate:
                    all_pos_outcome_ls.append(pos_outcome_curr_batch.view(-1))
                    all_neg_outcome_ls.append(neg_outcome.view(-1))
                    all_non_empty_ls.append(non_empty_curr_batch.view(-1))
                all_outcome_ls.append(outcome_pred_curr_batch.view(-1))
                if compute_ate:
                    all_gt_count_outcome_ls.append(count_Y.view(-1))
                    
                all_program_ls.append(program)
                all_program_str_ls.extend(program_str)
                all_program_col_ls.extend(program_col_ls)
                
                all_gt_treatment_ls.append(A.cpu().view(-1))
                all_gt_outcome_ls.append(Y.cpu().view(-1))
                
            all_treatment_pred_tensor = torch.cat(all_treatment_ls)
            all_outcome_pred_tensor = torch.cat(all_outcome_ls)
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
            all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
            
            if test_loader.dataset.y_scaler is not None:
                all_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_outcome_pred_tensor)
                all_gt_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_outcome_tensor)
            
            if compute_ate:
                all_concat_count_Y_tensor = torch.cat(all_gt_count_outcome_ls)
                all_pos_outcome_tensor = torch.cat(all_pos_outcome_ls)
                all_neg_outcome_tensor = torch.cat(all_neg_outcome_ls)
                all_non_empty_tensor = torch.cat(all_non_empty_ls)
                if test_loader.dataset.y_scaler is not None:
                    all_concat_count_Y_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_concat_count_Y_tensor)
                    all_pos_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_pos_outcome_tensor)
                    all_neg_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_neg_outcome_tensor)
            
            all_treatment_arr_np = all_gt_treatment_tensor.view(-1).numpy()
            
            

            outcome_error = F.mse_loss(all_outcome_pred_tensor.view(-1,1), all_gt_outcome_tensor.view(-1,1))# torch.sqrt(torch.mean((all_outcome_pred_tensor - all_gt_outcome_tensor)**2))

            assert torch.sum(torch.isnan(outcome_error)) == 0

            if not self.cont_treatment:
                if self.num_treatments > 2:
                    all_pred_treatment_arr_full_d = (all_treatment_pred_tensor.argmax(-1)).type(torch.long).view(-1).numpy()
                    treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
                    if len(np.unique(all_treatment_arr_np)) == 1:
                        treatment_auc = 0
                    else:
                        treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy(), multi_class="ovr")
                elif self.num_treatments == 2:
                    all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()
                    treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
                    if len(np.unique(all_treatment_arr_np)) == 1:
                        treatment_auc = 0
                    else:
                        treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                    # treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                
                print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            # else:
            #     treatment_acc = F.mse_loss(all_treatment_pred_tensor.view(-1,1), torch.from_numpy(all_treatment_arr_np).view(-1,1)).item()

            #     print("treatment error::%f, outcome error::%f"%(treatment_acc, outcome_error))
            print("evaluation outcome error::%f"%(outcome_error))
                
            if compute_ate:
                if all_concat_count_Y_tensor is not None and self.num_treatments == 2 and not self.cont_treatment:
                    gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_tensor.view(-1), all_gt_treatment_tensor.view(-1)], dim=1), all_concat_count_Y_tensor.reshape(-1,1))
                    
                    avg_ite, avg_ate = evaluate_treatment_effect_core(all_pos_outcome_tensor, all_neg_outcome_tensor, gt_treatment_outcome, gt_control_outcome)
                    print("average individualized treatment effect::%f"%avg_ite)
                    print("average treatment effect::%f"%avg_ate)
                    print("non empty rate::", torch.mean(all_non_empty_tensor.float()).item())
                    if not return_explanations:
                        return avg_ite, avg_ate, outcome_error
                    else:
                        return avg_ite, avg_ate, outcome_error, all_program_ls, all_program_str_ls, all_program_col_ls, all_outcome_pred_tensor, all_pos_outcome_tensor, all_neg_outcome_tensor
        if not return_explanations:
            return None, None, outcome_error
        else:
            return None, all_program_ls, all_program_str_ls, all_program_col_ls, all_outcome_pred_tensor

    def obtain_regularized_outcome(self, X_encoding, A, D, test=True):
        reg_full_pred = None
        if not self.cont_treatment:
            res =  self.dqn.policy_net.backbone_full_model(torch.ones_like(X_encoding), A, D, test=test, hidden=X_encoding)
            if test:
                _, reg_pred, reg_full_pred = res
            else:
                _, reg_pred = res
            # if not self.gpu_db:
            # reg_pred = reg_pred.detach().cpu()
            # else:
            reg_pred = reg_pred.detach()
            if reg_full_pred is not None:
                # if not self.gpu_db:
                    # reg_full_pred = reg_full_pred.detach().cpu()
                # else:
                reg_full_pred = reg_full_pred.detach()
        else:
            _, reg_pred =  self.dqn.policy_net.backbone_full_model(torch.ones_like(X_encoding), A, D, test=test, hidden=X_encoding)
            # if not self.gpu_db:
                # reg_pred = reg_pred.detach().cpu()
            # else:
            reg_pred = reg_pred.detach()
            
        return reg_pred, reg_full_pred

    def test(self, test_loader, return_explanations=False):
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            all_treatment_ls = []
            all_outcome_ls = []
            all_rule_outcome_ls = []
            all_rule_pos_outcome_ls = []
            all_rule_neg_outcome_ls = []
            all_gt_treatment_ls = []
            all_gt_outcome_ls = []
            all_gt_count_outcome_ls = []
            all_pos_outcome_ls = []
            all_neg_outcome_ls = []
            compute_ate = False
            all_program_ls = []
            all_program_str_ls = []
            all_program_col_ls = []
            empty_res = 0
            empty_pos = 0
            empty_neg = 0
            early_stop_samples=0
            total_sample_count = 0
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                X =X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
                    
                # if self.gpu_db:
                origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]
            
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                
                program = []
                outbound_mask_ls = []
                # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_str = []
                program_col_ls = []
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = origin_X#pd.concat(X_pd_ls)
                # all_transformed_expr_ls = []
                # all_treatment_pred = []
                # all_outcome_pred = []
                
                # prev_reward = torch.zeros([len(A), 2])
                
                outcome_pred_curr_batch=torch.zeros(len(A))
                outcome_pred_curr_batch[:] = torch.tensor(float('nan'))
                
                # if self.gpu_db:
                outcome_pred_curr_batch = outcome_pred_curr_batch.to(self.device)
                rule_outcome_pred_curr_batch = torch.zeros_like(outcome_pred_curr_batch)
                # reg_outcome_pred_curr_batch = torch.zeros(len(A))
                if compute_ate:    
                    pos_outcome_curr_batch=torch.zeros(len(A))
                    pos_outcome_curr_batch[:] = torch.tensor(float('nan'))
                    neg_outcome_curr_batch=torch.zeros(len(A))
                    neg_outcome_curr_batch[:] = torch.tensor(float('nan'))
                    # reg_pos_outcome_curr_batch=torch.zeros(len(A))
                    # reg_neg_outcome_curr_batch=torch.zeros(len(A))
                    # if self.gpu_db:
                    pos_outcome_curr_batch = pos_outcome_curr_batch.to(self.device)
                    neg_outcome_curr_batch = neg_outcome_curr_batch.to(self.device)
                    rule_pos_outcome_curr_batch = torch.zeros_like(pos_outcome_curr_batch)
                    rule_neg_outcome_curr_batch = torch.zeros_like(neg_outcome_curr_batch)
                # if self.gpu_db:
                stopping_conds = torch.ones_like(A).bool().to(self.device)
                # else:   
                #     stopping_conds = torch.ones_like(A.cpu()).bool()
                for arr_idx in range(self.program_max_len):
                    init = (len(program) == 0)
                    done = (arr_idx == self.program_max_len - 1)
                    # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
                    (treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, _, _), X_encoding = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, compute_ate=compute_ate, method_two=self.method_two, test=True, method_three=self.method_three, return_encoding=True)
                    if compute_ate:  
                        outcome_pred, pos_outcome, neg_outcome, _ = outcome_pred    
                    # x, t, d=None, test=False, hidden=None
                    if self.dqn.policy_net.backbone_full_model is not None:
                        reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D, test=True)
                    else:                            
                        reg_pred = torch.zeros_like(outcome_pred)
                        reg_full_pred = torch.zeros([len(outcome_pred), self.num_treatments]).to(outcome_pred.device)
                    # reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D)
                    # program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls = next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls
                    program, all_other_pats_ls, outbound_mask_ls = next_program, next_all_other_pats_ls, next_outbound_mask_ls
                    
                    # reg_outcome_pred_curr_batch[treatment_pred != -1] = reg_pred[treatment_pred != -1]
                    
                    # stopping_conds = determine_stopping_by_Q_values(atom_ls)

                    # if torch.any(stopping_conds < 0):
                    #     print()
                    
                    if compute_ate:    
                        stopping_conds = torch.logical_and(stopping_conds.view(-1), (treatment_pred != -1).view(-1))
                        
                        
                        pos_outcome_curr_batch[stopping_conds.view(-1)] = (pos_outcome[stopping_conds.view(-1)].view(-1) + self.regression_ratio*reg_full_pred[stopping_conds.view(-1), 1].view(-1))/(1 + self.regression_ratio)
                        neg_outcome_curr_batch[stopping_conds.view(-1)] = (neg_outcome[stopping_conds.view(-1)].view(-1) + self.regression_ratio*reg_full_pred[stopping_conds.view(-1), 0].view(-1))/(1 + self.regression_ratio)
                        outcome_pred_curr_batch[stopping_conds.view(-1)] = (outcome_pred[stopping_conds.view(-1)].view(-1) + self.regression_ratio*reg_pred[stopping_conds.view(-1)].view(-1))/(1 + self.regression_ratio)
                        rule_outcome_pred_curr_batch[stopping_conds.view(-1)] = outcome_pred[stopping_conds.view(-1)].view(-1)
                        rule_pos_outcome_curr_batch[stopping_conds.view(-1)] = pos_outcome[stopping_conds.view(-1)].view(-1)
                        rule_neg_outcome_curr_batch[stopping_conds.view(-1)] = neg_outcome[stopping_conds.view(-1)].view(-1)
                        # reg_pos_outcome_curr_batch[treatment_pred != -1] = reg_full_pred[treatment_pred != -1, 0]
                        # reg_neg_outcome_curr_batch[treatment_pred != -1] = reg_full_pred[treatment_pred != -1, 1]
                    else:
                        if not self.cont_treatment:
                            selected_treatment_pred = treatment_pred[torch.arange(treatment_pred.shape[0]), A.view(-1).cpu().long()]
                            stopping_conds = torch.logical_and(stopping_conds.view(-1), (selected_treatment_pred != -1).view(-1))
                            outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds].view(-1) + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                            rule_outcome_pred_curr_batch[stopping_conds] = outcome_pred[stopping_conds].view(-1)
                        else:
                            stopping_conds = torch.logical_and(stopping_conds.view(-1), (treatment_pred != -1).view(-1))
                            outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds].view(-1) + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                            rule_outcome_pred_curr_batch[stopping_conds] = outcome_pred[stopping_conds].view(-1)

                        
                    
                    if len(program_str) == 0:
                        program_str = next_program_str
                        program_col_ls = next_program_col_ls
                    else:
                        for l_idx in range(len(next_program_str)):
                            if stopping_conds[l_idx]:
                                program_str[l_idx] = next_program_str[l_idx]
                                program_col_ls[l_idx] = next_program_col_ls[l_idx]
                
                empty_res += torch.sum(torch.isnan(outcome_pred_curr_batch))
                outcome_pred_curr_batch[outcome_pred_curr_batch != outcome_pred_curr_batch] = 0
                early_stop_samples += (len(stopping_conds) - torch.sum(stopping_conds))
                total_sample_count += len(stopping_conds)
                all_treatment_ls.append(treatment_pred)
                if compute_ate:
                    empty_pos += torch.sum(torch.isnan(pos_outcome_curr_batch))
                    empty_neg += torch.sum(torch.isnan(neg_outcome_curr_batch))
                    pos_outcome_curr_batch[pos_outcome_curr_batch != pos_outcome_curr_batch] = 0
                    neg_outcome_curr_batch[neg_outcome_curr_batch != neg_outcome_curr_batch] = 0
                    # outcome_pred, pos_outcome, neg_outcome, _ = outcome_pred
                    all_pos_outcome_ls.append(pos_outcome_curr_batch.view(-1))
                    all_neg_outcome_ls.append(neg_outcome_curr_batch.view(-1))
                    all_rule_pos_outcome_ls.append(rule_pos_outcome_curr_batch.view(-1))
                    all_rule_neg_outcome_ls.append(rule_neg_outcome_curr_batch.view(-1))
                    
                all_outcome_ls.append(outcome_pred_curr_batch.view(-1))
                all_rule_outcome_ls.append(rule_outcome_pred_curr_batch.view(-1))
                if compute_ate:
                    all_gt_count_outcome_ls.append(count_Y.view(-1))
                    
                all_program_ls.append(program)
                all_program_str_ls.extend(program_str)
                all_program_col_ls.extend(program_col_ls)
                
                all_gt_treatment_ls.append(A.view(-1))
                all_gt_outcome_ls.append(Y.view(-1))
                
            all_treatment_pred_tensor = torch.cat(all_treatment_ls).cpu()
            all_outcome_pred_tensor = torch.cat(all_outcome_ls).cpu()
            all_rule_outcome_pred_tensor = torch.cat(all_rule_outcome_ls).cpu()
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls).cpu()
            all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls).cpu()
            
            if test_loader.dataset.y_scaler is not None:
                all_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_outcome_pred_tensor)
                all_gt_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_outcome_tensor)
                all_rule_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_rule_outcome_pred_tensor)
            
            if compute_ate:
                all_concat_count_Y_tensor = torch.cat(all_gt_count_outcome_ls)
                all_pos_outcome_tensor = torch.cat(all_pos_outcome_ls)
                all_neg_outcome_tensor = torch.cat(all_neg_outcome_ls)
                all_rule_pos_outcome_tensor = torch.cat(all_rule_pos_outcome_ls)
                all_rule_neg_outcome_tensor = torch.cat(all_rule_neg_outcome_ls)

                if test_loader.dataset.y_scaler is not None:
                    all_concat_count_Y_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_concat_count_Y_tensor)
                    all_pos_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_pos_outcome_tensor)
                    all_neg_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_neg_outcome_tensor)
                    all_rule_pos_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_rule_pos_outcome_tensor)
                    all_rule_neg_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_rule_neg_outcome_tensor)
            
            all_treatment_arr_np = all_gt_treatment_tensor.view(-1).numpy()
            
            

            outcome_error = F.mse_loss(all_outcome_pred_tensor.view(-1,1), all_gt_outcome_tensor.view(-1,1))# torch.sqrt(torch.mean((all_outcome_pred_tensor - all_gt_outcome_tensor)**2))

            assert torch.sum(torch.isnan(outcome_error)) == 0
            print("empty res::%d, empty pos::%d, empty neg::%d, early stopping sample count:%d, early stopping sample ratio:%f"%(empty_res, empty_pos, empty_neg, early_stop_samples, (early_stop_samples*1.0/total_sample_count)))
            
            if not self.cont_treatment:
                if self.num_treatments > 2:
                    all_pred_treatment_arr_full_d = (all_treatment_pred_tensor.argmax(-1)).type(torch.long).view(-1).numpy()
                    treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
                    if len(np.unique(all_treatment_arr_np))==1:
                        treatment_auc = 0
                    else:
                        all_treatment_pred_tensor[all_treatment_pred_tensor == -1] = 1.0/self.num_treatments
                        treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy(), multi_class="ovr")
                elif self.num_treatments == 2:
                    all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()
                    treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
                    if len(np.unique(all_treatment_arr_np)) == 1:
                        treatment_auc = 0
                    else:
                        treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                    # treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                
                print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            # else:
            #     treatment_acc = F.mse_loss(all_treatment_pred_tensor.view(-1,1), torch.from_numpy(all_treatment_arr_np).view(-1,1)).item()

            #     print("treatment error::%f, outcome error::%f"%(treatment_acc, outcome_error))
            print("evaluation outcome error::%f"%(outcome_error))
                
            if compute_ate:
                if all_concat_count_Y_tensor is not None and self.num_treatments == 2 and not self.cont_treatment:
                    gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_tensor.view(-1), all_gt_treatment_tensor.view(-1)], dim=1), all_concat_count_Y_tensor.reshape(-1,1))
                    
                    avg_ite, avg_ate = evaluate_treatment_effect_core(all_pos_outcome_tensor, all_neg_outcome_tensor, gt_treatment_outcome, gt_control_outcome)
                    avg_pehe = evaluate_treatment_effect_core2(all_pos_outcome_tensor, all_neg_outcome_tensor, gt_treatment_outcome, gt_control_outcome)
                    print("average individualized treatment effect::%f"%avg_ite)
                    print("average PEHE::%f"%avg_pehe)
                    if not return_explanations:
                        return avg_ite, avg_ate, outcome_error
                    else:
                        return avg_ite, avg_ate, outcome_error, all_program_ls, all_program_str_ls, all_program_col_ls, all_outcome_pred_tensor, all_rule_outcome_pred_tensor, all_pos_outcome_tensor, all_neg_outcome_tensor, all_rule_pos_outcome_tensor, all_rule_neg_outcome_tensor, gt_treatment_outcome, gt_control_outcome
        if not return_explanations:
            return None, None, outcome_error
        else:
            return None, all_program_ls, all_program_str_ls, all_program_col_ls, all_outcome_pred_tensor, all_rule_outcome_pred_tensor
    
    def test_cont(self, test_loader, small=False):
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            test_dataset = test_loader.dataset
            t_grid = test_dataset.t_grid
            if small:
                rand_sample_ids = torch.randperm(t_grid.shape[1])[0:10]
                t_grid = t_grid[:,rand_sample_ids]
            n_test = t_grid.shape[1]
            t_grid_hat = torch.zeros(2, n_test)
            t_grid_hat[0, :] = t_grid[0, :]

            

            for n in tqdm(range(n_test)):

                all_treatment_ls = []
                all_outcome_ls = []
                all_gt_treatment_ls = []
                all_gt_outcome_ls = []
                for step, batch in enumerate(test_loader):
                    
                        # batch = (x.cuda() for x in batch)
                    # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                    idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                    X =X.to(self.device)
                    Y = Y.to(self.device)

                    A *= 0
                    A += t_grid[0,n]
                    A = A.to(self.device)
                    if D is not None:
                        D = D.to(self.device)
                    # if self.gpu_db:
                    origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]
                    
                    all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                    
                    program = []
                    outbound_mask_ls = []
                    # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                    # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                    program_str = []
                    program_col_ls = []
                    # for p_k in range(len(program_str)):
                    #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                    #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                    
                    
                    X_pd_full = origin_X#pd.concat(X_pd_ls)
                    # all_transformed_expr_ls = []
                    # all_treatment_pred = []
                    # all_outcome_pred = []
                    
                    # prev_reward = torch.zeros([len(A), 2])

                    # if self.dqn.policy_net.backbone_model_name == "transtee":
                    #     X_encode = self.dqn.policy_net.backbone_model(graph_batch)
                    # else:
                    X_encode = self.dqn.policy_net.backbone_model(X)
                    outcome_pred_curr_batch=torch.zeros(len(A))
                    # if self.gpu_db:
                    stopping_conds = torch.ones_like(A).bool().to(self.device)
                    # else:   
                    #     stopping_conds = torch.ones_like(A.cpu()).bool()
                    
                    # if self.gpu_db:
                    outcome_pred_curr_batch = outcome_pred_curr_batch.to(self.device)
                    for arr_idx in range(self.program_max_len):
                        init = (len(program) == 0)
                        done = (arr_idx == self.program_max_len - 1)
                        # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
                        (treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls,_,_), X_encoding = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, X_encode=X_encode, return_encoding=True)
                        reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D)
                        stopping_conds = torch.logical_and(stopping_conds.view(-1), (treatment_pred != -1).view(-1))
                        outcome_pred_curr_batch[stopping_conds] = (outcome_pred[stopping_conds].view(-1) + self.regression_ratio*reg_pred[stopping_conds].view(-1))/(1 + self.regression_ratio)
                        
                        # if done:
                            
                        #     all_outcome_ls.append(outcome_pred.view(-1))
                            
                        program, program_str, program_col_ls, all_other_pats_ls, outbound_mask_ls = next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls
                    all_treatment_ls.append(treatment_pred.view(-1))
                    all_outcome_ls.append(outcome_pred_curr_batch.view(-1))
                    all_gt_treatment_ls.append(A.cpu().view(-1))
                    all_gt_outcome_ls.append(Y.cpu().view(-1))
                
                all_treatment_pred_tensor = torch.cat(all_treatment_ls)
                all_outcome_pred_tensor = torch.cat(all_outcome_ls)
                all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
                all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
                t_grid_hat[1,n] = all_outcome_pred_tensor.mean()
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data

            print("outcome loss::%f"%(mse))
            # all_treatment_arr_np = all_gt_treatment_tensor.view(-1).numpy()
            
            # all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()

            # outcome_error = F.mse_loss(all_outcome_pred_tensor.view(-1,1), all_gt_outcome_tensor.view(-1,1))# torch.sqrt(torch.mean((all_outcome_pred_tensor - all_gt_outcome_tensor)**2))

            # assert torch.sum(torch.isnan(outcome_error)) == 0

            # if not self.cont_treatment:
            #     treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
            #     if len(np.unique(all_treatment_arr_np)) <= 1:
            #         treatment_auc = 0
            #     else:
                    
            #         if self.num_treatments > 2:
            #             treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy(), multi_class="ovr")
            #         elif self.num_treatments == 2:
            #             treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
            #         # treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                
            #     print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            # else:
            #     treatment_acc = F.mse_loss(all_treatment_pred_tensor.view(-1,1), torch.from_numpy(all_treatment_arr_np).view(-1,1)).item()

            #     print("treatment error::%f, outcome error::%f"%(treatment_acc, outcome_error))
        return mse
    # def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5, epsilon=0, use_mlm=True):
          
    def run(self, train_dataset, valid_dataset, test_dataset):
        ''' Train the model'''

        # split data into two parts: one for training and the other for validation
        # idx = list(range(len(texts)))
        # random.shuffle(idx) # shuffle the index
        # n_train = int(len(texts)*0.8) 
        # n_val = len(texts)-n_train
        # idx_train = idx[0:n_train]
        # idx_val = idx[n_train:]

        # list of data
        # train_dataloader = self.build_dataloader(texts[idx_train], 
        #     treatments[idx_train], confounders[idx_train], outcomes[idx_train])
        # val_dataloader = self.build_dataloader(texts[idx_val], 
        #     treatments[idx_val], confounders[idx_val], outcomes[idx_val], sampler='sequential')
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=tabular_Dataset.collate_fn, drop_last=True)
        val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        

        # self.model.train() 
        # optimizer = AdamW(self.model.parameters(), lr = self.learning_rate, eps=1e-8)

        best_loss = 1e6
        epochs_no_improve = 0
        best_valid_ite = 1e6
        best_test_ite = 1e6
        best_train_ite = 1e6
        best_train_ate=1e6  
        best_valid_ate = 1e6
        best_test_ate=1e6
        best_val_outcome_error = 1e6
        best_test_outcome_error = 1e6
        best_train_outcome_error = 1e6
        best_performance_epoch=0
        print("evaluation on training set::")
        # self.test(train_dataloader)
        
        # print("evaluation on validation set::")
        self.test(test_dataloader)
        
        if self.is_log:
            print("evaluation on test set::") 
            # self.eval_emptiness(test_dataloader)
            
            if self.args.dataset_name == "synthetic":
            
                # train_sample_id_rule_feat_id_mapping_file = os.path.join(self.data_folder, self.args.dataset_name, "train_sample_feat_mappings")
                # valid_sample_id_rule_feat_id_mapping_file = os.path.join(self.data_folder, self.args.dataset_name, "valid_sample_feat_mappings")
                test_sample_id_rule_feat_id_mapping_file = os.path.join(self.data_folder, self.args.dataset_name, "test_sample_feat_mappings")
                
                # with open(train_sample_id_rule_feat_id_mapping_file, "rb") as f:
                #     train_sample_id_rule_feature_ids_mappings = pickle.load(f)
                
                # with open(valid_sample_id_rule_feat_id_mapping_file, "rb") as f:
                #     valid_sample_id_rule_feature_ids_mapping = pickle.load(f)
                
                with open(test_sample_id_rule_feat_id_mapping_file, "rb") as f:
                    test_sample_id_rule_feature_ids_mappings = pickle.load(f)

                unique_feat_ids=set()
                [unique_feat_ids.update(items) for items in test_sample_id_rule_feature_ids_mappings.values()]


            if self.train_dataset.count_outcome_attr is not None:
                # avg_treatment_effect, avg_ite, outcome_error, all_program_ls, all_program_str_ls, all_program_col_ls
                _,_,_, _, origin_explanation_str_ls,origin_explanation_col_ls, all_outcome_pred_tensor, all_rule_outcome_pred_tensor, all_pos_outcome_tensor, all_neg_outcome_tensor,all_rule_pos_outcome_tensor, all_rule_neg_outcome_tensor, all_gt_pos_outcome, all_gt_neg_outcome = self.test(test_dataloader, return_explanations=True)
                self.eval_sufficiency(test_dataloader, all_pos_outcome_tensor-all_neg_outcome_tensor, origin_explanation_str_ls)
            else:
                _,_, origin_explanation_str_ls,origin_explanation_col_ls, all_outcome_pred_tensor, all_rule_outcome_pred_tensor= self.test(test_dataloader, return_explanations=True)
                self.eval_sufficiency(test_dataloader, all_outcome_pred_tensor, origin_explanation_str_ls)
            

            if self.train_dataset.count_outcome_attr is not None:
                eval_consistency(self, test_dataloader, all_pos_outcome_tensor - all_neg_outcome_tensor, None, origin_explanation_str_ls, explanation_type="ours", out_folder=self.log_folder)
            else:
                eval_consistency(self, test_dataloader, all_outcome_pred_tensor, None, origin_explanation_str_ls, explanation_type="ours", out_folder=self.log_folder)

            with open(os.path.join(self.log_folder, "pred_treatment_effect.json"), "w") as f:
                pred_treatment_effect = (all_pos_outcome_tensor - all_neg_outcome_tensor).tolist()
                pred_treatment_effect_maps = {idx: pred_treatment_effect[idx] for idx in range(len(pred_treatment_effect))}
                json.dump(pred_treatment_effect_maps, f, indent=4)
            with open(os.path.join(self.log_folder, "rule_pred_treatment_effect.json"), "w") as f:
                pred_treatment_effect = (all_rule_pos_outcome_tensor - all_rule_neg_outcome_tensor).tolist()
                pred_treatment_effect_maps = {idx: pred_treatment_effect[idx] for idx in range(len(pred_treatment_effect))}
                json.dump(pred_treatment_effect_maps, f, indent=4)
            with open(os.path.join(self.log_folder, "gt_treatment_effect.json"), "w") as f:
                gt_treatment_effect = (all_gt_pos_outcome - all_gt_neg_outcome).tolist()
                gt_treatment_effect_maps = {idx: gt_treatment_effect[idx] for idx in range(len(gt_treatment_effect))}
                json.dump(gt_treatment_effect_maps, f, indent=4)
            
            if self.args.dataset_name == "synthetic":
                exp_exp_col_ls = list(test_sample_id_rule_feature_ids_mappings.values())
                e_count = 0
                for eidx in range(len(origin_explanation_col_ls)):
                    exp_cols = origin_explanation_col_ls[eidx][0]
                    exp_cols = [int(e.split("_")[1]) for e in exp_cols]
                    exp_exp_cols = set(exp_exp_col_ls[eidx])
                    diff_cols = exp_exp_cols.difference(set(exp_cols))
                    if len(diff_cols) == 0:
                       e_count += 1
                    else:
                        print()
                print(e_count) 
                    
                    
            # self.eval_aopc(test_dataloader)
            # if self.dataset_name == "synthetic":
                
            #     self.eval_faithfulness(test_dataloader)
            # self.eval_stability(test_dataloader, origin_explanation_str_ls, origin_explanation_col_ls)
            # if self.cont_treatment:
            #     self.test_cont(test_dataloader)
                
            # if self.cat_and_cont_treatment:
            #     mise, dpe, pe, ate = self.compute_eval_metrics(test_dataset.metainfo, test_dataset, self.num_treatments)
            #     print("Mise: %s" % str(mise))
            #     print("DPE: %s" % str(dpe))
            #     print("PE: %s" % str(pe))
            #     print("ATE: %s" % str(ate))
            exit(1)
        
        if self.args.eval and self.cat_and_cont_treatment:
                self.dqn.policy_net.load_state_dict(torch.load(os.path.join(self.log_folder, 'model_best')))

                mise, dpe, pe, ate = self.compute_eval_metrics(test_dataset.metainfo, test_dataset, self.num_treatments)
                print("Mise: %s" % str(mise))
                print("DPE: %s" % str(dpe))
                print("PE: %s" % str(pe))
                print("ATE: %s" % str(ate))
                

                mise, dpe, pe, ate = self.compute_eval_metrics(train_dataset.metainfo, train_dataset, self.num_treatments)
                print("Train Mise: %s" % str(mise))
                print("Train DPE: %s" % str(dpe))
                print("Train PE: %s" % str(pe))
                print("Train ATE: %s" % str(ate))
                exit(1)
                

        for epoch in range(self.epochs):
            losses = []
            self.dqn.policy_net.train()
            self.dqn.target_net.train()
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            all_treatment_pred = []
            all_outcome_pred = []
            all_treatment_gt = []
            all_outcome_gt = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                D_cpu = D
                if D is not None:
                    D = D.to(self.device)
                # D = None
                # if D is not None:
                #     D = D.to(self.device)
                # text_id_ls = text_id_ls.to(self.device)
                # text_mask_ls = text_mask_ls.to(self.device)
                # text_len_ls = text_len_ls.to(self.device)
                # Y = Y.to(self.device)
                # A = A.to(self.device)
                # if self.gpu_db:
                origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls)
                
                program = []
                outbound_mask_ls = []
                # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_str = []
                program_col_ls = []
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = origin_X#pd.concat(X_pd_ls)
                all_transformed_expr_ls = []
                
                
                prev_reward = torch.zeros([len(A), self.topk_act, 2])
                
                for arr_idx in range(self.program_max_len):
                    init = (len(program) == 0)
                    done = (arr_idx == self.program_max_len - 1)
                    # treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls
                    (treatment_pred, outcome_pred, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred), X_encoding = self.dqn.policy_net.forward(self, X, A, D, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=self.epsilon, eval=False, init=init, method_two=self.method_two, method_three=self.method_three, return_encoding=True)
                    if self.dqn.policy_net.backbone_full_model is not None:
                        reg_pred, reg_full_pred = self.obtain_regularized_outcome(X_encoding, A, D, test=False)
                    else:
                        reg_pred = 0
                        reg_full_pred = 0
                    ind_outcome_pred = (ind_outcome_pred + self.regression_ratio*reg_pred)/(1 + self.regression_ratio)
                    reward1, reward2 = self.obtain_reward(ind_treatment_pred, ind_outcome_pred, A, Y, epoch)
                    if done:
                        next_state = None
                    else:
                        next_state = (next_program, next_outbound_mask_ls)
                    
                    # namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))
                    full_reward = torch.stack([reward1, reward2], dim=-1).cpu()
                    transition = Transition((X.cpu(), A.cpu(), Y.cpu(), D_cpu), X_pd_full,(program, outbound_mask_ls), atom_ls, next_state, full_reward - prev_reward)
                    self.observe_transition(transition)
                    #update model
                    loss = self.dqn.optimize_model_ls0()
                    
                    if done:
                        all_treatment_pred.append(treatment_pred)
                        all_outcome_pred.append(outcome_pred)
                        all_treatment_gt.append(A.cpu())
                        all_outcome_gt.append(Y.cpu())
                    program = next_program
                    program_str = next_program_str
                    program_col_ls = next_program_col_ls
                    all_other_pats_ls = next_all_other_pats_ls
                    outbound_mask_ls = next_outbound_mask_ls
                    prev_reward = full_reward
                
                losses.append(loss)
            if epoch % self.target_update == 0:
                self.dqn.update_target()
            print("Epoch %d Training loss: %f"%(epoch, np.mean(np.array(losses)).item()))
            torch.cuda.empty_cache()
            all_treatment_pred_array = torch.cat(all_treatment_pred).cpu().numpy()
            all_treatment_gt_array = torch.cat(all_treatment_gt).cpu().numpy()
            if not self.cont_treatment:
                
                if self.num_treatments == 2:
                    training_acc = np.mean((all_treatment_pred_array > 0.5).reshape(-1).astype(float) == all_treatment_gt_array.reshape(-1))
                    if len(np.unique(all_treatment_gt_array)) == 1:
                        training_auc = 0
                    else:
                        training_auc = roc_auc_score(all_treatment_gt_array.reshape(-1), all_treatment_pred_array.reshape(-1))
                elif self.num_treatments > 2:
                    training_acc = np.mean((all_treatment_pred_array.argmax(-1)).reshape(-1).astype(float) == all_treatment_gt_array.reshape(-1))
                    if len(np.unique(all_treatment_gt_array)) == 1:
                        training_auc = 0
                    else:
                        all_treatment_pred_array[all_treatment_pred_array == -1] = 1.0/self.num_treatments
                        training_auc = roc_auc_score(all_treatment_gt_array.reshape(-1), all_treatment_pred_array, multi_class="ovr")
                    
                print("training auc score::", training_auc)
                print("training accuracy::", training_acc)

            # else:
                
            #     training_acc = F.mse_loss(torch.from_numpy(all_treatment_pred_array).view(-1,1),torch.from_numpy(all_treatment_gt_array).view(-1,1)).item()
            
            #     print("training errors::", training_acc)

            all_outcome_pred_array = torch.cat(all_outcome_pred).cpu()
            all_outcome_gt_array = torch.cat(all_outcome_gt).cpu()
            
            
            # self.reward_coeff = -torch.log(torch.tensor(0.1))/torch.max((all_outcome_pred_array.view(-1) - all_outcome_gt_array.view(-1))**2)
            if not self.no_hyper_adj:
                self.reward_coeff = -torch.log(torch.tensor(0.5))/torch.median((all_outcome_pred_array.view(-1) - all_outcome_gt_array.view(-1))**2)
            if self.cont_treatment:
                self.treatment_coeff = -torch.log(torch.tensor(0.5))/torch.median((torch.from_numpy(all_treatment_pred_array).view(-1) - torch.from_numpy(all_treatment_gt_array).view(-1))**2)
            
            if train_dataset.y_scaler is not None:
                all_outcome_pred_array = transform_outcome_by_rescale_back(train_dataset, all_outcome_pred_array)
                all_outcome_gt_array = transform_outcome_by_rescale_back(train_dataset, all_outcome_gt_array)
            # print("evaluation on training set::")
            # self.test(train_dataloader)
            print("training errors::", F.mse_loss(all_outcome_pred_array.view(-1,1), all_outcome_gt_array.view(-1,1)).item()) 
            print("update reward coefficient::", self.reward_coeff)
            torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.log_folder, 'model_' + str(epoch))) # save the best model
            print("evaluation on validation set::")
            val_ite, val_ate, val_outcome_err = self.test(val_dataloader)
            
            print("evaluation on test set::")
            test_ite, test_ate, test_outcome_err = self.test(test_dataloader)

            # if self.cat_and_cont_treatment:
            #     self.compute_eval_metrics(test_dataset.metainfo, test_dataloader, self.num_treatments)
            if val_ite is not None and test_ite is not None:
                train_ite, train_ate, train_outcome_err = self.test(train_dataloader)
                if val_outcome_err < best_val_outcome_error:
                    best_val_outcome_error = val_outcome_err
                    best_test_outcome_error = test_outcome_err
                    best_train_outcome_error = train_outcome_err
                    best_valid_ite = val_ite
                    best_test_ite = test_ite
                    best_train_ite = train_ite
                    best_train_ate = train_ate
                    best_valid_ate = val_ate
                    best_test_ate = test_ate
                    best_performance_epoch=epoch
                    torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.log_folder, 'model_best')) # save the best model
                print("best performance at epoch %d"%(best_performance_epoch))
                
                
                print("best train outcome error::", best_train_outcome_error)
                print("best valid outcome error::", best_val_outcome_error)
                print("best test outcome error::", best_test_outcome_error)
                
                print("best train ate::", best_train_ate)
                print("best valid ate::", best_valid_ate)
                print("best test ate::", best_test_ate)
                
                
                print("best train ite::", best_train_ite)
                print("best valid ite::", best_valid_ite)
                print("best test ite::", best_test_ite)
                
            # elif self.cat_and_cont_treatment:
            else:
                if val_outcome_err < best_val_outcome_error:
                    best_val_outcome_error = val_outcome_err
                    best_test_outcome_error = test_outcome_err
                    best_performance_epoch=epoch
                    torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.log_folder, 'model_best')) # save the best model
                print("best performance at epoch %d"%(best_performance_epoch))                
                print("best valid outcome error::", best_val_outcome_error)
                print("best test outcome error::", best_test_outcome_error)


            if self.cont_treatment and epoch % 100 == 99:
                if not self.dataset_name.startswith("news"):
                    cont_val_res = self.test_cont(val_dataloader)
                    cont_test_res = self.test_cont(test_dataloader)
                    if cont_val_res < best_valid_ite:
                        best_valid_ite = cont_val_res
                        best_test_ite = cont_test_res
                    print("best test error::", best_test_ite)
                    
                else:
                    self.test_cont(test_dataloader, small=True)
            
            
            self.epsilon *= self.epsilon_falloff

        self.dqn.policy_net.load_state_dict(torch.load(os.path.join(self.log_folder, 'model_best')))
        if self.cont_treatment:
            
            self.test_cont(test_dataloader)

        if self.cat_and_cont_treatment:
            # torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.log_folder, 'model_best'))

            mise, dpe, pe, ate = self.compute_eval_metrics(train_dataset.metainfo, train_dataset, self.num_treatments)
            print("Train Mise: %s" % str(mise))
            print("Train DPE: %s" % str(dpe))
            print("Train PE: %s" % str(pe))
            print("Train ATE: %s" % str(ate))
            
            mise, dpe, pe, ate = self.compute_eval_metrics(test_dataset.metainfo, test_dataset, self.num_treatments)
            print("Mise: %s" % str(mise))
            print("DPE: %s" % str(dpe))
            print("PE: %s" % str(pe))
            print("ATE: %s" % str(ate))
    

class baseline_trainer:
    def init_model(self, args, method, model_configs, input_size, train_dataset, reg_model=False):
        if method.lower() == "dragonnet":
            model = dragonet_model(input_size, model_configs["hidden_size"])
            self.shared_hidden_dim = model_configs["hidden_size"]
        elif method.lower() == "transtee":
            # if not args.dataset_name == "sw" and not args.dataset_name == "tcga2":
            if not args.cont_treatment and args.has_dose:
                cov_dim = model_configs["cov_dim"]
            else:
                cov_dim = input_size
            params = {'num_features': input_size, 'num_treatments': args.num_treatments,
            'h_dim': model_configs["hidden_size"], 'cov_dim':cov_dim}
            model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
            self.shared_hidden_dim = model_configs["hidden_size"]            

                
        elif method.lower() == "drnet":
            cfg_density = [(input_size, 100, 1, 'relu'), (100, 64, 1, 'relu')]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=model_configs["h"], num_t=args.num_treatments, has_dose=args.has_dose, cont_treatment=args.cont_treatment)
        elif method.lower() == "tarnet":
            cfg_density = [(input_size, 100, 1, 'relu'), (100, 64, 1, 'relu')]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=model_configs["h"], num_t=args.num_treatments, has_dose=args.has_dose, cont_treatment=args.cont_treatment)
        elif method.lower() == "vcnet":
            cfg_density = [(input_size, 100, 1, 'relu'), (100, 64, 1, 'relu')]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots, num_t=args.num_treatments, has_dose=args.has_dose, cont_treatment=args.cont_treatment)
        elif method.lower() == "nam":
            model = dragonet_model_nam(input_size, model_configs["hidden_size"], cont_treatment=args.cont_treatment, has_dose=args.has_dose)
            self.shared_hidden_dim = model_configs["hidden_size"]
        
        
        elif method.lower() == "attention":
            n_nns_per_feature=1
            attention_type = "simple"
            pairwise_fn = "mul"
            # feature_nns = [
            #     make_feature_nn()
            #     for _i in range(input_size * n_nns_per_feature)
            # ]
            model = AttentionExplanationModel(
                    n_samples=len(train_dataset),
                    n_attentions=1,
                    feature_nns=None,
                    target_nns=None,
                    n_nns_per_feature=n_nns_per_feature,
                    attention_type=attention_type,
                    pairwise_fn=pairwise_fn
                )
        elif method.lower() == "enrl":
            if train_dataset.feat_range_mappings is not None:
                feat_range_mappings = [train_dataset.feat_range_mappings[num_feat] for num_feat in train_dataset.num_cols]
            else:
                feat_range_mappings = [[0,1] for num_feat in train_dataset.num_cols]
            multihot_f_unique_num_ls = [len(train_dataset.cat_unique_count_mappings[cat_feat]) for cat_feat in train_dataset.cat_cols]
            
            model = ENRL(rule_len = args.program_max_len, rule_n = args.topk_act, numeric_f_num=len(train_dataset.num_cols), multihot_f_num=len(train_dataset.cat_cols), multihot_f_dim=sum(multihot_f_unique_num_ls), feat_range_mappings=feat_range_mappings, num_treatment=args.num_treatments, cont_treatment=args.cont_treatment, has_dose=args.has_dose)

        elif method.lower() == "prototype":
            model_configs["num_prototypes"] = 10
            model = ProtoVAE_tab(input_size, model_configs["hidden_size"], model_configs["num_prototypes"], args.num_treatments)

        if not reg_model:
            self.model = model
        else:
            self.reg_model = model

    def __init__(self, args, train_dataset, input_size, model_configs, backbone_model_config, device, outcome_regression=True):
        self.batch_size = args.batch_size
        self.method = args.method

        self.init_model(args, args.method, model_configs, input_size, train_dataset, reg_model=False)
        # if args.reg_method is not None:
        #     self.init_model(args, args.reg_method, model_configs, input_size, train_dataset, reg_model=True)
        # else:
        #     self.reg_model = None
        self.backbone_full_model = None
        self.backbone = args.backbone
        if args.backbone is not None:
            if args.backbone.lower() == "transtee":
                print("start loading transtee backbone")
                if not args.cont_treatment and args.has_dose:
                    cov_dim = backbone_model_config["cov_dim"]
                else:
                    cov_dim = input_size
                
                params = {'num_features': input_size, 'num_treatments': args.num_treatments,
                'h_dim': backbone_model_config["hidden_size"], 'cov_dim':cov_dim}
                # self.model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                self.shared_hidden_dim = backbone_model_config["hidden_size"]
                self.backbone_full_model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                # self.backbone_model = self.backbone_full_model.encoding_features
        
        if args.cached_backbone is not None and os.path.exists(args.cached_backbone):
            cached_backbone_state_dict = torch.load(args.cached_backbone)
            self.backbone_full_model.load_state_dict(cached_backbone_state_dict)
            if args.fix_backbone:
                for p in self.backbone_full_model.parameters():
                    p.requires_grad = False

        self.regression_ratio = args.regression_ratio
        
        if args.tr:
            if not args.method.lower() == "dragonnet":
                self.targetReg = DisCri(self.shared_hidden_dim, 50, args.num_treatments)
            else:
                self.targetReg = nn.Linear(model_configs["hidden_size"], 1)
            self.targetReg.cuda()
            tr_init_lr = 0.001
            tr_wd = 5e-3
            self.tr_optimizer = torch.optim.SGD(self.targetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
        self.tr = args.tr
        self.model.to(device)
        if self.backbone_full_model is not None:
            self.backbone_full_model = self.backbone_full_model.to(device)
        self.epochs = args.epochs
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.lr = args.lr
        self.alpha = args.alpha
        self.p = args.p
        self.num_treatments = args.num_treatments
        self.cont_treatment = args.cont_treatment
        self.outcome_regression = outcome_regression
        self.log_folder = args.log_folder
        self.data_folder, self.dataset_id = args.data_folder, args.dataset_id
        self.cat_and_cont_treatment = args.cat_and_cont_treatment
        self.has_dose = args.has_dose
        self.classification = not outcome_regression
        self.train_dataset =train_dataset
        
        # epislon layer not implemented here

    def initialize_model_parameters(self):
        for param in self.model.parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    # For weight parameters (e.g., linear layer weights), you can use torch.nn.init
                    nn.init.xavier_uniform_(param)
                else:
                    # For bias parameters, you can initialize them to zeros or any other suitable value
                    nn.init.zeros_(param)
    
    def regression_loss(self,concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]

        loss0 = torch.sqrt(torch.mean((1 - t_true) * torch.square(y_true - y0_pred)))
        loss1 = torch.sqrt(torch.mean(t_true * torch.square(y_true - y1_pred)))
        return loss0 + loss1

    def binary_classification_loss(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        t_pred = concat_pred[:, 2]
        t_pred = (t_pred + 0.001) / 1.002
        loss = torch.nn.functional.binary_cross_entropy(t_pred, t_true)
        return loss
    
    def treatment_pred_gt_comparison(self, concat_true, concat_pred):
        all_treatment_arr_np = concat_true.cpu().numpy()
        if self.num_treatments == 2:
            all_treatment_pred_tensor = concat_pred#[:, 2]
            all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()
            treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
        else:
            all_treatment_pred_tensor = concat_pred#[:, 2]
            all_pred_treatment_arr_full_d = torch.argmax(concat_pred, dim = -1)
            all_pred_treatment_arr_full_d = all_pred_treatment_arr_full_d.view(-1).numpy()
            treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
            
        if len(np.unique(all_treatment_arr_np)) <= 1:
            treatment_auc = 0
        else:
            if self.num_treatments > 2:
                if len(np.unique(all_treatment_arr_np)) == 1:
                    treatment_auc = 0
                else:
                    treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy(), multi_class="ovr")
            elif self.num_treatments == 2:
                if len(np.unique(all_treatment_arr_np)) == 1:
                    treatment_auc = 0
                else:
                    treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
        return treatment_acc, treatment_auc

    def dragonet_loss(self, concat_true, concat_pred):
        return self.regression_loss(concat_true, concat_pred) + self.binary_classification_loss(concat_true, concat_pred)

    def test_cont(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            # pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            
            test_dataset = test_loader.dataset
            t_grid = test_dataset.t_grid
            n_test = t_grid.shape[1]
            t_grid_hat = torch.zeros(2, n_test)
            t_grid_hat[0, :] = t_grid[0, :]

            

            for n in tqdm(range(n_test)):
                all_treatment_ls = []
                all_outcome_ls = []
                all_gt_treatment_ls = []
                all_gt_outcome_ls = []
                for step, batch in enumerate(test_loader):
                    
                        # batch = (x.cuda() for x in batch)
                    # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                    idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                    X =X.to(self.device)
                    Y = Y.to(self.device)

                    A *= 0
                    A += t_grid[0,n]
                    A = A.to(self.device)
                    if D is not None:
                        D = D.to(self.device)
                
                    if self.method == "ENRL":
                        X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                        X_num = X_num.to(self.device)
                        X_cat = X_cat.to(self.device)
                        pred, other_loss = self.model(X_num, X_cat, A, d=D)
                    else:
                
                        if not self.cont_treatment:
                            embedding, pred, full_pred = self.model(X, A, d=D, test=True)
                        else:
                            embedding, pred = self.model(X, A, d=D, test=True)
                    
                    if self.tr:
                        pred_treatment = self.targetReg(embedding.view(len(X), -1))
                        if self.num_treatments > 2:
                            pred_treatment = torch.softmax(pred_treatment, dim=1)
                        elif self.num_treatments == 1:
                            pred_treatment = torch.sigmoid(pred_treatment)
                        # else:
                        #     pred_treatment = pred_treatment
                        all_treatment_ls.append(pred_treatment.view(-1))
                    # all_treatment_ls.append(treatment_pred.view(-1))
                    all_outcome_ls.append(pred.view(-1))
                    
                    all_gt_treatment_ls.append(A.cpu().view(-1))
                    all_gt_outcome_ls.append(Y.cpu().view(-1))
                
                # all_treatment_pred_tensor = torch.cat(all_treatment_ls)
                all_outcome_pred_tensor = torch.cat(all_outcome_ls)
                # all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
                # all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
                t_grid_hat[1,n] = all_outcome_pred_tensor.mean()
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data

            print("outcome loss::%f"%(mse))

    def test(self, test_loader, cohort=False, y_scaler=None, return_pred=False):
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples
        if y_scaler is None:
            y_scaler = self.train_dataset.y_scaler
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            all_pred_outcome = []
            all_gt_outcome = []
            all_count_gt_outcome = []
            all_full_pred_outcome = []
            all_pred_treatment = []
            all_gt_treatment = []
            # all_concat_true = []
            # all_concat_pred = []
            # all_concat_count_Y = []
            for step, batch in pbar:
                
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A, count_Y, D = batch
                if not cohort:
                    idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                else:
                    if len(batch) == 3:
                        X, A, Y = batch
                        D = None
                    elif len(batch) == 4:
                        X, A, Y, D = batch
                    count_Y = None
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if D is not None:
                    D = D.to(self.device)
                if not self.method == "ENRL":
                    if not self.cont_treatment:
                        
                        embedding, pred, full_pred = self.model(X, A, d=D, test=True)
                            
                    else:
                        embedding, pred = self.model(X, A, d=D, test=True)
                else:
                    X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                    X_num = X_num.to(self.device)
                    X_cat = X_cat.to(self.device)
                    pred, full_pred = self.model(X_num, X_cat, A, d=D)
                    pred = pred.view(len(X), -1)
                    if full_pred is not None:
                        full_pred = full_pred.unsqueeze(-1)
                if self.backbone_full_model is not None:
                    if not self.cont_treatment:
                        pred, full_pred = reg_model_forward(self, X, A, D, origin_X, pred, test=True, full_out=full_pred)                    
                    else:
                        pred  = reg_model_forward(self, X, A, D, origin_X, pred, test=True)                    

                
                if self.tr:
                    pred_treatment = self.targetReg(embedding.view(len(X), -1))
                    if self.num_treatments > 2:
                        pred_treatment = torch.softmax(pred_treatment, dim=1)
                    elif self.num_treatments == 1:
                        pred_treatment = torch.sigmoid(pred_treatment)
                    # else:
                    #     pred_treatment = pred_treatment
                    all_pred_treatment.append(pred_treatment)
                all_gt_treatment.append(A)
                all_pred_outcome.append(pred.view(len(X), -1))
                all_gt_outcome.append(Y)
                if not self.cont_treatment:
                    all_full_pred_outcome.append(full_pred.view(len(X), -1))
                all_count_gt_outcome.append(count_Y)
                
                
                # concat_true = torch.cat([Y.view(len(Y), 1), A.view(len(Y), 1)], dim=1)
                # all_concat_true.append(concat_true.detach().cpu())
                # all_concat_pred.append(concat_pred.detach().cpu())
                # all_concat_count_Y.append(count_Y)


            # all_concat_true_tensor = torch.cat(all_concat_true)
            # all_concat_pred_tensor = torch.cat(all_concat_pred)
            
            if not self.cont_treatment:
                all_full_pred_outcome_tensor = torch.cat(all_full_pred_outcome).cpu()
            all_pred_outcome_tensor = torch.cat(all_pred_outcome).cpu()
            all_gt_outcome_tensor = torch.cat(all_gt_outcome).cpu()
            if self.tr:
                all_pred_treatment_tensor = torch.cat(all_pred_treatment).cpu()
            all_gt_treatment_tensor = torch.cat(all_gt_treatment).cpu()
            
            if y_scaler is not None:
                all_gt_outcome_tensor = transform_outcome_by_rescale_back0(y_scaler, all_gt_outcome_tensor)
                all_pred_outcome_tensor = transform_outcome_by_rescale_back0(y_scaler, all_pred_outcome_tensor)
                if not self.cont_treatment:
                    all_full_pred_outcome_tensor = torch.cat([transform_outcome_by_rescale_back0(y_scaler, all_full_pred_outcome_tensor[:,k]) for k in range(all_full_pred_outcome_tensor.shape[1])], dim=-1)
            
            all_concat_count_Y_tensor = None
            if all_count_gt_outcome[0] is not None:
                all_concat_count_Y_tensor = torch.cat(all_count_gt_outcome)
                if y_scaler is not None:
                    all_concat_count_Y_tensor = transform_outcome_by_rescale_back0(y_scaler, all_concat_count_Y_tensor)
            # regression_loss = self.regression_loss(all_concat_true_tensor, all_concat_pred_tensor)
                
            # mise = romb(np.square(all_gt_outcome_tensor.cpu().numpy() - all_pred_outcome_tensor.cpu().numpy()), dx=step_size)
            # inter_r = np.array(all_gt_outcome_tensor.numpy()) - all_pred_outcome_tensor.squeeze().numpy()
            # ite = np.mean(inter_r ** 2)
            if self.outcome_regression:
                outcome_error = F.mse_loss(all_gt_outcome_tensor.view(-1, 1).cpu(), all_pred_outcome_tensor.view(-1, 1).cpu()).item()
            else:
                outcome_error = F.binary_cross_entropy_with_logits(all_gt_outcome_tensor.view(-1, 1).cpu(), all_pred_outcome_tensor.view(-1, 1).cpu()).item()
                all_pred_outcome_tensor_probs = torch.sigmoid(all_pred_outcome_tensor)
                outcome_acc = np.mean((all_pred_outcome_tensor_probs > 0.5).view(-1).numpy() == all_gt_outcome_tensor.view(-1).numpy())
                if len(torch.unique(all_gt_outcome_tensor)) <= 1:
                    outcome_auc = 0
                else:
                    if len(torch.unique(all_gt_outcome_tensor)) == 1:
                        outcome_auc = 0
                    else:
                        outcome_auc = roc_auc_score(all_gt_outcome_tensor.view(-1).numpy(), all_pred_outcome_tensor_probs.view(-1).numpy())
                print("outcome accuracy::%f, outcome auc score::%f"%(outcome_acc, outcome_auc))
                
            if self.tr:
                treatment_acc, treatment_auc = self.treatment_pred_gt_comparison(all_gt_treatment_tensor, all_pred_treatment_tensor)
            
            
            if all_concat_count_Y_tensor is not None and self.num_treatments == 2 and not self.cont_treatment:
                gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.cat([all_gt_outcome_tensor, all_gt_treatment_tensor], dim=1), all_concat_count_Y_tensor)
                
                avg_ite, avg_ate = evaluate_treatment_effect_core(all_full_pred_outcome_tensor[:,1], all_full_pred_outcome_tensor[:,0], gt_treatment_outcome, gt_control_outcome)
                avg_pehe = evaluate_treatment_effect_core2(all_full_pred_outcome_tensor[:,1], all_full_pred_outcome_tensor[:,0], gt_treatment_outcome, gt_control_outcome)
                print("average individual treatment effect::%f"%avg_ite)
                print("average treatment effect::%f"%avg_ate)
                print("average pehe::", avg_pehe)
                if not return_pred:
                    return outcome_error, avg_ite, avg_ate, avg_pehe
                else:
                    return all_pred_outcome_tensor, all_full_pred_outcome_tensor[:,1], all_full_pred_outcome_tensor[:,0], gt_treatment_outcome, gt_control_outcome
            if self.tr:
                print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            else:
                print("outcome error::%f"%outcome_error)
        if not return_pred:
            return outcome_error
        else:
            return all_pred_outcome_tensor            

    def fit_decision_tree(self, train_dataset, max_depth):
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),desc='Testing')

            compute_ate = False
            
            all_feat_ls = []
            all_outcome_ls_by_treatment = dict()
            all_treatment_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
                _, origin_outcome_pred = self.model.forward(X, A, d=D)
                if train_dataset.y_scaler is not None:
                    origin_outcome_pred = transform_outcome_by_rescale_back(train_dataset, origin_outcome_pred.cpu())

                for treatment_id in range(self.num_treatments):
                    _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id, d=D)
                    if train_dataset.y_scaler is not None:
                        curr_outcome_pred = transform_outcome_by_rescale_back(train_dataset, curr_outcome_pred)

                    if treatment_id not in all_outcome_ls_by_treatment:
                        all_outcome_ls_by_treatment[treatment_id] = []
                    all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
                all_treatment_ls.append(A.cpu().view(-1))
                all_feat_ls.append(X.cpu())

            
            all_feat_tensor = torch.cat(all_feat_ls, dim=0)
            all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}

            all_tree_by_treatment_id = dict()
            for treatment_id in range(self.num_treatments):
                tree = DecisionTreeRegressor(max_depth=max_depth)
                tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
                all_tree_by_treatment_id[treatment_id] = tree

        return all_tree_by_treatment_id

    def eval_self_interpretable_models(self, test_dataset, train_dataset, max_depth, subset_count = 10, all_selected_feat_ls = None, explanation_type="decision_tree"):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        full_base_X = train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        # all_tree_by_treatment_id = self.fit_decision_tree(train_dataset, max_depth)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_feat_ls = []
            
            all_treatment_ls = []
            all_gt_outcome_ls= []
            all_orig_outcome_ls = []
            if not self.has_dose and not self.cont_treatment:
                all_outcome_ls_by_treatment = dict()
                all_topk_features_by_treatment = {k:[] for k in range(self.num_treatments)}
            else:
                all_outcome_ls_by_treatment = []
                all_topk_features_by_treatment =  []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)

                # X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                # X_num = X_num.to(self.device)
                # X_cat = X_cat.to(self.device)
                # pred, _ = self.model.forward(X_num, X_cat, A, d=D)
                
                _, pred = self.model.forward(X, A, d=D)
                
                if not self.cont_treatment and not self.has_dose:
                    for k in range(self.num_treatments):
                        topk_feature_ids = self.model.get_topk_features(X, torch.ones_like(A)*k, d=D, k=max_depth)
                        all_topk_features_by_treatment[k].append(topk_feature_ids.cpu())
                else:
                    all_topk_features_by_treatment.append(self.model.get_topk_features(X, A, d=D, k=max_depth).cpu())

                if test_loader.dataset.y_scaler is not None:
                    pred = transform_outcome_by_rescale_back(test_loader.dataset, pred.cpu())

                all_orig_outcome_ls.append(pred.cpu().view(-1))

                if not self.cont_treatment and not self.has_dose:
                    for treatment_id in range(self.num_treatments):
                        # _, curr_outcome_pred = self.model.forward(X, A, d=D)
                        _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id, d=D)
                        if test_loader.dataset.y_scaler is not None:
                            curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)

                        if treatment_id not in all_outcome_ls_by_treatment:
                            all_outcome_ls_by_treatment[treatment_id] = []
                        all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
                else:
                    all_outcome_ls_by_treatment.append(pred.cpu().view(-1))
                all_treatment_ls.append(A.cpu().view(-1))
                all_feat_ls.append(X.cpu())
                if test_loader.dataset.y_scaler is not None:
                    Y = transform_outcome_by_rescale_back(test_loader.dataset, Y.cpu())

                all_gt_outcome_ls.append(Y.cpu().view(-1))
            
            all_feat_tensor = torch.cat(all_feat_ls, dim=0)
            
            if not self.has_dose and not self.cont_treatment:
                all_topk_features_by_treatment = {k:torch.cat(all_topk_features_by_treatment[k], dim=0) for k in range(self.num_treatments)}
                all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}
            else:
                all_topk_features_by_treatment = torch.cat(all_topk_features_by_treatment, dim=0)
                all_outcome_tensor_by_treatment = torch.cat(all_outcome_ls_by_treatment)
            all_treatment_tensor = torch.cat(all_treatment_ls, dim=0)
            all_origin_outcome_tensor = torch.cat(all_orig_outcome_ls)
            # for treatment_id in range(self.num_treatments):
            #     tree = DecisionTreeRegressor(max_depth=max_depth)
            #     tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
            #     all_tree_by_treatment_id[treatment_id] = tree
            return all_topk_features_by_treatment, all_outcome_tensor_by_treatment, all_origin_outcome_tensor
            # evaluate apoc
            # all_ate_ls = []
            # for k in range(max_depth):
            #     curr_base_X_ls = []
            #     curr_gt_outcome_ls = []
            #     curr_pred_outcome_ls = []
            #     curr_outcome_ls = []
            #     curr_orig_outcome_ls = []
            #     for sample_id in tqdm(range(len(all_feat_tensor))):
            #         curr_feat = all_feat_tensor[sample_id]
            #         curr_pred_outcome = all_outcome_tensor_by_treatment[all_treatment_tensor[sample_id].item()][sample_id]
            #         curr_selected_col_ids_ls=[]
                    
            #         full_selected_col_ids = []
            #         for sample_id in tqdm(range(feature_tensor.shape[0])):
            #             out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
            #             # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
            #             selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
            #             # for col in  selected_col_ids:
            #             #     full_selected_col_ids
            #             full_selected_col_ids.append(selected_col_ids)
            #         decistion_rules_by_treatment[treatment] = full_selected_col_ids
            #         explainer_by_treatment[treatment] = explainer
                    
            #         for treatment_id0 in range(self.num_treatments):
            #             selected_col_ids = all_topk_features_by_treatment[treatment_id0][sample_id]
            #             curr_selected_col_ids_ls.append(selected_col_ids[0:k+1])
            #         # if explanation_type == "decision_tree":
            #         #     for treatment_id0 in range(self.num_treatments):
                            
            #         #         # curr_tree = all_tree_by_treatment_id[all_treatment_tensor[sample_id].item()]
            #         #         curr_tree = all_tree_by_treatment_id[treatment_id0]
            #         #         if all_selected_feat_ls is None:
            #         #             decision_rule, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, curr_feat.numpy(), return_features=True)

            #         #             selected_col_ids = torch.tensor(selected_col_ids)
            #         #         else:
            #         #             selected_col_ids = all_selected_feat_ls[sample_id]
            #         #         curr_selected_col_ids = selected_col_ids[:, 0:k+1].view(-1)
            #         #         curr_selected_col_ids_ls.append(curr_selected_col_ids)
            #         # elif explanation_type == "lime":
            #         #     self.model.device = self.device
            #         #     for treatment_id0 in range(self.num_treatments):
            #         #         # self.model.base_treatment=treatment_id0

            #         #         # explainer = Lime(all_feat_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_feat_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
            #         #         # out = explainer.explain(curr_feat.numpy(), num_features=max_depth)
            #         #         # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
            #         #         selected_col_ids = all_tree_by_treatment_id[treatment_id0]
            #         #         curr_selected_col_ids_ls.append(torch.tensor(selected_col_ids[0:k+1]))

            #         # if len(set(curr_selected_col_ids.tolist())) < k + 1:
            #         #     not_selected_col_ids = [k for k in range(curr_feat.shape[-1]) if k not in set(curr_selected_col_ids.tolist())]
            #         #     random.shuffle(not_selected_col_ids)
            #         #     curr_selected_col_ids = torch.cat([curr_selected_col_ids, torch.tensor(not_selected_col_ids[:k+1 - len(set(curr_selected_col_ids.tolist()))])], dim=0)


            #         # X, full_base_X, fid_ls
            #         curr_base_X = construct_base_x(curr_feat, full_base_X.cpu(), torch.cat(curr_selected_col_ids_ls))
            #         # curr_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_base_X.view(1, -1).numpy())))                  
            #         # curr_orig_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_feat.view(1, -1).numpy())))
            #         curr_base_X_ls.append(curr_base_X)
            #         curr_pred_outcome_ls.append(curr_pred_outcome.view(-1))
                
            #     # all_orig_outcome_tensor = torch.cat(curr_orig_outcome_ls)
            #     curr_base_X_tensor = torch.stack(curr_base_X_ls, dim=0)
            #     curr_base_X_tensor = curr_base_X_tensor.to(self.device)
            #     all_treatment_tensor = all_treatment_tensor.to(self.device)
            #     # curr_outcome_tensor = torch.cat(curr_outcome_ls)
            #     _, curr_outcome_tensor, full_outcome_tensor = self.model.forward(curr_base_X_tensor, all_treatment_tensor, test=True)
            #     pos_outcome_tensor = full_outcome_tensor[:,1].view(-1).cpu()
            #     neg_outcome_tensor = full_outcome_tensor[:,0].view(-1).cpu()
            #     curr_outcome_tensor = curr_outcome_tensor.cpu()
            #     # if test_loader.dataset.y_scaler is not None:
            #     #     curr_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_tensor)
            #     if test_loader.dataset.y_scaler is not None:
            #         pos_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, pos_outcome_tensor)
            #         neg_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, neg_outcome_tensor)
            #         curr_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_tensor)
            #     # ate_pred_diff = torch.abs(pos_outcome_tensor.view(-1).cpu() - neg_outcome_tensor.view(-1).cpu()) - torch.abs(all_outcome_tensor_by_treatment[1].view(-1) - all_outcome_tensor_by_treatment[0].view(-1))
            #     ate_pred_diff = torch.abs((pos_outcome_tensor.view(-1).cpu() - neg_outcome_tensor.view(-1).cpu()) - (all_outcome_tensor_by_treatment[1].view(-1) - all_outcome_tensor_by_treatment[0].view(-1)))
            #     # outcome_pred_diff = torch.abs(all_orig_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1))
                
            #     # all_outcome_diff_ls.append(outcome_pred_diff)

            #     all_ate_ls.append(ate_pred_diff)
            #             # for 
                                                    
            # # mean_outcome_diff = torch.mean(torch.stack(all_outcome_diff_ls, dim=0))        
                    
            # all_ate_diff = torch.mean(torch.stack(all_ate_ls, dim=0))
            
            # # print("mean outcome difference::", mean_outcome_diff.item())
            # print("aopc value::", all_ate_diff.item())
            # print()

    
    
    
            
    def eval_aopc(self, test_dataset, train_dataset, explainer_by_treatment, all_tree_by_treatment_id, max_depth, subset_count = 10, all_selected_feat_ls = None, explanation_type="decision_tree"):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        full_base_X = train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        # all_tree_by_treatment_id = self.fit_decision_tree(train_dataset, max_depth)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_feat_ls = []
            all_outcome_ls_by_treatment = dict()
            all_treatment_ls = []
            all_gt_outcome_ls= []
            all_orig_outcome_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
                _, origin_outcome_pred = self.model.forward(X, A, d=D)
                if test_loader.dataset.y_scaler is not None:
                    origin_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, origin_outcome_pred.cpu())
                all_orig_outcome_ls.append(origin_outcome_pred.cpu().view(-1))

                for treatment_id in range(self.num_treatments):
                    _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id, d=D)
                    if test_loader.dataset.y_scaler is not None:
                        curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)

                    if treatment_id not in all_outcome_ls_by_treatment:
                        all_outcome_ls_by_treatment[treatment_id] = []
                    all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
                all_treatment_ls.append(A.cpu().view(-1))
                all_feat_ls.append(X.cpu())
                if test_loader.dataset.y_scaler is not None:
                    Y = transform_outcome_by_rescale_back(test_loader.dataset, Y.cpu())

                all_gt_outcome_ls.append(Y.cpu().view(-1))
            
            all_feat_tensor = torch.cat(all_feat_ls, dim=0)
            all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}
            all_treatment_tensor = torch.cat(all_treatment_ls)
            all_orig_outcome_tensor = torch.cat(all_orig_outcome_ls)

            # all_tree_by_treatment_id = dict()
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
            # for treatment_id in range(self.num_treatments):
            #     tree = DecisionTreeRegressor(max_depth=max_depth)
            #     tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
            #     all_tree_by_treatment_id[treatment_id] = tree

            all_ate_ls = []
            for k in range(max_depth):
                curr_base_X_ls = []
                curr_gt_outcome_ls = []
                curr_pred_outcome_ls = []
                curr_outcome_ls = []
                curr_orig_outcome_ls = []
                for sample_id in tqdm(range(len(all_feat_tensor))):
                    curr_feat = all_feat_tensor[sample_id]
                    curr_pred_outcome = all_outcome_tensor_by_treatment[all_treatment_tensor[sample_id].item()][sample_id]
                    curr_selected_col_ids_ls=[]

                    if explanation_type == "decision_tree":
                        for treatment_id0 in range(self.num_treatments):
                            
                            # curr_tree = all_tree_by_treatment_id[all_treatment_tensor[sample_id].item()]
                            curr_tree = explainer_by_treatment[treatment_id0]
                            if all_selected_feat_ls is None:
                                decision_rule, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, curr_feat.numpy(), return_features=True)

                                selected_col_ids = torch.tensor(selected_col_ids)
                            else:
                                selected_col_ids = all_selected_feat_ls[sample_id]
                            curr_selected_col_ids = selected_col_ids[:, 0:k+1].view(-1)
                            curr_selected_col_ids_ls.append(curr_selected_col_ids)
                    elif explanation_type == "lime":
                        self.model.device = self.device
                        all_selected_col_score_by_ids = {}
                        for treatment_id0 in range(self.num_treatments):
                            # self.model.base_treatment=treatment_id0

                            # explainer = Lime(all_feat_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_feat_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                            # out = explainer.explain(curr_feat.numpy(), num_features=max_depth)
                            # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                            selected_col_score_by_ids = all_tree_by_treatment_id[treatment_id0][sample_id]
                            for col_key in selected_col_score_by_ids:
                                if col_key in all_selected_col_score_by_ids:
                                    all_selected_col_score_by_ids[col_key] += selected_col_score_by_ids[col_key]
                                else:
                                    all_selected_col_score_by_ids[col_key] = selected_col_score_by_ids[col_key]
                        
                        all_col_ids = [k for k in all_selected_col_score_by_ids]
                        all_col_importance = torch.tensor([all_selected_col_score_by_ids[k] for k in all_col_ids])
                        all_sorted_importance, all_sorted_col_ids = torch.sort(all_col_importance, descending=True)
                        all_col_ids = torch.tensor(all_col_ids)[all_sorted_col_ids]
                        curr_selected_col_ids_ls.append(all_col_ids[0:k+1])

                    # if len(set(curr_selected_col_ids.tolist())) < k + 1:
                    #     not_selected_col_ids = [k for k in range(curr_feat.shape[-1]) if k not in set(curr_selected_col_ids.tolist())]
                    #     random.shuffle(not_selected_col_ids)
                    #     curr_selected_col_ids = torch.cat([curr_selected_col_ids, torch.tensor(not_selected_col_ids[:k+1 - len(set(curr_selected_col_ids.tolist()))])], dim=0)


                    # X, full_base_X, fid_ls
                    curr_base_X = construct_base_x(curr_feat, full_base_X.cpu(), torch.cat(curr_selected_col_ids_ls))
                    # curr_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_base_X.view(1, -1).numpy())))                  
                    # curr_orig_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_feat.view(1, -1).numpy())))
                    curr_base_X_ls.append(curr_base_X)
                    curr_pred_outcome_ls.append(curr_pred_outcome.view(-1))
                
                # all_orig_outcome_tensor = torch.cat(curr_orig_outcome_ls)
                curr_base_X_tensor = torch.stack(curr_base_X_ls, dim=0)
                curr_base_X_tensor = curr_base_X_tensor.to(self.device)
                all_treatment_tensor = all_treatment_tensor.to(self.device)
                # curr_outcome_tensor = torch.cat(curr_outcome_ls)
                _, curr_outcome_tensor, full_outcome_tensor = self.model.forward(curr_base_X_tensor, all_treatment_tensor, test=True)
                pos_outcome_tensor = full_outcome_tensor[:,1].view(-1).cpu()
                neg_outcome_tensor = full_outcome_tensor[:,0].view(-1).cpu()
                curr_outcome_tensor = curr_outcome_tensor.cpu()
                # if test_loader.dataset.y_scaler is not None:
                #     curr_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_tensor)
                if test_loader.dataset.y_scaler is not None:
                    pos_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, pos_outcome_tensor)
                    neg_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, neg_outcome_tensor)
                    curr_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_tensor)
                # ate_pred_diff = torch.abs(pos_outcome_tensor.view(-1).cpu() - neg_outcome_tensor.view(-1).cpu()) - torch.abs(all_outcome_tensor_by_treatment[1].view(-1) - all_outcome_tensor_by_treatment[0].view(-1))
                ate_pred_diff = torch.abs((pos_outcome_tensor.view(-1).cpu() - neg_outcome_tensor.view(-1).cpu()) - (all_outcome_tensor_by_treatment[1].view(-1) - all_outcome_tensor_by_treatment[0].view(-1)))
                # outcome_pred_diff = torch.abs(all_orig_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1))
                
                # all_outcome_diff_ls.append(outcome_pred_diff)

                all_ate_ls.append(ate_pred_diff)
                        # for 
                                                    
            # mean_outcome_diff = torch.mean(torch.stack(all_outcome_diff_ls, dim=0))        
                    
            all_ate_diff = torch.mean(torch.stack(all_ate_ls, dim=0))
            
            # print("mean outcome difference::", mean_outcome_diff.item())
            print("aopc value::", all_ate_diff.item())
            print()

    # def eval_faithfulness(self, test_dataset, train_dataset, max_depth, subset_count = 10, all_selected_feat_ls = None):
    #     test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
    #     full_base_X = train_dataset.transformed_features.mean(dim=0)
    #     full_base_X = full_base_X.to(self.device)
    #     # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
    #     # random.shuffle(all_idx_set_ls)
        
    #     # selected_idx_set_ls = all_idx_set_ls[:subset_count]
    #     self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
    #     all_tree_by_treatment_id = self.fit_decision_tree(train_dataset, max_depth)
    #     with torch.no_grad():
    #         self.model.eval()
    #         pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

    #         compute_ate = False
            
    #         all_outcome_diff_ls = []
    #         all_ate_diff_ls = []

    #         all_explanation_perturbation_ls = []
    #         all_feat_ls = []
    #         all_outcome_ls_by_treatment = dict()
    #         all_treatment_ls = []
    #         all_gt_outcome_ls= []
    #         all_orig_outcome_ls = []
    #         all_relative_diff_ls = []
    #         for step, batch in pbar:
                
    #                 # batch = (x.cuda() for x in batch)
    #             # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
    #             # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
    #             # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
    #             idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
    #             X = X.to(self.device)
    #             Y = Y.to(self.device)
    #             A = A.to(self.device)
    #             if count_Y is not None:
    #                 compute_ate = True
                
    #             if D is not None:
    #                 D = D.to(self.device)
    #             _, origin_outcome_pred = self.model.forward(X, A, d=D)
    #             if test_loader.dataset.y_scaler is not None:
    #                 origin_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, origin_outcome_pred.cpu())
    #             all_orig_outcome_ls.append(origin_outcome_pred.cpu().view(-1))

    #             for treatment_id in range(self.num_treatments):
    #                 _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id, d=D)
    #                 if test_loader.dataset.y_scaler is not None:
    #                     curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)

    #                 if treatment_id not in all_outcome_ls_by_treatment:
    #                     all_outcome_ls_by_treatment[treatment_id] = []
    #                 all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
    #             all_treatment_ls.append(A.cpu().view(-1))
    #             all_feat_ls.append(X.cpu())
    #             if test_loader.dataset.y_scaler is not None:
    #                 Y = transform_outcome_by_rescale_back(test_loader.dataset, Y.cpu())

    #             all_gt_outcome_ls.append(Y.cpu().view(-1))
            
    #         all_feat_tensor = torch.cat(all_feat_ls, dim=0)
    #         all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}
    #         all_treatment_tensor = torch.cat(all_treatment_ls)
    #         all_orig_outcome_tensor = torch.cat(all_orig_outcome_ls)

    #         # all_tree_by_treatment_id = dict()
    #         all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
    #         # for treatment_id in range(self.num_treatments):
    #         #     tree = DecisionTreeRegressor(max_depth=max_depth)
    #         #     tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
    #         #     all_tree_by_treatment_id[treatment_id] = tree

    #         for k in range(max_depth):
    #             curr_base_X_ls = []
    #             curr_gt_outcome_ls = []
    #             curr_pred_outcome_ls = []
    #             curr_outcome_ls = []
    #             curr_orig_outcome_ls = []
    #             for sample_id in range(len(all_feat_tensor)):
    #                 curr_feat = all_feat_tensor[sample_id]
    #                 curr_pred_outcome = all_outcome_tensor_by_treatment[all_treatment_tensor[sample_id].item()][sample_id]
    #                 curr_tree = all_tree_by_treatment_id[all_treatment_tensor[sample_id].item()]
    #                 if all_selected_feat_ls is None:
    #                     decision_rule, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, curr_feat.numpy(), return_features=True)

    #                     selected_col_ids = torch.tensor(selected_col_ids)
    #                 else:
    #                     selected_col_ids = all_selected_feat_ls[sample_id]
    #                 curr_selected_col_ids = selected_col_ids[:, 0:k+1].view(-1)

    #                 # if len(set(curr_selected_col_ids.tolist())) < k + 1:
    #                 #     not_selected_col_ids = [k for k in range(curr_feat.shape[-1]) if k not in set(curr_selected_col_ids.tolist())]
    #                 #     random.shuffle(not_selected_col_ids)
    #                 #     curr_selected_col_ids = torch.cat([curr_selected_col_ids, torch.tensor(not_selected_col_ids[:k+1 - len(set(curr_selected_col_ids.tolist()))])], dim=0)


    #                 # X, full_base_X, fid_ls
    #                 curr_base_X = construct_base_x(curr_feat, full_base_X.cpu(), curr_selected_col_ids)
    #                 curr_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_base_X.view(1, -1).numpy())))                  
    #                 curr_orig_outcome_ls.append(torch.from_numpy(curr_tree.predict(curr_feat.view(1, -1).numpy())))
    #                 curr_base_X_ls.append(curr_base_X)
    #                 curr_pred_outcome_ls.append(curr_pred_outcome.view(-1))
                
    #             # all_orig_outcome_tensor = torch.cat(curr_orig_outcome_ls)
    #             curr_base_X_tensor = torch.stack(curr_base_X_ls, dim=0)
    #             curr_base_X_tensor = curr_base_X_tensor.to(self.device)
    #             all_treatment_tensor = all_treatment_tensor.to(self.device)
    #             curr_outcome_tensor = torch.cat(curr_outcome_ls)
    #             _, curr_outcome_tensor = self.model.forward(curr_base_X_tensor, all_treatment_tensor)
                
    #             if test_loader.dataset.y_scaler is not None:
    #                 curr_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_tensor)
               

    #             outcome_pred_diff = torch.abs(all_orig_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1))
    #             relative_pred_diff = (torch.abs(all_orig_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)) - torch.abs(curr_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)))/(torch.abs(all_orig_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1) + torch.abs(curr_outcome_tensor.view(-1) - all_gt_outcome_tensor.view(-1)) + 1e-5))
    #             all_relative_diff_ls.append(relative_pred_diff)
    #             all_outcome_diff_ls.append(outcome_pred_diff)
    #                     # for 
                                                    
    #         mean_outcome_diff = torch.mean(torch.stack(all_outcome_diff_ls, dim=0))        
                    
    #         mean_relative_diff = torch.mean(torch.cat(all_relative_diff_ls))
            
    #         print("mean outcome difference::", mean_outcome_diff.item())
    #         print("mean relative diff::", mean_relative_diff.item())
    #         print()
    
    def eval_faithfulness(self, test_dataset, train_dataset, max_depth, subset_count = 10, all_selected_feat_ls = None, explanation_type = 'decision_tree'):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        full_base_X = train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        sample_id_feat_id_mappings, sample_id_feat_name_mappings, rand_coeff_mappings = retrieve_gt_explanations(self.data_folder, self.dataset_id)
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        all_tree_by_treatment_id = self.fit_decision_tree(train_dataset, max_depth)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_feat_ls = []
            all_outcome_ls_by_treatment = dict()
            all_treatment_ls = []
            all_dose_ls = []
            all_gt_outcome_ls= []
            all_orig_outcome_ls = []
            all_relative_diff_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                if D is not None:
                    D = D.to(self.device)
                _, origin_outcome_pred = self.model.forward(X, A, d=D)
                if test_loader.dataset.y_scaler is not None:
                    origin_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, origin_outcome_pred.cpu())
                all_orig_outcome_ls.append(origin_outcome_pred.cpu().view(-1))

                for treatment_id in range(self.num_treatments):
                    _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id, d=D)
                    if test_loader.dataset.y_scaler is not None:
                        curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)

                    if treatment_id not in all_outcome_ls_by_treatment:
                        all_outcome_ls_by_treatment[treatment_id] = []
                    all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
                all_treatment_ls.append(A.cpu().view(-1))
                if D is not None:
                    all_dose_ls.append(D.cpu().view(-1))
                all_feat_ls.append(X.cpu())
                if test_loader.dataset.y_scaler is not None:
                    Y = transform_outcome_by_rescale_back(test_loader.dataset, Y.cpu())

                all_gt_outcome_ls.append(Y.cpu().view(-1))
            
            all_feat_tensor = torch.cat(all_feat_ls, dim=0)
            all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}
            all_treatment_tensor = torch.cat(all_treatment_ls)
            all_dose_tensor = None
            if len(all_dose_ls) > 0:
                all_dose_tensor = torch.cat(all_dose_ls)
            
            all_orig_outcome_tensor = torch.cat(all_orig_outcome_ls)

            all_tree_by_treatment_id = dict()
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
            for treatment_id in range(self.num_treatments):
                tree = DecisionTreeRegressor(max_depth=max_depth)
                tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
                all_tree_by_treatment_id[treatment_id] = tree

            # for k in range(max_depth):
            #     curr_base_X_ls = []
            #     curr_gt_outcome_ls = []
            #     curr_pred_outcome_ls = []
            #     curr_outcome_ls = []
            #     curr_orig_outcome_ls = []
            all_overlap_ls = []
            for sample_id in range(len(all_feat_tensor)):
                curr_feat = all_feat_tensor[sample_id]
                curr_pred_outcome = all_outcome_tensor_by_treatment[all_treatment_tensor[sample_id].item()][sample_id]
                if explanation_type == 'decision_tree':
                    curr_tree = all_tree_by_treatment_id[all_treatment_tensor[sample_id].item()]
                    decision_rule, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, curr_feat.numpy(), return_features=True)
                elif explanation_type == 'lime':
                    self.model.base_treatment=all_treatment_tensor[sample_id].item()
                    if all_dose_tensor is not None:
                        self.model.base_dose = all_dose_tensor[sample_id].item()
                    self.model.device = self.device
                    explainer = Lime(all_feat_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_feat_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                    out = explainer.explain(curr_feat.numpy(), num_features=max_depth)
                    
                    selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                    
               
                
                gt_explanation_col_ids = sample_id_feat_id_mappings[sample_id]
                
                
                
                
                # for selected_idx_set in selected_idx_set_ls:
                # selected_col_ids = torch.stack([all_atom_ls[k][col_id_key] for k in range(len(all_atom_ls))], dim=2)
                pred_explanation_col_ids_ls = torch.tensor([selected_col_ids[k] for k in range(len(selected_col_ids))]).view(-1).tolist()
                overlap_val =  len(set(gt_explanation_col_ids).intersection(set(pred_explanation_col_ids_ls)))/len(pred_explanation_col_ids_ls) #[len(set(gt_explanation_col_ids_ls[k]).intersection(set(pred_explanation_col_ids_ls[k])))*1.0/len(set(gt_explanation_col_ids_ls[k])) for k in range(len(gt_explanation_col_ids_ls))]
                
                all_overlap_ls.append(overlap_val)
                
            mean_overlap_val = torch.mean(torch.tensor(all_overlap_ls)).item()
            print("mean overlap val::", mean_overlap_val)
            print()
            


    def eval_stability(self, test_dataset, origin_explanation_str_ls, perturb_times=1, tree_depth=2, explanation_type="decision_tree"):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        with torch.no_grad():
            self.model.eval()
            
            all_explanation_perturbation_ls = []
            for _ in range(perturb_times):
                pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
                all_X =[]
                all_pert_X = []
                all_pert_out = []
                all_out = []
                all_treatment_ls = []
                curr_perturbation_ls = []
                for step, batch in pbar:
                    # batch = (x.cuda() for x in batch)
                    # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                    idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                    
                   
                    pert_origin_X = perturb_samples(origin_X,test_dataset)
                    pert_X = test_dataset.convert_feats_to_transformed_feats(pert_origin_X)
                

                    X = X.to(self.device)
                    pert_X =pert_X.to(self.device)
                    Y = Y.to(self.device)
                    A = A.to(self.device)
                    if count_Y is not None:
                        compute_ate = True
                    
                    if D is not None:
                        D = D.to(self.device)
                
                    _, pert_out = self.model.forward(pert_X, A, d=D)
                    _, out = self.model.forward(X, A, d=D)
                    all_X.append(X.detach().cpu())
                    all_pert_X.append(pert_X.detach().cpu())                                      
                    all_pert_out.append(pert_out.detach().cpu())
                    all_out.append(out.detach().cpu())
                    all_treatment_ls.append(A.detach().cpu())
                
                all_pert_X_tensor = torch.cat(all_pert_X)
                all_pert_out_tensor = torch.cat(all_pert_out)
                all_treatment_tensor = torch.cat(all_treatment_ls)
                all_X_tensor = torch.cat(all_X)
                all_out_tensor = torch.cat(all_out)
                
                # if explanation_type == 'lime':
                #     explainer = Lime(all_X_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_X_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")

                for sample_id in tqdm(range(len(all_pert_X_tensor))):
                    curr_all_perm_X_tensor = all_X_tensor.clone()
                    curr_all_perm_X_tensor[sample_id] = all_pert_X_tensor[sample_id]
                    curr_all_pert_out_tensor = all_out_tensor.clone()
                    curr_all_pert_out_tensor[sample_id] = all_pert_out_tensor[sample_id]
                    if explanation_type == 'decision_tree':
                        curr_tree = DecisionTreeRegressor(max_depth=tree_depth)
                        curr_tree.fit(curr_all_perm_X_tensor.detach().cpu(), curr_all_pert_out_tensor.detach().cpu())
                        
                        decision_rule = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, curr_all_perm_X_tensor[sample_id].numpy())                
                        curr_origin_explanations = origin_explanation_str_ls[all_treatment_tensor[sample_id].item()][sample_id]
                        curr_perturbation_ls.append(torch.tensor(evaluate_explanation_diff_single_pair(curr_origin_explanations, decision_rule)).item())
                    elif explanation_type == 'lime':
                        explainer = origin_explanation_str_ls[all_treatment_tensor[sample_id].item()]
                        out = explainer.explain(all_X_tensor[sample_id].numpy(), num_features=tree_depth)
                        origin_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                        out = explainer.explain(all_pert_X_tensor[sample_id].numpy(), num_features=tree_depth)
                        perturbed_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]

                        intersect_col_ids = set(origin_col_ids).intersection(set(perturbed_col_ids))
                        union_col_ids = set(origin_col_ids).union(set(perturbed_col_ids))
                        curr_perturbation_ls.append(len(intersect_col_ids)/len(union_col_ids))
                    
                    

                all_explanation_perturbation_ls.append(torch.tensor(curr_perturbation_ls))
            all_explanation_perturbation_tensor = torch.cat(all_explanation_perturbation_ls, dim=-1)
            mean_explanation_perturbation = torch.mean(all_explanation_perturbation_tensor)
            print("robustness score::", mean_explanation_perturbation.item())



    def posthoc_explain(self, test_dataset, tree_depth=2, explanation_type="decision_tree", subset_ids=None):
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        with torch.no_grad():
            
            self.model.eval()
            pbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Validating')
            

            feature_ls = []
            origin_feature_ls = []
            unique_treatment_set = set()
            unique_dose_set = set()
            unique_treatment_dose_set = set()


            for batch in pbar:

                idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                
                feature_ls.append(X.detach().cpu())
                origin_feature_ls.append(origin_X.detach().cpu())
                curr_treatment_ls = A.detach().cpu().view(-1).tolist()
                curr_dose_ls = None
                if D is not None:
                    curr_dose_ls = D.detach().cpu().view(-1).tolist()
                
                unique_treatment_set.update(curr_treatment_ls)
                if curr_dose_ls is not None:
                    unique_dose_set.update(curr_dose_ls)
                    unique_treatment_dose_set.update((zip(curr_treatment_ls, curr_dose_ls)))
            
            feature_tensor = torch.cat(feature_ls, dim=0)
            origin_feature_tensor = torch.cat(origin_feature_ls, dim=0)
        if not self.cont_treatment and not self.has_dose:
            unique_treatment_set = list(range(self.num_treatments))
        
        return obtain_post_hoc_explanatios_main(self, test_dataset, test_dataloader, unique_treatment_set, feature_tensor, origin_feature_tensor, tree_depth, explanation_type, subset_ids=subset_ids)

    def do_prediction(self, X, A, D, X_pd_full, origin_all_other_pats_ls, origin_X):
        _, out = self.model.forward(X, A, d=D)
        return out.detach().cpu()


    def run(self, train_dataset, valid_dataset, test_dataset, y_scaler=None, cohort=False):
        if not cohort:
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=tabular_Dataset.collate_fn, drop_last=True)
            val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=tabular_Dataset.collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=tabular_Dataset.collate_fn)
            if y_scaler is None:
                y_scaler = train_dataset.y_scaler
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=True)
            val_dataloader = DataLoader(valid_dataset, batch_size=len(test_dataset), shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        # print("evaluation on validation set::")
        # self.test(val_dataloader)
        if not cohort:
            print("evaluation on test set::")
            self.test(test_dataloader, cohort=cohort, y_scaler=y_scaler)
        best_train_outcome_error = 1e6
        best_val_outcome_error = 1e6
        best_test_outcome_error = 1e6
        
        best_train_ite=1e6
        best_val_ite = 1e6
        best_test_ite = 1e6
        
        
        best_train_ate=1e6
        best_val_ate=1e6
        best_test_ate=1e6
        
        best_train_pehe=1e6
        best_valid_pehe = 1e6
        best_test_pehe=1e6
        
        best_performance_epoch=0
        compute_ate = False
        if cohort:
            for epoch in range(self.epochs):
                losses = []
                self.model.train()        
                for step, batch in enumerate(train_dataloader):
                    
                        # batch = (x.cuda() for x in batch)
                    # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                    # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A, _, D = batch
                    
                    if len(batch) == 3:
                        X, A, Y = batch
                        D = None
                    elif len(batch) == 4:
                        X, A, Y, D = batch
                    count_Y = None
                    X = X.to(self.device)
                    Y=  Y.to(self.device)
                    A = A.to(self.device)
                    if D is not None:
                        D = D.to(self.device)
                    if count_Y is not None:
                        compute_ate = True

                    # if self.tr:
                        
                    #     embedding, out = self.model.forward(X, A, d=D)

                    #     set_requires_grad(self.targetReg, True)
                    #     self.tr_optimizer.zero_grad()
                    #     trg = self.targetReg(embedding.detach())
                    #     if self.num_treatments == 1:
                    #         loss_D = F.mse_loss(trg, A)
                    #     # elif self.p == 2:
                    #     #     loss_D = neg_guassian_likelihood(trg, A)
                    #     elif self.num_treatments == 2:
                    #         loss_D = F.binary_cross_entropy_with_logits(trg, A)
                    #     elif self.num_treatments > 2:
                    #         loss_D = F.cross_entropy(trg, A.view(-1).long())
                    #     loss_D.backward()
                    #     self.tr_optimizer.step()
                    
                    #     set_requires_grad(self.targetReg, False)
                    #     self.optimizer.zero_grad()
                    #     trg = self.targetReg(embedding)
                    #     if self.num_treatments == 1:
                    #         loss_D = F.mse_loss(trg, A)
                    #     # elif self.p == 2:
                    #     #     loss_D = neg_guassian_likelihood(trg, A)
                    #     elif self.num_treatments == 2:
                    #         loss_D = F.binary_cross_entropy_with_logits(trg, A)
                    #     elif self.num_treatments > 2:
                    #         loss_D = F.cross_entropy(trg, A.view(-1).long())
                    #     if self.outcome_regression:
                    #         loss = F.mse_loss(out, Y) - self.alpha * loss_D
                    #     else:
                    #         loss = F.binary_cross_entropy_with_logits(out, Y) - self.alpha * loss_D
                    #     # loss = criterion(out, y, alpha=alpha) - args.alpha * loss_D
                    #     loss.backward()
                    #     self.optimizer.step()
                    #     # adjust_learning_rate(self.optimizer, self.lr, epoch, lr_type='cos', num_epoch=self.epochs)
                            
                    # else:
                    self.optimizer.zero_grad()
                    _, out = self.model.forward(X, A, d=D)
                    if self.outcome_regression:
                        loss = F.mse_loss(out.view(-1,1), Y.view(-1,1))
                    else:
                        loss = F.binary_cross_entropy_with_logits(out, Y)
                    loss.backward()
                    self.optimizer.step()
                        # adjust_learning_rate(self.optimizer, self.lr, epoch, lr_type='cos', num_epoch=self.epochs)
                    
                    
                    
                    # concat_pred = self.model(X)
                    # concat_true = torch.cat([Y.view(len(Y), 1), A.view(len(Y), 1)], dim=1)
                    # # loss = self.dragonet_loss(concat_true, concat_pred)

                    # loss.backward()
                    # self.optimizer.step()
                
            return
        
        
        # if self.cat_and_cont_treatment:
        #     mise, dpe, pe, ate = compute_eval_metrics(train_dataset.metainfo, train_dataset, self.num_treatments, self.device, self.do_prediction)
        #     print("Train Mise: %s" % str(mise))
        #     print("Train DPE: %s" % str(dpe))
        #     print("Train PE: %s" % str(pe))
        #     print("Train ATE: %s" % str(ate))
            
        #     mise, dpe, pe, ate = compute_eval_metrics(test_dataset.metainfo, test_dataset, self.num_treatments, self.device, self.do_prediction)
        #     print("Mise: %s" % str(mise))
        #     print("DPE: %s" % str(dpe))
        #     print("PE: %s" % str(pe))
        #     print("ATE: %s" % str(ate))
        
        for epoch in range(1, self.epochs):
            losses = []
            self.model.train()        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A, _, D = batch
                if not cohort:
                    idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                else:
                    if len(batch) == 3:
                        X, A, Y = batch
                        D = None
                    elif len(batch) == 4:
                        X, A, Y, D = batch
                    count_Y = None
                X = X.to(self.device)
                Y=  Y.to(self.device)
                A = A.to(self.device)
                if D is not None:
                    D = D.to(self.device)
                if count_Y is not None:
                    compute_ate = True

                if self.tr:
                    
                    embedding, out = self.model.forward(X, A, d=D)
                    if self.reg_model is not None:
                        out = reg_model_forward(self, X, A, D, origin_X, out)

                    set_requires_grad(self.targetReg, True)
                    self.tr_optimizer.zero_grad()
                    trg = self.targetReg(embedding.detach())
                    if self.num_treatments == 1:
                        loss_D = F.mse_loss(trg, A)
                    # elif self.p == 2:
                    #     loss_D = neg_guassian_likelihood(trg, A)
                    elif self.num_treatments == 2:
                        loss_D = F.binary_cross_entropy_with_logits(trg, A)
                    elif self.num_treatments > 2:
                        loss_D = F.cross_entropy(trg, A.view(-1).long())
                    loss_D.backward()
                    self.tr_optimizer.step()
                
                    set_requires_grad(self.targetReg, False)
                    self.optimizer.zero_grad()
                    trg = self.targetReg(embedding)
                    if self.num_treatments == 1:
                        loss_D = F.mse_loss(trg, A)
                    # elif self.p == 2:
                    #     loss_D = neg_guassian_likelihood(trg, A)
                    elif self.num_treatments == 2:
                        loss_D = F.binary_cross_entropy_with_logits(trg, A)
                    elif self.num_treatments > 2:
                        loss_D = F.cross_entropy(trg, A.view(-1).long())
                    if self.outcome_regression:
                        loss = F.mse_loss(out, Y) - self.alpha * loss_D
                    else:
                        loss = F.binary_cross_entropy_with_logits(out, Y) - self.alpha * loss_D
                    # loss = criterion(out, y, alpha=alpha) - args.alpha * loss_D
                    loss.backward()
                    self.optimizer.step()
                    # adjust_learning_rate(self.optimizer, self.lr, epoch, lr_type='cos', num_epoch=self.epochs)
                        
                else:
                    other_loss = None
                    self.optimizer.zero_grad()
                    if not self.method == "ENRL":
                        _, out = self.model.forward(X, A, d=D)
                    else:
                        X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                        X_num = X_num.to(self.device)
                        X_cat = X_cat.to(self.device)
                        out, other_loss = self.model(X_num, X_cat, A, d=D)
                        out = out.view(len(out), 1)
                    if self.backbone_full_model is not None:
                        out = reg_model_forward(self, X, A, D, origin_X, out)                    
                    if self.outcome_regression:
                        loss = F.mse_loss(out.view(-1,1), Y.view(-1,1))
                    else:
                        loss = F.binary_cross_entropy_with_logits(out, Y)
                    if other_loss is not None:
                        loss += other_loss
                    loss.backward()
                    self.optimizer.step()
                    # adjust_learning_rate(self.optimizer, self.lr, epoch, lr_type='cos', num_epoch=self.epochs)
                
                
                
                # concat_pred = self.model(X)
                # concat_true = torch.cat([Y.view(len(Y), 1), A.view(len(Y), 1)], dim=1)
                # # loss = self.dragonet_loss(concat_true, concat_pred)

                # loss.backward()
                # self.optimizer.step()
            
                losses.append(loss.item())
            if not cohort:
                print("Training loss: ", np.mean(np.array(losses)))

            # print("evaluation on training set::")
            # self.test(train_dataloader)
            if not cohort:
                if not compute_ate:
                    
                    print("evaluation on validation set::")
                    val_outcome_error = self.test(val_dataloader, cohort=cohort, y_scaler=y_scaler)
                    
                    print("evaluation on test set::")
                    test_outcome_error = self.test(test_dataloader, cohort=cohort, y_scaler=y_scaler)

                    if val_outcome_error < best_val_outcome_error:
                        best_val_outcome_error = val_outcome_error
                        best_test_outcome_error = test_outcome_error
                        torch.save(self.model.state_dict(), os.path.join(self.log_folder, 'bestmod.pt'))
                    print("Performance at Epoch %d: validation outcome error::%f, test outcome error::%f"%(epoch, best_val_outcome_error, best_test_outcome_error))
                    print("best validation outcome error::%f, best test outcome error::%f"%(best_val_outcome_error, best_test_outcome_error))
                
                else:

                    print("evaluation on validation set::")
                    val_outcome_error, val_ite, val_ate, val_pehe = self.test(val_dataloader, cohort=cohort, y_scaler=y_scaler)
                    

                    print("evaluation on test set::")
                    test_outcome_error, test_ite, test_ate, test_pehe = self.test(test_dataloader, cohort=cohort, y_scaler=y_scaler)
                    if val_outcome_error < best_val_outcome_error:
                        print("evaluation on test set::")
                        train_outcome_error, train_ite, train_ate, train_pehe = self.test(train_dataloader, cohort=cohort, y_scaler=y_scaler)
                        
                        best_val_outcome_error = val_outcome_error
                        best_test_outcome_error = test_outcome_error
                        best_train_outcome_error = train_outcome_error
                        
                        best_val_ite = val_ite
                        best_test_ite = test_ite
                        best_train_ite = train_ite
                        
                        best_train_ate = train_ate
                        best_val_ate = val_ate
                        best_test_ate = test_ate
                        
                        best_train_pehe = train_pehe
                        best_valid_pehe = val_pehe
                        best_test_pehe = test_pehe
                        
                        best_performance_epoch = epoch
                        torch.save(self.model.state_dict(), os.path.join(self.log_folder, 'bestmod.pt'))

                    print("Performance at Epoch %d: validation ate::%f, test ate::%f"%(epoch, val_outcome_error, test_outcome_error))
                    
                    print("best performance at epoch %d"%(best_performance_epoch))
                    print("best train outcome error::%f"%(best_train_outcome_error))
                    print("best valid outcome error::%f"%(best_val_outcome_error))
                    print("best test outcome error::%f"%(best_test_outcome_error))
                    
                    print("best train ite::%f"%(best_train_ite))
                    print("best validation ite::%f"%(best_val_ite))
                    print("best test ite::%f"%(best_test_ite))
                    
                    print("best train ate::%f"%(best_train_ate))
                    print("best validation ate::%f"%(best_val_ate))
                    print("best test ate::%f"%(best_test_ate))
                    
                    print("best train pehe::%f"%(best_train_pehe))
                    print("best validation pehe::%f"%(best_valid_pehe))
                    print("best test pehe::%f"%(best_test_pehe))

                if self.cont_treatment and epoch%100==0:
                    self.test_cont(test_dataloader)
        if self.cont_treatment:
            self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
            self.test_cont(test_dataloader)
        
        if self.cat_and_cont_treatment:
            self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
            mise, dpe, pe, ate = compute_eval_metrics(train_dataset.metainfo, train_dataset, self.num_treatments, self.device, self.do_prediction)
            print("Train Mise: %s" % str(mise))
            print("Train DPE: %s" % str(dpe))
            print("Train PE: %s" % str(pe))
            print("Train ATE: %s" % str(ate))
            
            
            mise, dpe, pe, ate = compute_eval_metrics(test_dataset.metainfo, test_dataset, self.num_treatments, self.device, self.do_prediction)
            print("Mise: %s" % str(mise))
            print("DPE: %s" % str(dpe))
            print("PE: %s" % str(pe))
            print("ATE: %s" % str(ate))
