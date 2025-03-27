'''
Implementation of the TI estimator from
"Causal Estimation for Text Data with Apparent Overlap Violations"
'''
import math
import random
import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel

import pandas as pd
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys, os
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from rl_models.enc_dec import pred_probs_key, pred_Q_key, pred_v_key, prev_prog_key, col_id_key, select_num_feat_key, col_Q_key, min_Q_val
from rl_models.enc_dec import forward_main0_opt, create_deep_set_net_for_programs, TokenNetwork3, atom_to_vector_ls0_main, TokenNetwork2
from rl_models.rl_algorithm import ReplayMemory, process_curr_atoms0
from nlp_data_utils.nlp_dataset import NLP_Dataset

from collections import namedtuple, deque

from Qmod_utils import *

from treatment_prediction.utils_treatment import *#evaluate_treatment_effect_core, transform_outcome_by_rescale_back, split_treatment_control_gt_outcome, obtain_individual_predictions2, obtain_predictions2
from baseline_methods.self_interpretable_models.ENRL import ENRL
from baseline_methods.dragonnet import *
from baseline_methods.TransTEE.TransTEE import TransTEE
Transition = namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))


min_Q_val = -1.01

col_key = "col"

col_id_key = "col_id"

col_Q_key = "col_Q"

pred_Q_key = "pred_Q"

op_Q_key = "op_Q"

col_probs_key = "col_probs"

pred_probs_key = "pred_probs"

op_probs_key = "op_probs"

pred_v_key = "pred_v"

op_id_key = "op_id"
        
op_key = "op"

topk=3

further_sel_mask_key = "further_sel_mask"


MASK_IDX = 103


''' The first stage QNet'''
class CausalQNet(DistilBertPreTrainedModel):
    """ QNet model to estimate the conditional outcomes for the first stage
        Note the outcome Y is continuous """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.g_hat = nn.Linear(config.hidden_size, self.num_labels)
        self.Q_cls = nn.ModuleDict()
        for A in range(2):
          self.Q_cls['%d' % A] = nn.Sequential(
          nn.Linear(config.hidden_size, 200),
          nn.ReLU(),
          nn.Linear(200, 1))

        self.init_weights()

    def forward(self, text_ids, text_len, text_mask, A, Y, use_mlm=False):
        text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
        attention_mask_class = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
        target_words = torch.gather(text_ids, 1, mask)
        mlm_labels = torch.ones(text_ids.shape).long() * -100
        if torch.cuda.is_available():
            mlm_labels = mlm_labels.cuda()
        mlm_labels.scatter_(1, mask, target_words)
        text_ids.scatter_(1, mask, MASK_IDX)

        # distilbert output
        bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
        seq_output = bert_outputs[0]
        pooled_output = seq_output[:, 0]  # CLS token

        # masked language modeling objective
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = torch.nn.functional.gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss(reduction='sum')(prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = torch.tensor(0)

        # sm = nn.Softmax(dim=1)

        # A ~ text
        a_text = self.g_hat(pooled_output)
        a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.long().view(-1))
        # accuracy
        a_pred = a_text.argmax(dim=1)
        a_acc = (a_pred == A).sum().float()/len(A) 
        
        # Y ~ text+A
        # conditional expected outcomes
        Q0 = self.Q_cls['0'](pooled_output)
        Q1 = self.Q_cls['1'](pooled_output)

        if Y is not None:
            A0_indices = (A == 0).nonzero().squeeze()
            A1_indices = (A == 1).nonzero().squeeze()
            # Y loss
            y_loss_A1 = (((Q1.view(-1)-Y)[A1_indices])**2).sum()
            y_loss_A0 = (((Q0.view(-1)-Y)[A0_indices])**2).sum()
            y_loss = y_loss_A0 + y_loss_A1
        else:
            y_loss = 0.0

        return Q0, Q1, mlm_loss, y_loss, a_loss, a_acc


class CausalQNet_rl_all(torch.nn.Module):
    def init_without_feat_groups(self, lang,  program_max_len, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=1, continue_act=False):
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
                
        self.op_start_pos = -1
        self.num_start_pos = -1
        self.cat_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                for option in list(options_dict.keys()):        
                    if self.op_start_pos < 0:
                        self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:                    
                if decision in self.lang.syntax["num_feat"]:
                    if self.num_start_pos < 0:
                        self.num_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
                else:
                    if self.cat_start_pos < 0:
                        self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
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
        # self.all_input_feat_len = self.backbone_model.vocab_transform.out_features#self.num_feat_len+category_sum_count
        self.all_input_feat_len = self.num_feat_len+category_sum_count
        self.num_feats = num_features
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        # full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        full_input_size = self.all_input_feat_len + latent_size# self.ATOM_VEC_LENGTH
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
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.num_feats)
            # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.num_feats)
            # self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        self.to(self.device)
    
    def __init__(self, args, backbone_model_config, lang,  program_max_len, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = 1, continue_act=False, num_labels=2, num_treatments=2, cont_treatment=False, discretize_feat_value_count=20, use_mlm=True,removed_feat_ls= None):
        super(CausalQNet_rl_all, self).__init__()
        # self.backbone_model = CausalQNet_rl.from_pretrained(
        #     'distilbert-base-uncased',
        #     num_labels = num_labels,
        #     output_attentions=False,
        #     output_hidden_states=False)
        if args.backbone is not None:
            self.backbone_full_model = nn.ModuleDict()
            for A in range(2):
                self.backbone_full_model['%d' % A] = nn.Sequential(
                nn.Linear(num_feat_count+category_sum_count, 200),
                nn.ReLU(),
                nn.Linear(200, 1))
            self.backbone_full_model = self.backbone_full_model.cuda()


        self.regression_ratio = args.regression_ratio
        input_size = num_feat_count + category_sum_count
        # if args.backbone is not None:
        #     if args.backbone.lower() == "transtee":
        #         print("start loading transtee backbone")
        #         if not args.cont_treatment and args.has_dose:
        #             cov_dim = backbone_model_config["cov_dim"]
        #         else:
        #             cov_dim = input_size
                
        #         params = {'num_features': input_size, 'num_treatments': args.num_treatments,
        #         'h_dim': backbone_model_config["hidden_size"], 'cov_dim':cov_dim}
        #         # self.model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
        #         self.shared_hidden_dim = backbone_model_config["hidden_size"]
        #         self.backbone_full_model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
        #         self.backbone_model = self.backbone_full_model.encoding_features
        #         # if args.cached_backbone is not None:    
        #         #     self.backbone_model[0].load_state_dict(get_subset_modules_by_prefix(cached_backbone_state_dict, "linear1"))
        #         #     self.backbone_model[1].load_state_dict(get_subset_modules_by_prefix(cached_backbone_state_dict, "feature_weight"))
        #         #     self.backbone_model[2].load_state_dict(get_subset_modules_by_prefix(cached_backbone_state_dict, "encoder"))
        #         #     for p in self.backbone_model.parameters():
        #         #         p.requires_grad = False
        #         # self.postbone_model = 
        #     elif args.backbone.lower() == "nam":
        #         self.backbone_full_model = dragonet_model_nam(input_size, backbone_model_config["hidden_size"], num_class=args.num_class)
        #         self.backbone_model = self.backbone_full_model.encoding_features
        #         print("start loading drnet backbone")
        #         # cfg_density = [(input_size, 100, 1, 'relu'), (100, hidden_size, 1, 'relu')]
        #         # num_grid = 10
        #         # cfg = [[hidden_size, hidden_size, 1, 'relu'], [hidden_size, 1, 1, 'id']]
        #         # isenhance = 11
        #         # self.backbone_full_model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=backbone_model_config["h"], num_t=args.num_treatments, has_dose=args.has_dose, cont_treatment=args.cont_treatment)
        #         # self.backbone_model = self.backbone_full_model.encoding_features
                
        #         # cfg_density = [(num_feat_count + category_sum_count, 100, 1, 'relu'), (100, hidden_size, 1, 'relu')]
        #         # self.backbone_model =  nn.Sequential(*make_drnet_backbone(cfg_density))
        #     # self.backbone_model_name = "transtee"
        #     else:
        #         self.backbone_full_model = dragonet_model(input_size, backbone_model_config["hidden_size"])
        #         self.backbone_model = self.backbone_full_model.encoding_features
        #         # self.backbone_model = make_dragonet_backbone2(num_feat_count + category_sum_count, hidden_size)#
            
        #     if args.cached_backbone is not None and os.path.exists(args.cached_backbone):
        #         cached_backbone_state_dict = torch.load(args.cached_backbone)
        #         self.backbone_full_model.load_state_dict(cached_backbone_state_dict)
        #         if args.fix_backbone:
        #             for p in self.backbone_full_model.parameters():
        #                 p.requires_grad = False
        self.backbone = args.backbone
        self.fix_backbone = args.fix_backbone


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discretize_feat_value_count = discretize_feat_value_count
        self.id_attr, self.outcome_attr, self.treatment_attr = id_attr, outcome_attr, treatment_attr
        self.use_mlm = use_mlm
        self.removed_feat_ls = removed_feat_ls
        if self.removed_feat_ls is None:
            self.removed_feat_ls = []
        self.init_without_feat_groups(lang,  program_max_len, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=topk_act, continue_act=continue_act)
        self = self.to(self.device)
        self.num_treatments = num_treatments
        self.cont_treatment = cont_treatment
        self.classification = args.classification
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

    def obtain_predictions(self, data_ls, treatment_ls, id_attr, outcome_attr, treatment_attr, test=False):
        # if len(data) == 0:
        #     return 0
        all_treatment_pred = []
        all_outcome_pred = []
        for idx in range(len(data_ls)):
            sub_data_ls = data_ls[idx]
            sub_rwd_ls = []
            if not test:
                sub_treatment_pred = []
                sub_outcome_pred = []
                for df in sub_data_ls:
                    concat_data = pd.concat([df[[id_attr, outcome_attr, treatment_attr]] ])
                    concat_data.drop_duplicates(inplace=True)
                    treatment_pred_all = concat_data[treatment_attr]
                    if len(treatment_pred_all) <= 0:
                        treatment_pred=0.5
                    else:
                        treatment_pred=treatment_pred_all.mean()
                    
                    outcome_pred_all = concat_data.loc[concat_data[treatment_attr] == treatment_ls[idx].item(), outcome_attr]
                    if len(outcome_pred_all) <= 0:
                        outcome_pred = 0.5
                    else:
                        outcome_pred = outcome_pred_all.mean()
                    sub_treatment_pred.append(treatment_pred)
                    sub_outcome_pred.append(outcome_pred)
                all_treatment_pred.append(sub_treatment_pred)
                all_outcome_pred.append(sub_outcome_pred)
            else:
                concat_data = pd.concat([df[[id_attr, outcome_attr, treatment_attr]] for df in sub_data_ls])
                concat_data.drop_duplicates(inplace=True)
                treatment_pred_all = concat_data[treatment_attr]
                if len(treatment_pred_all) <= 0:
                    treatment_pred=0.5
                else:
                    treatment_pred=treatment_pred_all.mean()
                
                outcome_pred_all = concat_data.loc[concat_data[treatment_attr] == treatment_ls[idx].item(), outcome_attr]
                if len(outcome_pred_all) <= 0:
                    outcome_pred = 0.5
                else:
                    outcome_pred = outcome_pred_all.mean()
                all_treatment_pred.append(treatment_pred)
                all_outcome_pred.append(outcome_pred)
            
        return torch.tensor(all_treatment_pred), torch.tensor(all_outcome_pred)

    def forward_single_step(self, trainer, pooled_output, mlm_loss, A, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom, epsilon=0, eval=False, init=False, train=False, test=False , compute_ate=True, classification=False, method_two=False, method_three=False):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = [self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)]

            atom_ls = self.forward_ls0(pooled_output,X_pd_full, init_program, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, train=train)
        else:
            atom_ls = self.forward_ls0(pooled_output,X_pd_full, program, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, train=train)
        
        if not eval and program_str is not None:
        
            next_program, next_program_str, next_all_other_pats_ls, next_program_col_ls, next_outbound_mask_ls = process_curr_atoms0(trainer, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=[col_id_key, further_sel_mask_key])
            
            # next_all_other_pats_ls, transformed_expr_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)

            # treatment_pred, outcome_pred = self.obtain_predictions(next_all_other_pats_ls, A, self.id_attr, self.outcome_attr, self.treatment_attr, test=test)
            topk=3
            treatment_pred, outcome_pred, _ = obtain_predictions2(self, X_pd_full, A, None, trainer.lang, next_all_other_pats_ls, A, None, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, classification=classification, method_two=method_two, method_three=method_three)
            
            if compute_ate:
                pos_outcome, neg_outcome, _ = obtain_predictions2(self, X_pd_full, A, None, trainer.lang, next_all_other_pats_ls, A, None, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, classification=classification, compute_ate=True, method_two=method_two, method_three=method_three)
            
            ind_treatment_pred, ind_outcome_pred = obtain_individual_predictions2(self, X_pd_full, A, None, trainer.lang, next_all_other_pats_ls, A, None, self.id_attr, self.outcome_attr, self.treatment_attr, topk=topk, classification=classification, method_two=method_two, method_three=method_three)
            
            if compute_ate:
                return treatment_pred, (outcome_pred, pos_outcome, neg_outcome), mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
            else:
                return treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
            
            # treatment_pred, outcome_pred = torch.ones([len(next_all_other_pats_ls),1]), torch.ones([len(next_all_other_pats_ls), 1])
            # return treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls
        else:
            return atom_ls, mlm_loss
   
    def forward(self, trainer, text_ids, text_len, text_mask, A, X, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=False, train=False, test=False, compute_ate=True,classification=False, method_two=False, method_three=False):
        # pooled_output, mlm_loss = self.backbone_model(text_ids, text_len, text_mask, use_mlm=self.use_mlm)
        
        return self.forward_single_step(trainer, X, 0, A, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom, epsilon=epsilon, eval=eval, init=init, test=test, compute_ate=compute_ate, classification=classification, method_two=method_two, method_three=method_three)
    
    def atom_to_vector_ls0(self, atom_ls):
        return atom_to_vector_ls0_main(self, atom_ls)
    
class CausalQNet_rl(DistilBertPreTrainedModel):
    """ QNet model to estimate the conditional outcomes for the first stage
        Note the outcome Y is continuous """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # self.g_hat = nn.Linear(config.hidden_size, self.num_labels)
        # self.Q_cls = nn.ModuleDict()
        # for A in range(2):
        #   self.Q_cls['%d' % A] = nn.Sequential(
        #   nn.Linear(config.hidden_size, 200),
        #   nn.ReLU(),
        #   nn.Linear(200, 1))
        self.init_weights()
        
    def forward(self, text_ids, text_len, text_mask, use_mlm=True):
        text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
        attention_mask_class = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
        target_words = torch.gather(text_ids, 1, mask)
        mlm_labels = torch.ones(text_ids.shape).long() * -100
        if torch.cuda.is_available():
            mlm_labels = mlm_labels.cuda()
        mlm_labels.scatter_(1, mask, target_words)
        text_ids.scatter_(1, mask, MASK_IDX)

        # distilbert output
        bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
        seq_output = bert_outputs[0]
        pooled_output = seq_output[:, 0]  # CLS token

        # masked language modeling objective
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = torch.nn.functional.gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss(reduction='sum')(prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = None
        return pooled_output, mlm_loss
        # # sm = nn.Softmax(dim=1)

        # # A ~ text
        # a_text = self.g_hat(pooled_output)
        # a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.view(-1))
        # # accuracy
        # a_pred = a_text.argmax(dim=1)
        # a_acc = (a_pred == A).sum().float()/len(A) 
        
        # # Y ~ text+A
        # # conditional expected outcomes
        # Q0 = self.Q_cls['0'](pooled_output)
        # Q1 = self.Q_cls['1'](pooled_output)

        # if Y is not None:
        #     A0_indices = (A == 0).nonzero().squeeze()
        #     A1_indices = (A == 1).nonzero().squeeze()
        #     # Y loss
        #     y_loss_A1 = (((Q1.view(-1)-Y)[A1_indices])**2).sum()
        #     y_loss_A0 = (((Q0.view(-1)-Y)[A0_indices])**2).sum()
        #     y_loss = y_loss_A0 + y_loss_A1
        # else:
        #     y_loss = 0.0

        # return Q0, Q1, mlm_loss, y_loss, a_loss, a_acc
        
        # all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
        # program = []
        # outbound_mask_ls = []
        # # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        # # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
        # program_str = []
        # program_col_ls = []
        # # for p_k in range(len(program_str)):
        # #     program_str[p_k].append([[] for _ in range(self.topk_act)])
        # #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
        
        
        # X_pd_full = pd.concat(X_pd_ls)
        # all_transformed_expr_ls = []
        # all_treatment_pred = []
        # all_outcome_pred = []
        
        # prev_reward = torch.zeros([len(A), 2])
        
        # for arr_idx in range(self.program_max_len):
        #     init = (len(program) == 0)
        #     done = (arr_idx == self.program_max_len - 1)
        #     treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls = self.forward_single_step(text_ids, text_len, text_mask, X_pd_full, program, program_str, all_other_pats_ls, X_pd_ls, outbound_mask_ls, atom, device, use_mlm=use_mlm, epsilon=epsilon, eval=False, init=init)
        #     if train:
        #         reward1, reward2 = self.obtain_reward(treatment_pred, outcome_pred, A, Y)
        #         if done:
        #             next_state = None
        #         else:
        #             next_state = (next_program, next_outbound_mask_ls)
                
                
        #         transition = Transition((text_ids, text_len, text_mask), X_pd_full,(program, outbound_mask_ls), atom_ls, next_state, torch.stack([reward1, reward2], dim=1) - prev_reward)
        #         self.dqn.observe_transition(transition)
        #         #update model
        #         loss = self.dqn.optimize_model_ls0()
            
        #     if done:
        #         all_treatment_pred.append(treatment_pred)
        #         all_outcome_pred.append(outcome_pred)
        #     program = next_program
        #     program_str = next_program_str
        #     all_other_pats_ls = next_all_other_pats_ls
        #     outbound_mask_ls = next_outbound_mask_ls
            
            
            
            
        
        # a_text = self.g_hat(pooled_output)
        # a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.view(-1))
        # # accuracy
        # a_pred = a_text.argmax(dim=1)
        # a_acc = (a_pred == A).sum().float()/len(A) 
        
        # # Y ~ text+A
        # # conditional expected outcomes
        # Q0 = self.Q_cls['0'](pooled_output)
        # Q1 = self.Q_cls['1'](pooled_output)

        # if Y is not None:
        #     A0_indices = (A == 0).nonzero().squeeze()
        #     A1_indices = (A == 1).nonzero().squeeze()
        #     # Y loss
        #     y_loss_A1 = (((Q1.view(-1)-Y)[A1_indices])**2).sum()
        #     y_loss_A0 = (((Q0.view(-1)-Y)[A0_indices])**2).sum()
        #     y_loss = y_loss_A0 + y_loss_A1
        # else:
        #     y_loss = 0.0

        # return Q0, Q1, mlm_loss, y_loss, a_loss, a_acc

class QNet:
    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, train_df, valid_df, test_df, text_attr="text", treatment_attr="T", confoounder_attr="C", outcome_attr="Y", count_outcome_attr="count_Y",
                 a_weight = 1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None):
        # df['text'], df['T'], df['C'], df['Y']
        self.model = CausalQNet.from_pretrained(
            'distilbert-base-uncased',
            num_labels = 2,
            output_attentions=False,
            output_hidden_states=False)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir

        # idx = list(range(len(texts)))
        # random.shuffle(idx) # shuffle the index
        # n_train = int(len(texts)*0.8) 
        # n_val = len(texts)-n_train
        # idx_train = idx[0:n_train]
        # idx_val = idx[n_train:]

        self.train_text = train_df[text_attr]
        self.train_treatment = train_df[treatment_attr]
        self.train_confounder = train_df[confoounder_attr]
        self.train_outcome = train_df[outcome_attr]
        self.train_counter_outcome = train_df[count_outcome_attr]

        self.val_text = valid_df[text_attr]
        self.val_treatment = valid_df[treatment_attr]
        self.val_confounder = valid_df[confoounder_attr]
        self.val_outcome = valid_df[outcome_attr]
        self.val_counter_outcome = valid_df[count_outcome_attr]

        self.test_text = test_df[text_attr]
        self.test_treatment = test_df[treatment_attr]
        self.test_confounder = test_df[confoounder_attr]
        self.test_outcome = test_df[outcome_attr]
        self.test_counter_outcome = test_df[count_outcome_attr]


    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, counter_outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]
        if counter_outcomes is None:
            counter_outcomes = [-1 for _ in range(len(treatments))]
        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y, count_Y) in enumerate(zip(texts, treatments, confounders, outcomes, counter_outcomes)):
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
                                                max_length=128,
                                                truncation=True,
                                                pad_to_max_length=True)

            out['text_id'].append(encoded_sent['input_ids'])
            out['text_mask'].append(encoded_sent['attention_mask'])
            out['text_len'].append(sum(encoded_sent['attention_mask']))
            out['A'].append(A)
            out['C'].append(C)
            out['Y'].append(Y)
            out["count_Y"].append(count_Y)

        self.y_scaler = StandardScaler().fit(np.array(out["Y"] + out['count_Y']).reshape(-1,1))
        y = self.y_scaler.transform(np.array(out["Y"]).reshape(-1,1))
        out["Y"] = y.reshape(-1).tolist()

        count_y = self.y_scaler.transform(np.array(out["count_Y"]).reshape(-1,1))
        out["count_Y"] = count_y.reshape(-1).tolist()

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y', 'count_Y'])
        # if y_scaler is None:
        

        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def test(self,val_dataloader):
        # evaluate validation set
        self.model.eval()
        pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
        a_val_losses, y_val_losses, a_val_accs = [], [], []
    
        all_gt_outcome_ls = []
        all_gt_count_outcome_ls = []
        all_pos_pred_outcome_ls = []
        all_neg_pred_outcome_ls = []
        all_gt_treatment_ls = []


        for batch in pbar:
            if torch.cuda.is_available():
                batch = (x.cuda() for x in batch)
            text_id, text_len, text_mask, A, _, Y, count_Y = batch
            Q0, Q1, _, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
            
            a_val_losses.append(a_loss.item())
            y_val_losses.append(y_loss.item())

            # A accuracy
            a_acc = torch.round(a_acc*len(A))
            a_val_accs.append(a_acc.item())

            all_gt_outcome_ls.append(Y.detach().cpu())
            all_gt_count_outcome_ls.append(count_Y.detach().cpu())
            all_pos_pred_outcome_ls.append(Q1.detach().cpu())
            all_neg_pred_outcome_ls.append(Q0.detach().cpu())
            all_gt_treatment_ls.append(A.detach().cpu())
        
        all_gt_outcome_ls = torch.cat(all_gt_outcome_ls, dim=0)
        all_gt_count_outcome_ls = torch.cat(all_gt_count_outcome_ls, dim=0)
        all_pos_pred_outcome_ls = torch.cat(all_pos_pred_outcome_ls, dim=0)
        all_neg_pred_outcome_ls = torch.cat(all_neg_pred_outcome_ls, dim=0)
        all_gt_treatment_ls = torch.cat(all_gt_treatment_ls, dim=0)

        if not self.classification:
            if self.y_scaler is not None:
                all_pos_pred_outcome_ls = transform_outcome_by_rescale_back(self, all_pos_pred_outcome_ls)
                all_neg_pred_outcome_ls = transform_outcome_by_rescale_back(self, all_neg_pred_outcome_ls)
                all_gt_outcome_ls = transform_outcome_by_rescale_back(self, all_gt_outcome_ls)
                if len(all_gt_count_outcome_ls) > 0:
                    all_gt_count_outcome_ls = transform_outcome_by_rescale_back(self, all_gt_count_outcome_ls)

        if all_gt_count_outcome_ls is not None:
            gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_ls.view(-1), all_gt_treatment_ls.view(-1)], dim=1), all_gt_count_outcome_ls.reshape(-1,1))
            avg_treatment_effect = evaluate_treatment_effect_core(all_pos_pred_outcome_ls, all_neg_pred_outcome_ls, gt_treatment_outcome, gt_control_outcome)
            print("average treatment effect::%f"%(avg_treatment_effect))
        
        n_val = len(val_dataloader.dataset)
        a_val_loss = sum(a_val_losses)/n_val
        print('A Validation loss:',a_val_loss)

        y_val_loss = sum(y_val_losses)/n_val
        print('Y Validation loss:',y_val_loss)

        val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
        print('Validation loss:',val_loss)

        a_val_acc = sum(a_val_accs)/n_val
        print('A accuracy:',a_val_acc)
        return avg_treatment_effect
    
    def train(self, learning_rate=2e-5, epochs=1, patience=5):
        ''' Train the model'''

        # split data into two parts: one for training and the other for validation
        
        # list of data
        train_dataloader = self.build_dataloader(self.train_text, 
            self.train_treatment, self.train_confounder, self.train_outcome, self.train_counter_outcome)
        val_dataloader = self.build_dataloader(self.val_text, 
            self.val_treatment, self.val_confounder, self.val_outcome, self.val_counter_outcome, sampler='sequential')
        
        test_dataloader = self.build_dataloader(self.test_text,
            self.test_treatment, self.test_confounder, self.test_outcome, self.test_counter_outcome, sampler='sequential')

        self.train_dataloader = train_dataloader

        self.model.train() 
        optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps=1e-8)

        best_val_loss = 1e6
        best_test_loss = 1e6
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            losses = []
            self.model.train()
            all_pred_outcome_ls = []
            all_gt_outcome_ls = []
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, _, Y, count_Y = batch
            
                self.model.zero_grad()
                Q0, Q1, mlm_loss, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y)
                all_gt_outcome_ls.append(Y.detach().cpu())
                all_pred_outcome_ls.append((Q0.view(-1)*(Y==0).type(torch.float).view(-1) + Q1.view(-1)*(Y==1).type(torch.float).view(-1)).detach().cpu())
                # compute loss
                loss = self.loss_weights['a'] * a_loss + self.loss_weights['y'] * y_loss + self.loss_weights['mlm'] * mlm_loss
                
                       
                pbar.set_postfix({'Y loss': y_loss.item(),
                  'A loss': a_loss.item(), 'A accuracy': a_acc.item(), 
                  'mlm loss': mlm_loss.item()})

                # optimizaion for the baseline
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            all_gt_outcome_ls = torch.cat(all_gt_outcome_ls, dim=0)
            all_pred_outcome_ls = torch.cat(all_pred_outcome_ls, dim=0)
            if not self.classification:
                if self.y_scaler is not None:
                    all_pred_outcome_ls = transform_outcome_by_rescale_back(self, all_pred_outcome_ls)
                    all_gt_outcome_ls = transform_outcome_by_rescale_back(self, all_gt_outcome_ls)
            outcome_error = (torch.abs(all_pred_outcome_ls - all_gt_outcome_ls)).mean().item()
            print("training outcome error:", outcome_error)
            val_loss = self.test(val_dataloader)
            test_loss = self.test(test_dataloader)

            # early stop
            if val_loss < best_val_loss:
                torch.save(self.model, self.modeldir+'_bestmod.pt') # save the best model
                best_val_loss = val_loss
                best_test_loss = test_loss
                epochs_no_improve = 0              
            else:
                epochs_no_improve += 1
            
            print("best validation loss:", best_val_loss)
            print("best test loss:", best_test_loss)
            

            # if epoch >= 5 and epochs_no_improve >= patience:              
            #     print('Early stopping!' )
            #     print('The number of epochs is:', epoch)
            #     break

        # load the best model as the model after training
        self.model = torch.load(self.modeldir+'_bestmod.pt')

        return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get conditional expected outcomes Q0 and Q1 based on the training model'''
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = []  # E[Y|A=0, text]
        Q1s = []  # E[Y|A=1, text]
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Statistics computing")

        if not model_dir:
            self.model.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = self.model(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()
        else:
            Qmodel = torch.load(model_dir)
            Qmodel.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = Qmodel(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()

        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]
        As = np.array(As)
        Ys = np.array(Ys)
        Cs = np.array(Cs)

        return Q0s, Q1s, As, Ys, Cs



class DQN_all:
    def __init__(self, backbone_model_config, id_attr, outcome_attr, treatment_attr, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0, numeric_count=None, category_count=None, num_labels=2, category_sum_count = None, has_embeddings=False, use_mlm=True, topk_act=1, model_config=None, feat_group_names = None, removed_feat_ls=None, prefer_smaller_range = False, prefer_smaller_range_coeff=0.5, method_two=False, args = None, discretize_feat_value_count=20):
        self.mem_sample_size = mem_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.topk_act = topk_act
        torch.manual_seed(seed) 
        self.use_mlm = use_mlm           
        self.policy_net = CausalQNet_rl_all(args, backbone_model_config,lang,  program_max_len, model_config["latent_size"], dropout_p, numeric_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = topk_act, continue_act=False, num_labels=num_labels, discretize_feat_value_count=discretize_feat_value_count, use_mlm=use_mlm, removed_feat_ls= removed_feat_ls)
        # RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)
        
        self.target_net = CausalQNet_rl_all(args, backbone_model_config,lang,  program_max_len, model_config["latent_size"], dropout_p, numeric_count, category_sum_count, feat_range_mappings, id_attr, outcome_attr, treatment_attr, topk_act = topk_act, continue_act=False, num_labels=num_labels, discretize_feat_value_count=discretize_feat_value_count, use_mlm=use_mlm, removed_feat_ls= removed_feat_ls)
        # self.target_net = RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        # for p in self.policy_net.backbone_model.distilbert.parameters():
        #     p.requires_grad = False

        self.memory = ReplayMemory(replay_memory_capacity)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), learning_rate)

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
        text_ids, text_len, text_mask = features
        
        X, X_pd = data
        
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X), 1)
            # pred = self.target_net.forward(features, data, [init_program], outbound_mask_ls, ["formula"], 0, eval=False, init=True)
            
            pred, _ = self.target_net.forward(self.trainer, text_ids, text_len, text_mask, None, X, X_pd, [init_program], None, None, None, None, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=True)
            del init_program
        else:
            #program.sort()
            pred, _ = self.target_net.forward(self.trainer, text_ids, text_len, text_mask, None, X, X_pd, program, None, None, None, None, outbound_mask_ls, atom=None, epsilon=0, eval=False, init=False)
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
        
        else:
            # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors

            max_tensors = max_tensors/2
            
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

    def get_state_action_prediction_tensors_ls0(self, features, data, state, atom):
        # atom = atom_pair[0]
        # origin_atom = atom_pair[1]
        queue = list(atom.keys())
        
        program, outbound_mask_ls = state
        
        text_ids, text_len, text_mask = features
        
        # if atom[col_id_key].max() == 116:
        #     print()
        X, X_pd = data
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
            # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            # pred = self.policy_net.forward(features,X_pd, [init_program], outbound_mask_ls, atom, 0, eval=True, init=True)
            pred, mlm_loss = self.policy_net.forward(self.trainer, text_ids, text_len, text_mask, None, X, X_pd, [init_program], None, None, None, None, outbound_mask_ls, atom=atom, epsilon=0, eval=True, init=True)
            del init_program
        else:
            #program.sort()
            pred, mlm_loss = self.policy_net.forward(self.trainer, text_ids, text_len, text_mask, None, X, X_pd, program, None, None, None, None, outbound_mask_ls, atom=atom, epsilon=0, eval=True, init=False)
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
        else:
            assert torch.sum(atom_prediction_tensors*selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3
            
        atom_prediction_tensors = atom_prediction_tensors*selected_num_feat_tensor_bool + col_prediction_Q_tensor*(1-selected_num_feat_tensor_bool)

        # if atom_prediction_tensors.shape[0] < 4:
        #     print()
        # loss = torch.sum(atom_prediction_tensors)
        # self.optimizer.zero_grad()
        # loss.backward(retain_graph=True)

        return atom_prediction_tensors, mlm_loss
    
    # def get_state_action_prediction_tensors_ls0_medical(self, features, X_pd, state, atom):
    #     # atom = atom_pair[0]
    #     # origin_atom = atom_pair[1]
    #     queue = list(atom.keys())
        
    #     program = state
        
    #     if len(program) == 0:
    #         # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
    #         init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
    #         # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
    #         pred = self.policy_net.forward_ls0(features,X_pd, [init_program], atom, 0, eval=True, init=True)
    #         del init_program
    #     else:
    #         #program.sort()
    #         pred = self.policy_net.forward_ls0(features,X_pd, program, atom, 0, eval=True)

    #     x_idx = torch.tensor(list(range(len(X_pd))))
    
    #     _,col_tensor_indices = torch.topk(atom[col_Q_key].view(len(atom[col_Q_key]),-1), k=self.topk_act, dim=-1)


    #     col_prediction_Q_tensor_ls = []
        
    #     for k in range(self.topk_act):
    #         col_prediction_Q_tensor_ls.append(pred[col_Q_key].view(len(pred[col_Q_key]), -1)[x_idx, col_tensor_indices[:,k]])
        
    #     col_prediction_Q_tensor = torch.stack(col_prediction_Q_tensor_ls, dim=1)
    #     return col_prediction_Q_tensor
    
    
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
        if len(self.memory) < self.mem_sample_size: return 0.0

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
        # else:
        #     state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0_medical(f,d, p,a)) for f,d, p,a in state_action_batch]
        # state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
        state_action_values = torch.stack([t for t,l in state_action_pred])
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
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        return_loss = loss.detach().item()
        del loss
        return return_loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class QNet_rl:

    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, backbone_model_config, train_dataset, valid_dataset, test_dataset, id_attr, outcome_attr, treatment_attr, lang, learning_rate, gamma, dropout_p, feat_range_mappings, program_max_len, replay_memory_capacity, rl_config, model_config, numeric_count, category_count, category_sum_count, args, topk_act=1, num_labels=2, a_weight=1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None, log_folder = None, classification = False):

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
        self.feat_range_mappings = feat_range_mappings
        self.dqn = DQN_all(backbone_model_config, id_attr, outcome_attr, treatment_attr, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=rl_config["mem_sample_size"], seed=args.seed, numeric_count=numeric_count, category_count=category_count, num_labels=num_labels, category_sum_count = category_sum_count, topk_act=topk_act, model_config=model_config, args = args, discretize_feat_value_count=rl_config["discretize_feat_value_count"])
        self.dqn.trainer = self
        self.program_max_len = program_max_len
        self.target_update = rl_config["target_update"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification = classification
        # if torch.cuda.is_available():
        #     self.dq = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir
        print("model directory::", modeldir)
        self.do_medical = False
        self.log_folder = log_folder
        self.is_log = args.is_log
        self.method_two = args.method_two
        self.method_three = args.method_three
        self.has_dose = False
        # self.gpu_db = args.gpu_db
        self.num_class = len(lang.outcome_array.unique())
        # self.memory = ReplayMemory(replay_memory_capacity)

    def create_tabular_feature_for_text(self, encoded_sent):
        tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        model(**encoded_sent)

    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
                                                max_length=128,
                                                truncation=True,
                                                pad_to_max_length=True)

            self.create_tabular_feature_for_text(encoded_sent)
            out['text_id'].append(encoded_sent['input_ids'])
            out['text_mask'].append(encoded_sent['attention_mask'])
            out['text_len'].append(sum(encoded_sent['attention_mask']))
            out['A'].append(A)
            out['C'].append(C)
            out['Y'].append(Y)

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y'])
        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def obtain_reward(self, treatment_pred, outcome_pred, A, Y, epoch):
        treatment_pred[treatment_pred != treatment_pred] = -1
        # reward1 = ((treatment_pred > 0.5).view(-1).type(torch.float) == A.view(-1).type(torch.float)).view(-1).type(torch.float)
        reward1 = (treatment_pred > 0).type(torch.float)*(treatment_pred*(A == 1).type(torch.float) + (1-treatment_pred)*(A == 0).type(torch.float))
        # reward2 = outcome_pred.view(-1)*(Y == 1).view(-1).type(torch.float) + (1-outcome_pred.view(-1))*(Y == 0).view(-1).type(torch.float)
        if not self.classification:
            reward2 = (treatment_pred > 0).type(torch.float)*torch.exp(-self.reward_coeff*(outcome_pred - Y)**2)
        else:            
            selected_outcome_pred = outcome_pred[torch.arange(outcome_pred.shape[0]), :, Y.view(-1).long()]
        
            reward2 = (selected_outcome_pred >= 0).type(torch.float)*selected_outcome_pred.type(torch.float)
            # reward1 = reward2
        # reward1 = (treatment_pred > 0).type(torch.float)*torch.exp(-self.reward_coeff*(outcome_pred - Y)**2)
        # reward2 = treatment_pred*(A == 1).view(-1, 1).type(torch.float) + (1-treatment_pred)*(A == 0).view(-1, 1).type(torch.float)
        if epoch < 5:
            return reward1, reward2
        else:
            return reward2, reward2
   
    
    def copy_data_in_database(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(torch.clone(all_other_pats_ls[idx]))
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    
    
    def observe_transition(self, transition: Transition):
        self.dqn.memory.push(transition)
    
    
    def test(self, test_loader, return_explanations=False):
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            all_treatment_ls = []
            all_outcome_ls = []
            all_gt_treatment_ls = []
            all_gt_outcome_ls = []
            all_gt_count_outcome_ls=[]
            all_pos_outcome_pred_ls = []
            all_neg_outcome_pred_ls = []
            all_program_str_ls = []
            all_program_col_ls = []
            all_res_other_pat_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls, X), Y, A = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                text_id_ls = text_id_ls.to(self.device)
                text_mask_ls = text_mask_ls.to(self.device)
                text_len_ls = text_len_ls.to(self.device)
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
            
                # if self.gpu_db:
                origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]

                all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
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
                
                outcome_pred_curr_batch=torch.zeros([len(A), self.num_class])
                # if self.gpu_db:
                outcome_pred_curr_batch = outcome_pred_curr_batch.to(self.device)
                # reg_outcome_pred_curr_batch = torch.zeros(len(A))
                pos_outcome_curr_batch=torch.zeros([len(A), self.num_class])
                neg_outcome_curr_batch=torch.zeros([len(A), self.num_class])
                # reg_pos_outcome_curr_batch=torch.zeros(len(A))
                # reg_neg_outcome_curr_batch=torch.zeros(len(A))
                # if self.gpu_db:
                pos_outcome_curr_batch = pos_outcome_curr_batch.to(self.device)
                neg_outcome_curr_batch = neg_outcome_curr_batch.to(self.device)
                # if self.gpu_db:
                stopping_conds = torch.ones_like(A).bool().to(self.device)
                # else:   
                #     stopping_conds = torch.ones_like(A.cpu()).bool()

                for arr_idx in range(self.program_max_len):
                    init = (len(program) == 0)
                    done = (arr_idx == self.program_max_len - 1)
                    # treatment_pred, (outcome_pred, pos_outcome, neg_outcome), mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
                    treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred = self.dqn.policy_net.forward(self, text_id_ls, text_len_ls, text_mask_ls, A, X, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, test=True, compute_ate=True, classification=self.classification, method_two=self.method_two, method_three=self.method_three)
                    stopping_conds = torch.logical_and(stopping_conds.view(-1), (treatment_pred != -1).view(-1))
                    # if done:
                    outcome_pred, pos_pred, neg_pred = outcome_pred  
                    if self.dqn.policy_net.backbone_full_model is not None:
                        reg_outcome_pred = torch.zeros_like(outcome_pred)
                        reg_ind_outcome_pred = torch.zeros_like(ind_outcome_pred)
                        reg_pos_outcome_pred = torch.zeros_like(pos_pred)
                        reg_neg_outcome_pred = torch.zeros_like(neg_pred)
                        
                        # reg_neg_outcome_pred[torch.arange(len(Y)), Y.long().view(-1)] = self.dqn.policy_net.backbone_full_model[str(0)](X).view(-1)
                        for mid in range(2):
                            curr_reg_outcome = self.dqn.policy_net.backbone_full_model[str(mid)](X)
                            if mid == 1:
                                reg_pos_outcome_pred[torch.arange(len(Y)), Y.long().view(-1)] = curr_reg_outcome.view(-1)
                            else:
                                reg_neg_outcome_pred[torch.arange(len(Y)), Y.long().view(-1)] = curr_reg_outcome.view(-1)

                            reg_outcome_pred[A.view(-1) == mid, Y.long()[A==mid]] = curr_reg_outcome[A.view(-1)==mid].view(-1)
                            reg_ind_outcome_pred[A.view(-1) == mid] = ind_outcome_pred[A.view(-1)==mid]
                        

                        ind_outcome_pred = (ind_outcome_pred + self.dqn.policy_net.regression_ratio*reg_ind_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)
                        outcome_pred = (outcome_pred + self.dqn.policy_net.regression_ratio*reg_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)
                        pos_pred = (pos_pred + self.dqn.policy_net.regression_ratio*reg_pos_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)
                        neg_pred = (neg_pred + self.dqn.policy_net.regression_ratio*reg_neg_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)

                    outcome_pred_curr_batch[stopping_conds] = outcome_pred[stopping_conds]
                    pos_outcome_curr_batch[stopping_conds] = pos_pred[stopping_conds]
                    neg_outcome_curr_batch[stopping_conds] = neg_pred[stopping_conds]

                    program = next_program
                    program_str = next_program_str
                    program_col_ls = next_program_col_ls
                    all_other_pats_ls = next_all_other_pats_ls
                    outbound_mask_ls = next_outbound_mask_ls
                all_treatment_ls.append(treatment_pred)
                all_outcome_ls.append(outcome_pred)
                all_pos_outcome_pred_ls.append(pos_pred)
                all_neg_outcome_pred_ls.append(neg_pred)
                all_gt_treatment_ls.append(A.view(-1))
                all_gt_outcome_ls.append(Y.view(-1))
                all_program_str_ls.extend(program_str)
                all_program_col_ls.extend(program_col_ls)
                all_res_other_pat_ls.extend(all_other_pats_ls)
                if count_Y is not None:
                    all_gt_count_outcome_ls.append(count_Y.view(-1))
            all_treatment_pred_tensor = torch.cat(all_treatment_ls).cpu()
            all_outcome_pred_tensor = torch.cat(all_outcome_ls).cpu()
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls).cpu()
            all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls).cpu()
            all_gt_count_treatment_tensor = None
            all_pos_outcome_pred_tensor = None
            all_neg_outcome_pred_tensor = None
            if len(all_pos_outcome_pred_ls) > 0:
                all_pos_outcome_pred_tensor = torch.cat(all_pos_outcome_pred_ls).cpu()
            if len(all_neg_outcome_pred_ls) > 0:
                all_neg_outcome_pred_tensor = torch.cat(all_neg_outcome_pred_ls).cpu()
            if len(all_gt_count_outcome_ls) > 0:
                all_gt_count_treatment_tensor = torch.cat(all_gt_count_outcome_ls).cpu()


            if not self.classification:
                self.reward_coeff = -torch.log(torch.tensor(0.5))/torch.median((all_outcome_pred_tensor.view(-1) - all_gt_outcome_tensor.view(-1))**2)
                if test_loader.dataset.y_scaler is not None:
                    all_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_outcome_pred_tensor)
                    all_gt_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_outcome_tensor)
                    if len(all_gt_count_outcome_ls) > 0:
                        all_gt_count_treatment_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_count_treatment_tensor)
                    if all_pos_outcome_pred_tensor is not None:
                        all_pos_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_pos_outcome_pred_tensor)
                    if all_neg_outcome_pred_tensor is not None:
                        all_neg_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_neg_outcome_pred_tensor)
            all_treatment_arr_np = all_gt_treatment_tensor.view(-1).numpy()
            all_treatment_pred_tensor = all_treatment_pred_tensor.cpu()
            all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()
            treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
            if len(np.unique(all_treatment_arr_np)) <= 1:
                treatment_auc = 0
            else:
                treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                
            
    
            if not self.classification:
                outcome_error = F.mse_loss(all_outcome_pred_tensor.view(-1,1), all_gt_outcome_tensor.view(-1,1))
            else:
                outcome_error = F.cross_entropy(all_outcome_pred_tensor, all_gt_outcome_tensor.view(-1).long())

            

            assert torch.sum(torch.isnan(outcome_error)) == 0
            
            print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            if all_gt_count_treatment_tensor is not None:
                gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_tensor.view(-1), all_gt_treatment_tensor.view(-1)], dim=1), all_gt_count_treatment_tensor.reshape(-1,1))
                avg_treatment_effect = evaluate_treatment_effect_core(all_pos_outcome_pred_tensor, all_neg_outcome_pred_tensor, gt_treatment_outcome, gt_control_outcome)
                print("average treatment effect error::%f"%(avg_treatment_effect))
            else:
                if self.classification:
                    avg_treatment_effect = torch.mean(torch.norm(all_pos_outcome_pred_tensor - all_neg_outcome_pred_tensor, dim=-1))
                    print("average treatment effect::%f"%(avg_treatment_effect))
                else:
                    avg_treatment_effect = torch.mean(torch.abs(all_pos_outcome_pred_tensor - all_neg_outcome_pred_tensor))
                    print("average treatment effect::%f"%(avg_treatment_effect))
            if not return_explanations:
                return outcome_error, avg_treatment_effect
            else:
                return all_program_str_ls,all_program_col_ls, all_outcome_pred_tensor, all_pos_outcome_pred_tensor, all_neg_outcome_pred_tensor
    
    def eval_sufficiency(self, test_loader, predicted_y, origin_explanation_str_ls, fp=0.2):
        all_exp_ls = transform_explanation_str_to_exp(test_loader.dataset, origin_explanation_str_ls)
        
        with torch.no_grad():
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            all_matched_ratio_ls = []
            for sample_id in range(len(test_loader.dataset)):
                # idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = test_loader.dataset[sample_id]
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, _ = test_loader.dataset[sample_id]
                curr_exp_ls = all_exp_ls[sample_id]
                all_matched_features_boolean =  eval_booleans(curr_exp_ls, test_loader.dataset.origin_features)
                # all_matched_features_boolean = curr_exp[1](test_loader.dataset.features[curr_exp[0]], curr_exp[2])
                all_matched_pred_labels = predicted_y[all_matched_features_boolean]
                # if self.has_dose:
                #     all_dose_array = test_loader.dataset.dose_array[all_matched_features_boolean]
                #     topk_ids = torch.topk(torch.abs(all_dose_array - D).view(-1), k=topk, largest=False)[1]
                #     all_matched_pred_labels = all_matched_pred_labels[topk_ids]
                # else:
                all_treatment_array = test_loader.dataset.treatment_array[all_matched_features_boolean]
                topk_ids = torch.topk(torch.abs(all_treatment_array - A).view(-1), k=min(topk, len(all_treatment_array)), largest=False)[1]
                all_matched_pred_labels = all_matched_pred_labels[topk_ids]
                # print(all_matched_pred_labels)
                if self.classification:
                    matched_sample_count = torch.sum(torch.norm(all_matched_pred_labels - predicted_y[sample_id], dim=-1) < fp).item()-1
                else:
                    matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                matched_sample_count = max(matched_sample_count, 0)
                matched_ratio = matched_sample_count*1.0/(max(len(all_matched_pred_labels) - 1, 0)+1e-6)
                all_matched_ratio_ls.append(matched_ratio)
            
            sufficiency_score = np.array(all_matched_ratio_ls).mean()
            
            
            print("sufficiency score::", sufficiency_score)
    
    # def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5, epsilon=0, use_mlm=True):
    def train(self, train_dataset, valid_dataset, test_dataset):
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
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=NLP_Dataset.collate_fn, drop_last=True)
        val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        

        # self.model.train() 
        # optimizer = AdamW(self.model.parameters(), lr = self.learning_rate, eps=1e-8)

        best_loss = 1e6
        epochs_no_improve = 0
        if self.is_log:
            origin_explanation_str_ls,origin_explanation_col_ls, all_outcome_pred_tensor, all_pos_outcome_tensor, all_neg_outcome_tensor = self.test(test_dataloader, return_explanations=True)
            self.eval_sufficiency(test_dataloader, all_pos_outcome_tensor-all_neg_outcome_tensor, origin_explanation_str_ls)
            exit(1)
            # self.test(test_dataloader)

        # print("evaluation on training set::")
        # self.test(train_dataloader)
        
        # print("evaluation on validation set::")
        # self.test(val_dataloader)
        
        # print("evaluation on test set::")
        # self.test(test_dataloader)
        best_val_outcome_err = 1e6
        best_test_outcome_err = 1e6
        best_val_avg_treatment_effect = 1e6
        best_test_avg_treatment_effect = 1e6
        best_outcome_epoch = -1
        # self.test(val_dataloader)
        if not self.classification:
            print("update reward coefficient::", self.reward_coeff)
        for epoch in range(self.epochs):
            losses = []
            self.dqn.policy_net.train()
            self.dqn.target_net.train()
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            all_treatment_pred = []
            all_outcome_pred = []
            all_gt_treatment = []
            all_gt_outcome = []
            all_gt_count_outcome = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls, X), Y, A = batch
                text_id_ls = text_id_ls.to(self.device)
                text_mask_ls = text_mask_ls.to(self.device)
                text_len_ls = text_len_ls.to(self.device)
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
            
                # if self.gpu_db:
                origin_all_other_pats_ls = [x.to(self.device) for x in origin_all_other_pats_ls]

                all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
                program = []
                outbound_mask_ls = []
                # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_str = []
                program_col_ls = []
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = origin_X
                all_transformed_expr_ls = []                   

                
                prev_reward = torch.zeros([len(A), self.topk_act, 2])
                
                for arr_idx in range(self.program_max_len):
                    init = (len(program) == 0)
                    done = (arr_idx == self.program_max_len - 1)
                    # treatment_pred, (outcome_pred, pos_outcome, neg_outcome), mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
                    treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred = self.dqn.policy_net.forward(self, text_id_ls, text_len_ls, text_mask_ls, A, X, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=self.epsilon, eval=False, init=init, compute_ate=False, classification=self.classification)
                    
                    if self.dqn.policy_net.backbone_full_model is not None:
                        reg_outcome_pred = torch.zeros_like(outcome_pred)
                        reg_ind_outcome_pred = torch.zeros_like(ind_outcome_pred)
                        for mid in range(2):
                            curr_reg_outcome = self.dqn.policy_net.backbone_full_model[str(mid)](X)
                            reg_outcome_pred[A.view(-1) == mid, Y.long()[A==mid]] = curr_reg_outcome[A.view(-1)==mid].view(-1)
                            reg_ind_outcome_pred[A.view(-1) == mid] = ind_outcome_pred[A.view(-1)==mid]

                        ind_outcome_pred = (ind_outcome_pred + self.dqn.policy_net.regression_ratio*reg_ind_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)
                        outcome_pred = (outcome_pred + self.dqn.policy_net.regression_ratio*reg_outcome_pred)/(1 + self.dqn.policy_net.regression_ratio)
                        

                    reward1, reward2 = self.obtain_reward(ind_treatment_pred, ind_outcome_pred, A, Y, epoch)
                    if done:
                        next_state = None
                    else:
                        next_state = (next_program, next_outbound_mask_ls)
                    
                    
                    transition = Transition((text_id_ls, text_len_ls, text_mask_ls), (X, X_pd_full),(program, outbound_mask_ls), atom_ls, next_state, torch.stack([reward1, reward2], dim=-1).cpu() - prev_reward)
                    self.observe_transition(transition)
                    #update model
                    loss = self.dqn.optimize_model_ls0()
                    
                    if done:
                        all_treatment_pred.append(treatment_pred.cpu())
                        all_outcome_pred.append(outcome_pred.cpu())
                        all_gt_treatment.append(A.cpu())
                        all_gt_outcome.append(Y.cpu())
                        if count_Y is not None:
                            all_gt_count_outcome.append(count_Y.cpu())

                    program = next_program
                    program_str = next_program_str
                    program_col_ls = next_program_col_ls
                    all_other_pats_ls = next_all_other_pats_ls
                    outbound_mask_ls = next_outbound_mask_ls
                    
                    prev_reward = torch.stack([reward1, reward2], dim=-1).cpu()

                if step % self.target_update == 0:
                    self.dqn.update_target()
                
                losses.append(loss)
            print("Training loss: ", np.mean(np.array(losses)))
            all_treatment_pred_array = torch.cat(all_treatment_pred).numpy()
            all_treatment_gt_array = torch.cat(all_gt_treatment).numpy()
            training_acc = np.mean((all_treatment_pred_array > 0.5).reshape(-1).astype(float) == all_treatment_gt_array.reshape(-1))
            training_auc = roc_auc_score(all_treatment_gt_array.reshape(-1), all_treatment_pred_array.reshape(-1))
            
            print("training accuracy::", training_acc)
            print("training auc score::", training_auc)


            all_outcome_pred_array = torch.cat(all_outcome_pred)
            all_gt_outcome_pred_array = torch.cat(all_gt_outcome)
            if len(all_gt_count_outcome) > 0:
                all_gt_count_outcome_array = torch.cat(all_gt_count_outcome)                

            if not self.classification:
                self.reward_coeff = -torch.log(torch.tensor(0.5))/torch.median((all_outcome_pred_array.view(-1) - all_gt_outcome_pred_array.view(-1))**2)
                if train_dataset.y_scaler is not None:
                    all_outcome_pred_array = transform_outcome_by_rescale_back(train_dataset, all_outcome_pred_array)
                    all_gt_outcome_pred_array = transform_outcome_by_rescale_back(train_dataset, all_gt_outcome_pred_array)
                    if len(all_gt_count_outcome) > 0:
                        all_gt_count_outcome_array = transform_outcome_by_rescale_back(train_dataset, all_gt_count_outcome_array)
            
            if not self.classification:
                outcome_error = F.mse_loss(all_outcome_pred_array.view(-1,1), all_gt_outcome_pred_array.view(-1,1)).item()
            else:
                outcome_error = F.cross_entropy(all_outcome_pred_array, all_gt_outcome_pred_array.view(-1).long()).item()
            
            
            print("performance at epoch ", epoch)
            print("training outcome error::", outcome_error)
            # print("evaluation on training set::")
            # self.test(train_dataloader)
            
            print("evaluation on validation set::")
            val_error, val_ate = self.test(val_dataloader)
            
            print("evaluation on test set::")
            test_error, test_ate = self.test(test_dataloader)
            # self.dqn.update_target()
            
            self.epsilon *= self.epsilon_falloff

            if val_error < best_val_outcome_err:
                best_val_outcome_err = val_error
                best_test_outcome_err = test_error
                best_outcome_epoch = epoch
                best_val_ate = val_ate
                best_test_ate = test_ate
                torch.save(self.dqn.policy_net.state_dict(), self.modeldir+'_bestmod.pt')
            
            # print("outcome error at epoch %d::"%(epoch))
            print("best outcome epoch::", best_outcome_epoch)
            print("best validation outcome error::", best_val_outcome_err)
            print("best test outcome error::", best_test_outcome_err)
            print("best valid ate::", best_val_ate)
            print("best test ate::", best_test_ate)
            

        #     # evaluate validation set
        #     self.model.eval()
        #     pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
        #     a_val_losses, y_val_losses, a_val_accs = [], [], []
        
        #     for batch in pbar:
        #         if torch.cuda.is_available():
        #             batch = (x.cuda() for x in batch)
        #         text_id, text_len, text_mask, A, _, Y = batch
        #         _, _, _, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
                
        #         a_val_losses.append(a_loss.item())
        #         y_val_losses.append(y_loss.item())

        #         # A accuracy
        #         a_acc = torch.round(a_acc*len(A))
        #         a_val_accs.append(a_acc.item())


        #     a_val_loss = sum(a_val_losses)/n_val
        #     print('A Validation loss:',a_val_loss)

        #     y_val_loss = sum(y_val_losses)/n_val
        #     print('Y Validation loss:',y_val_loss)

        #     val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
        #     print('Validation loss:',val_loss)

        #     a_val_acc = sum(a_val_accs)/n_val
        #     print('A accuracy:',a_val_acc)


        #     # early stop
        #     if val_loss < best_loss:
        #         torch.save(self.model, self.modeldir+'_bestmod.pt') # save the best model
        #         best_loss = val_loss
        #         epochs_no_improve = 0              
        #     else:
        #         epochs_no_improve += 1
           
        #     if epoch >= 5 and epochs_no_improve >= patience:              
        #         print('Early stopping!' )
        #         print('The number of epochs is:', epoch)
        #         break

        # # load the best model as the model after training
        # self.model = torch.load(self.modeldir+'_bestmod.pt')

        # return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get conditional expected outcomes Q0 and Q1 based on the training model'''
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = []  # E[Y|A=0, text]
        Q1s = []  # E[Y|A=1, text]
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Statistics computing")

        if not model_dir:
            self.model.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = self.model(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()
        else:
            Qmodel = torch.load(model_dir)
            Qmodel.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = Qmodel(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()

        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]
        As = np.array(As)
        Ys = np.array(Ys)
        Cs = np.array(Cs)

        return Q0s, Q1s, As, Ys, Cs

class QNet_rl_baseline:

    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, train_dataset, valid_dataset, test_dataset, id_attr, outcome_attr, treatment_attr, lang, learning_rate, gamma, dropout_p, feat_range_mappings, program_max_len, replay_memory_capacity, rl_config, model_config, backbone_model_config, numeric_count, category_count, category_sum_count, args, topk_act=1, num_labels=2, a_weight=1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None, log_folder = None, classification = False):

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
        self.feat_range_mappings = feat_range_mappings
        # self.dqn = DQN_all(id_attr, outcome_attr, treatment_attr, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=rl_config["mem_sample_size"], seed=args.seed, numeric_count=numeric_count, category_count=category_count, num_labels=num_labels, category_sum_count = category_sum_count, topk_act=topk_act, model_config=model_config, args = args, discretize_feat_value_count=rl_config["discretize_feat_value_count"])
        # self.dqn.trainer = self
        if args.dataset_name == "EEEC":
            num_class = 5
        else:
            num_class = None
        
        if args.method.lower() == "enrl":
            if train_dataset.feat_range_mappings is not None:
                feat_range_mappings = [train_dataset.feat_range_mappings[num_feat] for num_feat in train_dataset.num_cols]
            else:
                feat_range_mappings = [[0,1] for num_feat in train_dataset.num_cols]
            multihot_f_unique_num_ls = [len(train_dataset.cat_unique_count_mappings[cat_feat]) for cat_feat in train_dataset.cat_cols]
            
            self.model = ENRL(rule_len = args.program_max_len, rule_n = args.topk_act, numeric_f_num=len(train_dataset.num_cols), multihot_f_num=len(train_dataset.cat_cols), multihot_f_dim=sum(multihot_f_unique_num_ls), feat_range_mappings=feat_range_mappings, num_treatment=args.num_treatments, cont_treatment=False, has_dose=False, num_class= num_class)
        elif args.method.lower() == "nam":
            print(model_config)
            self.model = dragonet_model_nam(numeric_count+category_sum_count, model_config["hidden_size"], num_class=num_class)
            self.shared_hidden_dim = model_config["hidden_size"]
        elif args.method.lower() == "transtee":
            # hparams = {'batch_size': 200, 'dropout': 0.1, 'bert_state_dict': None, 'label_size': 5, 'name': 'POMS_F'}
            # self.model = TransTEE_nlp(hparams)
            params = {'num_features': numeric_count+category_sum_count, 'num_treatments': args.num_treatments,
            # 'h_dim': model_config["hidden_size"], 'cov_dim':model_config["cov_dim"]}
            'h_dim': model_config["hidden_size"], 'cov_dim':numeric_count+category_sum_count}
            self.model = TransTEE(params, has_dose=False, cont_treatment = False, num_class= num_class )

        
        self.backbone_full_model = None
        self.backbone = args.backbone
        if args.backbone is not None:
            if args.backbone.lower() == "transtee":
                print("start loading transtee backbone")
                # if not args.cont_treatment and args.has_dose:
                #     cov_dim = backbone_model_config["cov_dim"]
                # else:
                cov_dim = numeric_count+category_sum_count
                
                params = {'num_features': numeric_count+category_sum_count, 'num_treatments': args.num_treatments,
                'h_dim': backbone_model_config["hidden_size"], 'cov_dim':cov_dim}
                # self.model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                self.shared_hidden_dim = backbone_model_config["hidden_size"]
                # self.backbone_full_model = TransTEE_image(params)
                self.backbone_full_model = TransTEE(params, has_dose=False, cont_treatment = False, num_class= num_class )
                # self.backbone_model = self.backbone_full_model.encoding_features
        
        if args.cached_backbone is not None and os.path.exists(args.cached_backbone):
            cached_backbone_state_dict = torch.load(args.cached_backbone)
            self.backbone_full_model.load_state_dict(cached_backbone_state_dict.state_dict())
            if args.fix_backbone:
                for p in self.backbone_full_model.parameters():
                    p.requires_grad = False
        if self.backbone_full_model is not None:
            self.backbone_full_model = self.backbone_full_model.cuda()

        self.regression_ratio = args.regression_ratio

        self.program_max_len = program_max_len
        self.target_update = rl_config["target_update"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification = classification
        # if torch.cuda.is_available():
        #     self.dq = self.model.cuda()
        self.model = self.model.cuda()
        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir
        print("model directory::", modeldir)
        self.do_medical = False
        self.log_folder = log_folder
        self.is_log = args.is_log
        self.method_two = args.method_two
        self.method_three = args.method_three
        self.method = args.method
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.has_dose = False
        self.cont_treatment = False
        self.outcome_regression = not classification
        self.num_treatments = args.num_treatments
        # self.memory = ReplayMemory(replay_memory_capacity)

    def eval_self_interpretable_models(self, test_dataset, train_dataset, max_depth, subset_count = 10, all_selected_feat_ls = None, explanation_type="decision_tree"):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        full_base_X = train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        self.model.load_state_dict(torch.load(self.log_folder+ '_bestmod.pt').state_dict())
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
            all_topk_features_by_treatment = {k:[] for k in range(self.num_treatments)}
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                # idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                D = None
                # if D is not None:
                #     D = D.to(self.device)

                # X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                # X_num = X_num.to(self.device)
                # X_cat = X_cat.to(self.device)
                # pred, _ = self.model.forward(X_num, X_cat, A, d=D)
                
                _, pred = self.model.forward(X, A, d=D)
                for k in range(self.num_treatments):
                    topk_feature_ids = self.model.get_topk_features(X, A, d=D, k=max_depth)
                    all_topk_features_by_treatment[k].append(topk_feature_ids.cpu())

                if test_loader.dataset.y_scaler is not None:
                    pred = transform_outcome_by_rescale_back(test_loader.dataset, pred.cpu())

                all_orig_outcome_ls.append(pred.cpu().view(-1))

                for treatment_id in range(self.num_treatments):
                    # _, curr_outcome_pred = self.model.forward(X, A, d=D)
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
            all_topk_features_by_treatment = {k:torch.cat(all_topk_features_by_treatment[k], dim=0) for k in range(self.num_treatments)}
            all_treatment_tensor = torch.cat(all_treatment_ls, dim=0)
            all_origin_outcome_tensor = torch.cat(all_orig_outcome_ls)
            # for treatment_id in range(self.num_treatments):
            #     tree = DecisionTreeRegressor(max_depth=max_depth)
            #     tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
            #     all_tree_by_treatment_id[treatment_id] = tree
            return all_topk_features_by_treatment, all_outcome_tensor_by_treatment, all_origin_outcome_tensor

    def create_tabular_feature_for_text(self, encoded_sent):
        tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        model(**encoded_sent)
    
    def test(self, test_loader, return_pred=False):
        with torch.no_grad():
            self.model.eval()
            # self.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')
            # all_treatment_ls = []
            all_outcome_ls = []
            all_gt_treatment_ls = []
            all_gt_outcome_ls = []
            all_gt_count_outcome_ls=[]
            all_pos_outcome_pred_ls = []
            all_neg_outcome_pred_ls = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls, X), Y, A = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                
                A = A.to(self.device)
                
                if self.method == "nam":
                    X = X.to(self.device)
                    A = A.to(self.device)
                    _, pred, full_pred = self.model.forward(X, A, test=True)
                elif self.method == "ENRL":
                    X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                    X_num = X_num.to(self.device)
                    X_cat = X_cat.to(self.device)
                    pred, full_pred = self.model(X_num, X_cat, A)
                elif self.method == "TransTEE":
                    # text_id_ls = text_id_ls.to(self.device)
                    # text_mask_ls = text_mask_ls.to(self.device)
                    X = X.to(self.device)
                    A = A.to(self.device)
                    Y = Y.to(self.device)
                    _, pred, full_pred = self.model(X, A, test=True)
                    # pred, _ = self.model.forward(X, A)

                # text_id_ls = text_id_ls.to(self.device)
                # text_mask_ls = text_mask_ls.to(self.device)
                # text_len_ls = text_len_ls.to(self.device)
                # # Y = Y.to(self.device)
                # # A = A.to(self.device)
            
                # all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
                # program = []
                # outbound_mask_ls = []
                # # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_str = []
                # program_col_ls = []
                # # for p_k in range(len(program_str)):
                # #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                # #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                # X_pd_full = origin_X#pd.concat(X_pd_ls)
                # # all_transformed_expr_ls = []
                # # all_treatment_pred = []
                # # all_outcome_pred = []
                
                # # prev_reward = torch.zeros([len(A), 2])
                
                # for arr_idx in range(self.program_max_len):
                #     init = (len(program) == 0)
                #     done = (arr_idx == self.program_max_len - 1)
                #     # treatment_pred, (outcome_pred, pos_outcome, neg_outcome), mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
                #     treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred = self.dqn.policy_net.forward(self, text_id_ls, text_len_ls, text_mask_ls, A, X, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=0, eval=False, init=init, test=True, compute_ate=True, classification=self.classification, method_two=self.method_two, method_three=self.method_three)
                #     if done:
                #         outcome_pred, pos_pred, neg_pred = outcome_pred             
                #         all_treatment_ls.append(treatment_pred)
                #         all_outcome_ls.append(outcome_pred)
                #         all_pos_outcome_pred_ls.append(pos_pred)
                #         all_neg_outcome_pred_ls.append(neg_pred)
                        
                #     program = next_program
                #     program_str = next_program_str
                #     program_col_ls = next_program_col_ls
                #     all_other_pats_ls = next_all_other_pats_ls
                #     outbound_mask_ls = next_outbound_mask_ls
                all_gt_treatment_ls.append(A.view(-1).cpu())
                all_gt_outcome_ls.append(Y.view(-1).cpu())
                all_outcome_ls.append(pred.view(len(Y), -1).cpu())
                all_pos_outcome_pred_ls.append(full_pred[:,1].view(len(Y), -1))
                all_neg_outcome_pred_ls.append(full_pred[:,0].view(len(Y), -1))
                if count_Y is not None:
                    all_gt_count_outcome_ls.append(count_Y.view(-1))
            # all_treatment_pred_tensor = torch.cat(all_treatment_ls)
            all_outcome_pred_tensor = torch.cat(all_outcome_ls)
            all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
            all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
            all_gt_count_treatment_tensor = None
            all_pos_outcome_pred_tensor = None
            all_neg_outcome_pred_tensor = None
            if len(all_pos_outcome_pred_ls) > 0:
                all_pos_outcome_pred_tensor = torch.cat(all_pos_outcome_pred_ls)
            if len(all_neg_outcome_pred_ls) > 0:
                all_neg_outcome_pred_tensor = torch.cat(all_neg_outcome_pred_ls)
            if len(all_gt_count_outcome_ls) > 0:
                all_gt_count_treatment_tensor = torch.cat(all_gt_count_outcome_ls)


            if not self.classification:
                self.reward_coeff = -torch.log(torch.tensor(0.5))/torch.median((all_outcome_pred_tensor.view(-1) - all_gt_outcome_tensor.view(-1))**2)
                if test_loader.dataset.y_scaler is not None:
                    all_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_outcome_pred_tensor)
                    all_gt_outcome_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_outcome_tensor)
                    if len(all_gt_count_outcome_ls) > 0:
                        all_gt_count_treatment_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_gt_count_treatment_tensor)
                    if all_pos_outcome_pred_tensor is not None:
                        all_pos_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_pos_outcome_pred_tensor)
                    if all_neg_outcome_pred_tensor is not None:
                        all_neg_outcome_pred_tensor = transform_outcome_by_rescale_back(test_loader.dataset, all_neg_outcome_pred_tensor)
            all_treatment_arr_np = all_gt_treatment_tensor.view(-1).numpy()
            
            # all_pred_treatment_arr_full_d = (all_treatment_pred_tensor > 0.5).type(torch.long).view(-1).numpy()
            # treatment_acc = np.mean(all_treatment_arr_np == all_pred_treatment_arr_full_d)
            # if len(np.unique(all_treatment_arr_np)) <= 1:
            #     treatment_auc = 0
            # else:
            #     treatment_auc = roc_auc_score(all_treatment_arr_np.astype(int), all_treatment_pred_tensor.numpy())
                
            
    
            if not self.classification:
                outcome_error = F.mse_loss(all_outcome_pred_tensor.view(-1,1), all_gt_outcome_tensor.view(-1,1))
            else:
                outcome_error = -F.cross_entropy(all_outcome_pred_tensor.cpu(), all_gt_outcome_tensor.long())

            

            assert torch.sum(torch.isnan(outcome_error)) == 0
            
                           
            
            # print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
            if all_gt_count_treatment_tensor is not None:
                gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_tensor.view(-1), all_gt_treatment_tensor.view(-1)], dim=1), all_gt_count_treatment_tensor.reshape(-1,1))
                avg_treatment_effect = evaluate_treatment_effect_core(all_pos_outcome_pred_tensor, all_neg_outcome_pred_tensor, gt_treatment_outcome, gt_control_outcome)
                print("average treatment effect error::%f"%(avg_treatment_effect))
                if return_pred:
                    return all_outcome_pred_tensor, all_pos_outcome_pred_tensor, all_neg_outcome_pred_tensor
            else:
                if self.classification:
                    all_pos_outcome_pred_tensor = torch.softmax(all_pos_outcome_pred_tensor, dim=-1)
                    all_neg_outcome_pred_tensor = torch.softmax(all_neg_outcome_pred_tensor, dim=-1)
                    avg_treatment_effect = torch.mean(torch.norm(all_pos_outcome_pred_tensor - all_neg_outcome_pred_tensor, dim=-1))
                    print("average treatment effect::%f"%(avg_treatment_effect))
                else:
                    avg_treatment_effect = torch.mean(torch.abs(all_pos_outcome_pred_tensor - all_neg_outcome_pred_tensor))
                    print("average treatment effect::%f"%(avg_treatment_effect))
                if return_pred:
                    return all_outcome_pred_tensor
            return outcome_error, avg_treatment_effect
    
    def posthoc_explain(self, test_dataset, tree_depth=2, explanation_type="decision_tree", subset_ids=None):
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        # test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        # self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        self.model.load_state_dict(torch.load(self.log_folder+'_bestmod.pt').state_dict())
        with torch.no_grad():
            
            self.model.eval()
            pbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Validating')
            

            feature_ls = []
            origin_feature_ls = []
            unique_treatment_set = set()
            unique_dose_set = set()
            unique_treatment_dose_set = set()


            for batch in pbar:

                # idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                
                feature_ls.append(X.detach().cpu())
                origin_feature_ls.append(origin_X.detach().cpu())
                curr_treatment_ls = A.detach().cpu().view(-1).tolist()
                curr_dose_ls = None
                # if D is not None:
                #     curr_dose_ls = D.detach().cpu().view(-1).tolist()
                
                unique_treatment_set.update(curr_treatment_ls)
                if curr_dose_ls is not None:
                    unique_dose_set.update(curr_dose_ls)
                    unique_treatment_dose_set.update((zip(curr_treatment_ls, curr_dose_ls)))
            
            feature_tensor = torch.cat(feature_ls, dim=0)
            origin_feature_tensor = torch.cat(origin_feature_ls, dim=0)

        return obtain_post_hoc_explanatios_main(self, test_dataset, test_dataloader, unique_treatment_set, feature_tensor, origin_feature_tensor, tree_depth, explanation_type, subset_ids=subset_ids)

    # def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5, epsilon=0, use_mlm=True):
    def train(self, train_dataset, valid_dataset, test_dataset):
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
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=NLP_Dataset.collate_fn, drop_last=True)
        val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        

        # self.model.train() 
        # optimizer = AdamW(self.model.parameters(), lr = self.learning_rate, eps=1e-8)

        best_loss = 1e6
        epochs_no_improve = 0
        
        # print("evaluation on training set::")
        # self.test(train_dataloader)
        
        # print("evaluation on validation set::")
        # self.test(val_dataloader)
        
        # print("evaluation on test set::")
        # self.test(test_dataloader)
        best_val_outcome_err = 1e6
        best_test_outcome_err = 1e6
        best_val_avg_treatment_effect = 1e6
        best_test_avg_treatment_effect = 1e6
        best_outcome_epoch = -1
        # self.test(val_dataloader)
        if not self.classification:
            print("update reward coefficient::", self.reward_coeff)
        for epoch in range(self.epochs):
            losses = []
            self.model.train()
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            all_treatment_pred = []
            all_outcome_pred = []
            all_gt_treatment = []
            all_gt_outcome = []
            all_gt_count_outcome = []
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                ids, origin_X, A, Y, count_Y, X, origin_all_other_pats_ls, (text_id_ls, text_mask_ls, text_len_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls, X), Y, A = batch
                text_id_ls = text_id_ls.to(self.device)
                text_mask_ls = text_mask_ls.to(self.device)
                text_len_ls = text_len_ls.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                other_loss = 0
                if self.method == "nam":
                    X = X.to(self.device)
                    A = A.to(self.device)
                    _, outcome_pred = self.model.forward(X, A)
                        
                elif self.method == "ENRL":
                    X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                    X_num = X_num.to(self.device)
                    X_cat = X_cat.to(self.device)
                    outcome_pred, other_loss = self.model(X_num, X_cat, A)
                    
                elif self.method == "TransTEE":
                    # text_id_ls = text_id_ls.to(self.device)
                    # text_mask_ls = text_mask_ls.to(self.device)
                    X = X.to(self.device)
                    A = A.to(self.device)
                    Y = Y.to(self.device)
                    _, outcome_pred = self.model(X, A)
                    # outcome_pred, _ = self.model.forward(text_id_ls, A, text_mask_ls)
                
                if self.backbone_full_model is not None:
                    outcome_pred = reg_model_forward(self, X, A, None, origin_X, outcome_pred)

                self.optimizer.zero_grad()
                if self.classification:
                    loss = F.cross_entropy(outcome_pred, Y.view(-1).long()) + other_loss
                else:
                    loss = F.mse_loss(outcome_pred.view(-1,1), Y) + other_loss
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                all_outcome_pred.append(outcome_pred.detach().cpu())
                all_gt_treatment.append(A.cpu())
                all_gt_outcome.append(Y.cpu())
                # all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
                # program = []
                # outbound_mask_ls = []
                # # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_str = []
                # program_col_ls = []
                # # for p_k in range(len(program_str)):
                # #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                # #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                # X_pd_full = origin_X
                # all_transformed_expr_ls = []

                
                # prev_reward = torch.zeros([len(A), self.topk_act, 2])
                
                # for arr_idx in range(self.program_max_len):
                #     init = (len(program) == 0)
                #     done = (arr_idx == self.program_max_len - 1)
                #     # treatment_pred, (outcome_pred, pos_outcome, neg_outcome), mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, transformed_expr_ls, atom_ls, ind_treatment_pred, ind_outcome_pred
                #     treatment_pred, outcome_pred, mlm_loss, next_program, next_program_str, next_program_col_ls, next_all_other_pats_ls, next_outbound_mask_ls, atom_ls, ind_treatment_pred, ind_outcome_pred = self.dqn.policy_net.forward(self, text_id_ls, text_len_ls, text_mask_ls, A, X, X_pd_full, program, program_str, program_col_ls, all_other_pats_ls, origin_X, outbound_mask_ls, None, epsilon=self.epsilon, eval=False, init=init, compute_ate=False, classification=self.classification)
                    
                #     reward1, reward2 = self.obtain_reward(ind_treatment_pred, ind_outcome_pred, A, Y, epoch)
                #     if done:
                #         next_state = None
                #     else:
                #         next_state = (next_program, next_outbound_mask_ls)
                    
                    
                #     transition = Transition((text_id_ls, text_len_ls, text_mask_ls), (X, X_pd_full),(program, outbound_mask_ls), atom_ls, next_state, torch.stack([reward1, reward2], dim=-1) - prev_reward)
                #     self.observe_transition(transition)
                #     #update model
                #     loss = self.dqn.optimize_model_ls0()
                    
                #     if done:
                #         all_treatment_pred.append(treatment_pred)

                #         if count_Y is not None:
                #             all_gt_count_outcome.append(count_Y)

                #     program = next_program
                #     program_str = next_program_str
                #     program_col_ls = next_program_col_ls
                #     all_other_pats_ls = next_all_other_pats_ls
                #     outbound_mask_ls = next_outbound_mask_ls
                    
                #     prev_reward = torch.stack([reward1, reward2], dim=-1)

                # if step % self.target_update == 0:
                #     self.dqn.update_target()
                
                # losses.append(loss)
            print("Training loss: ", np.mean(np.array(losses)))
            # all_treatment_pred_array = torch.cat(all_treatment_pred).numpy()
            # all_treatment_gt_array = torch.cat(all_gt_treatment).numpy()
            # training_acc = np.mean((all_treatment_pred_array > 0.5).reshape(-1).astype(float) == all_treatment_gt_array.reshape(-1))
            # training_auc = roc_auc_score(all_treatment_gt_array.reshape(-1), all_treatment_pred_array.reshape(-1))
            
            # print("training accuracy::", training_acc)
            # print("training auc score::", training_auc)


            all_outcome_pred_array = torch.cat(all_outcome_pred)
            all_gt_outcome_pred_array = torch.cat(all_gt_outcome)
            if len(all_gt_count_outcome) > 0:
                all_gt_count_outcome_array = torch.cat(all_gt_count_outcome)                

            if not self.classification:
                self.reward_coeff = -torch.log(torch.tensor(0.5))/torch.median((all_outcome_pred_array.view(-1) - all_gt_outcome_pred_array.view(-1))**2)
                if train_dataset.y_scaler is not None:
                    all_outcome_pred_array = transform_outcome_by_rescale_back(train_dataset, all_outcome_pred_array)
                    all_gt_outcome_pred_array = transform_outcome_by_rescale_back(train_dataset, all_gt_outcome_pred_array)
                    if len(all_gt_count_outcome) > 0:
                        all_gt_count_outcome_array = transform_outcome_by_rescale_back(train_dataset, all_gt_count_outcome_array)
            
            if not self.classification:
                outcome_error = F.mse_loss(all_outcome_pred_array.view(-1,1), all_gt_outcome_pred_array.view(-1,1)).item()
            else:
                outcome_error = F.cross_entropy(all_outcome_pred_array, all_gt_outcome_pred_array.view(-1).long()).item()
            
            
            print("performance at epoch ", epoch)
            print("training outcome error::", outcome_error)
            # print("evaluation on training set::")
            # self.test(train_dataloader)
            
            print("evaluation on validation set::")
            val_error, val_ate = self.test(val_dataloader)
            
            print("evaluation on test set::")
            test_error, test_ate = self.test(test_dataloader)
            # self.dqn.update_target()
            
            self.epsilon *= self.epsilon_falloff

            if val_error < best_val_outcome_err:
                best_val_outcome_err = val_error
                best_test_outcome_err = test_error
                best_outcome_epoch = epoch
                best_val_ate = val_ate
                best_test_ate = test_ate
                torch.save(self.model, self.modeldir+'_bestmod.pt')
            
            # print("outcome error at epoch %d::"%(epoch))
            print("best outcome epoch::", best_outcome_epoch)
            print("best validation outcome error::", best_val_outcome_err)
            print("best test outcome error::", best_test_outcome_err)
            print("best valid ate::", best_val_ate)
            print("best test ate::", best_test_ate)
            

        #     # evaluate validation set
        #     self.model.eval()
        #     pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
        #     a_val_losses, y_val_losses, a_val_accs = [], [], []
        
        #     for batch in pbar:
        #         if torch.cuda.is_available():
        #             batch = (x.cuda() for x in batch)
        #         text_id, text_len, text_mask, A, _, Y = batch
        #         _, _, _, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
                
        #         a_val_losses.append(a_loss.item())
        #         y_val_losses.append(y_loss.item())

        #         # A accuracy
        #         a_acc = torch.round(a_acc*len(A))
        #         a_val_accs.append(a_acc.item())


        #     a_val_loss = sum(a_val_losses)/n_val
        #     print('A Validation loss:',a_val_loss)

        #     y_val_loss = sum(y_val_losses)/n_val
        #     print('Y Validation loss:',y_val_loss)

        #     val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
        #     print('Validation loss:',val_loss)

        #     a_val_acc = sum(a_val_accs)/n_val
        #     print('A accuracy:',a_val_acc)


        #     # early stop
        #     if val_loss < best_loss:
        #         torch.save(self.model, self.modeldir+'_bestmod.pt') # save the best model
        #         best_loss = val_loss
        #         epochs_no_improve = 0              
        #     else:
        #         epochs_no_improve += 1
           
        #     if epoch >= 5 and epochs_no_improve >= patience:              
        #         print('Early stopping!' )
        #         print('The number of epochs is:', epoch)
        #         break

        # # load the best model as the model after training
        # self.model = torch.load(self.modeldir+'_bestmod.pt')

        # return self.model

def k_fold_fit_and_predict(read_data_dir, save_data_dir,
                        a_weight, y_weight, mlm_weight, model_dir, 
                        n_splits:int, lr=2e-5, batch_size=64):
    """
    Implements K fold cross-fitting for the model predicting the outcome Y. 
    That is, 
    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make predictions for each data point in fold j
    Returns two arrays containing the predictions for all units untreated, all units treated 
    """

    # get data
    df = pd.read_csv(read_data_dir)
    n_df = len(df)

    # initialize summary statistics
    Q0s = np.array([np.nan]*n_df, dtype = float)
    Q1s = np.array([np.nan]*n_df, dtype = float)
    As = np.array([np.nan]*n_df, dtype = float)
    Ys = np.array([np.nan]*n_df, dtype = float)
    Cs = np.array([np.nan]*n_df, dtype = float)


    # get k folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    idx_array = np.array(range(n_df))


    # train in k-folding fashion
    for train_index, test_index in kf.split(idx_array):
        # training df
        train_df = df.loc[train_index]
        train_df = train_df.reset_index()

        # test df
        test_df = df.loc[test_index]
        test_df = test_df.reset_index()

        # train the model with training data and train the propensitiy model with the testing data
        mod = QNet(batch_size=batch_size, a_weight=a_weight, y_weight=y_weight, mlm_weight=mlm_weight, modeldir=model_dir)
        mod.train(train_df['text'], train_df['T'], train_df['C'], train_df['Y'], epochs=20, learning_rate = lr)

        # g, Q, A, Y, C for the this test part (best model)
        Q0, Q1, A, Y, C = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'], model_dir=model_dir+'_bestmod.pt')
        Q0s[test_index] = Q0
        Q1s[test_index] = Q1
        As[test_index] = A
        Ys[test_index] = Y
        Cs[test_index] = C


        # delete models for this part
        os.remove(moddir + '_bestmod.pt')


    # if there's nan in Q0/Q1, raise error
    assert np.isnan(Q0s).sum() == 0
    assert np.isnan(Q1s).sum() == 0


    # save Q0, Q1, A, Y, C from the best model into a file
    stats_info = np.array(list(zip(Q0s, Q1s, As, Ys, Cs)))
    stats_info = pd.DataFrame(stats_info, columns=['Q0','Q1','A','Y','C'])
    stats_info.to_csv(save_data_dir, index = False)

    return


def get_agg_q(data_dir_dict, save_data_dir):
    '''Get aggregated conditional expected outcomes
        data_dir_dict is a dictionary that each seed has a corresponding data directory'''

    
    k = len(data_dir_dict)
    Q0s, Q1s, As, Ys, Cs = [], [], [], [], []

    for seed in data_dir_dict.keys():
        df = pd.read_csv(data_dir_dict[seed])
        Q0, Q1, A, Y, C = list(df['Q0']), list(df['Q1']), list(df['A']), list(df['Y']), list(df['C'])
        Q0s += [Q0]
        Q1s += [Q1]
        As += [A]
        Ys += [Y]
        Cs += [C]

    # check the data with the same index is the same one for all seeds
    for i in range(k-1):
        assert Ys[i]==Ys[i+1]
        assert As[i]==As[i+1]
        assert Cs[i]==Cs[i+1]

    Q0s, Q1s = np.array(Q0s), np.array(Q1s)
    A, Y, C = np.array(A), np.array(Y), np.array(C)
    Q0_agg, Q1_agg = np.sum(Q0s, axis=0)/k, np.sum(Q1s, axis=0)/k
        
    # save the aggregated data
    df_agg = df[['A', 'Y', 'C']].copy()
    df_agg['Q0'] = Q0_agg.copy()
    df_agg['Q1'] = Q1_agg.copy()
    df_agg.to_csv(save_data_dir, index=False)

    return
        


''' The second stage: propensity scores estimation '''
def get_propensities(As, Q0s, Q1s, model_type='GaussianProcessRegression', kernel=None, random_state=0, n_neighbors=100, base_estimator=None):
    """Train the propensity model directly on the data 
    and compute propensities of the data"""

    Q_mat = np.array(list(zip(Q0s, Q1s)))

    if model_type == 'GaussianProcessRegression':
        if kernel == None:
            kernel = DotProduct() + WhiteKernel()
        propensities_mod = GaussianProcessClassifier(kernel=kernel, random_state=random_state, warm_start=True)
        propensities_mod.fit(Q_mat, As)

        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'KNearestNeighbors':
        propensities_mod = KNeighborsClassifier(n_neighbors=n_neighbors)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'DecisionTree':
        propensities_mod = DecisionTreeClassifier(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'AdaBoost':
        propensities_mod = AdaBoostClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Bagging':
        propensities_mod = BaggingClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Logistic':
        propensities_mod = LogisticRegression(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    return gs


''' The third stage: get TI estimator '''
def get_TI_estimator(gs, Q0s, Q1s, As, Ys, error=0.05):
    '''Get TI estimator '''
    try:
        try_est = att_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)    
    except:
        print('There is 0/1 in propensity scores!')
    else:
        ti_estimate = pd.DataFrame(try_est, index = ['point estimate', 'standard error', 'confidence interval lower bound', 'confidence interval upper bound'])
        return ti_estimate


def get_estimands(gs, Q0s, Q1s, As, Ys, Cs=None, alpha=1, error=0.05, g_true=[0.8,0.6]):
    """ Get different estimands based on propensity scores, conditional expected outcomes, treatments and outcomes """
    estimands = []

    estimands.append(('unadj_T', [ATE_unadjusted(As, Ys)] + [np.nan] * 3))
    estimands.append(('adj_T', [ATE_adjusted(Cs, As, Ys)] + [np.nan] * 3))
    idx = (0.1 < gs) * (gs < 0.90)

    # Q only ATE
    ATE_Q = ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=None, weight=False, error_bound=error)
    estimands.append(('ate_Q', ATE_Q))


    # ATE AIPTW
    try:
        try_est = ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, weight=False,error_bound=error)    
    except:
        estimands.append(('ate_AIPTW', [np.nan]*4))
    else:
        estimands.append(('ate_AIPTW', try_est))

  
    # trimmed ATE AIPTW
    try:
        try_est = ate_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], weight=False,error_bound=error)    
    except:
        estimands.append(('trimmed ate_AIPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_AIPTW', try_est))


    # BMM
    try:
        bmm_ate = bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, alpha=alpha,error_bound=error)    
    except:
        estimands.append(('ate_BMM', [np.nan]*4))
    else:
        estimands.append(('ate_BMM', bmm_ate))


    # trimmed BMM
    try:
        bmm_ate = bmm(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], alpha=alpha,error_bound=error)    
    except:
        estimands.append(('trimmed ate_BMM', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_BMM', bmm_ate))


    # ATE IPTW
    try:
        try_est = ate_iptw(As, Ys, gs, error_bound=error)  
    except:
        estimands.append(('ate_IPTW', [np.nan]*4))
    else:
        estimands.append(('ate_IPTW', try_est))


    # trimmed ATE IPTW
    try:
        try_est = ate_iptw(As[idx], Ys[idx], gs[idx], error_bound=error)  
    except:
        estimands.append(('trimmed ate_IPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_IPTW', try_est))


    # ATT Q only
    try:
        try_est = att_q(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, error_bound=error)    
    except:
        estimands.append(('att_Q', [np.nan]*4))
    else:
        estimands.append(('att_Q', try_est))


    # ATT AIPTW
    try:
        try_est = att_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)    
    except:
        estimands.append(('att_AIPTW', [np.nan]*4))
    else:
        estimands.append(('att_AIPTW', try_est))


    # trimmed ATT AIPTW
    try:
        try_est = att_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], error_bound=error)    
    except:
        estimands.append(('trimmed att_AIPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed att_AIPTW', try_est))


    # ATT BMM
    try:
        try_est = att_bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)   
    except:
        estimands.append(('att_BMM', [np.nan]*4))
    else:
        estimands.append(('att_BMM', try_est))


    # trimmed ATT BMM
    try:
        try_est = att_bmm(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], error_bound=error)   
    except:
        estimands.append(('trimmed att_BMM', [np.nan]*4))
    else:
        estimands.append(('trimmed att_BMM', try_est))


    estimands = pd.DataFrame(data=dict(estimands))
    return estimands       




        
