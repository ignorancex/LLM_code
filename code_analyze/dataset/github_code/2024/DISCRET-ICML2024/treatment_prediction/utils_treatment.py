from sklearn.model_selection import train_test_split
import torch
import os
import yaml
import torch.nn.functional as F

import pandas as pd
import random
import numpy as np
from itertools import chain, combinations
from tabular.tabular_data_utils.tabular_dataset import tabular_Dataset
from torch.utils.data import TensorDataset
from psmpy import PsmPy
from psmpy.functions import cohenD
import warnings
import operator as op
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from posthoc_explanations.lime import Lime
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from anchor import anchor_tabular as anchor_tabular_class

from treatment_prediction.baseline_methods.anchor_reg.anchor_reg import anchor_tabular
from treatment_prediction.baseline_methods.lore_explainer_reg.lorem import LOREM_reg
from lore_explainer.lorem import LOREM

import shap
from treatment_prediction.baseline_methods.TransTEE.TransTEE import TransTEE, MonoTransTEE

# from miracle_local.impute import impute_train_eval_all
import json

def extract_decision_rules(tree, feature_names, sample, return_features =False):
    """
    Extract decision rules for an input test sample from a Decision Tree classifier.

    Parameters:
        - tree: The Decision Tree classifier model.
        - feature_names: A list of feature names.
        - sample: A single input test sample as a 1D NumPy array or list.

    Returns:
        - decision_rules: A list of strings representing the decision rules that lead to the prediction.
    """
    decision_rules = []
    if return_features:
        selected_features = []

    # Get the feature indices corresponding to feature names
    feature_indices = {feature: i for i, feature in enumerate(feature_names)}

    def _extract_rules(node, rule, features=None):
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
            # Leaf node, add the prediction
            # class_idx = tree.apply([sample])[0]
            # class_label = tree.classes_[class_idx]
            value = tree.apply([sample])[0]
            decision_rules.append(rule)
            if features is not None:
                selected_features.append(features)
            # decision_rules.append(rule.app" => outcome: {value}")
        else:
            # Non-leaf node, traverse left and right children
            feature_idx = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]

            feature_name = feature_names[feature_idx]
            sample_value = sample[feature_idx]

            if sample_value <= threshold:
                rule.append(f"{feature_name} <= {threshold}")
                if features is not None:
                    features.append(feature_idx)
                _extract_rules(tree.tree_.children_left[node], rule, features=features)
            else:
                rule.append(f"{feature_name} > {threshold}")
                if features is not None:
                    features.append(feature_idx)
                _extract_rules(tree.tree_.children_right[node], rule, features=features)

    if not return_features:
        _extract_rules(0, [])
    else:
        _extract_rules(0, [], [])

    if not return_features:
        return decision_rules
    else:
        return decision_rules, selected_features

def eval_sufficiency(trainer, test_loader, predicted_y, trees_by_treatment, explanation_by_treatment, fp=0.2, explanation_type='decision_tree', sub_sample_ids=None):
        # all_exp_ls = self.transform_explanation_str_to_exp(origin_explanation_str_ls)
        
        with torch.no_grad():
            trainer.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            all_matched_ratio_ls = []
            if sub_sample_ids is None:
                sample_ids = list(range(len(test_loader.dataset)))
            else:
                sample_ids = sub_sample_ids
            
            for idx, sample_id in enumerate(sample_ids):#range(len(test_loader.dataset)):
                # idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = test_loader.dataset[sample_id]
                
                all_matched_features_boolean = torch.ones(len(test_loader.dataset), dtype=torch.bool)
                
                if not trainer.has_dose and not trainer.cont_treatment:
                    if explanation_type == "decision_tree":
                        for k in range(trainer.num_treatments):
                            curr_tree = trees_by_treatment[k]
                            decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, test_loader.dataset.features[sample_id].numpy(), return_features=True)
                            decision_rule = transform_explanation_str_to_exp(test_loader.dataset, [decision_rule_str])
                            all_matched_features_boolean = torch.logical_and(all_matched_features_boolean, eval_booleans(decision_rule[0], test_loader.dataset.features))
                    elif explanation_type == "anchor" or explanation_type == "lore":
                        for k in range(trainer.num_treatments):
                            curr_tree = explanation_by_treatment[k][idx]
                            if len(curr_tree) <= 0:
                                all_matched_features_boolean = torch.zeros(len(test_loader.dataset), dtype=torch.bool)
                                all_matched_features_boolean[sample_id] = True
                            else:
                                
                                # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                                decision_rule = transform_explanation_str_to_exp(test_loader.dataset, [[curr_tree]])
                                all_matched_features_boolean = torch.logical_and(all_matched_features_boolean, eval_booleans(decision_rule[0], test_loader.dataset.features))
                else:
                    if explanation_type == "decision_tree":
                        curr_tree = trees_by_treatment[idx]
                        decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, test_loader.dataset.features[sample_id].numpy(), return_features=True)
                        decision_rule = transform_explanation_str_to_exp(test_loader.dataset, [decision_rule_str])
                        all_matched_features_boolean = torch.logical_and(all_matched_features_boolean, eval_booleans(decision_rule[0], test_loader.dataset.features))
                    elif explanation_type == "anchor" or explanation_type == "lore":
                        
                        curr_tree = explanation_by_treatment[idx]
                        # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                        if len(curr_tree) <= 0:
                            all_matched_features_boolean = torch.zeros(len(test_loader.dataset), dtype=torch.bool)
                            all_matched_features_boolean[sample_id] = True
                        else:
                            decision_rule = transform_explanation_str_to_exp(test_loader.dataset, [[curr_tree]])
                            all_matched_features_boolean = torch.logical_and(all_matched_features_boolean, eval_booleans(decision_rule[0], test_loader.dataset.features))

                # all_matched_features_boolean = curr_exp[1](test_loader.dataset.features[curr_exp[0]], curr_exp[2])
                # if sample_id == 0:
                #     print(torch.nonzero(all_matched_features_boolean.view(-1)).view(-1))
                
                all_matched_pred_labels = predicted_y[all_matched_features_boolean]
                # if not trainer.classification:
                #     matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                # else:
                #     matched_sample_count = torch.sum((torch.argmax(all_matched_pred_labels, dim=-1) == torch.argmax(predicted_y[sample_id], dim=-1))).item()-1
                if trainer.classification:
                    matched_sample_count = torch.sum(torch.norm(all_matched_pred_labels - predicted_y[sample_id], dim=-1) < fp).item()-1
                else:
                    matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                
                matched_sample_count = max(matched_sample_count, 0)
                matched_ratio = matched_sample_count*1.0/(max(len(all_matched_pred_labels)-1,0)+1e-6)
                all_matched_ratio_ls.append(matched_ratio)
            
            sufficiency_score = np.array(all_matched_ratio_ls).mean()
            # print(np.array(all_matched_ratio_ls))
            
            print("sufficiency score::", sufficiency_score)

def eval_consistency(trainer, test_loader, predicted_y, trees_by_treatment, explanations_by_treatment, fp=0.2, explanation_type='decision_tree', sub_sample_ids=None, out_folder = None):
        # all_exp_ls = self.transform_explanation_str_to_exp(origin_explanation_str_ls)
        
        with torch.no_grad():
            if not explanation_type == "ours":
                trainer.model.eval()
            else:
                trainer.dqn.policy_net.eval()
                trainer.dqn.target_net.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            all_matched_ratio_ls = []
            explanation_key_to_ids_mappings = dict()
            if sub_sample_ids is None:
                sample_ids = list(range(len(test_loader.dataset))) 
            else:
                sample_ids = sub_sample_ids
            
            for idx, sample_id in tqdm(enumerate(sample_ids)):#range(len(test_loader.dataset)):
                # idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = test_loader.dataset[sample_id]
                
                # all_matched_features_boolean = torch.ones(len(test_loader.dataset), dtype=torch.bool)
                
                if not trainer.has_dose and not trainer.cont_treatment:
                
                    if explanation_type == "decision_tree":
                        all_decision_key = ""
                        for k in range(trainer.num_treatments):
                            curr_rule = trees_by_treatment[k]
                            decision_rule_str, selected_col_ids = extract_decision_rules(curr_rule, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, test_loader.dataset.features[sample_id].numpy(), return_features=True)
                            
                            decision_key = " ".join(decision_rule_str[0])
                            
                            all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                    elif explanation_type == "lime" or explanation_type == "shap":
                        all_selected_col_score_by_ids = {}
                        for treatment_id0 in range(trainer.num_treatments):
                            # self.model.base_treatment=treatment_id0

                            # explainer = Lime(all_feat_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_feat_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                            # out = explainer.explain(curr_feat.numpy(), num_features=max_depth)
                            # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                            selected_col_score_by_ids = explanations_by_treatment[treatment_id0][idx]
                            for col_key in selected_col_score_by_ids:
                                if col_key in all_selected_col_score_by_ids:
                                    all_selected_col_score_by_ids[col_key] += selected_col_score_by_ids[col_key]
                                else:
                                    all_selected_col_score_by_ids[col_key] = selected_col_score_by_ids[col_key]
                        
                        all_col_ids = [k for k in all_selected_col_score_by_ids]
                        all_col_importance = torch.tensor([all_selected_col_score_by_ids[k] for k in all_col_ids])
                        all_sorted_importance, all_sorted_col_ids = torch.sort(all_col_importance, descending=True)
                        all_col_ids = torch.tensor(all_col_ids)[all_sorted_col_ids].tolist()
                        all_col_ids_str_ls = [str(k) for k in all_col_ids]
                        all_decision_key = " ".join(all_col_ids_str_ls)
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                        
                    elif explanation_type == "anchor" or explanation_type == "lore":
                        all_decision_key = ""
                        for k in range(trainer.num_treatments):
                            curr_rule = explanations_by_treatment[k][idx]
                            curr_rule.sort()
                            
                            # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                            
                            decision_key = " ".join(curr_rule)
                            
                            all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                        
                    elif explanation_type == "nam":
                        all_decision_key = ""
                        for k in range(trainer.num_treatments):
                            curr_rule = explanations_by_treatment[k][idx]
                            curr_rule = torch.sort(curr_rule)[0].tolist()
                            curr_rule_str_ls = [str(k) for k in curr_rule]
                            
                            # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                            
                            decision_key = " ".join(curr_rule_str_ls)
                            
                            all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                    
                    elif explanation_type == "ours":
                        curr_rule_ls = explanations_by_treatment[idx]
                        all_decision_key_ls = []
                        for curr_rule in curr_rule_ls:
                            sorted_rule = sorted(curr_rule)
                            curr_rule_str_ls = [str(k) for k in sorted_rule]
                            decision_key = " ".join(curr_rule_str_ls)
                            all_decision_key_ls.append(decision_key)
                        sorted_all_decision_key_ls = sorted(all_decision_key_ls)
                        all_decision_key = ",".join(sorted_all_decision_key_ls)
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                else:
                    if explanation_type == "decision_tree":
                        all_decision_key = ""
                        # for k in range(self.num_treatments):
                        curr_rule = trees_by_treatment[idx]
                        decision_rule_str, selected_col_ids = extract_decision_rules(curr_rule, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, test_loader.dataset.features[sample_id].numpy(), return_features=True)
                        
                        decision_key = " ".join(decision_rule_str[0])
                        
                        all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                    elif explanation_type == "lime" or explanation_type == "shap":
                        all_selected_col_score_by_ids = {}
                        # for treatment_id0 in range(self.num_treatments):
                            # self.model.base_treatment=treatment_id0

                            # explainer = Lime(all_feat_tensor.numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(all_feat_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                            # out = explainer.explain(curr_feat.numpy(), num_features=max_depth)
                            # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                        selected_col_score_by_ids = explanations_by_treatment[idx]
                        for col_key in selected_col_score_by_ids:
                            if col_key in all_selected_col_score_by_ids:
                                all_selected_col_score_by_ids[col_key] += selected_col_score_by_ids[col_key]
                            else:
                                all_selected_col_score_by_ids[col_key] = selected_col_score_by_ids[col_key]
                        
                        all_col_ids = [k for k in all_selected_col_score_by_ids]
                        all_col_importance = torch.tensor([all_selected_col_score_by_ids[k] for k in all_col_ids])
                        all_sorted_importance, all_sorted_col_ids = torch.sort(all_col_importance, descending=True)
                        all_col_ids = torch.tensor(all_col_ids)[all_sorted_col_ids].tolist()
                        all_col_ids_str_ls = [str(k) for k in all_col_ids]
                        all_decision_key = " ".join(all_col_ids_str_ls)
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                        
                    elif explanation_type == "anchor" or explanation_type == "lore":
                        all_decision_key = ""
                        # for k in range(self.num_treatments):
                        curr_rule = explanations_by_treatment[idx]
                        curr_rule.sort()
                        
                        # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                        
                        decision_key = " ".join(curr_rule)
                        
                        all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                        
                    elif explanation_type == "nam":
                        all_decision_key = ""
                        # for k in range(self.num_treatments):
                        curr_rule = explanations_by_treatment[idx]
                        curr_rule = torch.sort(curr_rule)[0].tolist()
                        curr_rule_str_ls = [str(k) for k in curr_rule]
                        
                        # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
                        
                        decision_key = " ".join(curr_rule_str_ls)
                        
                        all_decision_key += decision_key + " "
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)

                    elif explanation_type == "ours":
                        curr_rule_ls = explanations_by_treatment[idx]
                        all_decision_key_ls = []
                        for curr_rule in curr_rule_ls:
                            sorted_rule = sorted(curr_rule)
                            curr_rule_str_ls = [str(k) for k in sorted_rule]
                            decision_key = " ".join(curr_rule_str_ls)
                            all_decision_key_ls.append(decision_key)
                        sorted_all_decision_key_ls = sorted(all_decision_key_ls)
                        all_decision_key = ",".join(sorted_all_decision_key_ls)
                        if all_decision_key not in explanation_key_to_ids_mappings:
                            explanation_key_to_ids_mappings[all_decision_key] = []
                        explanation_key_to_ids_mappings[all_decision_key].append(sample_id)
                    
            for key in explanation_key_to_ids_mappings:
                sim_sample_ids = torch.tensor(explanation_key_to_ids_mappings[key])
                
                
                
                # all_matched_features_boolean = curr_exp[1](test_loader.dataset.features[curr_exp[0]], curr_exp[2])
                for sample_id in sim_sample_ids:
                    # if sample_id == 0:
                    #     print(sim_sample_ids)
                    all_matched_pred_labels = predicted_y[sim_sample_ids]
                    # if not trainer.classification:
                    #     matched_sample_count = torch.sum((all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                    # else:
                    #     matched_sample_count = torch.sum((torch.argmax(all_matched_pred_labels, dim=-1) - torch.argmax(predicted_y[sample_id], dim=-1)) < fp).item()-1
                    # if not trainer.classification:
                    #     matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                    # else:
                    #     matched_sample_count = torch.sum((torch.argmax(all_matched_pred_labels, dim=-1) == torch.argmax(predicted_y[sample_id], dim=-1))).item()-1
                    if trainer.classification:
                        matched_sample_count = torch.sum(torch.norm(all_matched_pred_labels - predicted_y[sample_id], dim=-1) < fp).item()-1
                    else:
                        matched_sample_count = torch.sum(torch.abs(all_matched_pred_labels - predicted_y[sample_id]) < fp).item()-1
                    
                    matched_sample_count = max(matched_sample_count, 0)
                    matched_ratio = matched_sample_count*1.0/(max(len(all_matched_pred_labels)-1,0)+1e-6)
                    all_matched_ratio_ls.append(matched_ratio)
            
            consistency_score = np.array(all_matched_ratio_ls).mean()
            # print(np.array(all_matched_ratio_ls))
            if out_folder is not None:
                output_file = os.path.join(out_folder, "explanation.json")
                with open(output_file, "w") as f:
                    json.dump(explanation_key_to_ids_mappings, f)
            print("consistency score::", consistency_score)

def eval_booleans(curr_exp_ls, data):
    final_res = torch.zeros(len(data)).bool()
    for sub_exp_ls in curr_exp_ls:
        curr_res = torch.ones(len(data)).bool()
        for curr_exp in sub_exp_ls:
            try:
                val = float(curr_exp[2])
            except:
                val = curr_exp[2]
            curr_res = torch.logical_and(curr_res, curr_exp[1](data[:,curr_exp[0]], val))
            
        final_res = torch.logical_or(final_res, curr_res)
    return final_res

def split_op_const(condn_str, op_str):
    if "=" in condn_str:
        if op_str == ">":
            op_symbol = op.__ge__
        else:
            op_symbol = op.__le__
    else:
        if op_str == ">":
            op_symbol = op.__gt__
        else:
            op_symbol = op.__lt__
    constant = condn_str.replace("=", "").strip()
    # if ">=" in condn_str:
    #     op_str=">="
    #     op_symbol = op.__ge__
    # elif ">" in condn_str:
    #     op_str=">"
    #     op_symbol = op.__gt__
    # elif "<=" in condn_str:
    #     op_str="<="
    #     op_symbol = op.__le__
    # elif "<" in condn_str:
    #     op_str="<"
    #     op_symbol = op.__lt__
    # else:
    #     op_str="="
    #     op_symbol = op.__eq__
    # constant = condn_str.split(op_str)[1].strip()
    return op_symbol, constant
    
def split_two_ops(string_with_cols, test_dataset):
    if ">" in string_with_cols:
        op_str=">"
    elif "<" in string_with_cols:
        op_str = "<"
    
    prev_cond, col_name, post_cond = string_with_cols.split(op_str)
    col_name = col_name.replace("=", "").strip()
    
    if col_name in test_dataset.num_cols:
        col_id = test_dataset.num_cols.index(col_name)
    elif col_name in test_dataset.cat_cols:
        col_id = test_dataset.cat_cols.index(col_name) - len(test_dataset.num_cols)
    
    pre_op_sym, pre_const = split_op_const(prev_cond, op_str)
    post_op_sym, post_const = split_op_const(post_cond, op_str)
    return [(col_id, pre_op_sym, pre_const), (col_id, post_op_sym, post_const)]
    
    

def transform_explanation_str_to_exp(test_dataset, origin_explanation_str_ls):
        all_exp_ls = []
        for sample_id in range(len(origin_explanation_str_ls)):
            curr_origin_explanations = origin_explanation_str_ls[sample_id]
            curr_exp_ls = []
            for k in range(len(curr_origin_explanations)):
                curr_curr_exp_ls = []
                for j in range(len(curr_origin_explanations[k])):
                    if curr_origin_explanations[k][j].count("<") > 1 or curr_origin_explanations[k][j].count(">") > 1:
                        curr_curr_exp_ls.extend(split_two_ops(curr_origin_explanations[k][j], test_dataset))
                        continue
                    
                    if ">=" in curr_origin_explanations[k][j]:
                        op_str=">="
                        op_symbol = op.__ge__
                    elif ">" in curr_origin_explanations[k][j]:
                        op_str=">"
                        op_symbol = op.__gt__
                    elif "<=" in curr_origin_explanations[k][j]:
                        op_str="<="
                        op_symbol = op.__le__
                    elif "<" in curr_origin_explanations[k][j]:
                        op_str="<"
                        op_symbol = op.__lt__
                    else:
                        op_str="="
                        op_symbol = op.__eq__
                    col_name = curr_origin_explanations[k][j].split(op_str)[0].strip()
                    if col_name in test_dataset.num_cols:
                        col_id = test_dataset.num_cols.index(col_name)
                    elif col_name in test_dataset.cat_cols:
                        col_id = test_dataset.cat_cols.index(col_name) - len(test_dataset.num_cols)
                    else:
                        assert ">=" in col_name or "<=" in col_name or ">" in col_name or "<" in col_name
                        split_two_ops(col_name)
                    # col_id = self.dqn.policy_net.grammar_token_to_pos[col_name]
                    constant = curr_origin_explanations[k][j].split(op_str)[1].strip()
                    
                    
                    curr_curr_exp_ls.append((col_id, op_symbol, constant))
                curr_exp_ls.append(curr_curr_exp_ls)
            
            all_exp_ls.append(curr_exp_ls)
        
        return all_exp_ls



def transform_treatment_ids(all_treatment_ids, A):
    return torch.nonzero(torch.tensor(all_treatment_ids).view(1,-1) == A.view(-1,1).cpu())[:,1]

def random_split_train_valid_test_ids(df_by_pat_mapping):
    pat_ids = list(df_by_pat_mapping.keys())
    pat_ids.sort()
    train_ids, test_ids = train_test_split(pat_ids, test_size=0.2, random_state=42)
    train_ids, valid_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    return train_ids, valid_ids, test_ids


def calculate_input_size(feat_to_onehot_embedding, cat_feat_ls, numer_feat_ls):
    input_size = 0
    input_size += len(numer_feat_ls)
    for feat in cat_feat_ls:
        input_size += len(list(feat_to_onehot_embedding[feat].values())[0])
    return input_size


def evaluate_treatment_effect_core(pred_treatment_outcome, pred_control_outcome, gt_treatment_outcome, gt_control_outcome):
    pred_treatment_effect = pred_treatment_outcome.view(-1,1) - pred_control_outcome.view(-1,1)
    gt_treatment_effect = gt_treatment_outcome.view(-1,1) - gt_control_outcome.view(-1,1)
    # return torch.mean((pred_treatment_effect - gt_treatment_effect)**2)
    return torch.mean(torch.abs(pred_treatment_effect- gt_treatment_effect)).item(), torch.abs(torch.mean(pred_treatment_effect- gt_treatment_effect)).item()

def evaluate_treatment_effect_core2(pred_treatment_outcome, pred_control_outcome, gt_treatment_outcome, gt_control_outcome):
    pred_treatment_effect = pred_treatment_outcome.view(-1,1) - pred_control_outcome.view(-1,1)
    gt_treatment_effect = gt_treatment_outcome.view(-1,1) - gt_control_outcome.view(-1,1)
    pehe = torch.sqrt(torch.mean((pred_treatment_effect.view(-1) - gt_treatment_effect.view(-1))**2))
    return pehe
    # return torch.mean((pred_treatment_effect - gt_treatment_effect)**2)
    # return torch.mean(torch.abs(pred_treatment_effect- gt_treatment_effect)).item(), torch.abs(torch.mean(pred_treatment_effect- gt_treatment_effect)).item()


def load_configs(args, root_dir=None):
    if args.model_config is None:
        args.model_config = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs/configs.yaml")

    # yamlfile_name = os.path.join(args.model_config, "model_config.yaml")
    # elif args.model_type == "csdi":
    #     yamlfile_name = os.path.join(model_config_file_path_base, "csdi_config.yaml")
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_dir, args.model_config), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        rl_config = config["rl"][args.rl_algorithm]
        model_config = config["model"][args.method.lower()]
        if args.backbone is not None and not args.backbone == "none":
            backbone_model_config = config["model"][args.backbone.lower()]
        else:
            backbone_model_config = None
    return rl_config, model_config, backbone_model_config

def load_dataset_configs(args, root_dir):
    
    # yamlfile_name = os.path.join(args.model_config, "model_config.yaml")
    # elif args.model_type == "csdi":
    #     yamlfile_name = os.path.join(model_config_file_path_base, "csdi_config.yaml")
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_dir, args.dataset_config), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        rl_config = config[args.dataset_name]
    return rl_config


def transform_outcome_by_rescale_back(train_dataset, all_outcome_pred_tensor):
    if train_dataset.y_scaler is not None:
        rescaled_output_tensor = torch.from_numpy(train_dataset.y_scaler.inverse_transform(all_outcome_pred_tensor.cpu().view(-1,1).numpy()))
    else:
        rescaled_output_tensor = all_outcome_pred_tensor.cpu().reshape(-1,1).numpy()
    return rescaled_output_tensor

def transform_outcome_by_rescale_back0(y_scaler, all_outcome_pred_tensor):
    rescaled_output_tensor = torch.from_numpy(y_scaler.inverse_transform(all_outcome_pred_tensor.cpu().view(-1,1).numpy()))
    return rescaled_output_tensor


def split_treatment_control_gt_outcome(all_concat_true_tensor, all_count_outcome_tensor):
    gt_treatment_tensor = all_concat_true_tensor[:, 0].view(-1)*(all_concat_true_tensor[:, 1] == 1).view(-1).type(torch.float) + all_count_outcome_tensor.view(-1)*(all_concat_true_tensor[:, 1] == 0).view(-1).type(torch.float)
    gt_control_tensor = all_concat_true_tensor[:, 0].view(-1)*(all_concat_true_tensor[:, 1] == 0).view(-1).type(torch.float) + all_count_outcome_tensor.view(-1)*(all_concat_true_tensor[:, 1] == 1).view(-1).type(torch.float)
    return gt_treatment_tensor, gt_control_tensor

def obtain_predictions(model, data_ls, treatment_ls, id_attr, outcome_attr, treatment_attr):
        # if len(data) == 0:
        #     return 0
        all_treatment_pred = []
        all_outcome_pred = []
        for idx in range(len(data_ls)):
            sub_data_ls = data_ls[idx]
            sub_rwd_ls = []
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



def obtain_sub_predictions2(model, lang, unioned_data, treatment_ls, dose_ls, idx, topk=1, compute_ate=False, classification=False, method_three=False):
    treatment_pred_all = lang.treatment_array[unioned_data]
    outcome_pred_all = lang.outcome_array[unioned_data]
    if compute_ate and not model.cont_treatment and model.num_treatments == 2:
        if method_three:
            return obtain_sub_predictions4(model, lang, unioned_data, treatment_ls, dose_ls, idx, topk=topk, compute_ate=compute_ate, classification=classification)

        pos_outcome = outcome_pred_all[treatment_pred_all == 1]
        neg_outcome = outcome_pred_all[treatment_pred_all == 0]
        if len(treatment_pred_all) > 0 and len(treatment_pred_all.unique()) >= len(lang.treatment_array.unique()):
            treatment_pred = treatment_pred_all.mean()
        else:
            treatment_pred = torch.ones_like(lang.treatment_array[0])*(-1)
        
        if not classification:
            if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
                return lang.outcome_array[lang.treatment_array==1].mean(), lang.outcome_array[lang.treatment_array==0].mean(), 0
            else:
                return pos_outcome.mean(), neg_outcome.mean(), 1
        else:
            unique_outcome_labels = lang.outcome_array.unique()
            if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
                return torch.tensor([1/len(unique_outcome_labels)]*len(unique_outcome_labels)).to(lang.treatment_array.device), torch.tensor([1/len(unique_outcome_labels)]*len(unique_outcome_labels)).to(lang.treatment_array.device), 0
            else:
                pos_outcome_mean = torch.tensor([torch.sum(pos_outcome == k).item()*1.0/len(pos_outcome) for k in range(len(unique_outcome_labels))]).to(lang.treatment_array.device)
                neg_outcome_mean = torch.tensor([torch.sum(neg_outcome == k).item()*1.0/len(neg_outcome) for k in range(len(unique_outcome_labels))]).to(lang.treatment_array.device)
                return pos_outcome_mean, neg_outcome_mean, 1
    
    non_empty=1
    # sub_rwd_ls = []
    # concat_data = pd.concat([df[[id_attr, outcome_attr, treatment_attr]] for df in sub_data_ls])
    # concat_data.drop_duplicates(inplace=True)
    # treatment_pred_all = concat_data[treatment_attr]
    tr_sorted_ids = None
    if len(treatment_pred_all) <= 0:
        if model.num_treatments == 2 or model.cont_treatment:
            treatment_pred=torch.ones_like(lang.treatment_array[0])*(-1)
        else:
            treatment_pred=(torch.ones_like(lang.treatment_array[0])*(-1)).view(-1).repeat(model.num_treatments)
        non_empty=0
    else:
        if not model.cont_treatment:
            if model.num_treatments == 2:
                if  len(treatment_pred_all.unique()) < model.num_treatments:
                    treatment_pred = torch.ones_like(lang.treatment_array[0])*(-1)
                    non_empty = 0
                else:
                    treatment_pred=treatment_pred_all.mean()
            elif model.num_treatments > 2:
                if  len(treatment_pred_all.unique()) < model.num_treatments:
                    # treatment_pred = [-1 for _ in range(model.num_treatments)] 
                    treatment_pred=(torch.ones_like(lang.treatment_array[0])*(-1)).view(-1).repeat(model.num_treatments)
                else:
                    # treatment_pred = [torch.sum(treatment_pred_all == k).item()*1.0/len(treatment_pred_all) for k in range(model.num_treatments)]
                    treatment_pred = torch.mean((treatment_pred_all.view(1,-1) == torch.tensor(list(range(model.num_treatments))).cuda().view(-1,1)).float(), dim=-1)


        else:
            treatment_pred = treatment_pred_all.mean()
    sorted_ids = None
    if dose_ls is not None:
        local_dose = lang.dose_array[unioned_data]
    
        _, sorted_ids = torch.sort((local_dose[treatment_pred_all == treatment_ls[idx].item()] - dose_ls[idx].item()).abs(), descending=False)
        
        
    
    # unioned_data = torch.logical_and(unioned_data, treatment_pred_all == treatment_ls[idx].item())
    
    if not model.cont_treatment:
        outcome_pred_curr_tr = outcome_pred_all[treatment_pred_all == treatment_ls[idx].item()]
    else:
        # tr_gap_sorted, tr_sorted_ids = torch.sort((treatment_pred_all - treatment_ls[idx].item()).abs(), descending=False)
        distance, tr_sorted_ids = torch.topk((treatment_pred_all - treatment_ls[idx].item()).abs(), k=min(topk, len(treatment_pred_all)), largest=False)
        outcome_pred_curr_tr = outcome_pred_all[tr_sorted_ids]#[tr_gap_sorted < 0.005]#
    if sorted_ids is not None:
        outcome_pred_curr_tr = outcome_pred_curr_tr[sorted_ids[0:topk]]
    # outcome_pred_all = concat_data.loc[concat_data[treatment_attr] == treatment_ls[idx].item(), outcome_attr]
    if len(outcome_pred_curr_tr) <= 0:
        # print(idx)
        if not classification:
            # if not model.cont_treatment:
            #     outcome_pred = lang.outcome_array[lang.treatment_array==treatment_ls[idx].item()].mean().item()
            # else:
            outcome_pred =  lang.outcome_array.mean()
        else:
            unique_outcome_labels = lang.outcome_array.unique()
            # outcome_pred = [1/len(unique_outcome_labels)]*len(unique_outcome_labels)
            outcome_pred = (torch.ones_like(lang.treatment_array[0])*(1/len(unique_outcome_labels))).view(-1).repeat(len(unique_outcome_labels))
            treatment_pred = torch.ones_like(lang.treatment_array[0])*(-1)
        non_empty = 0
    else:
        # if model.cont_treatment:
        #     # outcome_pred_all = outcome_pred_all[tr_sorted_ids][0:topk]#[tr_gap_sorted < 0.005]
        #     treatment_pred_all = treatment_pred_all[tr_sorted_ids][0:topk]
        #     sample_weight = ((treatment_pred_all[0] -  treatment_ls[idx].item()).abs().item() + 1e-4)/((treatment_pred_all.view(-1) - treatment_ls[idx].item()).abs().numpy()+ 1e-4)
        #     reg = LinearRegression().fit(treatment_pred_all.view(-1,1).numpy(), outcome_pred_all.view(-1,1).numpy(), sample_weight = sample_weight.reshape(-1))
        #     outcome_pred= reg.predict(treatment_ls[idx].cpu().view(1,-1).numpy()).item()
        # else:
        if not classification:
            outcome_pred = outcome_pred_curr_tr.mean() #.item()
        else:
            unique_outcome_labels = lang.outcome_array.unique()
            # outcome_pred = [torch.sum(outcome_pred_curr_tr == k).item()*1.0/len(outcome_pred_curr_tr) for k in range(len(unique_outcome_labels))]
            outcome_pred = torch.mean((outcome_pred_curr_tr.view(1,-1) == torch.tensor(list(range(len(unique_outcome_labels)))).cuda().view(-1,1)).float(), dim=-1)
    if not classification:
        assert outcome_pred == outcome_pred
    return treatment_pred, outcome_pred, non_empty

def obtain_sub_predictions3(model, test_X, test_treatment, test_dose, lang, unioned_data, treatment_ls, dose_ls, idx, topk=1, compute_ate=False, classification=False):
    treatment_pred_all = lang.treatment_array[unioned_data]
        
    outcome_pred_all = lang.outcome_array[unioned_data]
    
    data_pred_all = lang.transformed_features[unioned_data] 
    
    
    
    if len(treatment_pred_all) <= 0:
        if model.num_treatments == 2 or model.cont_treatment:
            treatment_pred=-1
        else:
            treatment_pred=[-1]*model.num_treatments
        return treatment_pred, -1, -1
    else:
        if not model.cont_treatment:
            if model.num_treatments == 2:
                treatment_pred=treatment_pred_all.mean().item()
            elif model.num_treatments > 2:
                treatment_pred = [torch.sum(treatment_pred_all == k).item()*1.0/len(treatment_pred_all) for k in range(model.num_treatments)]

        else:
            treatment_pred = treatment_pred_all.mean().item()
    
    
    if lang.dose_array is not None:
        dose_pred_all = lang.dose_array[unioned_data]
        # dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all, dose_pred_all)
    else:
        dose_pred_all = None
        # dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all)
        
        
        # torch.set_grad_enabled(True)
        # model.cohort_trainer.run(dataset_pred_all, dataset_pred_all, dataset_pred_all, cohort=True)
        # model.cohort_trainer.model.eval()
    
    if compute_ate and not model.cont_treatment and model.num_treatments == 2:
        # pos_outcome = outcome_pred_all[treatment_pred_all == 1]
        # neg_outcome = outcome_pred_all[treatment_pred_all == 0]
        if len(treatment_pred_all) > 0:
            treatment_pred = treatment_pred_all.mean().item()
        else:
            treatment_pred = -1
        
        with torch.no_grad():
            _,  pos_outcome = model.cohort_model(data_pred_all.to(test_treatment.device), torch.ones(len(data_pred_all)).view(-1).to(test_treatment.device))
            _,  neg_outcome = model.cohort_model(data_pred_all.to(test_treatment.device), torch.zeros(len(data_pred_all)).view(-1).to(test_treatment.device))
            return pos_outcome.mean().item(), neg_outcome.mean().item(), -1
        # if not classification:
        #     if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
        #         return lang.outcome_array[lang.treatment_array==1].mean().item(), lang.outcome_array[lang.treatment_array==0].mean().item()
        #     else:
        #         return pos_outcome.mean().item(), neg_outcome.mean().item()
        # else:
        #     unique_outcome_labels = lang.outcome_array.unique()
        #     if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
        #         return [1/len(unique_outcome_labels)]*len(unique_outcome_labels), [1/len(unique_outcome_labels)]*len(unique_outcome_labels)
        #     else:
        #         pos_outcome_mean = [torch.sum(pos_outcome == k).item()*1.0/len(pos_outcome) for k in range(len(unique_outcome_labels))]
        #         neg_outcome_mean = [torch.sum(neg_outcome == k).item()*1.0/len(neg_outcome) for k in range(len(unique_outcome_labels))]
        #         return pos_outcome_mean, neg_outcome_mean
        
    # sub_rwd_ls = []
    # concat_data = pd.concat([df[[id_attr, outcome_attr, treatment_attr]] for df in sub_data_ls])
    # concat_data.drop_duplicates(inplace=True)
    # treatment_pred_all = concat_data[treatment_attr]
    tr_sorted_ids = None
    
    # sorted_ids = None
    # if dose_ls is not None:
    #     local_dose = lang.dose_array[unioned_data]
    
    #     _, sorted_ids = torch.sort((local_dose[treatment_pred_all == treatment_ls[idx].item()] - dose_ls[idx].item()).abs(), descending=False)
        
        
    
    # unioned_data = torch.logical_and(unioned_data, treatment_pred_all == treatment_ls[idx].item())
    
    # if not model.cont_treatment:
    #     outcome_pred_curr_tr = outcome_pred_all[treatment_pred_all == treatment_ls[idx].item()]
    # else:
    #     # tr_gap_sorted, tr_sorted_ids = torch.sort((treatment_pred_all - treatment_ls[idx].item()).abs(), descending=False)
    #     distance, tr_sorted_ids = torch.topk((treatment_pred_all - treatment_ls[idx].item()).abs(), k=min(topk, len(treatment_pred_all)), largest=False)
    #     outcome_pred_curr_tr = outcome_pred_all[tr_sorted_ids]#[tr_gap_sorted < 0.005]#
    # # if sorted_ids is not None:
    # #     outcome_pred_curr_tr = outcome_pred_curr_tr[sorted_ids[0:topk]]
    # # outcome_pred_all = concat_data.loc[concat_data[treatment_attr] == treatment_ls[idx].item(), outcome_attr]
    # if len(outcome_pred_curr_tr) <= 0:
    #     # print(idx)
    #     if not classification:
    #         outcome_pred = lang.outcome_array[lang.treatment_array==treatment_ls[idx].item()].mean().item()
    #     else:
    #         unique_outcome_labels = lang.outcome_array.unique()
    #         outcome_pred = [1/len(unique_outcome_labels)]*len(unique_outcome_labels)
    # else:
    #     # if model.cont_treatment:
    #     #     # outcome_pred_all = outcome_pred_all[tr_sorted_ids][0:topk]#[tr_gap_sorted < 0.005]
    #     #     treatment_pred_all = treatment_pred_all[tr_sorted_ids][0:topk]
    #     #     sample_weight = ((treatment_pred_all[0] -  treatment_ls[idx].item()).abs().item() + 1e-4)/((treatment_pred_all.view(-1) - treatment_ls[idx].item()).abs().numpy()+ 1e-4)
    #     #     reg = LinearRegression().fit(treatment_pred_all.view(-1,1).numpy(), outcome_pred_all.view(-1,1).numpy(), sample_weight = sample_weight.reshape(-1))
    #     #     outcome_pred= reg.predict(treatment_ls[idx].cpu().view(1,-1).numpy()).item()
    #     # else:
    #     if not classification:
    #         outcome_pred = outcome_pred_curr_tr.mean().item()
    #     else:
    #         unique_outcome_labels = lang.outcome_array.unique()
    #         outcome_pred = [torch.sum(outcome_pred_curr_tr == k).item()*1.0/len(outcome_pred_curr_tr) for k in range(len(unique_outcome_labels))]
    with torch.no_grad():
        # if test_dose is not None:
        #     test_dose = test_dose.view(1,-1)
        if dose_pred_all is not None:
            dose_pred_all = dose_pred_all.to(test_treatment.device)
        _,  outcome_pred = model.cohort_model(data_pred_all.to(test_treatment.device), treatment_pred_all.to(test_treatment.device), d=dose_pred_all)
        outcome_pred = outcome_pred.mean().item()
    return treatment_pred, outcome_pred, -1

def get_rebalanced_array(binary_array):
    unique, counts = np.unique(binary_array, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Calculate the class with the fewer samples
    minority_class = min(class_distribution, key=class_distribution.get)
    minority_count = class_distribution[minority_class]

    # Calculate the class with the more samples
    majority_class = max(class_distribution, key=class_distribution.get)
    majority_count = class_distribution[majority_class]

    # Decide on the rebalancing strategy (undersampling or oversampling)
    if minority_count < majority_count:
        # Undersample the majority class to match the minority class
        majority_indices = np.where(binary_array == majority_class)[0]
        np.random.shuffle(majority_indices)
        majority_indices = majority_indices[:minority_count]

        # Combine the minority class indices with the sampled majority class indices
        balanced_indices = np.concatenate([np.where(binary_array == minority_class)[0], majority_indices])
    else:
        # Oversample the minority class to match the majority class
        minority_indices = np.where(binary_array == minority_class)[0]
        oversampled_indices = np.random.choice(minority_indices, size=majority_count, replace=True)

        # Combine the oversampled minority class indices with the majority class indices
        balanced_indices = np.concatenate([oversampled_indices, np.where(binary_array == majority_class)[0]])

    return balanced_indices

def obtain_sub_predictions4_back(model, lang, unioned_data, treatment_ls, dose_ls, idx, topk=1, compute_ate=False, classification=False):
    treatment_pred_all = lang.treatment_array[unioned_data]
        
    outcome_pred_all = lang.outcome_array[unioned_data]
    
    data_pred_all = lang.transformed_features[unioned_data]
    
    if len(treatment_pred_all) < 0:
        return -1, -1
    if not model.cont_treatment and treatment_pred_all.unique().shape[0] < model.num_treatments:
        return -1, -1

    # reb_index = get_rebalanced_array(treatment_pred_all.numpy())
    # treatment_pred_all = treatment_pred_all[reb_index]
    # data_pred_all = data_pred_all[reb_index]
    curr_df = pd.DataFrame(torch.cat([data_pred_all, treatment_pred_all.unsqueeze(-1)], dim=-1).numpy(), columns=["feat_" + str(i) for i in range(data_pred_all.shape[1])] + ["treatment"])
    
    curr_df.reset_index(inplace=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some values do not have a match. These are dropped for purposes of establishing a matched dataframe, and subsequent calculations and plots (effect size). If you do not wish this to be the case please set drop_unmatched=False")

        psm = PsmPy(curr_df, treatment='treatment', indx='index')
        
        psm.logistic_ps(balance = False)
        try:
            psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=False)
            matched_ids = psm.matched_ids.values
        except:
            matched_ids = None
    # model.cohort_trainer.initialize_model_parameters()
    # if len(treatment_pred_all) > 0:
    
    #     if lang.dose_array is not None:
    #         dose_pred_all = lang.dose_array[unioned_data]
    #         dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all, dose_pred_all)
    #     else:
    #         dose_pred_all = None
    #         dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all)
        
        
    #     torch.set_grad_enabled(True)
    #     model.cohort_trainer.run(dataset_pred_all, dataset_pred_all, dataset_pred_all, cohort=True)
    #     model.cohort_trainer.model.eval()
    
    if compute_ate and not model.cont_treatment and model.num_treatments == 2:
        # pos_outcome = outcome_pred_all[treatment_pred_all == 1]
        # neg_outcome = outcome_pred_all[treatment_pred_all == 0]
        # if len(treatment_pred_all) > 0:
        #     treatment_pred = treatment_pred_all.mean().item()
        # else:
        #     treatment_pred = -1
        
        # with torch.no_grad():
        #     _,  pos_outcome_mean = model.cohort_trainer.model(test_X.to(test_treatment.device).view(1,-1), torch.ones(1).to(test_treatment.device).view(1,-1).view(-1,1))
        #     _,  neg_outcome_mean = model.cohort_trainer.model(test_X.to(test_treatment.device).view(1,-1), torch.zeros(1).to(test_treatment.device).view(1,-1).view(-1,1))
        if matched_ids is not None:
            pos_outcome_mean = outcome_pred_all[matched_ids[:,0]].mean()
            neg_outcome_mean = outcome_pred_all[matched_ids[:,1]].mean()
            if treatment_pred_all[matched_ids[0,0]] == 1:
                return pos_outcome_mean.item(), neg_outcome_mean.item()
            else:
                return neg_outcome_mean.item(), pos_outcome_mean.item()
        else:
            pos_outcome = outcome_pred_all[treatment_pred_all == 1]
            neg_outcome = outcome_pred_all[treatment_pred_all == 0]
            return pos_outcome.mean().item(), neg_outcome.mean().item()
        
def obtain_sub_predictions4(model, lang, unioned_data, treatment_ls, dose_ls, idx, topk=1, compute_ate=False, classification=False):
    treatment_pred_all = lang.treatment_array[unioned_data]
        
    outcome_pred_all = lang.outcome_array[unioned_data]
    
    data_pred_all = lang.transformed_features[unioned_data]
    
    if len(treatment_pred_all) < 0:
        return -1, -1, -1
    if not model.cont_treatment and treatment_pred_all.unique().shape[0] < model.num_treatments:
        return -1, -1, -1

    propensity_model = LogisticRegression()
    
    propensity_model.fit(data_pred_all.numpy(), treatment_pred_all.numpy())
    
    treatment_pred_scores = propensity_model.predict_proba(data_pred_all.numpy())[:,1] + 1e-4
    treatment_pred_scores = torch.from_numpy(treatment_pred_scores)
    
    
    # reb_index = get_rebalanced_array(treatment_pred_all.numpy())
    # treatment_pred_all = treatment_pred_all[reb_index]
    # data_pred_all = data_pred_all[reb_index]
    # curr_df = pd.DataFrame(torch.cat([data_pred_all, treatment_pred_all.unsqueeze(-1)], dim=-1).numpy(), columns=["feat_" + str(i) for i in range(data_pred_all.shape[1])] + ["treatment"])
    
    # curr_df.reset_index(inplace=True)
    
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message="Some values do not have a match. These are dropped for purposes of establishing a matched dataframe, and subsequent calculations and plots (effect size). If you do not wish this to be the case please set drop_unmatched=False")

    #     psm = PsmPy(curr_df, treatment='treatment', indx='index')
        
    #     psm.logistic_ps(balance = False)
    #     try:
    #         psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=False)
    #         matched_ids = psm.matched_ids.values
    #     except:
    #         matched_ids = None
    # model.cohort_trainer.initialize_model_parameters()
    # if len(treatment_pred_all) > 0:
    
    #     if lang.dose_array is not None:
    #         dose_pred_all = lang.dose_array[unioned_data]
    #         dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all, dose_pred_all)
    #     else:
    #         dose_pred_all = None
    #         dataset_pred_all = TensorDataset(data_pred_all, treatment_pred_all, outcome_pred_all)
        
        
    #     torch.set_grad_enabled(True)
    #     model.cohort_trainer.run(dataset_pred_all, dataset_pred_all, dataset_pred_all, cohort=True)
    #     model.cohort_trainer.model.eval()
    
    if compute_ate and not model.cont_treatment and model.num_treatments == 2:
        # pos_outcome = outcome_pred_all[treatment_pred_all == 1]
        # neg_outcome = outcome_pred_all[treatment_pred_all == 0]
        # if len(treatment_pred_all) > 0:
        #     treatment_pred = treatment_pred_all.mean().item()
        # else:
        #     treatment_pred = -1
        
        # with torch.no_grad():
        #     _,  pos_outcome_mean = model.cohort_trainer.model(test_X.to(test_treatment.device).view(1,-1), torch.ones(1).to(test_treatment.device).view(1,-1).view(-1,1))
        #     _,  neg_outcome_mean = model.cohort_trainer.model(test_X.to(test_treatment.device).view(1,-1), torch.zeros(1).to(test_treatment.device).view(1,-1).view(-1,1))
        # if matched_ids is not None:
        #     pos_outcome_mean = outcome_pred_all[matched_ids[:,0]].mean()
        #     neg_outcome_mean = outcome_pred_all[matched_ids[:,1]].mean()
        #     if treatment_pred_all[matched_ids[0,0]] == 1:
        #         return pos_outcome_mean.item(), neg_outcome_mean.item()
        #     else:
        #         return neg_outcome_mean.item(), pos_outcome_mean.item()
        # else:
        pos_outcome = outcome_pred_all[treatment_pred_all == 1].view(-1)/treatment_pred_scores[treatment_pred_all == 1].view(-1)
        neg_outcome = outcome_pred_all[treatment_pred_all == 0].view(-1)/(1-treatment_pred_scores[treatment_pred_all == 0].view(-1))
        return pos_outcome.mean().item(), neg_outcome.mean().item(), 1
        # if not classification:
        #     if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
        #         return lang.outcome_array[lang.treatment_array==1].mean().item(), lang.outcome_array[lang.treatment_array==0].mean().item()
        #     else:
        #         return pos_outcome.mean().item(), neg_outcome.mean().item()
        # else:
        #     unique_outcome_labels = lang.outcome_array.unique()
        #     if len(pos_outcome) <= 0 or len(neg_outcome) <= 0:
        #         return [1/len(unique_outcome_labels)]*len(unique_outcome_labels), [1/len(unique_outcome_labels)]*len(unique_outcome_labels)
        #     else:
        #         pos_outcome_mean = [torch.sum(pos_outcome == k).item()*1.0/len(pos_outcome) for k in range(len(unique_outcome_labels))]
        #         neg_outcome_mean = [torch.sum(neg_outcome == k).item()*1.0/len(neg_outcome) for k in range(len(unique_outcome_labels))]
        #         return pos_outcome_mean, neg_outcome_mean
        
    # sub_rwd_ls = []
    # concat_data = pd.concat([df[[id_attr, outcome_attr, treatment_attr]] for df in sub_data_ls])
    # concat_data.drop_duplicates(inplace=True)
    # treatment_pred_all = concat_data[treatment_attr]
    # tr_sorted_ids = None
    # if len(treatment_pred_all) <= 0:
    #     if model.num_treatments == 2 or model.cont_treatment:
    #         treatment_pred=-1
    #     else:
    #         treatment_pred=[-1]*model.num_treatments
    # else:
    #     if not model.cont_treatment:
    #         if model.num_treatments == 2:
    #             treatment_pred=treatment_pred_all.mean().item()
    #         elif model.num_treatments > 2:
    #             treatment_pred = [torch.sum(treatment_pred_all == k).item()*1.0/len(treatment_pred_all) for k in range(model.num_treatments)]

    #     else:
    #         treatment_pred = treatment_pred_all.mean().item()
    # # sorted_ids = None
    # # if dose_ls is not None:
    # #     local_dose = lang.dose_array[unioned_data]
    
    # #     _, sorted_ids = torch.sort((local_dose[treatment_pred_all == treatment_ls[idx].item()] - dose_ls[idx].item()).abs(), descending=False)
        
        
    
    # # unioned_data = torch.logical_and(unioned_data, treatment_pred_all == treatment_ls[idx].item())
    
    # # if not model.cont_treatment:
    # #     outcome_pred_curr_tr = outcome_pred_all[treatment_pred_all == treatment_ls[idx].item()]
    # # else:
    # #     # tr_gap_sorted, tr_sorted_ids = torch.sort((treatment_pred_all - treatment_ls[idx].item()).abs(), descending=False)
    # #     distance, tr_sorted_ids = torch.topk((treatment_pred_all - treatment_ls[idx].item()).abs(), k=min(topk, len(treatment_pred_all)), largest=False)
    # #     outcome_pred_curr_tr = outcome_pred_all[tr_sorted_ids]#[tr_gap_sorted < 0.005]#
    # # # if sorted_ids is not None:
    # # #     outcome_pred_curr_tr = outcome_pred_curr_tr[sorted_ids[0:topk]]
    # # # outcome_pred_all = concat_data.loc[concat_data[treatment_attr] == treatment_ls[idx].item(), outcome_attr]
    # # if len(outcome_pred_curr_tr) <= 0:
    # #     # print(idx)
    # #     if not classification:
    # #         outcome_pred = lang.outcome_array[lang.treatment_array==treatment_ls[idx].item()].mean().item()
    # #     else:
    # #         unique_outcome_labels = lang.outcome_array.unique()
    # #         outcome_pred = [1/len(unique_outcome_labels)]*len(unique_outcome_labels)
    # # else:
    # #     # if model.cont_treatment:
    # #     #     # outcome_pred_all = outcome_pred_all[tr_sorted_ids][0:topk]#[tr_gap_sorted < 0.005]
    # #     #     treatment_pred_all = treatment_pred_all[tr_sorted_ids][0:topk]
    # #     #     sample_weight = ((treatment_pred_all[0] -  treatment_ls[idx].item()).abs().item() + 1e-4)/((treatment_pred_all.view(-1) - treatment_ls[idx].item()).abs().numpy()+ 1e-4)
    # #     #     reg = LinearRegression().fit(treatment_pred_all.view(-1,1).numpy(), outcome_pred_all.view(-1,1).numpy(), sample_weight = sample_weight.reshape(-1))
    # #     #     outcome_pred= reg.predict(treatment_ls[idx].cpu().view(1,-1).numpy()).item()
    # #     # else:
    # #     if not classification:
    # #         outcome_pred = outcome_pred_curr_tr.mean().item()
    # #     else:
    # #         unique_outcome_labels = lang.outcome_array.unique()
    # #         outcome_pred = [torch.sum(outcome_pred_curr_tr == k).item()*1.0/len(outcome_pred_curr_tr) for k in range(len(unique_outcome_labels))]
    # with torch.no_grad():
    #     if test_dose is not None:
    #         test_dose = test_dose.view(1,-1)
    #     _,  outcome_pred = model.cohort_trainer.model(test_X.to(test_treatment.device).view(1,-1), test_treatment.view(1,-1), d=test_dose)
    #     outcome_pred = outcome_pred.item()
    # return treatment_pred, outcome_pred


def obtain_predictions2(model, X, A, D, lang, data_ls, treatment_ls, dose_ls, id_attr, outcome_attr, treatment_attr, topk=1, all_treatment_ids=None, sim_treatment_ids=None, compute_ate=False,sim_treatment_probs=None, tr_str_two=False, treatment_graph_sim_mat=None, classification=False, method_two=False, method_three=False):
    # if len(data) == 0:
    #     return 0
    all_treatment_pred = []
    all_outcome_pred = []
    all_non_empty = []
    for idx in range(len(data_ls)):
        sub_data_ls = data_ls[idx]
        
        unioned_data = (torch.sum(torch.stack(sub_data_ls), dim=0) >= len(sub_data_ls))
        sub_D = None
        if D is not None:
            sub_D = D[idx]
        if method_two:
            treatment_pred, outcome_pred, non_empty = obtain_sub_predictions3(model, X[idx], A[idx], sub_D, lang, unioned_data, treatment_ls, dose_ls, idx, topk=topk, compute_ate=compute_ate, classification=classification)
        else:
            # if compute_ate:
            #     treatment_pred, outcome_pred = obtain_sub_predictions4(model, X[idx], A[idx], sub_D, lang, unioned_data, treatment_ls, dose_ls, idx, topk=topk, compute_ate=compute_ate, classification=classification)
                # torch.set_grad_enabled(False)
            # else:
            treatment_pred, outcome_pred, non_empty = obtain_sub_predictions2(model, lang, unioned_data, treatment_ls, dose_ls, idx, topk=topk, compute_ate=compute_ate, classification=classification, method_three=method_three)
        # else:
        #     # lang, unioned_data, treatment_ls, idx, all_treatment_ids, treatment_graph_sim_mat
        all_treatment_pred.append(treatment_pred.view(-1))
        all_outcome_pred.append(outcome_pred.view(-1))
        all_non_empty.append(non_empty)
        
    return torch.stack(all_treatment_pred), torch.stack(all_outcome_pred), torch.tensor(all_non_empty)

def obtain_individual_predictions2(model, X, A, D, lang, data_ls, treatment_ls, dose_ls, id_attr, outcome_attr, treatment_attr, topk=1, all_treatment_ids=None, sim_treatment_ids=None, sim_treatment_probs=None, tr_str_two=False, treatment_graph_sim_mat=None, classification=False, method_two=False, method_three=False):
    # if len(data) == 0:
    #     return 0
    all_treatment_pred = []
    all_outcome_pred = []
    for idx in range(len(data_ls)):
        sub_data_ls = data_ls[idx]
        
        sub_treatment_pred = []
        sub_outcome_pred = []
        
        for sub_data in sub_data_ls:
            sub_D = None
            if D is not None:
                sub_D = D[idx]
            # unioned_data = (torch.sum(torch.stack(sub_data_ls), dim=0) >= len(sub_data_ls))

            if method_two:
                treatment_pred, outcome_pred, _ = obtain_sub_predictions3(model, X[idx], A[idx], sub_D, lang, sub_data, treatment_ls, dose_ls, idx, topk=topk, classification=classification)
            else:
                treatment_pred, outcome_pred,_ = obtain_sub_predictions2(model, lang, sub_data, treatment_ls, dose_ls, idx, topk=topk, classification=classification, method_three=method_three)

            sub_treatment_pred.append(treatment_pred.view(-1))
            sub_outcome_pred.append(outcome_pred.view(-1))
        
        if "classification" in dir(model) and model.classification:
            all_treatment_pred.append(torch.stack(sub_treatment_pred).view(-1))
        else:
            all_treatment_pred.append(torch.stack(sub_treatment_pred))
        all_outcome_pred.append(torch.stack(sub_outcome_pred))
    
    if "classification" in dir(model) and model.classification:
        return torch.stack(all_treatment_pred), torch.stack(all_outcome_pred)
    else:
        if model.num_treatments > 2 and not model.cont_treatment:
            return torch.stack(all_treatment_pred), torch.cat(all_outcome_pred)
        else:
            return torch.cat(all_treatment_pred), torch.cat(all_outcome_pred)


def set_lang_data(lang, train_dataset, device, all_treatment_ids=None):
    lang.features = train_dataset.features
    lang.transformed_features = train_dataset.transformed_features
    lang.outcome_array = train_dataset.outcome_array
    lang.treatment_array = train_dataset.treatment_array        
    # if not train_dataset.treatment_graph is None:
    #     lang.treatment_graph = train_dataset.treatment_graph
    if all_treatment_ids is not None:
        lang.transformed_treatment_array = transform_treatment_ids(all_treatment_ids, lang.treatment_array)
    
    lang.count_outcome_array = train_dataset.count_outcome_array
    lang.dose_array = train_dataset.dose_array
    lang.data = train_dataset.data
    lang.dataset = train_dataset
    # if gpu_db and torch.cuda.is_available():
    lang.features = lang.features.to(device)
    lang.transformed_features = lang.transformed_features.to(device)
    lang.outcome_array = lang.outcome_array.to(device)
    lang.treatment_array = lang.treatment_array.to(device)
    if all_treatment_ids is not None:
        lang.transformed_treatment_array = lang.transformed_treatment_array.to(device)
    if train_dataset.count_outcome_array is not None:
        lang.count_outcome_array = lang.count_outcome_array.to(device)
    if train_dataset.dose_array is not None:
        lang.dose_array = lang.dose_array.to(device)
    return lang


def perturb_samples(X, dataset, num_epsilon=1e-3, cat_epsolon=1e-2):
    pert_X = X.clone()
    for fid in range(len(dataset.num_cols)):
        pert_X[:,fid] = pert_X[:,fid] + (torch.rand(pert_X[:,fid].shape)*2-1)*num_epsilon
    
    for fid in range(len(dataset.cat_cols)):
        
        unique_val_ids= list(dataset.cat_id_unique_vals_mappings[dataset.cat_cols[fid]].keys())      
        
        curr_cat_val_ids = list(pert_X[:,fid+len(dataset.num_cols)].view(-1).long())
        
        sample_ids = np.arange(len(curr_cat_val_ids))
        
        selected_sample_ids = sample_ids[0:int(cat_epsolon*len(curr_cat_val_ids))]
        
        for sample_id in selected_sample_ids:
            origin_val = curr_cat_val_ids[sample_id]
            perturbed_val = random.choice(unique_val_ids)
            while perturbed_val == origin_val:
                perturbed_val = random.choice(unique_val_ids)
            curr_cat_val_ids[sample_id] = perturbed_val
        pert_X[:,fid+len(dataset.num_cols)] = torch.tensor(curr_cat_val_ids)
    
    print("perturbation norm::", torch.norm(pert_X - X))
    return pert_X
    
def build_explanation_mappings(origin_exp):
    origin_exp_mappings={}
    for sub_idx in range(len(origin_exp)):
        if ">=" in origin_exp[sub_idx]:
            val_str = origin_exp[sub_idx].split(">=")[-1]
        elif "<=" in origin_exp[sub_idx]:
            val_str = origin_exp[sub_idx].split("<=")[-1]
        elif ">" in origin_exp[sub_idx]:
            val_str = origin_exp[sub_idx].split(">")[-1]
        elif "<" in origin_exp[sub_idx]:
            val_str = origin_exp[sub_idx].split("<")[-1]
        else:
            val_str = origin_exp[sub_idx].split("=")[-1]
        
        exp_key = origin_exp[sub_idx].split(val_str)[0]
        exp_key = exp_key.replace(" ", "")
        origin_exp_mappings[exp_key] = float(val_str)
    return origin_exp_mappings

def evaluate_explanation_diff_single_pair(origin_exp, pert_exp):
    curr_sim_score_ls = []
    for sub_idx1 in range(len(origin_exp)):
        curr_sub_sim_score_ls = []
        for sub_idx2 in range(len(pert_exp)):
            origin_exp_mappings = build_explanation_mappings(origin_exp[sub_idx1])
            pert_exp_mappings = build_explanation_mappings(pert_exp[sub_idx2])
            common_keys = set(origin_exp_mappings.keys()).intersection(set(pert_exp_mappings.keys()))
            sim_scores = 0
            for key in common_keys:
                origin_val = origin_exp_mappings[key]
                pert_val = pert_exp_mappings[key]
                if origin_val - pert_val == 0:
                    score = 1
                else:
                    score = 1 - (abs(origin_val - pert_val)/(abs(origin_val) + abs(pert_val)))
                sim_scores += score
            
            sim_scores = sim_scores/len(origin_exp_mappings)
            curr_sub_sim_score_ls.append(sim_scores)
        curr_sim_score_ls.append(curr_sub_sim_score_ls)
    return curr_sim_score_ls


def evaluate_explanation_dff(origin_exp_str_ls, pert_exp_str_ls):
    sim_score_ls = []
    for idx in range(len(origin_exp_str_ls)):
        origin_exp = origin_exp_str_ls[idx]
        pert_exp = pert_exp_str_ls[idx]
        curr_sim_score_ls = evaluate_explanation_diff_single_pair(origin_exp, pert_exp)
        sim_score_ls.append(torch.tensor(curr_sim_score_ls).max(dim=-1)[0].mean().item())
    return torch.tensor(sim_score_ls)
        
        
        
        
        
        
def construct_base_x(X, full_base_X, fid_ls):
    fid_ls = fid_ls.to(X.device)
    base_x = X.clone()
    base_x[fid_ls] = full_base_X[fid_ls]
    
    return base_x

def generate_all_subsets(input_set):
    """
    Generate all possible subsets of a given set.

    Parameters:
        - input_set: The input set for which to generate subsets.

    Returns:
        - subsets: A list of subsets, where each subset is represented as a tuple.
    """
    input_list = list(input_set)  # Convert the set to a list for indexing
    n = len(input_list)
    subsets = []

    for i in range(n + 1):
        # Generate all combinations of length i
        for combo in combinations(input_list, i):
            subsets.append(tuple(combo))

    return subsets

def obtain_post_hoc_explanatios_main(self, test_dataset, test_dataloader, unique_treatment_set, feature_tensor, origin_feature_tensor, tree_depth=2, explanation_type="decision_tree", subset_ids=None):
        outcome_ls_by_treatment = dict()
        if not self.has_dose and not self.cont_treatment:
            decistion_rules_by_treatment = dict()
            explainer_by_treatment = dict()
            for treatment in unique_treatment_set:
                feature_tensor = feature_tensor.to(self.device)
                treatment_tensor = torch.ones(len(feature_tensor), dtype=torch.long) * treatment
                treatment_tensor = treatment_tensor.to(self.device)
                embedding, out = self.model.forward(feature_tensor, treatment_tensor)
                outcome_ls_by_treatment[treatment] = out.detach().cpu()
                if explanation_type == 'decision_tree':
                    curr_tree = DecisionTreeRegressor(max_depth=tree_depth)
                    curr_tree.fit(origin_feature_tensor.detach().cpu(), out.detach().cpu())
                    decision_rule_ls = []
                    if subset_ids is None:
                        sample_ids = list(range(feature_tensor.shape[0]))
                    else:
                        sample_ids = subset_ids
                    for sample_id in sample_ids:
                        decision_rule = extract_decision_rules(curr_tree, test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols, origin_feature_tensor[sample_id].numpy())
                        decision_rule_ls.append(decision_rule)
                    decistion_rules_by_treatment[treatment] = decision_rule_ls
                    explainer_by_treatment[treatment] = curr_tree
                elif explanation_type == 'lime':
                    self.model.base_treatment=treatment
                    self.model.device = self.device

                    explainer = Lime(feature_tensor.cpu().numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(feature_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                    full_selected_col_ids = []
                    if subset_ids is None:
                        sample_ids = list(range(feature_tensor.shape[0]))
                    else:
                        sample_ids = subset_ids
                    for sample_id in tqdm(sample_ids):
                        out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                        # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                        selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                        # for col in  selected_col_ids:
                        #     full_selected_col_ids
                        full_selected_col_ids.append(selected_col_ids)
                    decistion_rules_by_treatment[treatment] = full_selected_col_ids
                    explainer_by_treatment[treatment] = explainer
                elif explanation_type == "anchor":
                    self.model.base_treatment=treatment
                    self.model.device = self.device
                    if not self.classification:
                        explainer = anchor_tabular.AnchorTabularExplainer(
                            ["low", "high"],
                            # ["feat_" + str(idx) for idx in range(feature_tensor.shape[-1])],
                            test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            test_dataset.features.numpy(),
                            discretizer='quartile')
                    else:
                        explainer =anchor_tabular_class.AnchorTabularExplainer(
                            [], # class names
                            test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            test_dataset.features.numpy(),
                            discretizer='quartile')
                    full_selected_col_ids = []
                    all_decision_rule_ls = []
                    if subset_ids is None:
                        sample_ids = list(range(feature_tensor.shape[0]))
                    else:
                        sample_ids = subset_ids
                    for sample_id in tqdm(sample_ids):
                        # out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                        if not self.classification:
                            out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), 
                                                self.model.predict_given_treatment_dose, 
                                                coverage_samples=100,
                                                prealloc_factor=100,
                                                #    coverage_samples=10000, # small: 10
                                                #    prealloc_factor=10000, # small: 100
                                                max_anchor_size=3,  
                                                delta=0.1,
                                                compute_label_fn=lambda y_pred, y: (1 - abs(y_pred - y)) ** 2,
                                                verbose=True,
                                                verbose_every=20)
                        else:
                            out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), 
                                               self.model.predict_given_treatment_dose2, 
                                               coverage_samples=100,
                                               delta=0.1,
                                               verbose=True,
                                               verbose_every=20)
                        # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                        # selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                        # for col in  selected_col_ids:
                        #     full_selected_col_ids
                        # full_selected_col_ids.append(selected_col_ids)
                        decision_rules = out.names()
                        decision_rules = decision_rules[0:tree_depth]
                        all_decision_rule_ls.append(decision_rules)
                    decistion_rules_by_treatment[treatment] = all_decision_rule_ls
                    explainer_by_treatment[treatment] = explainer
                elif explanation_type == "lore":
                    self.model.base_treatment=treatment
                    self.model.device = self.device
                    features_map = {}
                    for i, f in enumerate(test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols):
                        features_map[i] = { f"{f}": i }
                    config = {
                        'feature_names': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        'class_name': 'y_norm',
                        'class_values': test_dataloader.dataset.outcome_array,
                        'numeric_columns': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        'features_map': features_map,
                    }
                    config = {
                            'feature_names': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            'class_name': 'y_norm',
                            'class_values': test_dataloader.dataset.outcome_array,
                            'numeric_columns': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            'features_map': features_map,
                        }
                    if not self.classification:
                        
                        explainer = LOREM_reg(
                            feature_tensor.cpu().numpy(), 
                            self.model.predict_given_treatment_dose, 
                            config['feature_names'],
                            config['class_name'], 
                            config['class_values'],
                            config['numeric_columns'],
                            config['features_map'],
                            # note: `genetic`, `rndgen`, `random` & `closest` don't require model probabilites.
                            # if you use `geneticp` or `rndgenp`, then set bb_predict_proba
                            neigh_type='genetic', 
                            categorical_use_prob=True,
                            continuous_fun_estimation=False, 
                            size=1000,
                            ocr=0.1,
                            random_state=0, # ensure np.random.seed(0) is set before running
                            ngen=10,
                            verbose=True)
                    else:
                        # config = {
                        #     'feature_names': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        #     'class_name': 'y_norm',
                        #     'class_values': test_dataloader.dataset.outcome_array,
                        #     'numeric_columns': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        #     'features_map': features_map,
                        # }

                        explainer = LOREM(
                            feature_tensor.cpu().numpy(), 
                            self.model.predict_given_treatment_dose, 
                            config['feature_names'],
                            config['class_name'], 
                            config['class_values'],
                            config['numeric_columns'],
                            config['features_map'],
                            # note: `genetic`, `rndgen`, `random` & `closest` don't require model probabilites.
                            # if you use `geneticp` or `rndgenp`, then set bb_predict_proba
                            neigh_type='genetic', 
                            categorical_use_prob=True,
                            continuous_fun_estimation=False, 
                            size=1000,
                            ocr=0.1,
                            random_state=0, # ensure np.random.seed(0) is set before running
                            ngen=10,
                            verbose=True)
                    
                    full_selected_col_ids = []
                    all_decision_rule_ls = []
                    if subset_ids is None:
                        sample_ids = list(range(feature_tensor.shape[0]))
                    else:
                        sample_ids = subset_ids
                    for sample_id in tqdm(sample_ids):
                        # out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                        out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), samples=50)
                        # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                        # selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                        # for col in  selected_col_ids:
                        #     full_selected_col_ids
                        # full_selected_col_ids.append(selected_col_ids)
                        decision_rules = out.rule._pstr()
                        # decision_rules = out.names()
                        decision_rules = decision_rules.replace("{", "").replace("}", "")
                        decision_rules = decision_rules.split(",")
                        decision_rules = [decision_rules[did] for did in range(tree_depth)]
                        decision_rules = [rule.strip() for rule in decision_rules]
                        all_decision_rule_ls.append(decision_rules)
                    decistion_rules_by_treatment[treatment] = all_decision_rule_ls
                    explainer_by_treatment[treatment] = explainer
                elif explanation_type == "shap":
                    # e = shap.DeepExplainer(self.model, feature_tensor.cpu().numpy()) 
                    mono_model = MonoTransTEE(self.model, treatment)
                    # e = shap.DeepExplainer(mono_model, shap.utils.sample(feature_tensor, 100)) 
                    e = shap.DeepExplainer(mono_model, self.train_dataset.features.to(self.device)) 
                    all_shap_values = []
                    
                    if subset_ids is None:
                        sample_ids = list(range(feature_tensor.shape[0]))
                    else:
                        sample_ids = subset_ids
                    for sample_id in tqdm(sample_ids):
                    #     shap_values = torch.from_numpy(e.shap_values(feature_tensor[sample_id].view(1,-1)))
                    # for sample_id in tqdm(range(feature_tensor.shape[0])):
                        shap_val = e.shap_values(feature_tensor[sample_id].view(1,-1))
                        if type(shap_val) is list:
                        
                            shap_values = torch.from_numpy(shap_val[0])
                        else:
                            shap_values = torch.from_numpy(shap_val)
                        topk_s_val, topk_ids= torch.topk(shap_values.view(-1), k=tree_depth)
                        
                        selected_col_ids = {int(topk_ids[idx]):topk_s_val[idx] for idx in range(len(topk_ids))}
                        
                        
                        all_shap_values.append(selected_col_ids)
                    # all_shap_value_tensor = np.concatenate(all_shap_values, axis=0)
                    decistion_rules_by_treatment[treatment] = all_shap_values
                    explainer_by_treatment[treatment] = e
            return decistion_rules_by_treatment, explainer_by_treatment
        elif self.has_dose or self.cont_treatment:
            feature_tensor = feature_tensor.to(self.device)
            # treatment_tensor = torch.ones(len(feature_tensor), dtype=torch.long) * treatment
            # treatment_tensor = treatment_tensor.to(self.device)
            # embedding, out = self.model.forward(feature_tensor, treatment_tensor)
            # outcome_ls_by_treatment[treatment] = out.detach().cpu()
            
            decision_rule_ls = []
            explainer_ls = []
            if subset_ids is None:
                sample_ids = list(range(feature_tensor.shape[0]))
            else:
                sample_ids = subset_ids
            for sample_id in tqdm(sample_ids):
                if explanation_type == 'decision_tree':
                    feature_tensor = feature_tensor.to(self.device)
                    treatment_tensor = torch.ones(len(feature_tensor), dtype=torch.long) * test_dataset.treatment_array[sample_id]
                    treatment_tensor = treatment_tensor.to(self.device)
                    dose_tensor = None
                    if self.has_dose:
                        dose_tensor = torch.ones(len(feature_tensor), dtype=torch.long) * test_dataset.dose_array[sample_id]
                        dose_tensor = dose_tensor.to(self.device)
                    embedding, out = self.model.forward(feature_tensor, treatment_tensor, d=dose_tensor)
                    # outcome_ls_by_treatment[treatment] = out.detach().cpu()
                    curr_tree = DecisionTreeRegressor(max_depth=tree_depth)
                    curr_tree.fit(origin_feature_tensor.detach().cpu(), out.detach().cpu())
                    # decision_rule_ls = []
                
                    decision_rule = extract_decision_rules(curr_tree, test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols, origin_feature_tensor[sample_id].numpy())
                    decision_rule_ls.append(decision_rule)
                    explainer_ls.append(curr_tree)
                    # decistion_rules_by_treatment[treatment] = decision_rule_ls
                    # explainer_by_treatment[treatment] = curr_tree
                elif explanation_type == 'lime':
                    self.model.base_treatment=test_dataset.treatment_array[sample_id]
                    if self.has_dose:
                        self.model.base_dose=test_dataset.dose_array[sample_id]
                    self.model.device = self.device

                    explainer = Lime(feature_tensor.cpu().numpy(), self.model.predict_given_treatment_dose, feature_names=["feat_" + str(idx) for idx in range(feature_tensor.shape[-1])], mode="regression" if self.outcome_regression else "classification")
                    # full_selected_col_ids = []
                # for sample_id in tqdm(range(feature_tensor.shape[0])):
                    out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                    # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                    selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                    # for col in  selected_col_ids:
                    #     full_selected_col_ids
                    # full_selected_col_ids.append(selected_col_ids)
                    decision_rule_ls.append(selected_col_ids)
                    explainer_ls.append(explainer)
                elif explanation_type == "anchor":
                    self.model.base_treatment=test_dataset.treatment_array[sample_id]
                    if self.has_dose:
                        self.model.base_dose=test_dataset.dose_array[sample_id]
                    self.model.device = self.device
                    if not self.classification:
                        explainer = anchor_tabular.AnchorTabularExplainer(
                        ["low", "high"],
                        # ["feat_" + str(idx) for idx in range(feature_tensor.shape[-1])],
                        test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        test_dataset.features.numpy(),
                        discretizer='quartile')
                    
                    else:
                        explainer =anchor_tabular_class.AnchorTabularExplainer(
                            [], # class names
                            test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            test_dataset.features.numpy(),
                            discretizer='quartile')
                    # full_selected_col_ids = []
                    # all_decision_rule_ls = []
                # for sample_id in tqdm(range(feature_tensor.shape[0])):
                    # out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                    if not self.classification:
                        out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), 
                                        self.model.predict_given_treatment_dose, 
                                        coverage_samples=100,
                                        prealloc_factor=100,
                                        #    coverage_samples=10000, # small: 10
                                        #    prealloc_factor=10000, # small: 100
                                        max_anchor_size=3,  
                                        delta=0.1,
                                        compute_label_fn=lambda y_pred, y: (1 - abs(y_pred - y)) ** 2,
                                        verbose=True,
                                        verbose_every=20)
                    
                    else:
                            out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), 
                                               self.model.predict_given_treatment_dose2, 
                                               coverage_samples=100,
                                               delta=0.1,
                                               verbose=True,
                                               verbose_every=20)
                    # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                    # selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                    # for col in  selected_col_ids:
                    #     full_selected_col_ids
                    # full_selected_col_ids.append(selected_col_ids)
                    decision_rules = out.names()
                    # all_decision_rule_ls.append(decision_rules)
                    # decistion_rules_by_treatment[treatment] = all_decision_rule_ls
                    # explainer_by_treatment[treatment] = explainer
                    decision_rule_ls.append(decision_rules)
                    explainer_ls.append(explainer)
                elif explanation_type == "lore":
                    self.model.base_treatment=test_dataset.treatment_array[sample_id]
                    if self.has_dose:
                        self.model.base_dose=test_dataset.dose_array[sample_id]
                    self.model.device = self.device
                    features_map = {}
                    for i, f in enumerate(test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols):
                        features_map[i] = { f"{f}": i }
                    config = {
                            'feature_names': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            'class_name': 'y_norm',
                            'class_values': test_dataloader.dataset.outcome_array,
                            'numeric_columns': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                            'features_map': features_map,
                        }
                    if not self.classification:
                        
                        explainer = LOREM_reg(
                            feature_tensor.cpu().numpy(), 
                            self.model.predict_given_treatment_dose, 
                            config['feature_names'],
                            config['class_name'], 
                            config['class_values'],
                            config['numeric_columns'],
                            config['features_map'],
                            # note: `genetic`, `rndgen`, `random` & `closest` don't require model probabilites.
                            # if you use `geneticp` or `rndgenp`, then set bb_predict_proba
                            neigh_type='genetic', 
                            categorical_use_prob=True,
                            continuous_fun_estimation=False, 
                            size=1000,
                            ocr=0.1,
                            random_state=0, # ensure np.random.seed(0) is set before running
                            ngen=10,
                            verbose=True)
                    else:
                        # config = {
                        #     'feature_names': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        #     'class_name': 'y_norm',
                        #     'class_values': test_dataloader.dataset.outcome_array,
                        #     'numeric_columns': test_dataloader.dataset.num_cols + test_dataloader.dataset.cat_cols,
                        #     'features_map': features_map,
                        # }

                        self.explainer = LOREM(
                            feature_tensor.cpu().numpy(), 
                            self.model.predict_given_treatment_dose, 
                            config['feature_names'],
                            config['class_name'], 
                            config['class_values'],
                            config['numeric_columns'],
                            config['features_map'],
                            # note: `genetic`, `rndgen`, `random` & `closest` don't require model probabilites.
                            # if you use `geneticp` or `rndgenp`, then set bb_predict_proba
                            neigh_type='genetic', 
                            categorical_use_prob=True,
                            continuous_fun_estimation=False, 
                            size=1000,
                            ocr=0.1,
                            random_state=0, # ensure np.random.seed(0) is set before running
                            ngen=10,
                            verbose=True)
                    # full_selected_col_ids = []
                    # all_decision_rule_ls = []
                    # for sample_id in tqdm(range(feature_tensor.shape[0])):
                    # out = explainer.explain(feature_tensor[sample_id].cpu().numpy(), num_features=tree_depth)
                    out = explainer.explain_instance(feature_tensor[sample_id].cpu().numpy(), samples=50)
                    # selected_col_ids = [int(item[0].split("feat_")[1]) for item in out.as_list()]
                    # selected_col_ids = {int(item[0].split("feat_")[1]):item[1] for item in out.as_list()}
                    # for col in  selected_col_ids:
                    #     full_selected_col_ids
                    # full_selected_col_ids.append(selected_col_ids)
                    decision_rules = out.rule._pstr()
                    decision_rules = decision_rules.replace("{", "").replace("}", "")
                    decision_rules = decision_rules.split(",")
                    decision_rules = [rule.strip() for rule in decision_rules]
                    # decision_rules = out.names()
                    decision_rule_ls.append(decision_rules)
                    explainer_ls.append(explainer)
                elif explanation_type == "shap":
                    base_treatment=test_dataset.treatment_array[sample_id].to(self.device)
                    base_dose = None
                    if self.has_dose:
                        base_dose=test_dataset.dose_array[sample_id].to(self.device)
                    # e = shap.DeepExplainer(self.model, feature_tensor.cpu().numpy()) 
                    mono_model = MonoTransTEE(self.model, base_treatment, base_dose)
                    e = shap.DeepExplainer(mono_model, shap.utils.sample(feature_tensor.to(self.device), 100)) 
                    # all_shap_values = []
                    # for sample_id in tqdm(range(feature_tensor.shape[0])):
                    # shap_values = torch.from_numpy(e.shap_values(feature_tensor[sample_id].view(1,-1)))
                    shap_val = e.shap_values(feature_tensor[sample_id].view(1,-1))
                    if type(shap_val) is list:
                        
                        shap_values = torch.from_numpy(shap_val[0])
                    else:
                        shap_values = torch.from_numpy(shap_val)
                    topk_s_val, topk_ids= torch.topk(shap_values.view(-1), k=tree_depth)
                    
                    selected_col_ids = {int(topk_ids[idx]):topk_s_val[idx] for idx in range(len(topk_ids))}
                    
                    
                    # all_shap_values.append(selected_col_ids)
                    # all_shap_value_tensor = np.concatenate(all_shap_values, axis=0)
                    decision_rule_ls.append(selected_col_ids)
                    explainer_ls.append(e)
                    # decision_rule_ls.append(all_decision_rule_ls)
                    # explainer_by_treatment[treatment] = explainer
            
            return decision_rule_ls, explainer_ls
            #, outcome_ls_by_treatment     


def determine_stopping_by_Q_values(atom_ls):
    pred_Q = torch.max(atom_ls["pred_Q"], dim=-1)[0]
    col_Q = torch.max(atom_ls["col_Q"], dim=-1)[0]
    pred_Q = pred_Q.view(pred_Q.shape[0], -1)
    
    all_Q = pred_Q + col_Q
    return torch.sum(all_Q > 0, dim = -1) == all_Q.shape[-1]
    

def reg_model_forward(model, X, A, D, origin_X, out, test=False, full_out=None):
        if not model.backbone == "ENRL":
            if not model.cont_treatment and test:
                _, reg_out, full_reg_out = model.backbone_full_model(X, A, d=D, test=test)
            else:
                _, reg_out = model.backbone_full_model(X, A, d=D, test=test)
        else:
            X_num, X_cat = origin_X[:,0:model.model.numeric_f_num], origin_X[:,model.model.numeric_f_num:]
            X_num = X_num.to(model.device)
            X_cat = X_cat.to(model.device)
            reg_out, other_loss = model.backbone_full_model(X_num, X_cat, A, d=D)
        out = (out + model.regression_ratio*reg_out.reshape(out.shape))/(model.regression_ratio + 1)
        if not test:
            return out
        else:
            if not model.cont_treatment:
                full_out = (full_out + model.regression_ratio*full_reg_out.reshape(full_out.shape))/(model.regression_ratio + 1)
                return out, full_out
            else:
                return out
            
def reg_model_forward_image(model, data, X, A, D, origin_X, out, test=False, full_out=None):
        if not model.backbone == "ENRL":
            if not model.cont_treatment and test:
                reg_out, full_reg_out = model.backbone_full_model(data, X, A, d=D, test=test)
            else:
                reg_out = model.backbone_full_model(data, X, A, d=D, test=test)
        else:
            X_num, X_cat = origin_X[:,0:model.model.numeric_f_num], origin_X[:,model.model.numeric_f_num:]
            X_num = X_num.to(model.device)
            X_cat = X_cat.to(model.device)
            reg_out, other_loss = model.backbone_full_model(data, X_num, X_cat, A, d=D)
        out = (out.view(-1) + model.regression_ratio*reg_out.view(-1))/(model.regression_ratio + 1)
        if not test:
            return out
        else:
            if not model.cont_treatment:
                full_out = (full_out.view(-1) + model.regression_ratio*full_reg_out.view(-1))/(model.regression_ratio + 1)
                return out, full_out
            else:
                return out