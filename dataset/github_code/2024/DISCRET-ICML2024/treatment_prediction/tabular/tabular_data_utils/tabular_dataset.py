from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel

from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.preprocessing import StandardScaler
# from causalforge.data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
import networkx as nx
import pickle
from torch_geometric.data import Data
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from torch_geometric.data import Batch
model_name_str="name"
model_out_count_str="out_count"

model_property_mappings=[{model_name_str: "SamLowe/roberta-base-go_emotions", model_out_count_str:28}]


def create_treatment_graph_batch(units, treatment_graphs, treatment_ids):
    test_unit_pt_dataset = create_pt_geometric_dataset(
            units=units, treatment_graphs=treatment_graphs, treatment_ids=treatment_ids
        )
    
    batch = Batch.from_data_list(test_unit_pt_dataset)
    return batch

def populate_text_attrs(text):
    logits_ls = []    
    for model_idx in range(len(model_property_mappings)):
        model_name = model_property_mappings[model_idx][model_name_str]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, max_length=1024)
        inputs = tokenizer(text, return_tensors="pt",  max_length=1024, truncation=True)
        with torch.no_grad():
            inputs["input_ids"] = inputs["input_ids"][:,0:512]
            inputs["attention_mask"] = inputs["attention_mask"][:,0:512]
            logits = model(**inputs).logits
            logits_ls.append(logits)
    
    return torch.cat(logits_ls).numpy()

def populate_text_attrs_all(all_text, df):
    attr_name_ls = []
    for model_idx in range(len(model_property_mappings)):
        attr_name_ls.extend([model_property_mappings[model_idx][model_name_str] + "_" + str(k) for k in range(model_property_mappings[model_idx][model_out_count_str])])
    
    all_text_ls = list(all_text)
    for idx in tqdm(range(len(all_text_ls))):
        text = df.loc[idx, "text"]# all_text_ls[idx]
        curr_attr_values = populate_text_attrs(text)
        df.loc[idx, attr_name_ls] = curr_attr_values.reshape(-1)
    
    return df

def populate_text_attrs_all_2(all_text, df, ngram=1, max_features=500):
    ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                        stop_words='english', max_features=max_features)
    vectorized_data = ngram_vectorizer.fit_transform(all_text).toarray()
    
    feat_name_ls = ["ngram_" + str(k) for k in range(max_features)]
    
    # df[feat_name_ls] = pd.DataFrame(vectorized_data)
    for idx in tqdm(range(len(vectorized_data))):
        text = df.loc[idx, "text"]# all_text_ls[idx]
        assert text == all_text[idx]
        df.loc[idx, feat_name_ls] = vectorized_data[idx].tolist()
    return df

def split_train_valid_test_df(df, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    random_sample_ids = list(range(len(df)))
    random.shuffle(random_sample_ids)
    random_sample_ids = np.array(random_sample_ids)
    train_count = int(len(df)*train_ratio)
    valid_count = int(len(df)*valid_ratio)
    test_count = int(len(df)*test_ratio)
    train_ids = random_sample_ids[0:train_count]
    valid_ids = random_sample_ids[train_count: train_count + valid_count]
    test_ids = random_sample_ids[train_count + valid_count:train_count + valid_count+test_count]
    train_df = df.iloc[train_ids]
    valid_df = df.iloc[valid_ids]
    test_df = df.iloc[test_ids]
    return train_df, valid_df, test_df

def remove_nan_rows(df):
    df = df.dropna()
    return df

def load_aicc_dataset(data_folder):
    covariates = pd.read_csv(os.path.join(data_folder, "x.csv"))
    all_covariates_with_treatment = []
    
    censor_folder_ls = [os.path.join(data_folder, "scaling"), ]
    
    f_count = 0
    
    for censor_folder in censor_folder_ls:
        for file in os.listdir(censor_folder):
            if not "_cf." in file:
                continue
            file = file.split("_cf.")[0] + ".csv"
            
            full_file_name = os.path.join(censor_folder, file)
            curr_treatment_info = pd.read_csv(full_file_name)
            covariates_with_treatment = pd.merge(covariates, curr_treatment_info, on="sample_id")
            
            cf_full_file_name = os.path.join(censor_folder, file.split(".csv")[0] + "_cf.csv")
            cf_curr_treatment_info = pd.read_csv(cf_full_file_name)
            cf_covariates_with_treatment = pd.merge(covariates_with_treatment, cf_curr_treatment_info, on="sample_id")
            
            control_outcome_ls = list(cf_covariates_with_treatment["y0"])
            treatment_outcome_ls = list(cf_covariates_with_treatment["y1"])
            treatment_ls = list(cf_covariates_with_treatment["z"])
            count_outcome_ls = [treatment_outcome_ls[i] if treatment_ls[i] == 0 else control_outcome_ls[i] for i in range(len(treatment_ls))]
            
            cf_covariates_with_treatment["count_Y"] = count_outcome_ls
            
            cf_covariates_with_treatment.drop(columns=["y0", "y1"], inplace=True)
            all_covariates_with_treatment.append(cf_covariates_with_treatment)
            f_count += 1
            if f_count >= 3:
                break
    all_covariates_with_treatment_df = pd.concat(all_covariates_with_treatment)
    all_data = all_covariates_with_treatment_df.reset_index().drop(columns=["index", "sample_id"]).reset_index()
    all_data = all_data.sample(frac=0.2)
    return all_data


def load_ihdp_cont_dataset(data_folder, dataset_idx):
    data_matrix = torch.load(data_folder + '/train_matrix.pt')
    test_data_matrix = torch.load(data_folder + '/data_matrix.pt')
    data_grid = torch.load(data_folder + '/t_grid.pt')
    train_idx = torch.load(data_folder + '/eval/' + str(dataset_idx) + '/idx_train.pt')
    test_idx = torch.load(data_folder + '/eval/' + str(dataset_idx) + '/idx_test.pt')
    train_matrix = data_matrix[train_idx]
    test_matrix = test_data_matrix[test_idx].numpy()
    t_grid = data_grid[:,test_idx]
    train_t_grid = data_grid[:, train_idx]
    
    train_matrix, valid_matrix, train_t_grid, valid_t_grid = train_test_split(train_matrix.numpy(), np.transpose(train_t_grid), test_size=0.2, random_state=42)
    
    all_attr = ["Treatment"] + ["x_" + str(k) for k in range(1, train_matrix.shape[-1]-1)] + ["outcome"]
    
    train_df = pd.DataFrame(train_matrix, columns=all_attr)
    valid_df = pd.DataFrame(valid_matrix, columns=all_attr)
    test_df = pd.DataFrame(test_matrix, columns=all_attr)
    train_df = train_df.reset_index().rename(columns={"index": "id"})
    valid_df = valid_df.reset_index().rename(columns={"index": "id"})
    test_df = test_df.reset_index().rename(columns={"index": "id"})
    all_data = pd.concat([train_df, valid_df, test_df])
    
    return train_df, valid_df, test_df, all_data, (np.transpose(valid_t_grid), t_grid)

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

def get_treatment_graphs(treatment_ids: list, id_to_graph_dict: dict):
    return [id_to_graph_dict[i] for i in treatment_ids]

def transform_treatment_graph(node_features, edges):
    graph = nx.Graph()
    for node_idx in range(len(node_features)):
        graph.add_node(node_idx)
        graph.nodes[node_idx]["features"] = node_features[node_idx].item()
    
    graph.add_edges_from(edges)
    return graph


def compute_treatment_similarity_mat(all_treatment_ids, treatment_id_graph_mappings):
    treatment_similarity_mat = np.zeros((len(all_treatment_ids), len(all_treatment_ids)))
    upper_bound = 50
    for idx in range(len(all_treatment_ids)):

        for idx2 in tqdm(range(len(all_treatment_ids))):
            if idx < idx2:
                tid = all_treatment_ids[idx]
                tid2 = all_treatment_ids[idx2]
                treatment_graph = treatment_id_graph_mappings[tid]
                treatment_graph2 = treatment_id_graph_mappings[tid2]
                tr_g = transform_treatment_graph(treatment_graph["node_features"], treatment_graph["edges"])
                tr_g2 = transform_treatment_graph(treatment_graph2["node_features"], treatment_graph2["edges"])
                # ged=gm.GraphEditDistance(1,1,1,1)
                # ged.set_attr_graph_used('features',None)
                # result=ged.compare([tr_g,tr_g2],None) 
                # cost = ged.distance(result)
                res = nx.optimize_edit_paths(tr_g, tr_g2, node_match=lambda x,y: np.linalg.norm(x["features"] - y["features"]) < 0.005, upper_bound=upper_bound, timeout=2)
                cost = -1
                for c in res:
                    cost = c[-1]
                    break
                if cost < 0:
                    cost = upper_bound
                treatment_similarity_mat[idx, idx2] = cost
        # if upper_bound is not None:
        #     upper_bound = max(treatment_similarity_mat[idx].max(), upper_bound)
        # else:
        #     upper_bound = treatment_similarity_mat[idx].max()
    max_cost = np.max(treatment_similarity_mat)
    treatment_similarity_mat = 1 - treatment_similarity_mat / max_cost

    return treatment_similarity_mat

def randomly_assign_nan(array, portion=0.2):
    total_elements = array.size
    num_nan_elements = int(portion * total_elements)

    # Generate random indices to assign NaN values
    nan_indices = np.random.choice(total_elements, size=num_nan_elements, replace=False)

    # Create a mask to identify elements to be set to NaN
    nan_mask = np.full_like(array, fill_value=False, dtype=bool)
    nan_mask.flat[nan_indices] = True

    # Assign NaN values to the specified portion using the mask
    array_with_nan = np.where(nan_mask, np.nan, array)

    return array_with_nan





def load_ihdp_dataset2(data_folder, dataset_id, subset_sum, missing_ratio=None):
    
    feat_name_ls = ["bw","b.head","preterm","birth.o","nnhealth","momage","sex","twin","b.marr","mom.lths","mom.hs","mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was"]
    
    col_names = ["treatment", "y_factual", "y_cfactual", "mu0","mu1"] + feat_name_ls#["x_" + str(k) for k in range(1,26)]
    # perturbed_ids = np.random.permutation(50)
    # file_ids = [perturbed_ids[0] + 1]
    file_ids = [dataset_id]
    # valid_ids = perturbed_ids[35:40] + 1 
    # test_ids = perturbed_ids[40:50] + 1
    
    
    if subset_sum is None:
        all_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in file_ids]
    else:
        dataset_ids = [dataset_id, dataset_id + 1, dataset_id + 2]
        dataset_ids = [k if k <= 50 else k-50 for k in dataset_ids]
        all_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in dataset_ids]
    
    # valid_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in valid_ids]
    # test_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in test_ids]
    
    
    
    all_df = pd.concat(all_data_ls)
    all_df.drop(columns=["mu0", "mu1"], inplace=True)


    all_df = all_df.reset_index().rename(columns={"index": "id"})
    
    random_sample_ids = np.random.permutation(len(all_df))
    train_ids = random_sample_ids[0:int(len(all_df)*0.7)]
    valid_ids = random_sample_ids[int(len(all_df)*0.7):int(len(all_df)*0.8)]
    test_ids = random_sample_ids[int(len(all_df)*0.8):]
    
    train_data = all_df.iloc[train_ids]
    valid_data = all_df.iloc[valid_ids]
    test_data = all_df.iloc[test_ids]
    # valid_data = pd.concat(valid_data_ls)
    # valid_data.drop(columns=["mu0", "mu1"], inplace=True)
    # valid_data = valid_data.reset_index().rename(columns={"index": "id"})
    # test_data = pd.concat(test_data_ls)
    # test_data.drop(columns=["mu0", "mu1"], inplace=True)
    # test_data = test_data.reset_index().rename(columns={"index": "id"})
    # all_df = pd.concat([train_data, valid_data, test_data])
    return train_data, valid_data, test_data, all_df

def sample_sets(feat_name_ls, num_sets=10, samples_per_set=4):
    """
    Sample 'num_sets' sets of 'samples_per_set' samples from 'samples'.
    
    Args:
    samples (list or set): The set of samples to draw from.
    num_sets (int): The number of sets to generate.
    samples_per_set (int): The number of samples in each set.
    
    Returns:
    list of lists: A list containing 'num_sets' sets of samples.
    """
    if len(feat_name_ls) < samples_per_set:
        raise ValueError("Cannot sample more items per set than available in 'samples'.")
    
    result_sets = []
    result_feat_id_sets = []
    
    for _ in range(num_sets):
        sampled_set = random.sample(list(range(len(feat_name_ls))), samples_per_set)
        sampled_feat_names = [feat_name_ls[k] for k in sampled_set]
        result_sets.append(sampled_feat_names)
        result_feat_id_sets.append(list(sampled_set))
    
    return result_sets, result_feat_id_sets


def get_outcome_by_selected_feats(curr_subset_df, curr_feat_set, treatment_feat):
    features = np.array(curr_subset_df[curr_feat_set])
    treatment_array = np.array(curr_subset_df[treatment_feat])
    random_coeff = np.random.random(len(curr_feat_set))
    outcome_array = np.sum(features*random_coeff.reshape(1,-1),axis=-1) + treatment_array*features[:,0] + np.random.random(len(curr_subset_df))*0.1
    count_outcome_array = np.sum(features*random_coeff.reshape(1,-1),axis=-1) + (1-treatment_array)*features[:,0] + np.random.random(len(curr_subset_df)).reshape(1,-1)*0.1
    return outcome_array, count_outcome_array, random_coeff



def synthesize_outcome_df(all_df, num_sets, result_sets, sub_feat_id_sets, treatment_feat, outcome_feat, count_outcome_feat, dataset_id):
    all_sample_ids = list(range(len(all_df)))
    random.seed(dataset_id)
    random.shuffle(all_sample_ids)
    sample_size_per_set = int(len(all_df)/num_sets) + 1
    all_outcome_array_ls = []
    all_count_outcome_array_ls = []
    meta_data_mappings = [None]*len(all_df)
    batch_ids = 0
    for k in range(0, len(all_df), sample_size_per_set):
        start_idx = k
        end_idx = start_idx + sample_size_per_set
        if end_idx >= len(all_df):
            end_idx = len(all_df)
        curr_subset_sample_ids = all_sample_ids[start_idx: end_idx]
        curr_subset_df = all_df.iloc[curr_subset_sample_ids]
        curr_feat_set = result_sets[batch_ids]
        curr_feat_id_set = sub_feat_id_sets[batch_ids]
        if 265 in curr_subset_sample_ids:
            print()
        outcome_array, count_outcome_array, random_coeff = get_outcome_by_selected_feats(curr_subset_df, curr_feat_set, treatment_feat)
        all_outcome_array_ls.append(outcome_array.reshape(-1))
        all_count_outcome_array_ls.append(count_outcome_array.reshape(-1))
        for sample_id in curr_subset_sample_ids:
            meta_data_mappings[sample_id] = dict()
            meta_data_mappings[sample_id]["feat_names"] = curr_feat_set
            meta_data_mappings[sample_id]["feat_ids"] = curr_feat_id_set
            meta_data_mappings[sample_id]["coeff"] = random_coeff
        batch_ids += 1
        
    all_outcome_array_numpy = np.concatenate(all_outcome_array_ls)
    all_count_outcome_array_numpy = np.concatenate(all_count_outcome_array_ls)
    all_df.loc[all_sample_ids, outcome_feat] = all_outcome_array_numpy
    all_df.loc[all_sample_ids, count_outcome_feat] = all_count_outcome_array_numpy

    # all_df_copy = all_df.iloc[all_sample_ids].copy()
    # all_df_copy[outcome_feat] = all_outcome_array_numpy
    # all_df_copy[count_outcome_feat] = all_count_outcome_array_numpy
    return all_df, meta_data_mappings

def load_synthetic_data(data_folder, dataset_id, num_sets=5, samples_per_set=2):
    
    # feat_name_ls = ["bw","b.head","preterm","birth.o","nnhealth","momage","sex","twin","b.marr","mom.lths","mom.hs","mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was"]
    feat_name_ls = ["x_" + str(k) for k in range(1,26)]
    
    col_names = ["treatment", "y_factual", "y_cfactual", "mu0","mu1"] + feat_name_ls#["x_" + str(k) for k in range(1,26)]
    # perturbed_ids = np.random.permutation(50)
    # file_ids = [perturbed_ids[0] + 1]
    file_ids = [dataset_id]
    # valid_ids = perturbed_ids[35:40] + 1 
    # test_ids = perturbed_ids[40:50] + 1
    
    
    
    all_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in file_ids]
    
    # valid_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in valid_ids]
    # test_data_ls = [pd.read_csv(os.path.join(data_folder, "ihdp_npci_" + str(k) + ".csv"), names=col_names) for k in test_ids]
    
    
    
    all_df = pd.concat(all_data_ls)
    all_df.drop(columns=["mu0", "mu1"], inplace=True)
    all_df = all_df.reset_index().rename(columns={"index": "id"})

    sub_feat_name_ls, sub_feat_id_ls = sample_sets(feat_name_ls, num_sets=num_sets, samples_per_set=samples_per_set)
    outcome_attr = "y_factual"
    treatment_attr = "treatment"
    count_outcome_attr = "y_cfactual"

    sub_data_folder = os.path.join(data_folder, str(dataset_id))
    os.makedirs(sub_data_folder, exist_ok=True)
    meta_data_file = os.path.join(sub_data_folder, "synthetic_meta_info")
    data_file = os.path.join(sub_data_folder, "synthetic_data.csv")
    if os.path.exists(meta_data_file):
        with open(meta_data_file, "rb") as f:
            meta_data_mappings = pickle.load(f)
        all_df = pd.read_csv(data_file)
    else:
        all_df, meta_data_mappings = synthesize_outcome_df(all_df, num_sets, sub_feat_name_ls, sub_feat_id_ls, treatment_attr, outcome_attr, count_outcome_attr, dataset_id)
        all_df.to_csv(data_file)
        with open(meta_data_file, "wb") as f:
            pickle.dump(meta_data_mappings, f)
    
    random_sample_ids = np.random.permutation(len(all_df))
    train_ids = random_sample_ids[0:int(len(all_df)*0.7)]
    train_meta_data_file = [meta_data_mappings[idx] for idx in train_ids]
    
    valid_ids = random_sample_ids[int(len(all_df)*0.7):int(len(all_df)*0.8)]
    valid_meta_data_file = [meta_data_mappings[idx] for idx in valid_ids]
    
    test_ids = random_sample_ids[int(len(all_df)*0.8):]
    test_meta_data_file = [meta_data_mappings[idx] for idx in test_ids]
    with open(os.path.join(sub_data_folder, "train_synthetic_meta_info"), "wb") as f:
        pickle.dump(train_meta_data_file, f)
        
    with open(os.path.join(sub_data_folder, "valid_synthetic_meta_info"), "wb") as f:
        pickle.dump(valid_meta_data_file, f)
        
    with open(os.path.join(sub_data_folder, "test_synthetic_meta_info"), "wb") as f:
        pickle.dump(test_meta_data_file, f)
    
    
    train_data = all_df.iloc[train_ids]
    valid_data = all_df.iloc[valid_ids]
    test_data = all_df.iloc[test_ids]
    # valid_data = pd.concat(valid_data_ls)
    # valid_data.drop(columns=["mu0", "mu1"], inplace=True)
    # valid_data = valid_data.reset_index().rename(columns={"index": "id"})
    # test_data = pd.concat(test_data_ls)
    # test_data.drop(columns=["mu0", "mu1"], inplace=True)
    # test_data = test_data.reset_index().rename(columns={"index": "id"})
    # all_df = pd.concat([train_data, valid_data, test_data])
    return train_data, valid_data, test_data, all_df


def get_other_satisfiable_samples(all_features, feature_ids, feature_values, other_sample_ids, all_feature_values_in_rules, sample_id_rule_feature_ids_mappings):
    other_satisfiable_sample_boolean = np.ones(len(other_sample_ids), dtype=bool)
    for idx in range(len(feature_ids)):
        other_satisfiable_sample_boolean = other_satisfiable_sample_boolean & (all_features[other_sample_ids, feature_ids[idx]] == feature_values[idx])
    other_satisfiable_sample_ids = other_sample_ids[other_satisfiable_sample_boolean]
    perturb_feature_values = feature_values.copy()
    perturb_feature_values = 1 - perturb_feature_values

    for sid in other_satisfiable_sample_ids:
        curr_sample_rule_feature_ids = set(sample_id_rule_feature_ids_mappings[sid])
        curr_sample_covered_feature_ids = curr_sample_rule_feature_ids.intersection(set(feature_ids.tolist()))
        curr_perturb_features = perturb_feature_values.copy()
        for fid in curr_sample_covered_feature_ids:
            curr_perturb_features[np.where(feature_ids == fid)[0][0]] = all_feature_values_in_rules[fid]
        # for fid in range(len(curr_sample_rule_feature_ids)):
        #     if feature_ids[fid] in all_feature_values_in_rules:
        #         perturb_feature_values[fid] = all_feature_values_in_rules[feature_ids[fid]]
        all_features[sid, feature_ids] = curr_perturb_features
        assert np.sum(np.abs(perturb_feature_values - feature_values)) > 0
    
    # for fid in range(len(feature_ids)):
    #     if feature_ids[fid] in all_feature_values_in_rules:
    #         perturb_feature_values[fid] = all_feature_values_in_rules[feature_ids[fid]]
    #     all_features[other_satisfiable_sample_ids, feature_ids[fid]] = perturb_feature_values[fid]
    
    return all_features
    

def generate_data_by_rules(sample_count, rule_count = 4, rule_len = 2, feature_count=8):
    rule_ls = []
    
    sample_count_per_patch = int(sample_count/rule_count)
    all_features = np.random.choice([0,1], size=(sample_count, feature_count), replace=True)
    all_feature_values_in_rules = {}
    all_pos_outcome_array = np.zeros(sample_count)
    all_neg_outcome_array = np.zeros(sample_count)
    sample_id_rule_feature_ids_mappings = dict()

    for rid in range(rule_count):
        feature_ids = np.random.choice(feature_count, size=rule_len, replace=False)
        feature_ids = np.sort(feature_ids)
        feature_values = np.random.choice([0,1], size=rule_len, replace=True)
        for fid in range(len(feature_ids)):
            if feature_ids[fid] in all_feature_values_in_rules:
                feature_values[fid] = all_feature_values_in_rules[feature_ids[fid]]

        start_sample_id = rid*sample_count_per_patch
        end_sample_id = start_sample_id + sample_count_per_patch
        if end_sample_id >= sample_count:
            end_sample_id = sample_count
        all_features[start_sample_id:end_sample_id, feature_ids] = feature_values

        for fid in range(len(feature_ids)):
            all_feature_values_in_rules[feature_ids[fid]] = feature_values[fid]

        for sid in range(start_sample_id, end_sample_id):
            sample_id_rule_feature_ids_mappings[sid] = feature_ids.tolist()

        rule_ls.append((feature_ids.tolist(), feature_values.tolist()))

    for rid in range(rule_count):
        feature_ids, feature_values = rule_ls[rid]
        start_sample_id = rid*sample_count_per_patch
        end_sample_id = start_sample_id + sample_count_per_patch
        if end_sample_id >= sample_count:
            end_sample_id = sample_count

        other_sample_ids = np.array(list(range(start_sample_id)) + list(range(end_sample_id, sample_count)))
        all_features = get_other_satisfiable_samples(all_features, np.array(feature_ids), np.array(feature_values), other_sample_ids, all_feature_values_in_rules, sample_id_rule_feature_ids_mappings)

        # outcome_feat_coeff = np.random.random(len(feature_ids))
        outcome_feat_coeff = np.random.random(feature_count)
        outcome_feat_coeff[feature_ids] = outcome_feat_coeff[feature_ids]*5
        for fid in range(feature_count):
            if fid not in feature_ids:
                outcome_feat_coeff[fid] = 0
        
        treatment_feat_coeff = np.random.random(1)
        pos_outcome_array = np.sum(all_features[start_sample_id:end_sample_id]*outcome_feat_coeff.reshape(1,-1) + 5*treatment_feat_coeff, axis=-1)
        neg_outcome_array = np.sum(all_features[start_sample_id:end_sample_id]*outcome_feat_coeff.reshape(1,-1) - 5*treatment_feat_coeff, axis=-1)
        all_pos_outcome_array[start_sample_id:end_sample_id] = pos_outcome_array
        all_neg_outcome_array[start_sample_id:end_sample_id] = neg_outcome_array


    return rule_ls, all_features, all_feature_values_in_rules, all_pos_outcome_array, all_neg_outcome_array, sample_id_rule_feature_ids_mappings


def generate_data_by_rules2(sample_count, rule_count = 4, rule_len = 2, feature_count=8):
    rule_ls = []
    
    sample_count_per_patch = int(sample_count/rule_count)
    all_features = np.random.choice([0,1], size=(sample_count, feature_count), replace=True)
    all_feature_values_in_rules = {}
    all_pos_outcome_array = np.zeros(sample_count)
    all_neg_outcome_array = np.zeros(sample_count)
    sample_id_rule_feature_ids_mappings = dict()

    all_feature_ids = set(list(range(feature_count)))

    for rid in range(rule_count):
        feature_ids = np.random.choice(list(all_feature_ids), size=rule_len, replace=False)
        feature_ids = np.sort(feature_ids)
        feature_values = np.random.choice([0,1], size=rule_len, replace=True)
        for fid in range(len(feature_ids)):
            if feature_ids[fid] in all_feature_values_in_rules:
                feature_values[fid] = all_feature_values_in_rules[feature_ids[fid]]

        start_sample_id = rid*sample_count_per_patch
        end_sample_id = start_sample_id + sample_count_per_patch
        if end_sample_id >= sample_count:
            end_sample_id = sample_count
        all_features[start_sample_id:end_sample_id, feature_ids] = feature_values

        for fid in range(len(feature_ids)):
            all_feature_values_in_rules[feature_ids[fid]] = feature_values[fid]

        for sid in range(start_sample_id, end_sample_id):
            sample_id_rule_feature_ids_mappings[sid] = feature_ids.tolist()

        rule_ls.append((feature_ids.tolist(), feature_values.tolist()))
        
        all_feature_ids = all_feature_ids.difference(set(feature_ids.tolist()))

    for rid in range(rule_count):
        feature_ids, feature_values = rule_ls[rid]
        start_sample_id = rid*sample_count_per_patch
        end_sample_id = start_sample_id + sample_count_per_patch
        if end_sample_id >= sample_count:
            end_sample_id = sample_count

        other_sample_ids = np.array(list(range(start_sample_id)) + list(range(end_sample_id, sample_count)))
        all_features = get_other_satisfiable_samples(all_features, np.array(feature_ids), np.array(feature_values), other_sample_ids, all_feature_values_in_rules, sample_id_rule_feature_ids_mappings)

        outcome_feat_coeff = np.random.random(len(feature_ids))
        treatment_feat_coeff = np.random.random(1)
        pos_outcome_array = np.sum(all_features[start_sample_id:end_sample_id,np.array(feature_ids)]*outcome_feat_coeff.reshape(1,-1) + 5*treatment_feat_coeff, axis=-1)
        neg_outcome_array = np.sum(all_features[start_sample_id:end_sample_id,np.array(feature_ids)]*outcome_feat_coeff.reshape(1,-1) - 2*treatment_feat_coeff, axis=-1)
        all_pos_outcome_array[start_sample_id:end_sample_id] = pos_outcome_array
        all_neg_outcome_array[start_sample_id:end_sample_id] = neg_outcome_array


    return rule_ls, all_features, all_feature_values_in_rules, all_pos_outcome_array, all_neg_outcome_array, sample_id_rule_feature_ids_mappings

def load_synthetic_data0(data_folder, dataset_id, feature_count=10,sample_count=2000):
    # feat_name_ls = ["bw","b.head","preterm","birth.o","nnhealth","momage","sex","twin","b.marr","mom.lths","mom.hs","mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was"]
    feat_name_ls = ["x_" + str(k) for k in range(0,feature_count)]
    
    col_names = ["treatment", "y_factual", "y_cfactual"] + feat_name_ls#["x_" + str(k) for k in range(1,26)]

    # sub_feat_name_ls, sub_feat_id_ls = sample_sets(feat_name_ls, num_sets=num_sets, samples_per_set=samples_per_set)
    outcome_attr = "y_factual"
    treatment_attr = "treatment"
    count_outcome_attr = "y_cfactual"

    # sub_data_folder = os.path.join(data_folder, str(dataset_id))
    sub_data_folder = data_folder
    os.makedirs(sub_data_folder, exist_ok=True)
    sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "sample_feat_mappings")
    data_file = os.path.join(sub_data_folder, "synthetic_data.csv")
    rule_file = os.path.join(sub_data_folder, "rule_file")
    if os.path.exists(data_file) and os.path.exists(rule_file) and os.path.exists(sample_id_rule_feat_id_mapping_file):
        # with open(meta_data_file, "rb") as f:
        #     meta_data_mappings = pickle.load(f)
        all_df = pd.read_csv(data_file)
        with open(rule_file, "rb") as f:
            rule_ls = pickle.load(f)
        with open(sample_id_rule_feat_id_mapping_file, "rb") as f:
            sample_id_rule_feature_ids_mappings = pickle.load(f)
    else:
        rule_ls, all_features, all_feature_values_in_rules, all_pos_outcome_array, all_neg_outcome_array, sample_id_rule_feature_ids_mappings = generate_data_by_rules(sample_count, feature_count=feature_count)

        treatment_arr = np.random.choice([0,1], size=sample_count, replace=True) 

        y_fact_col = all_pos_outcome_array*treatment_arr + all_neg_outcome_array*(1-treatment_arr)
        y_cfact_col = all_pos_outcome_array*(1-treatment_arr) + all_neg_outcome_array*treatment_arr

        all_data = np.concatenate([treatment_arr.reshape(-1,1), y_fact_col.reshape(-1,1), y_cfact_col.reshape(-1,1), all_features], axis=-1)
        all_df = pd.DataFrame(all_data, columns=col_names)
        
        all_df = all_df.reset_index().rename(columns={"index": "id"})
        # all_df, meta_data_mappings = synthesize_outcome_df(all_df, num_sets, sub_feat_name_ls, sub_feat_id_ls, treatment_attr, outcome_attr, count_outcome_attr, dataset_id)
        all_df.to_csv(data_file)
        with open(rule_file, "wb") as f:
            pickle.dump(rule_ls, f)
        with open(sample_id_rule_feat_id_mapping_file, "wb") as f:
            pickle.dump(sample_id_rule_feature_ids_mappings, f)
        # with open(meta_data_file, "wb") as f:
        #     pickle.dump(meta_data_mappings, f)
    
    random_sample_ids = np.random.permutation(len(all_df))
    train_ids = random_sample_ids[0:int(len(all_df)*0.7)]
    train_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in train_ids}
    # train_meta_data_file = [meta_data_mappings[idx] for idx in train_ids]
    
    valid_ids = random_sample_ids[int(len(all_df)*0.7):int(len(all_df)*0.8)]
    valid_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in valid_ids}
    # valid_meta_data_file = [meta_data_mappings[idx] for idx in valid_ids]
    
    test_ids = random_sample_ids[int(len(all_df)*0.8):]
    test_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in test_ids}
    
    train_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "train_sample_feat_mappings")
    valid_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "valid_sample_feat_mappings")
    test_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "test_sample_feat_mappings")
    
    with open(train_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(train_sample_id_rule_feature_ids_mappings, f)
    
    with open(valid_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(valid_sample_id_rule_feature_ids_mappings, f)
    
    with open(test_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(test_sample_id_rule_feature_ids_mappings, f)
    
    # test_meta_data_file = [meta_data_mappings[idx] for idx in test_ids]
    # with open(os.path.join(sub_data_folder, "train_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(train_meta_data_file, f)
        
    # with open(os.path.join(sub_data_folder, "valid_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(valid_meta_data_file, f)
        
    # with open(os.path.join(sub_data_folder, "test_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(test_meta_data_file, f)
    
    
    train_data = all_df.iloc[train_ids]
    valid_data = all_df.iloc[valid_ids]
    test_data = all_df.iloc[test_ids]
    # valid_data = pd.concat(valid_data_ls)
    # valid_data.drop(columns=["mu0", "mu1"], inplace=True)
    # valid_data = valid_data.reset_index().rename(columns={"index": "id"})
    # test_data = pd.concat(test_data_ls)
    # test_data.drop(columns=["mu0", "mu1"], inplace=True)
    # test_data = test_data.reset_index().rename(columns={"index": "id"})
    # all_df = pd.concat([train_data, valid_data, test_data])
    return train_data, valid_data, test_data, all_df

def load_synthetic_data0_2(data_folder, dataset_id, feature_count=12,sample_count=2000):
    # feat_name_ls = ["bw","b.head","preterm","birth.o","nnhealth","momage","sex","twin","b.marr","mom.lths","mom.hs","mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was"]
    feat_name_ls = ["x_" + str(k) for k in range(0,feature_count)]
    
    col_names = ["treatment", "y_factual", "y_cfactual"] + feat_name_ls#["x_" + str(k) for k in range(1,26)]

    # sub_feat_name_ls, sub_feat_id_ls = sample_sets(feat_name_ls, num_sets=num_sets, samples_per_set=samples_per_set)
    outcome_attr = "y_factual"
    treatment_attr = "treatment"
    count_outcome_attr = "y_cfactual"

    # sub_data_folder = os.path.join(data_folder, str(dataset_id))
    sub_data_folder = data_folder
    os.makedirs(sub_data_folder, exist_ok=True)
    sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "sample_feat_mappings")
    data_file = os.path.join(sub_data_folder, "synthetic_data.csv")
    rule_file = os.path.join(sub_data_folder, "rule_file")
    if os.path.exists(data_file) and os.path.exists(rule_file) and os.path.exists(sample_id_rule_feat_id_mapping_file):
        # with open(meta_data_file, "rb") as f:
        #     meta_data_mappings = pickle.load(f)
        all_df = pd.read_csv(data_file)
        with open(rule_file, "rb") as f:
            rule_ls = pickle.load(f)
        with open(sample_id_rule_feat_id_mapping_file, "rb") as f:
            sample_id_rule_feature_ids_mappings = pickle.load(f)
    else:
        rule_ls, all_features, all_feature_values_in_rules, all_pos_outcome_array, all_neg_outcome_array, sample_id_rule_feature_ids_mappings = generate_data_by_rules2(sample_count, feature_count=feature_count)

        treatment_arr = np.random.choice([0,1], size=sample_count, replace=True) 

        y_fact_col = all_pos_outcome_array*treatment_arr + all_neg_outcome_array*(1-treatment_arr)
        y_cfact_col = all_pos_outcome_array*(1-treatment_arr) + all_neg_outcome_array*treatment_arr

        all_data = np.concatenate([treatment_arr.reshape(-1,1), y_fact_col.reshape(-1,1), y_cfact_col.reshape(-1,1), all_features], axis=-1)
        all_df = pd.DataFrame(all_data, columns=col_names)
        
        all_df = all_df.reset_index().rename(columns={"index": "id"})
        # all_df, meta_data_mappings = synthesize_outcome_df(all_df, num_sets, sub_feat_name_ls, sub_feat_id_ls, treatment_attr, outcome_attr, count_outcome_attr, dataset_id)
        all_df.to_csv(data_file)
        with open(rule_file, "wb") as f:
            pickle.dump(rule_ls, f)
        with open(sample_id_rule_feat_id_mapping_file, "wb") as f:
            pickle.dump(sample_id_rule_feature_ids_mappings, f)
        # with open(meta_data_file, "wb") as f:
        #     pickle.dump(meta_data_mappings, f)
    
    random_sample_ids = np.random.permutation(len(all_df))
    train_ids = random_sample_ids[0:int(len(all_df)*0.7)]
    train_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in train_ids}
    # train_meta_data_file = [meta_data_mappings[idx] for idx in train_ids]
    
    valid_ids = random_sample_ids[int(len(all_df)*0.7):int(len(all_df)*0.8)]
    valid_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in valid_ids}
    # valid_meta_data_file = [meta_data_mappings[idx] for idx in valid_ids]
    
    test_ids = random_sample_ids[int(len(all_df)*0.8):]
    test_sample_id_rule_feature_ids_mappings = {k: sample_id_rule_feature_ids_mappings[k] for k in test_ids}
    
    train_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "train_sample_feat_mappings")
    valid_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "valid_sample_feat_mappings")
    test_sample_id_rule_feat_id_mapping_file = os.path.join(sub_data_folder, "test_sample_feat_mappings")
    
    with open(train_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(train_sample_id_rule_feature_ids_mappings, f)
    
    with open(valid_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(valid_sample_id_rule_feature_ids_mappings, f)
    
    with open(test_sample_id_rule_feat_id_mapping_file, "wb") as f:
        pickle.dump(test_sample_id_rule_feature_ids_mappings, f)
    
    # test_meta_data_file = [meta_data_mappings[idx] for idx in test_ids]
    # with open(os.path.join(sub_data_folder, "train_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(train_meta_data_file, f)
        
    # with open(os.path.join(sub_data_folder, "valid_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(valid_meta_data_file, f)
        
    # with open(os.path.join(sub_data_folder, "test_synthetic_meta_info"), "wb") as f:
    #     pickle.dump(test_meta_data_file, f)
    
    
    train_data = all_df.iloc[train_ids]
    valid_data = all_df.iloc[valid_ids]
    test_data = all_df.iloc[test_ids]
    # valid_data = pd.concat(valid_data_ls)
    # valid_data.drop(columns=["mu0", "mu1"], inplace=True)
    # valid_data = valid_data.reset_index().rename(columns={"index": "id"})
    # test_data = pd.concat(test_data_ls)
    # test_data.drop(columns=["mu0", "mu1"], inplace=True)
    # test_data = test_data.reset_index().rename(columns={"index": "id"})
    # all_df = pd.concat([train_data, valid_data, test_data])
    return train_data, valid_data, test_data, all_df


# def load_ihdp_dataset(trial=1):
#     # covariates = pd.read_csv(os.path.join(data_folder, "x.csv"))
#     r = DataLoader.get_loader('IHDP').load()
#     X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r

#     selected_trial_ids = np.random.choice(X_tr.shape[-1], trial, replace=False)
    
#     selected_X_tr = np.transpose(X_tr[:,:,selected_trial_ids], (0,2,1)).reshape(-1, X_tr.shape[1])
#     selected_T_tr = T_tr[:,selected_trial_ids].reshape(-1, 1)
#     selected_YF_tr = YF_tr[:,selected_trial_ids].reshape(-1, 1)
#     selected_YCF_tr = YCF_tr[:,selected_trial_ids].reshape(-1, 1)
#     selected_mu_0_tr = mu_0_tr[:,selected_trial_ids].reshape(-1, 1)
#     selected_mu_1_tr = mu_1_tr[:,selected_trial_ids].reshape(-1, 1)
#     selected_X_te = np.transpose(X_te[:,:,selected_trial_ids], (0,2,1)).reshape(-1, X_te.shape[1])
#     selected_T_te = T_te[:,selected_trial_ids].reshape(-1, 1)
#     selected_YF_te = YF_te[:,selected_trial_ids].reshape(-1, 1)
#     selected_YCF_te = YCF_te[:,selected_trial_ids].reshape(-1, 1)
#     selected_mu_0_te = mu_0_te[:,selected_trial_ids].reshape(-1, 1)
#     selected_mu_1_te = mu_1_te[:,selected_trial_ids].reshape(-1, 1)
    
#     train_data = np.concatenate([selected_X_tr, selected_T_tr, selected_YF_tr, selected_YCF_tr], axis=-1)
#     test_data = np.concatenate([selected_X_te, selected_T_te, selected_YF_te, selected_YCF_te], axis=-1)
#     feature_name_ls = ["x_" + str(k) for k in range(selected_X_tr.shape[-1])] + ["Treatment", "outcome", "counter_outcome"]
    
    
#     train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
#     train_df = pd.DataFrame(train_data, columns=feature_name_ls)
#     valid_df = pd.DataFrame(valid_data, columns=feature_name_ls)
#     test_df = pd.DataFrame(test_data, columns=feature_name_ls)
#     train_df = train_df.reset_index().rename(columns={"index": "id"})
#     valid_df = valid_df.reset_index().rename(columns={"index": "id"})
#     test_df = test_df.reset_index().rename(columns={"index": "id"})
#     all_df = pd.concat([train_df, valid_df, test_df])

    
#     return train_df, valid_df, test_df, all_df

    
class tabular_Dataset(Dataset):
    def __init__(self, data, drop_cols, lang, id_attr, outcome_attr, treatment_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, normalize_y=True, y_scaler=None, count_outcome_attr=None, dose_attr=None, treatment_graph=None):
        self.data = data
        self.drop_cols = drop_cols
        self.id_attr, self.outcome_attr, self.treatment_attr, self.count_outcome_attr, self.dose_attr = id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr
        self.data.index = list(range(len(self.data)))
        self.patient_ids = self.data[id_attr].unique().tolist()
        self.lang = lang
        self.other_data = other_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.other_data is not None:
            self.other_data = self.other_data.to(self.device)
        self.cat_cols = list(set(self.lang.CAT_FEATS))
        self.cat_cols.sort()
        self.num_cols = [col for col in data.columns if col not in self.lang.CAT_FEATS and not col == id_attr and not col in drop_cols and not col == outcome_attr and not col == treatment_attr and not col == outcome_attr and not col == count_outcome_attr and not col == dose_attr] 
        if cat_unique_count_mappings is not None and cat_unique_vals_id_mappings is not None and cat_id_unique_vals_mappings is not None:
            self.cat_unique_count_mappings = cat_unique_count_mappings
            self.cat_unique_vals_id_mappings = cat_unique_vals_id_mappings
            self.cat_id_unique_vals_mappings = cat_id_unique_vals_mappings
        else:
            self.cat_unique_count_mappings = {}
            self.cat_unique_vals_id_mappings = {}
            self.cat_id_unique_vals_mappings = {}
            self.create_cat_feat_mappings()
        self.cat_unique_val_count_ls = [self.cat_unique_count_mappings[col] for col in self.cat_cols]
        
        self.cat_sum_count = sum([len(self.cat_unique_vals_id_mappings[feat]) for feat in self.cat_unique_vals_id_mappings])
        self.init_onehot_mats()
        if normalize_y:
            if y_scaler is None:
                y_scaler = StandardScaler().fit(self.data[outcome_attr].to_numpy().reshape(-1,1))
                y = y_scaler.transform(self.data[outcome_attr].to_numpy().reshape(-1,1))
                self.data[outcome_attr] = y
                self.y_scaler = y_scaler
            else:
                y = y_scaler.transform(self.data[outcome_attr].to_numpy().reshape(-1,1))
                self.data[outcome_attr] = y
                self.y_scaler = y_scaler
            if count_outcome_attr is not None:
                self.data[count_outcome_attr] = y_scaler.transform(self.data[count_outcome_attr].to_numpy().reshape(-1,1))
        else:
            self.y_scaler = None
        self.treatment_data=  None
        if treatment_graph is not None:
            self.treatment_data = treatment_graph[0]
            self.id_to_graph_dict = treatment_graph[1]
        
        # self.convert_text_to_tokens()
        # if balance:
        #     most = self.data['label'].value_counts().max()
        #     for label in list(self.data['label'].value_counts().index):
        #         if type(label) is not list:
        #             match  = self.data.loc[self.data['label'] == label]['PAT_ID'].to_list()
        #         else:
        #             match = self.data.loc[np.sum(np.array(list(self.data['label'])) == np.array(label), axis=-1) == len(self.data['label'][0])]['PAT_ID'].to_list()
        #         samples = [random.choice(match) for _ in range(most-len(match))]
        #         self.patient_ids.extend(samples)
                
        # random.shuffle(self.patient_ids)
    
    # def convert_text_to_tokens(self, tokenizer=None):
    #     if tokenizer is None:
    #         tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    #     self.encoded_text_dict = dict()        
    #     # for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
    #     for idx in tqdm(range(len(self.data))):
    #         appts = self.data.loc[self.data[self.id_attr] == self.patient_ids[idx]]
            
    #         encoded_sent = tokenizer.encode_plus(appts[self.text_attr].tolist()[0], add_special_tokens=True,
    #                                             max_length=128,
    #                                             truncation=True,
    #                                             pad_to_max_length=True)

    #         self.encoded_text_dict[self.patient_ids[idx]] = [torch.tensor(encoded_sent['input_ids']), torch.tensor(encoded_sent['attention_mask']), sum(encoded_sent['attention_mask'])]

    
    def init_onehot_mats(self):
        self.feat_onehot_mat_mappings = dict()
        for cat_feat in self.cat_unique_count_mappings:            
            self.feat_onehot_mat_mappings[cat_feat] = torch.eye(self.cat_unique_count_mappings[cat_feat])

    def create_cat_feat_mappings(self):
        for cat_feat in self.lang.CAT_FEATS:
            unique_vals = list(self.data[cat_feat].unique())
            self.cat_unique_count_mappings[cat_feat] = len(unique_vals)
            self.cat_unique_vals_id_mappings[cat_feat] = dict()
            self.cat_id_unique_vals_mappings[cat_feat] = dict()
            unique_vals.sort()
            for val_idx, unique_val in enumerate(unique_vals):
                self.cat_id_unique_vals_mappings[cat_feat][val_idx] = unique_val
                self.cat_unique_vals_id_mappings[cat_feat][unique_val] = val_idx

    # def init_treatment_graph(self):
    #     self.graph_ls = []
    #     for idx in range(len(self.treatment_graph)):
    #         graph = nx.Graph()
    #         node_features = self.treatment_graph[idx]["node_features"]
    #         for node_idx in range(len(node_features)):
    #             graph.add_node(node_idx)
    #             graph.nodes[node_idx]["features"] = node_features[node_idx]
            
    #         graph.add_edges_from(self.treatment_graph[idx]["edges"])
    #         self.graph_ls.append(graph)

    def convert_cat_vals_to_onehot(self, X_cat_tar):
        X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat_tar[:, idx].type(torch.long)] for idx in range(len(self.cat_cols))]
            
        X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1)        
        return X_cat_onehot

    def convert_feats_to_transformed_feats(self, X):
        X_num_tar = X[:,0:len(self.num_cols)]
        if len(self.cat_cols) > 0:
            X_cat_tar = X[:,len(self.num_cols):]
            X_cat_onehot = self.convert_cat_vals_to_onehot(X_cat_tar)
            return torch.cat([X_num_tar, X_cat_onehot], dim=-1)
        else:
            return X_num_tar
    
    def rescale_back(self, X_num_tar):
        origin_X_num_ls = []
        for idx in range(len(self.num_cols)):
            col = self.num_cols[idx]
            col_min, col_max = self.feat_range_mappings[col]
            origin_X_num_ls.append(X_num_tar[:,idx]*(col_max - col_min) + col_min)
        return torch.stack(origin_X_num_ls, dim=1)

    def init_data(self):
        # feat_onehot_mat_mappings = [item[0][6] for item in data][0]
        X_num_tar = torch.from_numpy(self.r_data[self.num_cols].to_numpy()).type(torch.float)
        origin_X_num_tar = torch.from_numpy(self.data[self.num_cols].to_numpy()).type(torch.float)

        if len(self.cat_cols) > 0:
            X_cat_tar = torch.from_numpy(self.r_data[self.cat_cols].to_numpy()).type(torch.float)
            origin_X_cat_tar = torch.from_numpy(self.data[self.cat_cols].to_numpy()).type(torch.float)
            
            X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat_tar[:, idx].type(torch.long)] for idx in range(len(self.cat_cols))]
            
            X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1)        
            self.transformed_features = torch.cat([X_num_tar, X_cat_onehot], dim=-1)
            self.features = torch.cat([X_num_tar, X_cat_tar], dim=-1)
            self.origin_features = torch.cat([origin_X_num_tar, origin_X_cat_tar], dim=-1)
        else:
            self.transformed_features = X_num_tar
            self.features = X_num_tar
            self.origin_features = origin_X_num_tar
        if self.treatment_attr is not None:
            self.treatment_array = torch.from_numpy(self.r_data[self.treatment_attr].to_numpy()).type(torch.float)
        # if self.treatment_graph is not None:
        #     self.init_treatment_graph()

        
        self.outcome_array = torch.from_numpy(self.r_data[self.outcome_attr].to_numpy()).type(torch.float)
        if self.count_outcome_attr is not None:
            self.count_outcome_array = torch.from_numpy(self.r_data[self.count_outcome_attr].to_numpy()).type(torch.float)
        else:
            self.count_outcome_array = None
        
        if self.dose_attr is not None:
            self.dose_array = torch.from_numpy(self.r_data[self.dose_attr].to_numpy()).type(torch.float)
        else:
            self.dose_array = None
        
        self.imputed_features = torch.clone(self.features)

    def rescale_data(self, feat_range_mappings):
        self.r_data = self.data.copy()
        self.feat_range_mappings = feat_range_mappings
        if feat_range_mappings is not None:
            # for feat in feat_range_mappings:
            for feat in self.num_cols:
                if feat == "label":
                    continue
                # if not feat == 'PAT_ID' and (not feat in self.lang.CAT_FEATS):
                # if feat in self.num_cols:
                lb, ub = feat_range_mappings[feat][0], feat_range_mappings[feat][1]
                if lb < ub:
                    self.r_data[feat] = (self.data[feat]-lb)/(ub-lb)
                else:
                    self.r_data[feat] = 0
            for feat in self.cat_cols:
                # else:
                    # if feat in self.cat_cols:
                    self.r_data[feat] = self.data[feat]
                    for unique_val in self.cat_unique_vals_id_mappings[feat]:
                        self.r_data.loc[self.r_data[feat] == unique_val, feat] = self.cat_unique_vals_id_mappings[feat][unique_val]
                            
        self.init_data()

    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        # appts = self.r_data.loc[self.r_data[self.id_attr] == self.patient_ids[idx]]
        appts2 = self.imputed_features[idx]
        trans_appts2 = self.transformed_features[idx]
        treatment2 = self.treatment_array[idx]

        outcome2 = self.outcome_array[idx]
        if self.count_outcome_array is not None:
            count_outcome2 = self.count_outcome_array[idx]
        else:
            count_outcome2 = None
        
        if self.dose_array is not None:
            dose_val2 = self.dose_array[idx]
        else:
            dose_val2 = None
        if self.other_data is None:  
            all_other_pats2 = torch.ones(len(self.features)).bool()  #torch.cat([self.features[0:idx], self.features[idx+1:]], dim=0)
            all_other_pats2[idx] = False
        else:
            all_other_pats2 = torch.ones(len(self.other_data)).bool()
        
        if self.treatment_data is not None:
            graph_data = self.treatment_data[idx]
            graph2 = self.id_to_graph_dict[treatment2.long().item()]
            graph_data.covariates = appts2
        else:
            graph2 = None
        # if self.other_data is None:
        #     all_other_pats = self.r_data.loc[self.r_data[self.id_attr] != self.patient_ids[idx]]
        # else:
        #     all_other_pats = self.other_data
        
        # # full_pats = self.r_data#.loc[self.r_data['PAT_ID'] != self.patient_ids[idx]]
        # # text_cln = appts[self.text_attr].tolist()[0]
            
        # # text_id, text_mask, text_len = self.encoded_text_dict[self.patient_ids[idx]]
        # # m = [appts[self.outcome_attr].max()]
        # y = torch.tensor(list(appts[self.outcome_attr]), dtype=torch.float)
        # if self.count_outcome_attr is not None:
        #     count_y = torch.tensor(list(appts[self.count_outcome_attr]), dtype=torch.float)
        # else:
        #     count_y = None
        
        # if self.dose_attr is not None:
        #     dose_val = torch.tensor(list(appts[self.dose_attr]), dtype=torch.float)
        # else:
        #     dose_val = None
        # treatment = torch.tensor(list(appts[self.treatment_attr]), dtype=torch.float)
        # X_pd = appts.drop(self.drop_cols, axis=1)
        
        # X_num = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.num_cols].to_numpy(dtype=np.float64)][0]
        # X_cat = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.cat_cols].to_numpy(dtype=np.float64)][0].type(torch.long)
        
        # X_num = torch.from_numpy(X_pd[self.num_cols].to_numpy()).type(torch.float)
        # X_cat = torch.from_numpy(X_pd[self.cat_cols].to_numpy()).type(torch.long)
        
        # X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat.view(-1)[idx]] for idx in range(len(self.cat_cols))]
        
        # X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1).unsqueeze(0)
        
        # X = [torch.cat([X_num, X_cat_onehot], dim=-1)]
        #zero pad
        # X.extend([torch.tensor([0]*len(X[0]), dtype=torch.float) ]*(len(X)-self.patient_max_appts))
        # return (idx, self.patient_ids[idx], all_other_pats, appts, self.num_cols, self.cat_cols, self.feat_onehot_mat_mappings), y, treatment, count_y, dose_val, (appts2, treatment2, outcome2, count_outcome2, dose_val2, trans_appts2, all_other_pats2)
        if graph2 is None:
            return idx, appts2, treatment2, outcome2, count_outcome2, dose_val2, trans_appts2, all_other_pats2
        else:
            return idx, appts2, treatment2, outcome2, count_outcome2, dose_val2, trans_appts2, all_other_pats2, graph_data, graph2
    
    def create_imputed_data(self):
        self.imputed_features = torch.clone(self.features)

        if torch.sum(torch.isnan(self.imputed_features)) > 0:
            self.transform_imputed_data()
    
    def transform_imputed_data(self):
        start_cln_idx = len(self.num_cols)
        self.imputed_features[:,0:start_cln_idx] = self.transformed_features[:, 0:start_cln_idx]

        if len(self.cat_cols) > 0:
            imputed_cat_feat_ls = []
            for idx in range(len(self.cat_cols)):
                curr_cat_unique_count = self.cat_unique_count_mappings[self.cat_cols[idx]]
                imputed_cat_feat_ls.append(torch.argmax(self.transformed_features[:,start_cln_idx:start_cln_idx + curr_cat_unique_count], dim=1))
                start_cln_idx = start_cln_idx + curr_cat_unique_count

            self.imputed_features[:,len(self.num_cols):] = torch.stack(imputed_cat_feat_ls, dim=1)

    @staticmethod
    def collate_fn(data):
        
        # X_sample_ids = [item[0][0] for item in data]
        # patient_id_ls = [item[0][1] for item in data]
        # all_other_pats_ls = [item[0][2] for item in data]
        
        # X_pd_ls = [item[0][3] for item in data]
        # X_pd_array = pd.concat(X_pd_ls)
        # # text_id_ls = torch.stack([item[0][4] for item in data])
        # # text_mask_ls = torch.stack([item[0][5] for item in data])
        # # text_len_ls = torch.tensor([item[0][6] for item in data])
        
        # # X_ls = [item[0][2][0].view(1,-1) for item in data]
        
        # # X_num_tensor = torch.cat([item[0][4][0] for item in data])
        # # X_cat_tensor = torch.cat([item[0][4][1] for item in data])
        
        # # # full_data = [item[0][6] for item in data][0]
        # num_cols = [item[0][4] for item in data][0]
        # cat_cols = [item[0][5] for item in data][0]
        # X_num_tar = torch.from_numpy(X_pd_array[num_cols].to_numpy()).type(torch.float)

        # if len(cat_cols) > 0:
        #     feat_onehot_mat_mappings = [item[0][6] for item in data][0]

        #     X_cat_tar = torch.from_numpy(X_pd_array[cat_cols].to_numpy()).type(torch.float)
            
        #     X_cat_onehot_ls = [feat_onehot_mat_mappings[cat_cols[idx]][X_cat_tar[:, idx].type(torch.long)] for idx in range(len(cat_cols))]
            
        #     X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1)
            
        #     X = torch.cat([X_num_tar, X_cat_onehot], dim=-1)
        # else:
        #     X = X_num_tar
        # # X_tensor = torch.cat(X_ls)
        # # assert torch.norm(X_tensor - X) <= 0
        # y_ls = [item[1].view(1,-1) for item in data]
        # y_tensor = torch.cat(y_ls)
        # treatment_ls = [item[2].view(1,-1) for item in data]
        # treatment_tensor = torch.cat(treatment_ls)
        
        
        
        # if data[0][3] is not None:
        #     count_y_ls = [item[3].view(1,-1) for item in data]
        #     count_y_tensor = torch.cat(count_y_ls)
        # else:
        #     count_y_tensor = None
        
        
        # if data[0][4] is not None:
        #     dose_val_ls = [item[4].view(1,-1) for item in data]
        #     dose_tensor = torch.cat(dose_val_ls)
        # else:
        #     dose_tensor = None
        
        # X_sample_ids_tensor = torch.tensor(X_sample_ids)
        # patient_id_tensor = torch.tensor(patient_id_ls)
        unzipped_data_ls = list(zip(*data))

        ids_ls = unzipped_data_ls[0]
        appts2_ls = unzipped_data_ls[1]
        treatment2_ls = unzipped_data_ls[2] 
        outcome2_ls = unzipped_data_ls[3]
        count_outcome2_ls = unzipped_data_ls[4]
        dose_val2_ls = unzipped_data_ls[5]
        trans_appts2_ls = unzipped_data_ls[6]
        all_other_pats2_ls = unzipped_data_ls[7]
        if len(unzipped_data_ls) > 8:
            graph_data_ls = unzipped_data_ls[8]
            units = torch.stack([graph_data_ls[idx].covariates.view(-1) for idx in range(len(graph_data_ls))])
            graph2_ls = unzipped_data_ls[9]
            graph_batch = create_pt_geometric_dataset(
                    units=units, treatment_graphs=graph2_ls, outcomes=torch.tensor(outcome2_ls)
                )
            graph_batch = Batch.from_data_list(graph_batch)

            
            
        appts2 = torch.stack(appts2_ls)
        if treatment2_ls[0] is not None:
            treatment2=torch.tensor(treatment2_ls).view(-1,1)
        else:
            treatment2=None
        
        outcome2= torch.tensor(outcome2_ls).view(-1,1) 
        if data[0][4] is not None:
            count_outcome2 = torch.tensor(count_outcome2_ls).view(-1,1)  
        else:
            count_outcome2= None
        if data[0][5] is not None:
            dose_val2 = torch.tensor(dose_val2_ls).view(-1,1)
        else:
            dose_val2 = None
        trans_appts2 = torch.stack(trans_appts2_ls)  
        all_other_pats2 = all_other_pats2_ls
        ids = torch.tensor(ids_ls)
        if len(unzipped_data_ls) > 8:
            return ids, appts2, treatment2, outcome2, count_outcome2, dose_val2, trans_appts2, all_other_pats2, graph_batch
        else:
            return ids, appts2, treatment2, outcome2, count_outcome2, dose_val2, trans_appts2, all_other_pats2
        
        # return (X_sample_ids_tensor, patient_id_tensor, all_other_pats_ls, X_pd_ls, X), y_tensor, treatment_tensor, count_y_tensor, dose_tensor

def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.data.columns)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

# def rescale_outcome(train_dataset, y_scaler, outcome_attr, count_outcome_attr):
#     train_dataset.r_data[outcome_attr] =  y_scaler.transform(np.array(train_dataset.r_data[outcome_attr]).reshape(-1,1))
#     train_dataset.r_data[count_outcome_attr] =  y_scaler.transform(np.array(train_dataset.r_data[count_outcome_attr]).reshape(-1,1))
#     train_dataset.y_scaler = y_scaler

def create_dataset(dataset_name, all_df, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, drop_cols, count_outcome_attr=None, dose_attr = None, normalize_y = True, extra_info=None):
    if all_df is not None:
        all_dataset = tabular_Dataset(all_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, normalize_y=normalize_y)
        feat_range_mappings = obtain_feat_range_mappings(all_dataset)   
    else:
        feat_range_mappings = None
    
    # outcome_arr = list(all_dataset.data[outcome_attr])
    # y_scaler = StandardScaler().fit(np.array(outcome_arr).reshape(-1,1))

    y_scaler = all_dataset.y_scaler if all_df is not None else None
    # if normalize_y and all_df is not None:
    #     if count_outcome_attr is not None:
    #         outcome_arr = list(all_dataset.data[outcome_attr]) + list(all_dataset.data[count_outcome_attr])
    #     else:
    #         outcome_arr = list(all_dataset.data[outcome_attr])
        # y_scaler = StandardScaler().fit(np.array(outcome_arr).reshape(-1,1))
    
    train_dataset = tabular_Dataset(train_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, y_scaler=y_scaler, normalize_y=normalize_y)
    train_dataset.rescale_data(feat_range_mappings)
    # feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    valid_dataset = tabular_Dataset(valid_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, cat_unique_count_mappings=train_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=train_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=train_dataset.cat_id_unique_vals_mappings, other_data=train_dataset.features, y_scaler=y_scaler, normalize_y=normalize_y)
    test_dataset = tabular_Dataset(test_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, cat_unique_count_mappings=train_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=train_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=train_dataset.cat_id_unique_vals_mappings, other_data=train_dataset.features, y_scaler=y_scaler, normalize_y=normalize_y)
    # else:
    #     train_dataset.rescale_data(feat_range_mappings)
    #     # feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    #     # test_dataset = tabular_Dataset(test_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, cat_unique_count_mappings=train_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=train_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=train_dataset.cat_id_unique_vals_mappings, other_data=train_dataset.features, y_scaler=y_scaler, normalize_y=normalize_y, treatment_graph=extra_info[2])
    #     test_dataset = None
    
    
    
    valid_dataset.rescale_data(feat_range_mappings)
    if test_dataset is not None:
        test_dataset.rescale_data(feat_range_mappings)
        if "cont" in dataset_name:
            valid_dataset.t_grid = extra_info[0]
            test_dataset.t_grid = extra_info[1]
    
    if dataset_name == "tcga":
        test_dataset.metainfo = extra_info
        train_dataset.metainfo = extra_info

    # rescale_outcome(train_dataset, y_scaler, outcome_attr, count_outcome_attr)
    # rescale_outcome(valid_dataset, y_scaler, outcome_attr, count_outcome_attr)
    # rescale_outcome(test_dataset, y_scaler, outcome_attr, count_outcome_attr)
    
    return train_dataset, valid_dataset, test_dataset, feat_range_mappings
    