from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset
import os, sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict



from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from treatment_prediction.create_language import Language
import synthetic_lang
from simulation import run_simulation
from featurization import convert_text_to_features, transform_text_to_tokens
from treatment_prediction.utils_treatment import set_lang_data


def calculate_silhouette_score(data, labels, dist_func=torch.cdist):
    # Calculate pairwise distances between data points
    distances = dist_func(data, data)

    # Calculate the mean distance for each sample to other samples in its cluster
    a = torch.tensor([distances[i, labels == labels[i]].mean() for i in range(len(data))])

    # Calculate the mean distance for each sample to samples in other clusters
    b = torch.tensor([min(distances[i, labels != labels[i], labels == cluster].mean()
                          for cluster in torch.unique(labels)) for i in range(len(data))])

    # Calculate silhouette score
    silhouette_scores = (b - a) / torch.max(a, b)

    return silhouette_scores.mean()

def kmeans_plusplus_init(data, k):
    device = data.device
    
    # Choose the first centroid randomly from the data points
    initial_indices = torch.randint(0, len(data), (1,), device=device)
    centroids = data[initial_indices]
    
    while centroids.shape[0] < k:
        # Calculate the squared distance from each point to its nearest centroid
        distances = torch.min(torch.norm(data[:, None, :] - centroids, dim=-1)**2, dim=1).values
        
        # Choose the next centroid from the data points with probability proportional to squared distance
        probabilities = distances / distances.sum()
        next_centroid_index = np.random.choice(len(data), p=probabilities.cpu().numpy())
        next_centroid = data[next_centroid_index]
        
        centroids = torch.cat((centroids, next_centroid.unsqueeze(0)), dim=0)
    
    return centroids

def k_means_pytorch(data, k, dist_func=torch.cdist, max_iters=1000):
    # Initialize centroids randomly from the data points
    # centroids = data[torch.randperm(data.size(0))[:k]]
    centroids = kmeans_plusplus_init(data, k)

    for _ in tqdm(range(max_iters)):
        # Calculate distances between data points and centroids
        distances = dist_func(data, centroids)

        # Assign each data point to the nearest centroid
        _, labels = torch.min(distances, dim=1)

        # Update centroids based on the mean of assigned data points
        new_centroids = torch.stack([data[labels == i].mean(0) for i in range(k)])
        print(torch.norm(new_centroids-centroids))
        # Check for convergence
        if torch.norm(new_centroids-centroids) < 1e-4:
            break

        centroids = new_centroids

    return centroids, labels, distances

def split_train_valid_test_df(df, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2, split_ids=None):
    if split_ids is not None:
        train_ids, valid_ids, test_ids = split_ids
        train_df = df.iloc[train_ids]
        valid_df = df.iloc[valid_ids]
        test_df = df.iloc[test_ids]    
        return train_df,  valid_df, test_df
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
    

    
class NLP_Dataset(Dataset):
    def __init__(self, dataset_name, tokenizer, data, drop_cols, lang, id_attr, outcome_attr, treatment_attr, text_attr, count_outcome_attr=None, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, y_scaler=None, classification=False):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.data = data
        self.drop_cols = drop_cols
        self.id_attr, self.outcome_attr, self.treatment_attr = id_attr, outcome_attr, treatment_attr
        self.text_attr = text_attr
        self.data.index = list(range(len(self.data)))
        self.patient_ids = self.data[id_attr].unique().tolist()
        self.lang = lang
        self.other_data = other_data
        self.cat_cols = list(set(self.lang.CAT_FEATS))
        self.cat_cols.sort()
        self.num_cols = [col for col in data.columns if col not in self.lang.CAT_FEATS and not col == id_attr and not col in drop_cols and not col == outcome_attr and not col == treatment_attr and not col == outcome_attr and not col == text_attr and not col == count_outcome_attr] 
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
        self.text_ls = self.data[self.text_attr].tolist()
        self.convert_text_to_tokens()
        self.count_outcome_attr=count_outcome_attr
        if not classification:
            if y_scaler is None:
                if self.count_outcome_attr is not None:
                    all_outcome_arr = np.concatenate([self.data[self.outcome_attr].to_numpy().reshape(-1,1), self.data[self.count_outcome_attr].to_numpy().reshape(-1,1)], axis=0)
                else:
                    all_outcome_arr = self.data[self.outcome_attr].to_numpy().reshape(-1,1)

                y_scaler = StandardScaler().fit(all_outcome_arr.reshape(-1,1))
                y = y_scaler.transform(self.data[outcome_attr].to_numpy().reshape(-1,1))
                self.data[outcome_attr] = y
                self.y_scaler = y_scaler

                if self.count_outcome_attr is not None:
                    y = y_scaler.transform(self.data[self.count_outcome_attr].to_numpy().reshape(-1,1))
                    self.data[self.count_outcome_attr] = y
            else:
                y = y_scaler.transform(self.data[outcome_attr].to_numpy().reshape(-1,1))
                self.data[outcome_attr] = y

                if self.count_outcome_attr is not None:
                    y = y_scaler.transform(self.data[self.count_outcome_attr].to_numpy().reshape(-1,1))
                    self.data[self.count_outcome_attr] = y
                
                self.y_scaler = y_scaler
        else:
            self.y_scaler = None
        
        self.classification = classification            
        self.dose_array = None

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
            self.origin_features =origin_X_num_tar
        if self.treatment_attr is not None:
            self.treatment_array = torch.from_numpy(self.r_data[self.treatment_attr].to_numpy()).type(torch.float)
        # if self.treatment_graph is not None:
        #     self.init_treatment_graph()

        
        self.outcome_array = torch.from_numpy(self.r_data[self.outcome_attr].to_numpy()).type(torch.float)
        if self.count_outcome_attr is not None:
            self.count_outcome_array = torch.from_numpy(self.r_data[self.count_outcome_attr].to_numpy()).type(torch.float)
        else:
            self.count_outcome_array = None

    def convert_text_to_tokens(self):
        # if tokenizer is None:
        #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.encoded_text_dict = dict()        
        # for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
        for idx in tqdm(range(len(self.data))):
            # appts = self.data.loc[self.data[self.id_attr] == self.patient_ids[idx]]
            curr_text = self.text_ls[idx]
            text_ids, text_mask, text_len = transform_text_to_tokens(self.tokenizer, curr_text, self.dataset_name)
            # encoded_sent = tokenizer.encode_plus(curr_text, add_special_tokens=True,
            #                                     max_length=128,
            #                                     truncation=True,
            #                                     pad_to_max_length=True)

            # self.encoded_text_dict[self.patient_ids[idx]] = [torch.tensor(encoded_sent['input_ids']), torch.tensor(encoded_sent['attention_mask']), sum(encoded_sent['attention_mask'])]
            self.encoded_text_dict[idx] = [text_ids, text_mask, text_len]
            
    
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

    def rescale_data(self, feat_range_mappings):
        self.r_data = self.data.copy()
        self.feat_range_mappings = feat_range_mappings
        for feat in feat_range_mappings:
            if feat == "label":
                continue
            # if not feat == 'PAT_ID' and (not feat in self.lang.CAT_FEATS):
            if feat in self.num_cols:
                lb, ub = feat_range_mappings[feat][0], feat_range_mappings[feat][1]
                if lb < ub:
                    self.r_data[feat] = (self.data[feat]-lb)/(ub-lb)
                else:
                    self.r_data[feat] = 0
            else:
                if feat in self.cat_cols:
                    self.r_data[feat] = self.data[feat]
                    for unique_val in self.cat_unique_vals_id_mappings[feat]:
                        self.r_data.loc[self.r_data[feat] == unique_val, feat] = self.cat_unique_vals_id_mappings[feat][unique_val]
        self.init_data()

    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        # appts = self.r_data.loc[self.r_data[self.id_attr] == self.patient_ids[idx]]
        appts2 = self.features[idx]
        trans_appts2 = self.transformed_features[idx]
        treatment2 = self.treatment_array[idx]

        outcome2 = self.outcome_array[idx]
        if self.count_outcome_array is not None:
            count_outcome2 = self.count_outcome_array[idx]
        else:
            count_outcome2 = None
        
        if self.other_data is None:  
            all_other_pats2 = torch.ones(len(self.features)).bool()  #torch.cat([self.features[0:idx], self.features[idx+1:]], dim=0)
            all_other_pats2[idx] = False
        else:
            all_other_pats2 = torch.ones(len(self.other_data)).bool()

        text_id, text_mask, text_len = self.encoded_text_dict[idx]

        return idx, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, (text_id, text_mask, text_len) 
        # # m = [appts[self.outcome_attr].max()]
        # y = torch.tensor(list(appts[self.outcome_attr]), dtype=torch.float)
        # treatment = torch.tensor(list(appts[self.treatment_attr]), dtype=torch.float)
        # X_pd = appts.drop(self.drop_cols, axis=1)
        
        # # X_num = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.num_cols].to_numpy(dtype=np.float64)][0]
        # # X_cat = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.cat_cols].to_numpy(dtype=np.float64)][0].type(torch.long)
        
        # X_num = torch.from_numpy(X_pd[self.num_cols].to_numpy()).type(torch.float)

        # if len(self.cat_cols) > 0:
        #     X_cat = torch.from_numpy(X_pd[self.cat_cols].to_numpy()).type(torch.long)
            
        #     X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat.view(-1)[idx]] for idx in range(len(self.cat_cols))]
            
        #     X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1).unsqueeze(0)
            
        #     X = [torch.cat([X_num, X_cat_onehot], dim=-1)]
        # else:
        #     X = X_num
        # #zero pad
        # # X.extend([torch.tensor([0]*len(X[0]), dtype=torch.float) ]*(len(X)-self.patient_max_appts))
        # return (idx, self.patient_ids[idx], all_other_pats, appts, text_id, text_mask, text_len, X.view(-1)), y, treatment
    
    @staticmethod
    def collate_fn(data):
        # idx, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, (text_id, text_mask, text_len) 
        unzipped_data_ls = list(zip(*data))

        ids_ls = unzipped_data_ls[0]
        appts2_ls = unzipped_data_ls[1]
        treatment2_ls = unzipped_data_ls[2] 
        outcome2_ls = unzipped_data_ls[3]
        count_outcome2_ls = unzipped_data_ls[4]
        trans_appts2_ls = unzipped_data_ls[5]
        all_other_pats2_ls = unzipped_data_ls[6]

        text_id_ls = torch.stack([item[-1][0] for item in data])
        text_mask_ls = torch.stack([item[-1][1] for item in data])
        text_len_ls = torch.tensor([item[-1][2] for item in data])
            
            
        appts2 = torch.stack(appts2_ls)
        if treatment2_ls[0] is not None:
            treatment2=torch.tensor(treatment2_ls).view(-1,1)
        else:
            treatment2=None
        
        outcome2= torch.tensor(outcome2_ls).view(-1,1) 
        if count_outcome2_ls[0] is not None:
            count_outcome2 = torch.tensor(count_outcome2_ls).view(-1,1)  
        else:
            count_outcome2= None

        trans_appts2 = torch.stack(trans_appts2_ls)  
        all_other_pats2 = all_other_pats2_ls
        ids = torch.tensor(ids_ls)
        # idx, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, (text_id, text_mask, text_len) 
        return ids, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, (text_id_ls, text_mask_ls, text_len_ls) 

def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.num_cols)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

def create_dataset(dataset_name, tokenizer, all_df, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, text_attr, drop_cols, count_outcome_attr=None, classification=False):
    all_dataset = NLP_Dataset(dataset_name, tokenizer, all_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, text_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, count_outcome_attr=count_outcome_attr, classification=classification)
    feat_range_mappings = obtain_feat_range_mappings(all_dataset)   
    
    
    train_dataset = NLP_Dataset(dataset_name, tokenizer, train_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, text_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, y_scaler=all_dataset.y_scaler, count_outcome_attr=count_outcome_attr, classification=classification)
    # feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    train_dataset.rescale_data(feat_range_mappings)
    valid_dataset = NLP_Dataset(dataset_name, tokenizer, valid_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, text_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=train_dataset.r_data, y_scaler=all_dataset.y_scaler, count_outcome_attr=count_outcome_attr, classification=classification)
    valid_dataset.rescale_data(feat_range_mappings)
    test_dataset = NLP_Dataset(dataset_name, tokenizer, test_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, text_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=train_dataset.r_data, y_scaler=all_dataset.y_scaler,count_outcome_attr=count_outcome_attr, classification=classification)
    test_dataset.rescale_data(feat_range_mappings)
    return train_dataset, valid_dataset, test_dataset, feat_range_mappings
    
def construct_dataset_main(args):
    DROP_FEATS=None
    args.data_path = os.path.join(args.data_folder, args.dataset_name)
    if args.dataset_name == "music":
        raw_df = pd.read_csv(os.path.join(args.data_path, 'music.csv'))
        df, offset =run_simulation(raw_df, propensities=[0.8, 0.6], 
                                    beta_t=1.0, 
                                    beta_c=50.0,
                                    gamma=1.0   ,
                                    cts=True)
        
            #     df, offset =run_simulation(raw_df, propensities=[0.8, 0.6], 
    #                                 beta_t=1.0, 
    #                                 beta_c=50.0,
    #                                 gamma=1.0   ,
    #                                 cts=True)
        id_attr = "index"
        outcome_attr = "Y"
        count_outcome_attr = "count_Y"
        treatment_attr = "T"
        text_attr = "text"
        DROP_FEATS=['C','Unnamed: 0', 'Unnamed: 0.1']
        split_ids = None
        args.classification=False
    elif args.dataset_name == "EEEC":
        
        
        
        train_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_train.csv'))
        valid_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_dev.csv'))
        test_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_test.csv'))
        
        
        df = pd.concat([train_df, valid_df, test_df])
        train_df.reset_index(inplace=True)
        valid_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        df.reset_index(inplace=True)
        
        train_ids = list(range(len(train_df)))
        valid_ids = list(range(len(train_df), len(train_df)+len(valid_df)))
        test_ids = list(range(len(train_df)+len(valid_df), len(train_df)+len(valid_df)+len(test_df)))
        split_ids = [train_ids, valid_ids, test_ids]
        id_attr = "index"
        outcome_attr = "POMS_label"
        count_outcome_attr = None
        treatment_attr = args.treatment_opt + "_F_label"
        text_attr = "Sentence_F"
        if args.treatment_opt == "Gender":
            # ID_F,ID_CF,Person_F,Person_CF,Sentence_F,Sentence_CF,Gender_F_label,Gender_CF_label,Template,Race,Race_label,Emotion_word,Emotion,POMS_label
            DROP_FEATS=['Unnamed: 0','ID_F', 'ID_CF', 'Person_F', 'Person_CF', 'Sentence_CF', args.treatment_opt, "Template", "Race", "Race_label", "Emotion_word", "Emotion", "Gender_F_label"]
        else:
            # ID_F,ID_CF,Person_F,Person_CF,Sentence_F,Sentence_CF,Race_F_label,Race_CF_label,Template,Gender,Gender_label,Emotion_word,Emotion,POMS_label
            DROP_FEATS=['Unnamed: 0','ID_F', 'ID_CF', 'Person_F', 'Person_CF', 'Sentence_CF', "Template", "Gender", "Race", "Gender_label", "Emotion_word", "Emotion", "Race_F_label"]
        args.classification=True
    df, tokenizer = convert_text_to_features(args, df, text_attr, treatment_attr, outcome_attr)
    
    train_df, valid_df, test_df = split_train_valid_test_df(df, split_ids=split_ids)
    
    # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang, num_feats=None
    synthetic_lang.DROP_FEATS=DROP_FEATS
    lang = Language(df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, None, text_attr, precomputed=None, lang=synthetic_lang)
       
    train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(args.dataset_name, tokenizer, df, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, text_attr, synthetic_lang.DROP_FEATS, count_outcome_attr=count_outcome_attr, classification=args.classification)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lang = set_lang_data(lang, train_dataset, device=device)
    
    return train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, treatment_attr, text_attr, count_outcome_attr, lang