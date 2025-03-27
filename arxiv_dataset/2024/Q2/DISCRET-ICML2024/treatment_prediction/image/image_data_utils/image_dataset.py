from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset
import os, sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import io, color, segmentation
from skimage import io, color, segmentation, measure

from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

import pickle

from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from treatment_prediction.create_language import Language
import synthetic_lang
from simulation import run_simulation
from featurization import k_means
from treatment_prediction.utils_treatment import set_lang_data
from image_data_utils.image_loader import CausalDataset

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

def split_train_valid_test_df(df, image_data, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2, split_ids=None):
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
    train_image_data = image_data[train_ids]
    valid_df = df.iloc[valid_ids]
    valid_image_data = image_data[valid_ids]
    test_df = df.iloc[test_ids]
    test_image_data = image_data[test_ids]
    return train_df, valid_df, test_df, train_image_data, valid_image_data, test_image_data
    

    
class Image_Dataset(Dataset):
    def __init__(self, dataset_name, image_array, data, drop_cols, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr=None, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, y_scaler=None, classification=False):
        self.image_array = image_array
        self.dataset_name = dataset_name
        self.data = data
        self.drop_cols = drop_cols
        self.id_attr, self.outcome_attr, self.treatment_attr = id_attr, outcome_attr, treatment_attr
        self.data.index = list(range(len(self.data)))
        self.patient_ids = self.data[id_attr].unique().tolist()
        self.lang = lang
        self.other_data = other_data
        self.cat_cols = list(set(self.lang.CAT_FEATS))
        self.cat_cols.sort()
        self.num_cols = [col for col in data.columns if col not in self.lang.CAT_FEATS and not col == id_attr and not col in drop_cols and not col == outcome_attr and not col == treatment_attr and not col == outcome_attr and not col == count_outcome_attr] 
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
        # self.convert_text_to_tokens()
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

        image_vals = self.image_array[idx]

        return idx, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, image_vals
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

        image_ls = np.stack([item[-1] for item in data])
        
            
            
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
        return ids, appts2, treatment2, outcome2, count_outcome2, trans_appts2, all_other_pats2, image_ls

def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.num_cols)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

def create_dataset(dataset_name, image_data_array, train_image_data, valid_image_data, test_image_data, all_df, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, drop_cols, count_outcome_attr=None, classification=False):
    all_dataset = Image_Dataset(dataset_name, image_data_array, all_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, count_outcome_attr=count_outcome_attr, classification=classification)
    feat_range_mappings = obtain_feat_range_mappings(all_dataset)   
    
    
    train_dataset = Image_Dataset(dataset_name, train_image_data, train_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, y_scaler=all_dataset.y_scaler, count_outcome_attr=count_outcome_attr, classification=classification)
    # feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    train_dataset.rescale_data(feat_range_mappings)
    valid_dataset = Image_Dataset(dataset_name, valid_image_data, valid_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=train_dataset.r_data, y_scaler=all_dataset.y_scaler, count_outcome_attr=count_outcome_attr, classification=classification)
    valid_dataset.rescale_data(feat_range_mappings)
    test_dataset = Image_Dataset(dataset_name, test_image_data, test_df, drop_cols, lang, id_attr, outcome_attr, treatment_attr, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=train_dataset.r_data, y_scaler=all_dataset.y_scaler,count_outcome_attr=count_outcome_attr, classification=classification)
    test_dataset.rescale_data(feat_range_mappings)
    return train_dataset, valid_dataset, test_dataset, feat_range_mappings

def load_uganda_dataset(args):
    image_data_file = os.path.join(args.data_folder, 'images.npy')
    image_data = np.load(image_data_file)
    covariates =pd.read_csv(os.path.join(args.data_folder, 'X.csv'))[["V1","V2","V3","V4","V5","V6","V7"]]
    treatment_array = np.array(pd.read_csv(os.path.join(args.data_folder, 'obsW.csv'))["x"])
    outcome_array = np.array(pd.read_csv(os.path.join(args.data_folder, 'obsY.csv'))["x"])
    print("image_data.shape", image_data.shape)
    return image_data, covariates, treatment_array, outcome_array

def obtain_unique_images(image_data):
    unique_images = []
    
    image_to_unique_image_idx = dict()
    print("obtain unique images")
    for i in tqdm(range(image_data.shape[0])):
        if len(unique_images) <= 0:
            unique_images.append(image_data[i])
            image_to_unique_image_idx[i] = 0
        else:
            hit = False
            for j in range(len(unique_images)):
                difference = np.linalg.norm(image_data[i] - unique_images[j])
                if difference <= 0:
                    image_to_unique_image_idx[i] = j
                    hit = True
                    break
            # if j >= len(unique_images):
            if not hit:
                image_to_unique_image_idx[i] = len(unique_images)
                unique_images.append(image_data[i])
                
            
    return unique_images, image_to_unique_image_idx
   

# image_id=50
def get_segments_and_embeddings_for_all_images(image_ls):
    outputs_ls=[]
    all_regions_ls=[]
    all_image_ids = []
    image_id_to_patch_ids = []
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for image_id in tqdm(range(len(image_ls))):
        # array_3d = np.stack([c_dataset[image_id]['band1'], c_dataset[image_id]['band2'], c_dataset[image_id]['band3']], axis=-1).astype(np.uint8)
        array_3d = image_ls[image_id]
        
        image = Image.fromarray(array_3d.astype(np.uint8))
        lab_image = color.rgb2lab(image)

        segments_slic = slic(lab_image, n_segments=20, compactness=200, max_num_iter=200)
        # print(np.unique(segments_slic))
        regions = measure.regionprops(segments_slic)
        start_idx = len(outputs_ls)
        curr_patch_ids = []
        bbox_ls = []
        for patch_id in range(len(regions)):
            x,y,w,h = regions[patch_id].bbox[0], regions[patch_id].bbox[1], regions[patch_id].bbox[2], regions[patch_id].bbox[3]
            assert w >= x and h >= y
            patch = Image.fromarray(array_3d[x:w, y:h].astype(np.uint8))
            inputs = processor(images=patch, return_tensors="pt", padding=False)
            inputs.to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                outputs_ls.append(outputs)
                all_image_ids.append(image_id)
            curr_patch_ids.append(start_idx)
            start_idx += 1
            bbox_ls.append([x,y,w,h])
        image_id_to_patch_ids.append(curr_patch_ids)
        all_regions_ls.extend(bbox_ls)
    return outputs_ls, all_regions_ls, all_image_ids, image_id_to_patch_ids


def obtain_patch_cluster_ids_per_image(image_id_to_patch_ids, cluster_assignments, cluster_count):
    image_id_to_cluster_ids = []
    for image_id in range(len(image_id_to_patch_ids)):
        patch_ids = image_id_to_patch_ids[image_id]
        cluster_ids = cluster_assignments[torch.tensor(patch_ids)]
        cluster_id_count_ls = []
        for cluster_id in range(cluster_count):
            cluster_id_count_ls.append(torch.sum(cluster_ids == cluster_id).item())
        image_id_to_cluster_ids.append(np.array(cluster_id_count_ls))
    return image_id_to_cluster_ids

def construct_full_image_to_cluster_id_mappings(image_id_to_cluster_ids, image_to_unique_image_idx):
    full_image_id_to_cluster_ids = []
    for img_idx in range(len(image_to_unique_image_idx)):
        unique_img_idx = image_to_unique_image_idx[img_idx]
        cluster_ids = image_id_to_cluster_ids[unique_img_idx]
        full_image_id_to_cluster_ids.append(cluster_ids)
    return full_image_id_to_cluster_ids

def subset_image_array(image_array, curr_image_ids, image_ids):
    id2= 0
    id1 = 0
    curr_image_array = []
    while(id1 < len(curr_image_ids) and id2 < len(image_ids)):
        if curr_image_ids[id1] == image_ids[id2]:
           id1 += 1
           id2 += 1 
           curr_image_array.append(image_array[id2])
        else:
            id2 += 1
    return curr_image_array

def construct_dataset_main(args, cluster_count=20):
    if not args.method == "ours" and not args.method == "nam" and not args.method == "TransTEE_tab":
        DROP_FEATS=["cluster_id_"+str(i) for i in range(cluster_count)] + ["Unnamed: 0"]
    else:
        DROP_FEATS=["Unnamed: 0"]

    full_df_file_name = os.path.join(args.data_folder, "full_df.csv")
    image_data, df, treatment_array, outcome_array = load_uganda_dataset(args)
    image_data = torch.from_numpy(image_data).permute(3,0,1,2)
    image_data = image_data.numpy()
    count_outcome_attr = None
    
    image_processed_id_file_name = os.path.join(args.data_folder, "uganda/UgandaDataProcessed.csv")
        
    image_processed_id_mappings = pd.read_csv(image_processed_id_file_name)
    
    curr_image_processed_id_mappings_file_name = os.path.join(args.data_folder, "KeysOfObservations.csv")
    
    curr_image_processed_id_mappings = pd.read_csv(curr_image_processed_id_mappings_file_name)
    
    image_data = subset_image_array(image_data, np.array(curr_image_processed_id_mappings["x"]), np.array(image_processed_id_mappings['geo_long_lat_key']))
    
    image_data = np.stack(image_data, axis=0)
    image_data_file = os.path.join(args.data_folder, 'real_images')
    # with open(image_data_file, "wb") as f:
    #     pickle.dump(image_data, f, protocol=4)
    print("number of images::", len(image_data))
    if os.path.exists(full_df_file_name):
        df = pd.read_csv(full_df_file_name)

        treatment_attr = "treatment"
        
        outcome_attr = "outcome"

        id_attr = "id"
    else:
      
        
        
        
        image_to_unique_image_idx_file_name = os.path.join(args.data_folder, "image_to_unique_image_idx.pkl")
        
        unique_image_file_name = os.path.join(args.data_folder, "unique_images.pkl")
        
        if not os.path.exists(image_to_unique_image_idx_file_name) or not os.path.exists(unique_image_file_name):
        
            unique_images, image_to_unique_image_idx = obtain_unique_images(image_data)
        
            with open(image_to_unique_image_idx_file_name, "wb") as f:
                pickle.dump(image_to_unique_image_idx, f)
            
            with open(unique_image_file_name, "wb") as f:
                pickle.dump(unique_images, f)
        else:
            with open(image_to_unique_image_idx_file_name, "rb") as f:
                image_to_unique_image_idx = pickle.load(f)
            
            with open(unique_image_file_name, "rb") as f:
                unique_images = pickle.load(f)
        
        
        
        output_embedding_file_name = os.path.join(args.data_folder, "output_embeddings_ls")
        all_region_file_name = os.path.join(args.data_folder, "all_region_ls")
        patch_id_to_unique_img_file_name = os.path.join(args.data_folder, "patch_id_to_unique_img_id")
        image_id_to_patch_id_file_name = os.path.join(args.data_folder, "image_id_to_patch_ids")
                
        
        if not os.path.exists(output_embedding_file_name) or not os.path.exists(all_region_file_name) or not os.path.exists(patch_id_to_unique_img_file_name) or not os.path.exists(image_id_to_patch_id_file_name):
            output_embeddings_ls, all_regions_ls, patch_id_to_unique_img_id, image_id_to_patch_ids = get_segments_and_embeddings_for_all_images(unique_images)
            with open(output_embedding_file_name, "wb") as f:
                pickle.dump(output_embeddings_ls, f)
            with open(all_region_file_name, "wb") as f:
                pickle.dump(all_regions_ls, f)
            with open(patch_id_to_unique_img_file_name, "wb") as f:
                pickle.dump(patch_id_to_unique_img_id, f)
            with open(image_id_to_patch_id_file_name, "wb") as f:
                pickle.dump(image_id_to_patch_ids, f)
        else:
            with open(output_embedding_file_name, "rb") as f:
                output_embeddings_ls = pickle.load(f)
            with open(all_region_file_name, "rb") as f:
                all_regions_ls = pickle.load(f)
            with open(patch_id_to_unique_img_file_name, "rb") as f:
                patch_id_to_unique_img_id = pickle.load(f)
            with open(image_id_to_patch_id_file_name, "rb") as f:
                image_id_to_patch_ids = pickle.load(f)
        
        
        outputs_tensor = torch.cat(output_embeddings_ls)
        outputs_tensor = outputs_tensor/torch.norm(outputs_tensor, dim=1, keepdim=True)


        centroids, cluster_assignments = k_means(outputs_tensor, cluster_count)

        image_id_to_cluster_ids = obtain_patch_cluster_ids_per_image(image_id_to_patch_ids, cluster_assignments, cluster_count)
        
        full_image_id_to_cluster_ids = construct_full_image_to_cluster_id_mappings(image_id_to_cluster_ids, image_to_unique_image_idx)
        
        cluster_id_cols = ["cluster_id_"+str(i) for i in range(cluster_count)]
        
        df[cluster_id_cols] = np.stack(full_image_id_to_cluster_ids, axis=0)[0:len(df)]
        
        treatment_attr = "treatment"
        
        outcome_attr = "outcome"
        patch_ls, full_image_ls = obtain_image_segment_concepts(outputs_tensor, centroids, all_regions_ls, patch_id_to_unique_img_id, unique_images)
        patch_ls_file = os.path.join(args.data_folder, "patch_ls")
        patch_full_image_ls = os.path.join(args.data_folder, "patch_full_image_ls")
        with open(patch_ls_file, "wb") as f:
            pickle.dump(patch_ls, f)
        with open(patch_full_image_ls, "wb") as f:
            pickle.dump(full_image_ls, f)
        # patch_cluster_id_file_name = os.path.join(args.da)
        # with open()
        
        df[treatment_attr] = treatment_array
        df[outcome_attr] = outcome_array
        df.reset_index(inplace=True)
        id_attr = "id"
        df.rename(columns={"index": id_attr}, inplace=True)
        df.to_csv(full_df_file_name)
        
    nan_indexes = df.index[df.isna().any(axis=1)].tolist()

    df.drop(nan_indexes, inplace=True)

    image_data = np.delete(image_data, nan_indexes, axis=0)


    train_df, valid_df, test_df, train_image_data, valid_image_data, test_image_data = split_train_valid_test_df(df, image_data)
    
    train_image_data_file = os.path.join(args.data_folder, 'real_train_images')
    with open(train_image_data_file, "wb") as f:
        pickle.dump(train_image_data, f)
    
    valid_image_data_file = os.path.join(args.data_folder, 'real_valid_images')
    with open(valid_image_data_file, "wb") as f:
        pickle.dump(valid_image_data, f)
        
    test_image_data_file = os.path.join(args.data_folder, 'real_test_images')
    with open(test_image_data_file, "wb") as f:
        pickle.dump(test_image_data, f)
    
    # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang, num_feats=None
    synthetic_lang.DROP_FEATS=DROP_FEATS
    lang = Language(df, id_attr, outcome_attr, None, treatment_attr, None, None, precomputed=None, lang=synthetic_lang)
       
    train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(args.dataset_name, image_data, train_image_data, valid_image_data, test_image_data, df, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, synthetic_lang.DROP_FEATS, count_outcome_attr=count_outcome_attr, classification=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lang = set_lang_data(lang, train_dataset, device=device)
    
    return train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, count_outcome_attr, treatment_attr, lang


def obtain_image_segment_concepts(outputs_tensor, centroids, all_regions_ls, all_image_ids, image_data):
    dist = torch.cdist(outputs_tensor, centroids)
    closest_centroid_sample_idx = dist.argmin(dim=0)
    patch_ls = []
    full_image_ls = []
    for idx, patch_id in enumerate(closest_centroid_sample_idx.tolist()):
        # x1,y1,x2,y2 = all_regions_ls[patch_id].bbox[0], all_regions_ls[patch_id].bbox[1], all_regions_ls[patch_id].bbox[2], all_regions_ls[patch_id].bbox[3]
        x1,y1,x2,y2 = all_regions_ls[patch_id][0], all_regions_ls[patch_id][1], all_regions_ls[patch_id][2], all_regions_ls[patch_id][3]
        print(x1,y1,x2,y2)
        image_id = all_image_ids[patch_id]
        array_3d = image_data[image_id] # np.stack([c_dataset[image_id]['band1'], c_dataset[image_id]['band2'], c_dataset[image_id]['band3']], axis=-1).astype(np.uint8)
        patch = Image.fromarray(array_3d[x1:x2, y1:y2].astype(np.uint8))
        full_image_ls.append(Image.fromarray(array_3d.astype(np.uint8)))
        patch_ls.append(patch)
    return patch_ls, full_image_ls