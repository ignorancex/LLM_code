# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import os
import torch
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm
import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from tqdm import tqdm

import pandas as pd
import numpy as np
#from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
import json
#import _pickle as cPickle
import pickle
from collections import Counter


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Node():
    def __init__(self, prompt=None, prompt_encoding=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        #self.attribute = attribute
        self.prompt = prompt
        self.prompt_encoding = prompt_encoding
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, prompts, encoded_prompts, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        #a photo of a <attribute> object
        self.prompts = prompts
        self.objects = None
        self.encoded_prompts = encoded_prompts
        
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        print(f'build_tree: curr_depth: {curr_depth}, max_depth: {self.max_depth}')

        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                prompt_idx = best_split["promt_index"] 
                prompt = self.prompts[prompt_idx]
                prompt_encoding = self.encoded_prompts[prompt_idx]
                threshold = best_split["threshold"]
                info_gain = best_split["info_gain"]
                return Node(prompt=prompt, prompt_encoding=prompt_encoding, threshold=threshold, 
                            left=left_subtree, right=right_subtree, info_gain=info_gain)
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        best_split["info_gain"] = max_info_gain
        
        # loop over all the attributes
        #should be normalized text encoded attribute prompt (a photo of <attr> object )
        for prompt_idx, prompt_encoding in enumerate(tqdm(self.encoded_prompts)):

            #feature_values = dataset[:, feature_index]

            #TODO:keep one value for each image ? (e.g. l2 norm)
            #for each image we have the clip embeddings
            img_embeds = dataset[:, 0:-1]
            # converting list to array
            #embeds = np.array(embeds)
            img_embeds = torch.from_numpy(img_embeds)
            img_embeds = img_embeds.to(device=device)
            prompt_encoding = prompt_encoding.to(device=device)
            prompt_encoding = prompt_encoding.double()
            #print(f'\nprompt_encoding: {prompt_encoding}\nimg_embeds: {img_embeds}')
            similarity = prompt_encoding @ img_embeds.t()
            similarity = similarity.to('cpu')
            similarity = similarity.numpy()

            possible_thresholds = np.unique(similarity)
            #looping over all possible_thresholds is too slow(takes ~5 hours) 
            #so choose a few cut points
            sorted_thresholds = np.sort(possible_thresholds)
            thresholds_len = len(sorted_thresholds)
            num_thresholds = 4
            #print(f'\nthresholds_len: {thresholds_len}, num_thresholds: {num_thresholds}')
            if thresholds_len > num_thresholds + 1:
                cut_point_size = thresholds_len // (num_thresholds + 1)
                chosen_thresholds = [sorted_thresholds[i*cut_point_size] for i in range(1,num_thresholds+1)]
                #print(f'\ncut_point_size: {cut_point_size}, \nchosen_thresholds: {chosen_thresholds}')
            else:
                chosen_thresholds = sorted_thresholds    

            #print(f'len(chosen_thresholds): {len(chosen_thresholds)}')

            # loop over all the data values 
            #for threshold in tqdm(possible_thresholds):
            #for threshold in tqdm(chosen_thresholds):
            chosen_thresholds = [0.3]
            for threshold in chosen_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, similarity, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        #best_split["attribute"] = attr
                        best_split["promt_index"] = prompt_idx
                        #best_split["prompt_encoding"] = prompt_encoding
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, similarity, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([dataset[idx] for idx, sim in enumerate(similarity) if sim <= threshold])
        dataset_right = np.array([dataset[idx] for idx, sim in enumerate(similarity) if sim > threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        #print(f'calculate_leaf_value - Y: \n{Y}')
        Y = list(Y)
        #print(f'calculate_leaf_value - list(Y): \n{Y}')

        c = Counter(Y)

        #TODO: return softmax over Y?
        #return max(Y, key=Y.count)
        most_common = c.most_common(1)
        #print(f'c.most_common(1): {most_common}')
        most_common = most_common[0]
        #print(f'most_common[0]: {most_common}')
        most_common_elem = most_common[0]
        #print(f'most_common[0][0]: {most_common_elem}')

        #assert False
        return most_common_elem
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f"prompt: {tree.prompt}, threshold: {tree.threshold},  info_gain: {tree.info_gain}")
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        #print(f'\nX[0]: \n{X[0]}')

        img_embeds = torch.from_numpy(X)
        img_embeds = img_embeds.to(device=device)
        #print(f'\nimg_embeds[0]: \n{img_embeds[0]}')

        preditions = [self.make_prediction(x, self.root) for x in img_embeds]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value is not None: 
            return tree.value
        #x is an encoded image (shape 1X512)
        #tree.prompt_encoding is encoded text (shape 1X512)
        prompt_encoding = tree.prompt_encoding.to(device=device)
        prompt_encoding = prompt_encoding.double()
        #similarity = tree.prompt_encoding @ x.t()
        similarity = prompt_encoding @ x.t()
        #feature_val = x[tree.feature_index]
        #if feature_val<=tree.threshold:
        if similarity.item() <=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

def save_tree(filename, tree_clf):
    #outfile = open(filename,'wb')
    #pickle.dump(tree_clf,outfile)
    #outfile.close()
    with open(filename,'wb') as f:
        pickle.dump(tree_clf,f)

def load_tree(filename):
    #infile = open(filename,'rb')
    #tree_clf = pickle.load(infile)
    #infile.close()
    #return tree_clf
    with open(filename,'rb') as f:
        tree_clf = pickle.load(f)
        return tree_clf

def get_data_and_labels(csv_file_name, num_features = 512):
    
    print(f'get_data_and_labels from csv_file_name: {csv_file_name}')
    df = pd.read_csv(csv_file_name)
    #print(f'\ndataset: \n{dataset}')
    #print(f'\ndataset.info(): \n')
    #dataset.info()
    #print(f'\ndataset.head(): \n{dataset.head()}')
    dataset = df.to_numpy()
    #X = dataset.values[:, 0:num_features]
    
    #X = dataset.drop(columns="label")
    #feature_names = X.columns
    #Y = dataset.values[:, num_features]
    #Y = dataset["label"].astype('int16')

    X = dataset[:,:-1]
    Y = dataset[:,-1].astype('int')

    #clip_imagenet_train.csv:
    #X.shape: (1281167, 512), Y.shape: (1281167,)
    
    
    
    print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
    #print(f'Y[0:5]: {Y[0:5]}')
    #print(f'\nX[0:5]: \n{X[0:5]}')
    #print(f'\nfeature_names: \n{feature_names}')

    #return X, Y, feature_names
    print('dataset loaded')
    return X, Y

def get_chunk(lst, chunck_size):
     
    # looping till length l
    for i in range(0, len(lst), chunck_size):
        yield lst[i:i + chunck_size]

def main():
    
    #train_filename = 'clip_imagenet_train.csv'
    #train_dataset = pd.read_csv(train_filename)
    #train_dataset.head()

    num_features = 512
    #X_train = train_dataset[:, 0:num_features]
    #Y_train = train_dataset[:, num_features]

    X_train, Y_train = get_data_and_labels('clip_imagenet_train.csv', num_features)
    #X_test, Y_test = get_data_and_labels('clip_imagenet_test.csv', num_features)

    Y_train = [x.item() for x in Y_train]
    class_label_counter = Counter(Y_train)
    print(f'class_label_counter: \n{class_label_counter}')

    with open(f"image_net_class_label_counter.json", "w") as fp:
        json.dump(class_label_counter, fp)

    assert False

    with open('full_adjectives.json', 'rb') as fp:
        attributes = json.load(fp)

    attributes_prompts = [f'a photo of a {attr} object' for attr in attributes]
    print(f'len(attributes): {len(attributes)}')
    
    model_name = "ViT-B/32"
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)
    
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in attributes_prompts]).to(device)
    
    

    prompt_chunk_size = 6000
    encoded_prompts = torch.empty(0, X_train.shape[1]).to(device=device)
    tokenized_prompts = tokenized_prompts.to(device=device)
    with torch.no_grad():
        #for img_embeds_chunk in get_chunk(tokenized_prompts, prompt_chunk_size):
        for img_embeds_chunk in tokenized_prompts.chunk(3):
            encoded_prompts_chunck = clip_model.encode_text(img_embeds_chunk)
            encoded_prompts_chunck /= encoded_prompts_chunck.norm(dim=-1, keepdim=True)   
            encoded_prompts = torch.cat([encoded_prompts, encoded_prompts_chunck]).to(device)
            #print(f'encoded_prompts.shape: {encoded_prompts.shape}')

    print(f'encoded_prompts.shape: {encoded_prompts.shape}')
    #assert False
    
    apperances_threshold = 0.35
    images_chunk_size = 10000
    img_embeds = torch.from_numpy(X_train)
    img_embeds = img_embeds.to(device=device)
    img_classes = torch.from_numpy(Y_train)
    #img_classes = img_classes.to(device=device)
    #print(f'img_classes.shape: {img_classes.shape}')
    attribute_num_per_class = dict()
    for idx, prompt in enumerate(tqdm(encoded_prompts)):
        #for img_embeds_chunk in get_chunk(, images_chunk_size):
        attr = attributes[idx]
        #attribute_num_appearances[attr] = 0
        #for img_embeds_chunk in img_embeds.chunk(200):
            #img_embeds = torch.from_numpy(img_embeds_chunk)
            #img_embeds = img_embeds.to(device=device)
        prompt = prompt.to(device=device)
        prompt = prompt.double()
        similarity = prompt @ img_embeds.t()
        #print(f'similarity.shape: {similarity.shape}')
        
            #similarity = similarity.to('cpu')
            #similarity = similarity.numpy()
            #print(f'img_embeds_chunk.shape: {img_embeds_chunk.shape}, similarity.shape: {similarity.shape}')
        
        similarity = similarity.to('cpu')
        apperances = Y_train[similarity >= apperances_threshold]
        apperances = [x.item() for x in apperances]
        #print(f'apperances[0:100]: {apperances[0:100]}')
        count_per_class = Counter(apperances)
        #print(f'{attr} count_per_class: {count_per_class}')
        #num_apperances = torch.numel(apperances)
        #attribute_num_appearances[attr] = num_apperances
        attribute_num_per_class[attr] = count_per_class
        #print(f'attribute: {attr}, num_apperances: {num_apperances}, apperances: {apperances}')

    #attribute_num_appearances = {k: v for k, v in sorted(attribute_num_appearances.items(), key=lambda item: item[1])}
    #print(f'apperances_threshold: {apperances_threshold}, attribute_num_appearances: \n{attribute_num_appearances}')
    with open(f"num_attr_appearances_per_class_threshold_{apperances_threshold}.json", "w") as fp:
        json.dump(attribute_num_per_class, fp)

    assert False

    #clf = DecisionTreeClassifier(prompts=attributes_prompts, encoded_prompts=encoded_prompts, min_samples_split=2, max_depth =3)
    clf = DecisionTreeClassifier(prompts=attributes_prompts, encoded_prompts=encoded_prompts, min_samples_split=2, max_depth=4)

    Y_train = np.expand_dims(Y_train, axis=1)
    clf.fit(X_train, Y_train)

    preds = clf.predict(X_test)
    #clf.fit(X_temp, Y_temp)
    #preds = clf.predict(X_temp)


    accuracy = accuracy_score(Y_test, preds)
    
    #tree_rules = export_text(clf, feature_names = list(feature_names))
    #print(f"\ncsp_cgqa decision tree before loading from file:\n")
    #print(f"\ncsp_cgqa decision tree:\n")
    print(f"\nimage net decision tree:\n")

    clf.print_tree()

    #print(f'\ntree is: \n{tree_rules}')
    #print(f"\nDecisionTreeClassifier accuracy on csp_cgqa test set: {accuracy*100:.2f}%")
    print(f"\nDecisionTreeClassifier accuracy on image net test set: {accuracy*100:.2f}%")


    #filename = 'csp_cgqa_depth_5_decision_tree.pkl'
    filename = 'imagenet_15_attr_depth_4_decision_tree.pkl'
    
    save_tree(filename, clf)

    #loaded_tree = load_tree(filename)
    #print(f"\ncsp_cgqa decision tree after loading from file {filename}:\n")

    #loaded_tree.print_tree()

    #preds = loaded_tree.predict(X_test)
    #accuracy = accuracy_score(Y_test, preds)

    #print(f"\nafter loading from file accuracy on csp_cgqa test set: {accuracy*100:.2f}%")



if __name__ == "__main__":
    main()
