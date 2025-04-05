# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import os
import sys
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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Node():
    def __init__(self, prompt_indexes=None, left=None, right=None, class_index=None):
        ''' constructor ''' 
        
        # for decision node
        self.prompt_indexes = prompt_indexes
        self.left = left
        self.right = right
        
        # for leaf node
        self.class_index = class_index

class DecisionTreeClassifier():
    def __init__(self, prompts, encoded_prompts, nun_leaves=1000, children=None):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        self.nun_leaves = nun_leaves
        self.children = children
        self.prompts = prompts
        self.encoded_prompts = encoded_prompts

       

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
        
        if tree.class_index is not None:
            #print(f'make_prediction - leaf node. return class_index: {tree.class_index}') 
            return tree.class_index

        #print(f'make_prediction - tree.right.prompt_indexes: {tree.right.prompt_indexes}')
        #print(f'make_prediction - tree.left.prompt_indexes: {tree.left.prompt_indexes}') 
 


        #x is an encoded image (shape 1X512)
        #tree.prompt_encoding is encoded text (shape 1X512)
        right_prompt_encoding = self.encoded_prompts[tree.right.prompt_indexes].to(device=device)
        right_prompt_encoding = right_prompt_encoding.double()
        left_prompt_encoding = self.encoded_prompts[tree.left.prompt_indexes].to(device=device)
        left_prompt_encoding = left_prompt_encoding.double()
        #similarity = tree.prompt_encoding @ x.t()
        right_similarity = right_prompt_encoding @ x.t()
        left_similarity = left_prompt_encoding @ x.t()

        #TODO: check the sum or average ? something else?

        #feature_val = x[tree.feature_index]
        #if feature_val<=tree.threshold:
        #if similarity.item() <=tree.threshold:
        #if torch.sum(left_similarity) > torch.sum(right_similarity):
        #if torch.mean(left_similarity) > torch.mean(right_similarity):
        if torch.max(left_similarity) > torch.max(right_similarity):
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


    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.class_index is not None:
            print(tree.class_index)

        else:
            #print(f"prompt: {tree.prompt}, threshold: {tree.threshold},  info_gain: {tree.info_gain}")
            print(f"classes: {tree.prompt_indexes}")
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self):
        ''' function to train the tree '''
        #root index is the number of decision nodes(nun_leaves - 1) + index of the last leaf node (nun_leaves - 1)
        root_index = 2*(self.nun_leaves - 1)
        self.root = self.build_tree(node_index=root_index)

    def build_tree(self, node_index):
        ''' recursive function to build the tree ''' 
        
        if node_index < self.nun_leaves:
            #leaf node
            # return leaf node
            return Node(class_index=node_index, prompt_indexes=[node_index])

        child_index = node_index - self.nun_leaves
        
        left_child_index = self.children[child_index][0]
        right_child_index = self.children[child_index][1]
        
        left_subtree = self.build_tree(left_child_index)
        right_subtree = self.build_tree(right_child_index)
        prompt_indexes = []
        prompt_indexes.extend(left_subtree.prompt_indexes)
        prompt_indexes.extend(right_subtree.prompt_indexes)
        return Node(left=left_subtree, right=right_subtree, prompt_indexes=prompt_indexes)
        

def get_data_and_labels(csv_file_name, num_features = 512, subset_size=-1):
    
    print(f'get_data_and_labels from csv_file_name: {csv_file_name}')
    df = pd.read_csv(csv_file_name)
    #print(f'\ndataset: \n{dataset}')
    #print(f'\ndataset.info(): \n')
    #dataset.info()
    #print(f'\ndataset.head(): \n{dataset.head()}')
    dataset = df.to_numpy()
    #X = dataset.values[:, 0:num_features]
    
    if subset_size != -1:
        np.random.shuffle(dataset)
        dataset = dataset[0:subset_size]

    #X = dataset.drop(columns="label")
    #feature_names = X.columns
    #Y = dataset.values[:, num_features]
    #Y = dataset["label"].astype('int16')

    X = dataset[:,:-1]
    Y = dataset[:,-1].astype('int16')


    print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
    #print(f'Y[0:5]: {Y[0:5]}')
    #print(f'\nX[0:5]: \n{X[0:5]}')
    #print(f'\nfeature_names: \n{feature_names}')

    #return X, Y, feature_names
    return X, Y

def main():
    model_name = "ViT-B/32"
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)

    imagenet_test_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='val', transform=preprocess)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #print(f'imagenet_test_dataset.classes: {imagenet_test_dataset.classes}')
    
    #prompts = [f"a photo of a {c[0]}" for c in imagenet_test_dataset.classes]
    prompts = []
    
    #replace duplicate 'crane' class with 'crane bird' and 'crane machine'
    found_crane = False
    for idx, c in enumerate(imagenet_test_dataset.classes):
        if c[0] == 'crane':
            #print(f'found crane idx: {idx}')
            if found_crane:
                prompts.append(f"a photo of a crane machine")
            else:
                prompts.append(f"a photo of a crane bird")
                found_crane = True
        else:
            prompts.append(f"a photo of a {c[0]}")

    #print(f'imagenet prompts: {prompts}')
    #print(f'prompt idx, \tprompt')
    #for idx, p in enumerate(prompts):
    #    print(f'{idx}, \t{p}')
    #assert False

    #text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_test_dataset.classes]).to(device)
    
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    num_leaves = len(imagenet_test_dataset.classes)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.to('cpu')

    #linkage in ("ward", "average", "complete", "single")
    #clustering = AgglomerativeClustering(linkage="ward", n_clusters=10)
    #clustering = AgglomerativeClustering(linkage="ward", n_clusters=num_leaves-1)
    
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(text_features)

    #clusters = shc.linkage(text_features, 
    #        method='ward', 
    #        metric="euclidean")

    #plt.figure(figsize=(30, 7))
    #plt.title("ImageNet classes Dendrogram")
    #shc.dendrogram(Z=clusters)
    #plt.savefig("Dendrogram3.png")
    #plt.show()

    #print(f'\nclustering.labels_: {clustering.labels_}\n\n')
    

    #print(f'class name, \tcluster ID\n\n')
    #for idx, class_name in enumerate(imagenet_test_dataset.classes):
    #    print(f'{class_name}, \t{clustering.labels_[idx]}')


    #print(f'cluster index, \tchildren_indexes\n\n')
    #for idx, child in enumerate(clustering.children_):
    #    print(f'{idx}, \t{child}')
     
    #print(f'\n\nclustering.children_: {clustering.children_}\n\n')

    clf = DecisionTreeClassifier(prompts=prompts, encoded_prompts=text_features, nun_leaves=num_leaves, children=clustering.children_)

    print(f'before fit')
    clf.fit()
    print(f'after fit')

    print(f"\nimage net classes tree diagram:\n")
    clf.print_tree()

    original_stdout = sys.stdout 

    with open('imagenet_classes_decision_tree_diagram.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file 
        clf.print_tree()
        sys.stdout = original_stdout # Reset the standard output to its original value

    X_test, Y_test = get_data_and_labels('clip_imagenet_test.csv')
    preds = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, preds)

    print(f"\n ImageNet classes decision tree accuracy on image net test set: {accuracy*100:.2f}%")
    with open('imagenet_classes_decision_tree_accuracy.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file 
        print(f"\n ImageNet classes decision tree accuracy on image net test set: {accuracy*100:.2f}%")
        sys.stdout = original_stdout # Reset the standard output to its original value



if __name__ == "__main__":
    main()
