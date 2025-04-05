import numpy as np
import csv
import random
import torch 
import collections

class Pool():
    def __init__(self, pt_file):
        self.features, self.labels = torch.load(pt_file)
        self.features = self.features.cpu().numpy()
        self.labels = self.labels.cpu().numpy()
        print(f'Pool size: {self.features.shape}')
        
        index_lt = [i for i in range(len(self.labels))]
        random.shuffle(index_lt)
        index_lt = np.array(index_lt)
        self.features = self.features[index_lt]
        self.labels = self.labels[index_lt]

        print(f'Pool shuffle') 
    
    def random_query_label(self):
        i = random.randint(0, len(self.labels)-1)
        assert i < len(self.labels)
        feat = self.features[i]
        label = self.labels[i]
        self.features = np.delete(self.features, i, 0)
        self.labels = np.delete(self.labels, i, 0)
        return feat, label   

    def random_query_labels(self, K=1000):
        index_lt = [i for i in range(len(self.labels))]
        i_lt = random.sample(index_lt, K)
        assert len(i_lt) > 0
        feat_lt, label_lt = [], []
        for i in i_lt:
            feat = self.features[i]
            label = self.labels[i]
            feat_lt.append(feat)
            label_lt.append(label)
        
        # delete rows
        self.features= np.delete(self.features, i_lt, 0)
        self.labels = np.delete(self.labels, i_lt, 0)
        return feat_lt, label_lt

    def keep(self, K=1000):
        delete_num = len(self.labels) - K
        index_lt = [i for i in range(len(self.labels))]
        i_lt = random.sample(index_lt, delete_num)
 
        self.features= np.delete(self.features, i_lt, 0)
        self.labels = np.delete(self.labels, i_lt, 0)
        #return feat_lt, label_lt


    def random_query_balanced_labels(self, K=2):
        counter = collections.Counter(self.labels)
        index_lt = [[] for i in counter]

        for i in range(len(self.labels)):
            index_lt[self.labels[i]].append(i)
        
        sample_index_lt = []
        
        for i in counter:
            sample_index_lt += random.sample(index_lt[i], K)
        
        return self.query_labels(sample_index_lt)

    def random_query_labels_spec(self, sample_each_class_lt):
        counter = collections.Counter(self.labels)
        index_lt = [[] for i in counter]

        for i in range(len(self.labels)):
            index_lt[self.labels[i]].append(i)
        
        sample_index_lt = []
        
        for i in counter:
            sample_index_lt += random.sample(index_lt[i], sample_each_class_lt[i])
        
        return self.query_labels(sample_index_lt)


    def query_label(self, i):
        assert i < len(self.labels)
        feat = self.features[i]
        label = self.labels[i]
        self.features = np.delete(self.features, i, 0)
        self.labels = np.delete(self.labels, i, 0)
        return feat, label

    def query_labels(self, i_lt):
        assert len(i_lt) > 0
        feat_lt, label_lt = [], []
        for i in i_lt:
            feat = self.features[i]
            label = self.labels[i]
            feat_lt.append(feat)
            label_lt.append(label)
        
        # delete rows
        self.features= np.delete(self.features, i_lt, 0)
        self.labels = np.delete(self.labels, i_lt, 0)
        return feat_lt, label_lt

    def get_feat(self, i):
        assert i < len(self.labels)
        return self.features[i]
    
    def get_all_features(self):
        return self.features

