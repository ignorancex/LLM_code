import numpy as np
import csv
import random
import torch 
import collections
import os

class2label = {'airplane':0,  'automobile':1,  'bird':2,  'cat':3,
            'deer':4,  'dog':5,  'frog':6,  'horse':7,  'ship':8,  'truck':9}

class Pool():
    def __init__(self, pt_file):
        #self.features, self.labels = torch.load(pt_file)
        #self.features = self.features.cpu().numpy()
        #self.labels = self.labels.cpu().numpy()
        self.image_name_lt = []
        self.labels = []
        fin = open(pt_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            path = os.path.join('./data/', line)
            self.image_name_lt.append(path)
            class_name = os.path.basename(os.path.dirname(line))
            label = class2label[class_name]
            self.labels.append(label)
        
        self.image_name_lt = np.array(self.image_name_lt)
        self.labels = np.array(self.labels)
        print(f'Pool size: {len(self.image_name_lt)}')
        
        #index_lt = [i for i in range(len(self.labels))]
        #random.shuffle(index_lt)
        #index_lt = np.array(index_lt)
        #self.features = self.features[index_lt]
        #self.labels = self.labels[index_lt]
        #print(f'Pool shuffle') 
    
    def random_query_label(self):
        i = random.randint(0, len(self.labels)-1)
        assert i < len(self.labels)
        path = self.image_name_lt[i]
        label = self.labels[i]
        self.image_name_lt = np.delete(self.image_name_lt, i, 0)
        self.labels = np.delete(self.labels, i, 0)
        return path, label   

    def random_query_labels(self, K=1000):
        index_lt = [i for i in range(len(self.labels))]
        i_lt = random.sample(index_lt, K)
        assert len(i_lt) > 0
        image_name_lt, label_lt = [], []
        for i in i_lt:
            path = self.image_name_lt[i]
            label = self.labels[i]
            image_name_lt.append(path)
            label_lt.append(label)
        
        # delete rows
        self.image_name_lt = np.delete(self.image_name_lt, i_lt, 0)
        self.labels = np.delete(self.labels, i_lt, 0)
        return image_name_lt, label_lt

    def keep(self, K=1000):
        delete_num = len(self.labels) - K
        index_lt = [i for i in range(len(self.labels))]
        i_lt = random.sample(index_lt, delete_num)
 
        self.image_name_lt = np.delete(self.image_name_lt, i_lt, 0)
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

    def query_label(self, i):
        assert i < len(self.labels)
        image_name = self.image_name_lt[i]
        label = self.labels[i]
        self.image_name_lt = np.delete(self.image_name_lt, i, 0)
        self.labels = np.delete(self.labels, i, 0)
        return image_name, label

    def query_labels(self, i_lt):
        assert len(i_lt) > 0
        image_name_lt, label_lt = [], []
        for i in i_lt:
            image_name = self.image_name_lt[i]
            label = self.labels[i]
            image_name_lt.append(image_name)
            label_lt.append(label)
        
        # delete rows
        self.image_name_lt = np.delete(self.image_name_lt, i_lt, 0)
        self.labels = np.delete(self.labels, i_lt, 0)
        return image_name_lt, label_lt

    def get_image_name(self, i):
        assert i < len(self.labels)
        return self.image_name_lt[i]
    
    def get_all_image_name(self):
        return self.image_name_lt

