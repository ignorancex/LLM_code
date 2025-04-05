import numpy as np
import csv
import random
import torch 
import collections
import scipy

class Pool():
    def __init__(self, x_file, y_file):
        #self.features, self.labels = torch.load(pt_file)
        #self.features = self.features.cpu().numpy()
        #self.labels = self.labels.cpu().numpy()
        
        features = []
        labels = []
        ids = []
        '''
        with open(x_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                miss = False
                for i in range(0, len(row)):
                    if row[i] == '?':
                        miss = True
                        break

                if miss == True:
                    continue

                #ids.append(row[0])
                print(row)
                feat = [float(row[i]) for i in range(2, len(row))]
                features.append(feat)
        '''
        fin = open(x_file, 'r') 
        while True:
            line = fin.readline().strip()
            if not line:
                break
            array = line.split()
            feat = [float(array[i]) for i in range(len(array))]
            features.append(feat)
        fin.close()

        fin = open(y_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            label = int(line)-1
            labels.append(label)
        fin.close()

        features = np.array(features, np.float32)
        
        features = scipy.stats.zscore(features, axis=0, nan_policy='omit')
        
        #print(np.mean(features, axis=0))
        self.features = features
        self.labels = np.array(labels)
        self.ids = ids
        
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

