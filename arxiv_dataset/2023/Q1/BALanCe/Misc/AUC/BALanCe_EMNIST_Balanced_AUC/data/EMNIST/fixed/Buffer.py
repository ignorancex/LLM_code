import numpy as np
import collections
import random

class Buffer():
    def __init__(self):
        self.features = []
        self.labels = []
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
    
    def print_class_num(self):
        counter = collections.Counter(list(self.labels.squeeze()))
        print(f'Class num')
        print(counter)

    def __init__from_pt(self, pt_file):
        print('Buffer init from pt file')
        self.features, self.labels = torch.load(pt_file)
        self.features = self.features.cpu().numpy()
        self.labels = self.labels.cpu().numpy()

    def add_row(self, feat, label):
        if len(self.features) == 0:
            self.features = np.array([feat])
            self.labels = np.array([label])
        else:
            feat = np.array([feat])
            self.features = np.vstack((self.features, feat))
            self.labels = np.vstack((self.labels, label))
 
    def add_rows(self, feat_lt, label_lt):
        assert len(feat_lt) == len(label_lt)
        for i in range(len(feat_lt)):
            feat = feat_lt[i]
            label = label_lt[i]
            self.add_row(feat, label)

    def balance_class(self):
        class2index = {}
        print(f'balance class')
        print(f'sample num before: {self.labels.shape}')

        for i in range(len(self.labels)):
            label = self.labels[i][0]
            if label in class2index:
                class2index[label].append(i)
            else:
                class2index[label] = [i]

        class_num = [len(class2index[label]) for label in class2index]
        print(f'class num before: {class_num}')

        samples_each_class = min(class_num)
        index_lt  = []
        for label in class2index:
            t_lt = class2index[label]
            random.shuffle(t_lt)
            index_lt += t_lt[:samples_each_class]

        index_lt = np.array(index_lt)

        self.features = self.features[index_lt]
        self.labels = self.labels[index_lt]
        
        print(f'sample num after: {self.labels.shape}')




