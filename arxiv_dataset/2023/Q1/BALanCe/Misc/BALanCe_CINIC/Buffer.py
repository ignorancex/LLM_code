import numpy as np
import torch
from Pool import class2label
import os

class Buffer():
    def __init__(self):
        self.image_name_lt = []
        self.labels = []
        self.image_name_lt = np.array(self.image_name_lt)
        self.labels = np.array(self.labels)
    
    def add_from_pt(self, pt_file):
        print('Buffer init from image path list file')
        self.image_name_lt = []
        self.labels = []
        fin = open(pt_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            path = os.path.join('./data/', line)
            self.image_name_lt.append([path])
            class_name = os.path.basename(os.path.dirname(line))
            label = class2label[class_name]
            self.labels.append([label])

        self.image_name_lt = np.array(self.image_name_lt)
        self.labels = np.array(self.labels)
        
        print(f'Pool size: {len(self.image_name_lt)}')

    def add_row(self, image_name, label):
        if len(self.image_name_lt) == 0:
            self.image_name_lt = np.array([image_name])
            self.labels = np.array([label])
        else:
            self.image_name_lt = np.vstack((self.image_name_lt, image_name))
            self.labels = np.vstack((self.labels, label))
 
    def add_rows(self, image_name_lt, label_lt):
        assert len(image_name_lt) == len(label_lt)
        for i in range(len(image_name_lt)):
            image_name = image_name_lt[i]
            label = label_lt[i]
            self.add_row(image_name, label)


