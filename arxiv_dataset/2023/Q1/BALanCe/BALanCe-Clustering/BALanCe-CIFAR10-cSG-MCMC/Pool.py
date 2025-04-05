import numpy as np
import csv
import random
import torch 
import collections
import sys
import json
import glob
import os

class Pool():
    def __init__(self, folder, label_file, bitmap=None):
        #label_file = folder + '/trainLabels.csv'
        path_lt, label_lt = [], []
        fin = open(label_file, 'r')
        count = 0
        while True:
            line = fin.readline().strip()
            if not line:
                break
            count += 1
            if count == 1:
                continue
            
            array = line.split('\t')
            image_name = array[0]
            label = int(array[1])
        
            image_path = folder + '/' + image_name
            path_lt.append(image_path)
            label_lt.append(label)

        fin.close()
        
        self.path_lt = path_lt
        self.labels = label_lt
        '''
        self.cluster2pos = {}
        self.pos2cluster = {}
        self.cluster2acc = {}
        '''
        if bitmap != None:
            if isinstance(bitmap, str):
                with open(bitmap, 'r') as f:
                    self.bitmap = json.load(f)
            elif isinstance(bitmap, list):
                self.bitmap = bitmap
            else:
                print(f'Not identify bitmap: {bitmap}')
                sys.exit() 
        else:
            self.bitmap = [i for i in range(len(self.labels))]
        
        print(f'Pool size: {len(self.bitmap)}')
    
    def size(self):
        return len(self.bitmap)

    def query_label(self, pos):
        if pos in self.bitmap:
            self.bitmap.remove(pos)
            '''
            if pos in self.pos2cluster:
                cluster_label = self.pos2cluster[pos]
                self.cluster2pos[cluster_label].remove(pos)
                self.pos2cluster.pop(pos)
            else:
                print(f'Pos: {pos} is not in pos2cluster')
                sys.exit()
            '''
        else:
            print(f'Pos: {pos} is not in bitmpa')
            sys.exit()

    def query_labels(self, pos_lt):
        for pos in pos_lt:
            if pos not in self.bitmap:
                print(f'Pos: {pos} is not in bitmap')
                sys.exit()

        for pos in pos_lt:
            self.bitmap.remove(pos)
            '''
            if pos in self.pos2cluster:
                cluster_label = self.pos2cluster[pos]
                self.cluster2pos[cluster_label].remove(pos)
                self.pos2cluster.pop(pos)
            else:
                print(f'Pos: {pos} is not in pos2cluster')
                sys.exit()
            '''

    def add_sample(self, pos):
        if pos in self.bitmap:
            print(f'Pos: {pos} already in bitmap')
            sys.exit()
        else:
            self.bitmap.append(pos)

    def add_samples(self, pos_lt):
        for pos in pos_lt:
            if pos in self.bitmap:
                print(f'Pos: {pos} already in bitmap')
                sys.exit()
        self.bitmap += pos_lt



