#!/usr/bin/env python
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
    Inspiration: https://github.com/studywolf/blog/tree/master/RL
"""

import sys, re, operator, math, time, random
random.seed(a = 0, version = 2)
from numpy import setdiff1d, rint, ndarray, uint64, array, logical_and, transpose
from itertools import chain, combinations
from os.path import exists
from glob import glob
from collections import OrderedDict
from scipy.spatial.distance import cosine, euclidean, correlation 
from scipy.stats import pearsonr
from pandas import DataFrame, concat, read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from utilities import fmax_score, bps2string, get_path_bag_weight, get_path, get_stacker_size, get_bps_rnd_and_weights
from yule import q_score, kappa_score

class Agent:
    def updatePosition(self, pos):  #, pos=None):
        self.pos = pos  
        self.explored = set()
        self.ens = (-1,)      

    def setWorld(self, world):
        self.world = world

    def calcState(self):
        return self.pos

    def possibleActions(self):
        return self.world.possible_actions(self.pos)

    def getEnsPerf(self):
        selected_ens = self.getEns()
        return self.world.get_perf(selected_ens)

    # return True if successfully moved in that direction
    def goInDirection(self, dir):
        target = dir
        self.pos = dir
        return True

    def findPolicy(self, start_vertex, end_vertex, policy = []):
        policy = policy + [start_vertex]
        if start_vertex == end_vertex:
            return policy
        actions = self.world.possible_actions(start_vertex)
        visited = self.getVisitedStates()
        explored = self.getExploredStates()
        valid = [element for element in actions if element in visited]
        possib = [self.ai.getQ(start_vertex, a) for a in valid]
        if len(possib) > 0:
            maxV = max(possib)
            for a in valid:
                if self.ai.getQ(start_vertex, a) == maxV:
                    extd_policy = self.findPolicy(a, end_vertex, policy)
        else:
            return policy
        return extd_policy
    
    def getVisitedStates(self):
        visited = []
        visited.append(self.world.start_node)
        for key, value in self.ai.q.items():
            state, action = key
            visited += tuple([action])
        return set(visited)

    def getExploredStates(self):
        return self.explored

    def getPolicy(self):
        return self.findPolicy(self.world.start_node, self.world.exit_node, [])

    '''def getEns(self): # original fuction
        ens = self.world.start_node
        path = self.getPolicy()
        for a in path:
            if self.world.perf[a] > self.world.perf[ens]:
                ens = a
        self.ens = ens
        return ens'''

    def getEns(self):
        ens = self.world.start_node
        policy = self.getPolicy()
        if self.strategy == "gsernd":
            #path/RL/apply/STACK_RESULTS/ORDER0/stack_bp10_seed0_L1Log.fmax
            path = get_path(self.world.bps[1])
            enforced_size = get_stacker_size(path, self.world.np, self.world.seed, self.world.fold)
            for curr_ens in policy:
                if (len(curr_ens) == enforced_size):
                    ens = curr_ens
        else:
            for curr_ens in policy:
                if self.world.perf[curr_ens] > self.world.perf[ens]:
                    ens = curr_ens
        self.ens = ens
        return ens

    def utility(self):
        policy = self.getPolicy()
        utility = []
        for i in range(0, len(policy) - 1):
            utility += [self.ai.getQ(policy[i], policy[i+1])]
        return utility  

    def getTestPerf(self, node):
        test_perf = -1
        if node == (0,): # probably RL_pessimistic was trained for too little time and all ensembles of size 2 are performing worse than the individual base predictors
            force_picks = [val for val in [tuple([y]) for y in range(1, self.world.np+1)] if val in self.getExploredStates()]
            select_from = {x:self.world.perf[x] for x in force_picks}
            index = max(select_from, key=select_from.get)[0]             
            node = tuple([index])
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]
        elif (len(node) == 1):
            index = node[0]
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]
        else:
            index = list(node)[0]
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]  
            for index in list(node)[1:]:
                test_pred += (self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index])
        #print("denom = ", denom)
        denom = sum([self.world.bps_weight[index] for index in list(node)])
        test_pred /= denom
        #print(test_perf[:3])
        test_labels = self.get_fold_probs(node[0], 'test', 'label') 
        #print(" - - - -")
        if self.world.metric == "fmax":
            test_perf   = fmax_score(test_labels, test_pred)
        if self.world.metric == "f1score":
            test_perf   = f_score(test_labels, test_pred)
        if self.world.metric == "auROC":
            test_perf   = roc_auc_score(test_labels, test_pred)
        if self.world.metric == "auPRC": 
            test_perf   = average_precision_score(test_labels, test_pred)
        if self.world.metric == "pearsonr":
            test_perf   = float(pearsonr(test_labels, test_pred)[0])

    def select_most_diverse(self, state, actions):
        if state == self.world.start_node or actions[0] == ('exit',):
            action = random.choice(actions)
        else:
            select_from = {x:self.world.cosine_dist(state, x) for x in actions}
            keys = [key for key, value in select_from.items() if value == max(select_from.values())]
            #print select_from
            #print keys
            action = random.choice(keys) #randomly if max ties
        #print "from %s we selected action = %s\n" % ((state,), (action,))
        return action

    def select_most_diverse_euclidean(self, state, actions):
        if state == self.world.start_node or actions[0] == ('exit',):
            action = random.choice(actions)
        else:
            select_from = {x:self.world.euclidean_dist(state, x) for x in actions}
            keys = [key for key, value in select_from.items() if value == max(select_from.values())]
            action = random.choice(keys) #randomly if max ties
        #print "selected action = %s" % (action,)
        return action

    def select_most_diverse_correlation(self, state, actions):
        if state == self.world.start_node or actions[0] == ('exit',):
            action = random.choice(actions)
        else:
            select_from = {x:self.world.correlation_dist(state, x) for x in actions}
            keys = [key for key, value in select_from.items() if value == max(select_from.values())]
            action = random.choice(keys) #randomly if max ties
        #print "selected action = %s" % (action,)
        return action

    def select_most_diverse_yule(self, state, actions):
        if state == self.world.start_node or actions[0] == ('exit',):
            action = random.choice(actions)
        else:
            select_from = {x:self.world.yule_dist(state, x) for x in actions}
            keys = [key for key, value in select_from.items() if value == max(select_from.values())]
            action = random.choice(keys) #randomly if max ties
	#print "selected action = %s" % (action,)
        return action

    def select_most_diverse_kappa(self, state, actions):
        if state == self.world.start_node or actions[0] == ('exit',):
            action = random.choice(actions)
        else:
            select_from = {x:self.world.kappa_dist(state, x) for x in actions}
            keys = [key for key, value in select_from.items() if value == max(select_from.values())]
            action = random.choice(keys) #randomly if max ties
        #print "selected action = %s" % (action,)
        return action


    def get_fold_probs(self, model_idx, set, col):
        assert set in ['valid', 'test']
        assert col in ['label', 'prediction']
        path, bag, x = get_path_bag_weight(self.world.bps[model_idx])
        test_df      = read_csv('%s/%s-b%i-f%i-s%i.csv.gz' % (path, set, bag, self.world.fold, self.world.seed), skiprows = 1, compression = 'gzip')
        test_pred    = test_df[col]	    
        return test_pred

    def getBestBPPerf_onTest(self):
        best_bp = self.world.getBestBP()
        bp_test = self.getTestPerf(best_bp)
        return bp_test

    def getFEPerf_onTest(self):
        fe_node = tuple(list(range(1, self.world.np+1)))
        fe_test = self.getTestPerf(fe_node)
        return fe_test



class World:
    def __init__(self, np, starting_point, exit_val, input_dir, seed, fold, rule, metric):
        self.np          = np
        self.start_node  = tuple([-1])
        self.input_dir 	 = input_dir
        self.seed        = seed
        self.fold        = fold
        self.metric      = metric
        self.rule        = rule
        self.graph       = {}
        self.perf        = {}
        self.div         = {}
        self.bps         = {}
        self.bps_weight  = {}
        self.predictors  = {}
        self.bps_weighted_pred_df = DataFrame(columns=range(0, (np + 1))) #already weighted! (#cumulative)
        self.cwan        = () #cumulative weighted average numerator; used only for rule == "WA"
        self.initialize_bps()
        
        #connecting start_node with all the individual base predictors
        if starting_point == '0':
            self.start_node = tuple([0])
            self.perf[self.start_node] = float(0)
            bps = []
            for i in range(1, np+1):
                bps.append(tuple([i])) 
            self.graph[self.start_node] = bps
        else:
            self.start_node = max(self.perf, key=self.perf.get)
            self.cwan = self.start_node
            self.bps_weighted_pred_df[self.np+1] = self.bps_weighted_pred_df[self.start_node[0]] 
            self.graph[self.start_node] = []

        #connecting the full ensemble with the exit_node (intermediate nodes will be added as explored by agent)
        self.exit_node = tuple(['exit'])
        self.perf[self.exit_node] = exit_val
        self.graph[tuple(list(range(1, np+1)))] = [self.exit_node]



    def cosine_dist(self, node, action):
        index      = list(node)[0]
        node_preds = self.bps_weighted_pred_df[index]
        for index in list(node)[1:]:
            node_preds   = node_preds.add(self.bps_weighted_pred_df[index])
        bp = setdiff1d(action, node)[0]
        action_preds = node_preds + self.bps_weighted_pred_df[bp]
        cosine_distance = cosine(node_preds, action_preds)
        #print ("%r %r %f" % (node, action, cosine_distance))
        return cosine_distance

    def euclidean_dist(self, node, action):
        index      = list(node)[0]
        node_preds = self.bps_weighted_pred_df[index]
        for index in list(node)[1:]:
            node_preds   = node_preds.add(self.bps_weighted_pred_df[index])
        bp = setdiff1d(action, node)[0]
        action_preds = node_preds + self.bps_weighted_pred_df[bp]
        euclidean_distance = euclidean(node_preds, action_preds)
        #print ("%r %r %f" % (node, action, euclidean_distance))
        return euclidean_distance

    def correlation_dist(self, node, action):
        index      = list(node)[0]
        node_preds = self.bps_weighted_pred_df[index]
        for index in list(node)[1:]:
            node_preds   = node_preds.add(self.bps_weighted_pred_df[index])
        bp = setdiff1d(action, node)[0]
        action_preds = node_preds + self.bps_weighted_pred_df[bp]
        correlation_distance = correlation(node_preds, action_preds)
        #print ("%r %r %f" % (node, action, correlation_distance))
        return correlation_distance

    def yule_dist(self, node, action):
        index      = list(node)[0]
        node_preds = self.bps_weighted_pred_df[index]
        denom = 0.0
        for index in list(node)[1:]:
            node_preds  = node_preds.add(self.bps_weighted_pred_df[index])
            denom       = sum([self.bps_weight[index] for index in list(node)])
        bp = setdiff1d(action, node)[0]
        action_preds = (node_preds + self.bps_weighted_pred_df[bp])/(denom + self.bps_weight[bp])

        THRESHOLD = 0.5
        labels = (self.bps_weighted_pred_df['label']).astype(uint64)
        a = (action_preds >= THRESHOLD).astype(uint64)
        b = (node_preds >= THRESHOLD).astype(uint64)
        a_labeled = [1 if i == j else 0 for i, j in zip(a, labels)]
        b_labeled = [1 if i == j else 0 for i, j in zip(b, labels)]

        yule_distance = 1-q_score(array(a), array(b))
        #print ("yule(%r, %r) = %f" % (node, action, yule_distance))
        return yule_distance

    def kappa_dist(self, node, action):
        index      = list(node)[0]
        node_preds = self.bps_weighted_pred_df[index]
        denom = 0.0
        for index in list(node)[1:]:
            node_preds  = node_preds.add(self.bps_weighted_pred_df[index])
            denom       = sum([self.bps_weight[index] for index in list(node)])
        bp = setdiff1d(action, node)[0]
        action_preds = (node_preds + self.bps_weighted_pred_df[bp])/(denom + self.bps_weight[bp])

        THRESHOLD = 0.5
        labels = (self.bps_weighted_pred_df['label']).astype(uint64)
        a = (action_preds >= THRESHOLD).astype(uint64)
        b = (node_preds >= THRESHOLD).astype(uint64)
        a_labeled = [1 if i == j else 0 for i, j in zip(a, labels)]
        b_labeled = [1 if i == j else 0 for i, j in zip(b, labels)]

        kappa_distance = kappa_score(array(a), array(b))
        #print ("kappa(%r, %r) = %f" % (node, action, kappa_distance))
        return kappa_distance

    def compute_performance(self, true_labels, predictions):
        if self.metric == "fmax":
            performance = fmax_score(true_labels, predictions)
        if self.metric == "f1score":
            performance = f_score(true_labels, predictions)
        if self.metric == "auROC":
            performance = roc_auc_score(true_labels, predictions)
        if self.metric == "auPRC": 
            performance = average_precision_score(true_labels, predictions)
        if self.metric == "pearsonr":
            performance = float(pearsonr(true_labels, predictions)[0])
        return performance


    def get_perf(self, node):
        if (node not in self.perf):
            self.comp_perf(node)
        return self.perf[node]


    def comp_perf(self, node):
        if (self.rule == "WA"):
            if (len(node) > len(self.cwan) and len(setdiff1d(node, self.cwan)) == 1 and node != ['exit']):
                bp2add = setdiff1d(node, self.cwan)
                bp = bp2add[0] 
                self.bps_weighted_pred_df[(self.np+1)] = self.bps_weighted_pred_df[(self.np+1)].add(self.bps_weighted_pred_df[bp])
                self.cwan = node
                denom = sum([self.bps_weight[index] for index in list(node)])
                y_score = self.bps_weighted_pred_df[(self.np + 1)]
                y_score /= denom # new y_score variable because col_np+1 needs to be the numerator
                performance = self.compute_performance(self.bps_weighted_pred_df['label'], y_score) 
                self.perf[node] = performance
                if (len(node) == self.np):
                    self.reset_cwan_col()
            else:  #de novo...
                index = list(node)[0]
                y_score = self.bps_weighted_pred_df[index]
                for index in list(node)[2:]:
                    y_score += self.bps_weighted_pred_df[index]
                self.bps_weighted_pred_df[(self.np+1)] = y_score
                self.cwan = node
                denom = sum([self.bps_weight[index] for index in list(node)])
                y_score /= denom
                performance = self.compute_performance(self.bps_weighted_pred_df['label'], y_score) 
                self.perf[node] = performance
        if (self.rule == "L2"):
            num_predictors = len(node)
            train_dataset = self.bps_weighted_pred_df[list(node)]
            train_labels = self.bps_weighted_pred_df['label']
            stacker = LogisticRegression(random_state = self.seed, penalty='l2', solver = 'liblinear')
            stacker.fit(train_dataset, train_labels.values.ravel())
            test_labels = train_labels.values.ravel()
            y_score = DataFrame(stacker.predict_proba(train_dataset)[:,1])
            performance = self.compute_performance(test_labels, y_score)
            self.perf[node] = performance
            
    def reset_cwan_col(self):
        if (self.start_node) == (0,):
            self.bps_weighted_pred_df[(self.np+1)] = 0
            self.cwan = ()
        else:
            self.bps_weighted_pred_df[self.np+1] =  self.bps_weighted_pred_df[self.start_node[0]]
            self.cwan = self.start_node

    def getBestBP(self):
        return (max(self.bps_weight, key=self.bps_weight.get),)
    
    def initialize_bps(self):
        if (self.rule == "WA"):
            self.initialize_bps_wa()
        elif (self.rule == "L2"):
            self.initialize_bps_l2()

    def initialize_bps_l2(self):
        self.predictors, self.bps_weight = get_bps_rnd_and_weights(self.input_dir, self.seed, self.metric, self.np, self.fold)       
        for i in range(len(self.predictors)):
            index = i + 1
            self.perf[tuple([index])] = float(self.predictors[i].split(",")[1])
            self.bps[index] = self.predictors[i]
            path, bag = get_path_bag_weight(self.predictors[i])[:2]
            valid_df = read_csv('%s/valid-b%i-f%i-s%i.csv.gz' % (path, bag, self.fold, self.seed), skiprows = 1, compression = 'gzip')
            valid_label = valid_df['label']
            valid_pred = valid_df['prediction']
            self.bps_weighted_pred_df[index] = valid_pred #no multiplication by any weight
        self.bps_weighted_pred_df[0] = valid_label
        self.bps_weighted_pred_df.rename(columns={0:'label'}, inplace=True)


    def initialize_bps_wa(self):
        self.predictors, self.bps_weight = get_bps_rnd_and_weights(self.input_dir, self.seed, self.metric, self.np, self.fold)       
        for i in range(len(self.predictors)):
            index = i + 1
            self.perf[tuple([index])] = float(self.predictors[i].split(",")[1])
            self.bps[index] = self.predictors[i]
            path, bag = get_path_bag_weight(self.predictors[i])[:2]
            valid_df = read_csv('%s/valid-b%i-f%i-s%i.csv.gz' % (path, bag, self.fold, self.seed), skiprows = 1, compression = 'gzip')
            valid_label = valid_df['label']
            valid_pred = valid_df['prediction']
            self.bps_weighted_pred_df[index] = valid_pred * self.bps_weight[index] 
        self.bps_weighted_pred_df[0] = valid_label
        self.bps_weighted_pred_df.rename(columns={0:'label'}, inplace=True)
              

    def possible_actions(self, pos):
        if pos == tuple([0]) or pos == tuple(list(range(1, self.np+1))):
            return self.graph[pos]
        else:
            predictors = range(1, self.np+1)
            possibilities = [(pos + (i,)) for i in predictors if i not in pos]
            pos_act = [tuple(sorted(p)) for p in possibilities]
            return pos_act
 
    def printGraph(self):
        print("\nworld.graph:")
        for key, value in sorted(self.graph.items()):
            print ('\tNode %r --> %r' % (key, value))
        print ("* * * * *\n")

    def printPerf(self):
        print("\nworld.perf:")
        for key, value in sorted(self.perf.items()):
            print ('\t%s[%r] = %.6f' % (self.metric, key, value))
        print ("* * * * *\n")

    def printClassifiers(self):
        print("\nworld.bps:")
        for key, value in sorted(self.bps.items()):
            print ('\tNode %r :: %s' % (key, value))
        print ("* * * * *\n")

    def getGraph(self):
        return self.graph

