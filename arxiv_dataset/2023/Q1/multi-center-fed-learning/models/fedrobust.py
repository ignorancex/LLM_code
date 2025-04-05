# note to audience, fedrobust is based on the work of Median-of-means K-means,
# author is Camille Saumard, his email is camille.brunet@gmail.com
import importlib
import numpy as np
import os
import sys
import random

from sklearn.datasets import make_blobs
from math import modf, log
from scipy.spatial.distance import cdist
from kbmom.kmedianpp import euclidean_distances, kmedianpp_init

# tensorflow is required for our experiment, please install tf 1.5 not 2
import tensorflow as tf
import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server, MDLpoisonServer
from model import ServerModel
from utils.constants import DATASETS
from robust_main import KbMOM

from mlhead_utilfuncs import log_history, save_historyfile


STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def online(clients):
    """We assume all users are always online."""
    return clients

def save_model(server_model, dataset, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)


def print_metrics(metrics, weights):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 90th percentile %g' \
              % (metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))
    fom = [metrics[c][metric_names[0]] for c in sorted(metrics)]
    final = np.average(fom, weights=ordered_weights)
    return final

class Fedrobust_Trainer:
    
    def __init__(self, users, groups, train_data, test_data):
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.num_clients_per_round = 0
        self.config = []
        self.server = []
        self.all_clients = []
        self.iter = 0
        self.acc = 0.0
        self._bic = 0.0
        self._db_score = 0.0

        
    def model_config(self, config, dataset, my_model):   
        shared_model = my_model
        model_path = '%s/%s.py' % (dataset, shared_model)
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')
        model_path = '%s.%s' % (dataset, shared_model)

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientModel')  
        # Suppress tf warnings
        tf.logging.set_verbosity(tf.logging.WARN)

        # Create 2 models
        model_params = MODEL_PARAMS[model_path]
        model_params_list = list(model_params)
        model_params_list.insert(0, config["seed"])
        model_params_list[1] = config["lr"]
        model_params = tuple(model_params_list)
        tf.reset_default_graph()
        client_model = ClientModel(*model_params)


        self.config = config

        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        clients =  [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                    for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(clients)) 
        self.all_clients = clients
        
        if config['poisoning'] == True:
            num_agents = int(config["num_agents"] * len(clients)) 
            clients_per_round = config["clients-per-round"]            
            server_ = MDLpoisonServer(client_model, clients, num_agents, model_params_list[2], clients_per_round)
        else:
            # Create server
            server_ = Server(client_model)
        self.Server = server_            
        return clients, server_, client_model
    
    def fed_train(self, init_prms, client_in_block):
        #server.select_clients(possible_clients, num_clients=len(possible_clients))
        #c_ids, c_groups, c_num_samples = server.get_clients_info(None)
        eval_every = self.config["eval-every"]
        epochs_per_round = self.config['epochs']
        batch_size = self.config['batch-size']
        print("Start training on these clients:", client_in_block)
        block_clients = [self.all_clients[i] for i in client_in_block]
        sys_metrics, updates = self.Server.train_model(single_center=init_prms, num_epochs=epochs_per_round, batch_size=batch_size, minibatch=None, clients = block_clients, apply_prox=False)
        return sys_metrics, updates
         
    def fed_update(self):
        return self.Server.update_model_nowmode()
    
    def fed_test(self, nearest_centroid, robust):
        accs_ = [0] * len(set(nearest_centroid))
        for k,nc in enumerate(set(nearest_centroid)):
            cl_within_clus = []
            for i, v in enumerate(self.all_clients):
                if nearest_centroid[i] == nc:
                    cl_within_clus.append(self.all_clients[i])
            self.Server.model = robust.centers[nc]
            stat_metrics = self.Server.test_model(cl_within_clus)
            c_ids, c_groups, c_num_samples = self.Server.get_clients_info(cl_within_clus)
            accs_[k] = print_metrics(stat_metrics, c_num_samples)
        print("--- Acc: ",  np.average(accs_), " ---")
        log_history(self.iter+1, np.average(accs_), np.average(accs_), cl_within_clus)
        self.iter += 1
        self.acc = np.average(accs_)

    def evaluate(self, rbclf, config):
        print("Evaluation metrics below: ")
        bic = rbclf.sil_score()
        db_score = rbclf.DB_score()
        print("BIC: ", bic)
        print("DB_score:", db_score)
        return (bic, db_score)
        
    def begins(self, config, args):
        
        def shout(text):
            return text.upper()
        
        clients, server, client_model = self.model_config(config, args.dataset, 'cnn')  
        
        K = config["num-clusters"]
        num_rounds = config["num-rounds"]
        eval_every = config["eval-every"]
        epochs_per_round = config['epochs']
        batch_size = config['batch-size']
        clients_per_round = config["clients-per-round"]
        n_layers = config[args.dataset + "-num-layers"]
        n_blocks = config["num-blocks"]
        coef_ech = config["coff-each"]
        
        all_ids, all_groups, all_num_samples = server.get_clients_info(clients)
        # train all clients one round 
        _, tmp_data = server.train_model(None, 1, batch_size, None, clients, False)
        all_cl_models = [x[1] for x in tmp_data ]
        #print("the shape of a model is: ", len(all_cl_models[0][1]))
        print("last layer of a model is:", all_cl_models[0][n_layers].shape)
            
        # self.all_cl_models, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
        robust_helper = KbMOM(X = all_cl_models, K = K, max_iter = num_rounds, nbr_blocks = n_blocks, coef_ech = int(len(clients) * coef_ech) , quantile=0.5, init_type='kmedianpp', n_layers = n_layers)
        robust_helper.set_E_func(self.fed_train)
        robust_helper.set_M_func(self.fed_update)
        robust_helper.set_T_func(self.fed_test)
        print("*** Robust algorithm training started ***")
        robust_helper.fit(all_cl_models)
        #clients, server, client_model = self.model_config(config, args.dataset, 'cnn_prox')  
        centroids = robust_helper.predict(all_cl_models)
        self.fed_test(centroids, robust_helper)
        
        if config["benchmark"] == 1:
            vals = self.evaluate(robust_helper, config)
            self._bic = vals[0]
            self._db_score = vals[1]
        return self.acc
    
    def ends(self):
        save_historyfile()
        print("experiment of Fed Robust is finished.")
        return
