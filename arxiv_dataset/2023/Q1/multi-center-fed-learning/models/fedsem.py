import importlib
import numpy as np
import os
import sys
import time
import pickle
import copy

import random
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from baseline_constants import ACCURACY_KEY



import metrics.writer as metrics_writer
STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'


# below import fedmc necessary lib
from mh_constants import VARIABLE_PARAMS
from mlhead_clus_server import Mlhead_Clus_Server
from mlhead_utilfuncs import get_tensor_from_localmodels,count_num_point_from, log_history, save_historyfile

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from mlhead_client import Client
from server import Server, MDLpoisonServer
from model import ServerModel
from fedprox_optimizer import PerturbedGradientDescent
from kbmom.utils import loglikelihood, BIC
from sklearn.metrics import davies_bouldin_score, silhouette_score


def mlhead_print_totloss(k, eval_every, rounds, prefix, accuracy, cluster, stack_list, client_list):
    print("acc list dimension: ", accuracy.ndim)
#     micro_acc = np.mean(accuracy)
    micro_acc = np.max(accuracy)
    print('micro (with weight) test_%s: %g' % (prefix, micro_acc) )
    #save_metric_csv(k+1, micro_acc, stack_list)
    macro_acc = np.mean([stack_list[cl] for cl in stack_list])
    print('macro (overall) test_%s: %g' % (prefix, macro_acc) )
    log_history(k+1, micro_acc, macro_acc, client_list)
    return micro_acc


def mlhead_print_stats(
    num_round, server, clients, num_samples, config, stack_list, prepare_test, acc_array = None):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')

    test_stat_metrics = server.test_model(clients, set_to_use='test')
    for k in test_stat_metrics:
        stack_list[k] = test_stat_metrics[k][ACCURACY_KEY]
  
    test_acc =  print_metrics(test_stat_metrics, num_samples, prefix='test_')

    # We also wants to evaluate a macro value (accuracy & loss)
    c = max(config["num-clusters"], 1)
    if acc_array is not None:
        return np.append([acc_array], [test_acc])
    else:
        return np.array(test_acc)

def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    
    micro_acc = metric_names[0]
    miacc_metric = [metrics[c][micro_acc] for c in sorted(metrics)]
    to_ret = np.average(miacc_metric, weights=ordered_weights)
    for metric in metric_names[:2]:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))

    return to_ret

class Fedsem_Trainer():
    
    def __init__(self, users, groups, train_data, test_data): 
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.last_round = 0.
        self._bic = 0.
        self._db_score = 0.


    def center_init(self, num_clusters, client_model):
        for i in range(num_clusters):
            if os.path.exists("cnn-C{}.pb".format(i)):
                with open("cnn-C{}.pb".format(i), "rb") as f:
                    self.center_models[i] = pickle.load(f)
            else:
                self.center_models[i] = copy.deepcopy(client_model.get_params())
                
    def default_cluster(self):
        group = len(self.clients), self.clients
        default_list = list()
        default_list.append(group)
        return default_list
    
    def clustering_function(self, points):
        start_time = time.time()
        # comment one clustering to use another
        # this is outlier ones
        iter_stop = 0
#         learned_cluster = self.mlhead_cluster.outlier_clustering(points)
#         while (self.mlhead_cluster.is_unbalanced_clus(learned_cluster)) and (iter_stop < 2):
#             iter_stop += 1
#             learned_cluster = self.mlhead_cluster.outlier_clustering(points)
                   
        learned_cluster = self.mlhead_cluster.run_clustering(points)     
        end_time = time.time() - start_time
        self.kmeans_cost.append(end_time) 
        return learned_cluster
    
    def evaluate(self, points):
        data = [points[k] for k in points]
        label = self.mlhead_cluster._clusterModel.assign_clusters(data)
        print("Evaluation metrics below: ")
        
        _sil_score = silhouette_score(data, label, metric='euclidean')
        db_score = davies_bouldin_score(data, label)
        print("BIC: ", _sil_score)
        print("DB_score:", db_score)       
        return (_sil_score, db_score)
        
    
    def model_config(self, config, dataset, my_model, seed):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        model_path = '%s/%s.py' % (dataset, my_model)
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')
        model_path = '%s.%s' % (dataset, my_model)

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientModel')


        model_params = MODEL_PARAMS[model_path]
        if config["lr"] != -1:
            model_params_list = list(model_params)
            model_params_list.insert(0, config["seed"])
            model_params_list[1] = config["lr"]
            model_params = tuple(model_params_list)

        # Create client model, and share params with server model
        tf.reset_default_graph()
        client_model = ClientModel(*model_params)
        num_clusters = config["num-clusters"]
        

        
        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        clients =  [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                    for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(clients)) 
        
        if config['poisoning'] == True:
            num_agents = int(config["num_agents"] * len(clients)) 
            clients_per_round = config["clients-per-round"]
            server = MDLpoisonServer(client_model, clients, num_agents, model_params[2], clients_per_round)
        else:
            # Create server
            server = Server(client_model)            

        client_ids, client_groups, all_num_samples = server.get_clients_info(clients)
        # Create our SEM modeling agl
        print('--- Random Initialization ---')

        print("--- Do training and initilized clusting server---")
        self.mlhead_cluster = Mlhead_Clus_Server(client_model, dataset, "cnn", num_clusters, len(clients), seed)
        self.mlhead_cluster.select_clients(seed,  clients)
        self.center_models = [None] * num_clusters
        self.center_init(num_clusters, client_model)
        
        return clients, server, client_model, all_num_samples

    def begins(self, config, args):
        clients, server, client_model, all_num_samples = self.model_config(config, args.dataset, 'cnn', config["seed"])         
        """
            A trainer different from the baseline
            Then all need to do is replace different optimizer
        """
        num_clusters = config["num-clusters"]
        num_rounds = config["num-rounds"]
        eval_every = config["eval-every"]
        epochs_per_round = config['epochs']
        batch_size = config['batch-size']
        clients_per_round = config["clients-per-round"]
        update_head_every = config['update-center-every']
        n_layers = config[args.dataset + "-num-layers"]
        
        print("----- Multi-center Federated Training -----")
        prev_score = None
        self.kmeans_cost = []
        for k in range(num_rounds):
            best_kept = None
            stack_list = {}
            client_list = {}
            if prev_score is None: # This is the first iteration
                if num_clusters == -1 :
                    write_file = False
                    learned_cluster = self.default_cluster()
                else:
                    print("----- First time center rendering  -----")
                    write_file = True
                               
                    c_wts = self.mlhead_cluster.get_init_point_data()
                    learned_cluster = self.clustering_function(c_wts)
                    prev_score = len(c_wts)

            #print('--- Round %d of %d: Training %d Clients ---' % (k + 1, num_rounds, clients_per_round))
            print('--- Round %d of %d: Training assigned to %d Cluster ' % (k + 1, num_rounds, 
                                                                            len(learned_cluster)), 
                "<", count_num_point_from(learned_cluster), "> ---")

            joined_clients = dict()
            for c_idx, group in enumerate(learned_cluster):
                server = Server(None)
                if group[0] <= 1:
                    print("Skip cluster %d as number of client not enough" % c_idx)
                    continue
                # if not explicit clients per round given, then default all
                # clients in this group participarte training
                if clients_per_round != -1: 
                    num = min(clients_per_round, group[0])
                    active_clients = np.random.choice(group[1], num, replace=False)
                else:
                    active_clients = group[1]
                
                client_list[c_idx] = [c.id for c in active_clients ]
                c_ids, c_groups, c_num_samples = server.get_clients_info(active_clients)
                sys_metrics, updates = server.train_model(self.center_models[c_idx], 
                                                          num_epochs=epochs_per_round, 
                                                          batch_size=batch_size, 
                                                          clients = active_clients)
                
                for c, up in zip(active_clients, updates):
                    joined_clients[c.id] = up[1]
                # Thinking how to do a distance as weight averging
                if  ("mode" not in config) or (config["mode"])  == "no_size":
                    self.center_models[c_idx] = server.update_model_wmode()
                else:
                    server.update_model_nowmode()

                #_, _, client_num_samples = server.get_clients_info(active_clients)
                best_kept = mlhead_print_stats(k + 1, server, clients, all_num_samples, config, 
                                                   stack_list, True, best_kept)
            # end iterate clusters
            self.last_round = mlhead_print_totloss(
                k, eval_every, num_rounds, "accuracy",  best_kept, learned_cluster, stack_list, client_list)

            # Update the center point when k = local training * a mulitplier
            if not num_clusters == -1 and not k == (num_rounds -1) and (k + 1) % update_head_every == 0:
                tmp = {cli_id: joined_clients[cli_id][n_layers].flatten()  for cli_id in joined_clients}
                c_wts.update(tmp)
                learned_cluster = self.clustering_function(c_wts) # cwts is N (clients) x x_dimensions
                joined_clients.clear()
                print("----- center update performed -----")
                prev_score = len(c_wts)
                
        client_model.close()
        
        if config["benchmark"] == 1:
            vals = self.evaluate(c_wts)
            self._bic = vals[0]
            self._db_score = vals[1]        
        return self.last_round
                

    def ends(self):
        save_historyfile()
        print("experiment of Fedsem finished.")
        return
        # save history file
        
        # Save server model        
#         ckpt_path = os.path.join('checkpoints', args.dataset)
#         if not os.path.exists(ckpt_path):
#             os.makedirs(ckpt_path)

#         for i, server in enumerate(self.head_server_stack):
#             # {}-K{}-C{}, K stands for number of clusters and C stands for ith center
#             save_path = server.save_model(os.path.join(ckpt_path, '{}-K{}-C{}.ckpt'.format(args.model, args.num_clusters, i+1)))
#         print('Model saved in path: %s' % save_path)
#        print('{} rounds kmeans used {:.3f}'.format(self.num_rounds, np.average(self.kmeans_cost, weights=None)))
#         for i, server in enumerate(self.center_models):
#             head_weights = server
#             with open('./{}-C{}.pb'.format(args.model, i), 'wb+') as f:
#                 pickle.dump(head_weights, f)
                
#         for s in self.head_server_stack:
#             s.close_model()
