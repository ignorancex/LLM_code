import torch
import numpy as np
import time
from datasets import *
import torch.utils.data as data
import pdb
from tqdm import tqdm

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import ndcg_score

import concurrent.futures  as futures

from model import NCF, MF, NCFUser, NCFUserFedHM
from util import *
from client import *
from FakeRoast.FedOrchestrator import FedOrchestrator
import copy

import os
DEBUG=os.getenv("DEBUG")

def uniform_sample(n, k, compressions):
    '''
        create samples that are uniform across compressions
    '''
    compressions = np.array(compressions)

    cts = {}
    for c in np.unique(compressions):
        cts[c] = np.sum(compressions == c) / n * k

    sample = []
    for c in cts.keys():
        ids = np.argwhere(compressions == c).reshape(-1)
        sample = sample + list(np.random.choice(ids, int(np.ceil(cts[c]))))
    return sample


def get_compression_for_users(user_ids, cmp_str):
    '''
        helper to give concise input for compression config
    '''
    compressions = []
    if cmp_str.startswith('2:'):
        x = cmp_str.split(':')[1]
        powers = [int(a) for a in x.split('-')]
        for i in range(len(user_ids)):
            p = int(i * len(powers) / len(user_ids))
            compressions.append(1.0/(2**powers[p]))
    elif cmp_str.startswith('s2:'): # s2:0.9:0-1 # 90% is 1x and 10% is 2x
        prop = float(cmp_str.split(':')[1])
        x = cmp_str.split(':')[2]
        powers = [int(a) for a in x.split('-')]
        assert (len(powers) == 2)
        for i in range(len(user_ids)):
            p = int(float(i) / len(user_ids) > prop)
            compressions.append(1.0/(2**powers[p]))
    else:
        raise NotImplementedError
    return compressions

def get_server_model(args, total_users, total_items):
    '''
        create a full noncompressed model from args
    '''
    model = None
    if args.model == "NCF":
        model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    else:
        raise NotImplementedError
    return model

def get_client_model(args, total_users, total_items, compression, seed):
    ''' create a compressable model from args'''
    model = None
    if args.model == "NCF":
        if compression < 1.0:
            if args.fair_use_fedhm:
                model =   NCFUserFedHM(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 
                        compression, seed)

            else:

                model = NCFUser(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 
                        compression, seed)
        else:
            model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    else:
        raise NotImplementedError
    if DEBUG:
        print(model)
    return model


def setup_and_run_client(user_id, compression, device,
                full_train_dataset, full_test_dataset, args, server_params,
                total_users, total_items):
    # create client (dataset) 
    client = Client(user_id, compression, device,
                full_train_dataset, full_test_dataset, args)
    # create raw model
    raw_model = get_client_model(args, total_users, total_items, compression,
                          user_id if args.no_consistent_hashing else 2023 ) # hash for consistency
    # set model
    client.download(raw_model, server_params)
    
    if DEBUG:
        print(raw_model)
        print("compression", compression, count_parameters(raw_model))
    
    # train model
    client.train()
    
    return client

    
def fair(args):
      
    # data 
    DATAFILE="/home/apd10/FLCF/src/data/"+args.dataset+"/"
    train_file = DATAFILE + "train.txt"
    test_file = DATAFILE + "test.txt"

    full_train_dataset = NCFData(train_file, num_neg=args.num_neg)
    full_test_dataset = NCFData(test_file, num_neg=0) # no need for neg samples
        
    print("#data= ", full_train_dataset.__len__())
    print("#test= ", full_test_dataset.__len__())
    # get the assignments user to compression
    user_ids = full_train_dataset.unique_users
    compressions = get_compression_for_users(user_ids, args.fair_compressions)
    if args.fair_randomize_user :
        gen = np.random.RandomState(101)
        gen.shuffle(compressions)
    print("#compressions#", compressions)

    if args.fair_data_loss_compression is not None:
        print("Truncation of devices")
        print("OLD:")
        print("users", user_ids)
        print("compressions", compressions)
        max_compression = get_compression(args.fair_data_loss_compression)
        trunc_users = []
        trunc_compressions = []
        for i in range(len(user_ids)):
            if compressions[i] >= max_compression:
                trunc_users.append(user_ids[i])
                trunc_compressions.append(compressions[i])

        user_ids = trunc_users
        compressions = trunc_compressions
        print("NEW:")
        print("users", user_ids)
        print("compressions", compressions)
                
            
        

    total_users = max(full_test_dataset.user_max, full_train_dataset.user_max) + 1
    total_items = max(full_test_dataset.item_max, full_train_dataset.item_max) + 1
    
    total_cuda_count = torch.cuda.device_count()

    server_model = get_server_model(args, total_users, total_items)
    full_wts = FedOrchestrator.get_wts_full_single(server_model, is_global=False)

        
    earlystop = EarlyStop(args.early_trend_window, True, args.early_stop_thold)

    for t in tqdm(range(args.T)):
        #print("ROUND BEGIN:", t, flush=True)
        if args.fair_uniform_compression_sample:
            sample = uniform_sample(len(user_ids), args.K, compressions)
        else:
            sample = np.random.choice(np.arange(len(user_ids)), args.K)
        
        clients = []
        for i in range(args.K):
            s = sample[i]
            user_id = user_ids[s]
            compression = compressions[s]
            device = i % total_cuda_count
            r = setup_and_run_client(user_id, compression, device,
                                 full_train_dataset, full_test_dataset, args, full_wts,
                                 total_users, total_items)
            clients.append(r)
        torch.cuda.synchronize()

        local_norm_weights = np.array([c.train_dataset.__len__() for c in clients])
        local_norm_weights = local_norm_weights / np.sum(local_norm_weights)
        # issue in user embeddings TODO
        
        full_wts = FedOrchestrator.get_wts([c.model for c in clients], False, False, local_norm_weights)
        param_avg = copy.deepcopy(full_wts)
        FedOrchestrator.set_wts_full(server_model, param_avg)
        if t % args.eval_every == 0 and t > 0:
            server_ndcg = evaluate(server_model, full_train_dataset, full_test_dataset, 
                      total_items, 0, user_batch=10, full=True)

            print("\n[",t,"/",args.T, "ndcg",server_ndcg, flush=True)
            if earlystop.step(server_ndcg):
                  print("Early Stop Triggered")
                  break

