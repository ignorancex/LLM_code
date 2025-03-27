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
from util import *

from model import NCF, MF, NCFUser

def get_model(args, total_users, total_items, compression, seed):
    ''' create a compressable model from args'''
    model = None
    if args.model == "NCF":
        if compression < 1.0:
            model = NCFUser(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 
                        compression, seed)
        else:
            model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    else:
        raise NotImplementedError
    return model


def central(args):
    print(args)

    DATAFILE="/home/apd10/FLCF/src/data/"+args.dataset+"/"
    train_file = DATAFILE + "train.txt"
    test_file = DATAFILE + "test.txt"

    full_train_ds = NCFData(train_file, num_neg=args.num_neg)
    full_test_ds = NCFData(test_file, num_neg=0) # no need for neg samples
        
    full_train_loader = data.DataLoader(
                dataset=full_train_ds, batch_size=args.batch_size, shuffle=False
                )
    full_test_loader = data.DataLoader(
                dataset=full_test_ds, batch_size=1024, shuffle=False
                )
    print("#data= ", full_train_ds.__len__())
    print("#test= ", full_test_ds.__len__())

    total_users = max(full_test_ds.user_max, full_train_ds.user_max) + 1
    total_items = max(full_test_ds.item_max, full_train_ds.item_max) + 1
    item_dim = args.emb_dim
    user_dim = args.emb_dim
    total_cuda_count = torch.cuda.device_count()
    
    if args.model == "NCF":
        model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    elif args.model == "MF":
        model = MF(total_users, total_items, args.emb_dim)
    elif args.model == "NCFU":
        model = NCFUser(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 0.5, 2023)

    model = get_model(args, total_users, total_items, get_compression(args.central_compression), args.seed)
    print(model)
    print("count of parameters", count_parameters(model))

    torch.manual_seed(10101)
    np.random.seed(10111)

    device = torch.device("cuda:" + str(0))
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    earlystop = EarlyStop(args.early_trend_window, True, args.early_stop_thold)
    print("epochs", args.T * args.E)
    for epoch in tqdm(range(args.T*args.E)):
        model = model.to(device)
        model.train()
        losses = []
        norms = []
        for batch_idx, (users, pos, neg) in enumerate(full_train_loader):
            model.zero_grad()
            loss1, loss2 = model.bpr_loss(torch.LongTensor(users).cuda(), torch.LongTensor(pos).cuda(), torch.LongTensor(neg).cuda())
            loss = loss1 + args.reg_lambda * loss2
            loss.backward()
            optimizer.step()
            losses.append(float(loss1.detach().cpu()))
            norms.append(float(loss2.detach().cpu()))
        #if epoch > 0 and epoch % 1 == 0:
        #    evaluate(model, full_train_loader, full_test_loader, total_items, device)
        if epoch % args.eval_every == 0:
            server_ndcg = evaluate(model, full_train_ds, full_test_ds, 
                      total_items, device, user_batch=10, full=True)
            print("\n[",epoch,"/",args.T*args.E, "ndcg",server_ndcg, flush=True)
            if earlystop.step(server_ndcg):
                  print("Early Stop Triggered")
                  break
        if DEBUG:
            print(epoch,  "-->", np.mean(losses), np.mean(norms))
