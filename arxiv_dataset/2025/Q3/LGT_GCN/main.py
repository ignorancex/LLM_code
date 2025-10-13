import argparse
from utils import train, val_and_test
from data import load_data
import torch
import random
import os
import time
import numpy as np
from LGT_GCN import LGT_GCN

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#parameter
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora,citeseer,pubmed,AmazonPhoto, CoauthorCS}.')
parser.add_argument('--model', type=str, default='LGT_GCN', help='{LGT_GCN}')
parser.add_argument('--hid', type=int, default= 256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay .')
parser.add_argument('--nlayer', type=int, default=2,  help='Number of layers, works for Deep model.')
parser.add_argument("--seed",type=int,default=30,help="seed for model")
parser.add_argument('--rank', type=int, default=3, help='lamda.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate.')
args = parser.parse_args()
set_seed(args.seed)

all_test_acc = []
all_test_acc = []
all_mad = []
all_metric = []
start_time = time.time()
for i in range(5):
    data = load_data(args.data,round_no=i)
    nfeat = data.x.shape[1]
    nclass = int(data.y.max()) + 1
    net = LGT_GCN(args, nfeat, nclass, data)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()

    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val=0
    best_test=0
    best_mad = 0
    best_metric=[0,0]

    depth = 0
    patience = 30
    num_no_improvement = 0
    inner_loop = 0
    inner_best_acc=0
    for epoch in range(args.nlayer * args.epochs):
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data,depth=depth)
        test_acc,val_acc,val_loss,test_loss,mad,metric1,metric2= val_and_test(net, args.model, criterion, data, args.nlayer,depth=depth, epoch=epoch)
        print("======>  depth:{}  val_acc:{}".format(depth, val_acc))
        if best_val < val_acc:
            best_val = val_acc
            best_test = test_acc
            best_mad = mad
            best_metric = [metric1,metric2]

        if inner_best_acc < val_acc:
            inner_best_acc = val_acc
            num_no_improvement = 0
        else:
            num_no_improvement+=1

        if num_no_improvement > patience or inner_loop >= args.epochs:
            num_no_improvement = 0
            inner_best_acc = 0
            depth += 1
            inner_loop = 0

        inner_loop += 1

        if isinstance(best_test, int):
            best_test_output = round(best_test, 4)
        else:
            best_test_output = round(best_test.tolist(), 4)
        if isinstance(mad, int):
            best_mad_output = round(mad, 4)
        else:
            best_mad_output = round(mad.tolist(), 4)
        local_time = time.localtime(time.time())
        local_time_s = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        print("{} epoch:{} train_acc:{},train_loss:{},val_acc:{},val_loss:{},"
              "test_acc:{},test_loss:{},best_acc:{},best_mad:{},mad:{}"
              .format(local_time_s,epoch,round(train_acc.tolist(),4),
                      round(train_loss.tolist(),4),round(val_acc.tolist(),4),
                      round(val_loss.tolist(),4),round(test_acc.tolist(),4),
                      round(test_loss.tolist(),4),best_test_output,best_mad_output,mad))

        if depth >= args.nlayer:
            break

    print("\ncurrent best test acc",best_test)
    all_test_acc.append(best_test)
    all_mad.append(best_mad)
    all_metric.append(best_metric)

end_time = time.time()
print("cost time", end_time-start_time)