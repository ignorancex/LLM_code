import argparse
import numpy as np

import torch
import torch.nn as nn
from opacus import PrivacyEngine
import time
import os 

from models import StandardizeLayer
from train_utils import get_device, train, test,test_for_membership
from data import get_data
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence
from log import Logger
from gdp import *
from sklearn.metrics import roc_auc_score
import random
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

NUMBER_OF_CLASSES=10

@torch.no_grad()
def mixup_classes_gpu(X,Y):# with cuda
    if args.classes>NUMBER_OF_CLASSES:
        assert 0
    # X=torch.from_numpy(X)
    Y=torch.from_numpy(Y)
    # sort the labels
    ind=torch.argsort(Y)
    print(ind)
    Y=Y[ind]
    X=X[ind].float()
    X=X.cuda()
    # Y=Y.cuda()
    print(X.shape)
    # assert 0
    print(ind.shape,Y,X.shape)
    n,dx=X.shape
    one_hot=torch.zeros(Y.shape[0],NUMBER_OF_CLASSES).scatter_(1,Y.reshape(-1,1),1).cuda()
    if args.m<0:
        return X.cpu(),one_hot.cpu()
    MXs=[]
    MYs=[]
    for j in range(100):
        MC=(torch.rand(( int(args.T/100) ,NUMBER_OF_CLASSES),device="cuda")<(args.classes/NUMBER_OF_CLASSES) )
        MC=MC[torch.sum(MC,dim=1)>0].float()
        new_t=MC.shape[0]
        # print(MC.shape)
        MS=[]
        for i in range(NUMBER_OF_CLASSES):
            sample_rate=(args.m/n)/(args.classes/NUMBER_OF_CLASSES)
            Ms=(torch.rand((new_t,int(n/NUMBER_OF_CLASSES)),device="cuda" )<sample_rate).float()
            Ms=Ms*MC[:,i:i+1]
            MS.append(Ms)
            # print(Ms.shape)
        M=torch.cat(MS,dim=1)*float(1/args.m)
        MX=M.mm(X)
        MX+=torch.randn_like(MX)*( (args.noise_coefx*(args.Cx) ) / float(args.m) )
        MY=M.mm(one_hot)
        MY=MY+torch.randn_like(MY)*( args.noise_coefy*(args.Cy) / float(args.m) )
        MXs.append(MX.cpu())
        MYs.append(MY.cpu())
    # MY=torch.clamp(MY, min=0) 
    return torch.cat(MXs),torch.cat(MYs)


def clipping(ts,C,normalize=False):
    ts_reshape=ts.reshape(ts.shape[0],-1)
    if normalize:
        ts_reshape=ts_reshape-ts_reshape.mean(dim=1,keepdim=True)
    l2_norm=torch.norm(ts_reshape,p=2,dim=1,keepdim=True)
    l2_norm[l2_norm<C]=C
    l2_norm=l2_norm.div(C)
    ts_reshape=ts_reshape.div(l2_norm)
    return ts_reshape.reshape(ts.shape)

def main(feature_path=None, batch_size=2048, mini_batch_size=256,
         lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1,
         max_grad_norm=0.1, max_epsilon=None, epochs=100, logdir=None,**k):
    global NUMBER_OF_CLASSES
    if args.dataset=="cifar10":
        NUMBER_OF_CLASSES=10
    if args.dataset=="cifar100" or  args.dataset=="miniimagenet":
        NUMBER_OF_CLASSES=100
    if args.dataset=="imagenet" :
        NUMBER_OF_CLASSES=1000
    logger = Logger(logdir)
    mini_batch_size=256
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    setup_seed(random.randint(1,100000))
    writer = SummaryWriter("tb_simclr_{}_classes/eps{}/lamb{}{}_{}_C{}{}{}{}".format(args.dataset,args.eps,args.lamb,args.feature_path, args.m ,args.Cx,args.Cy ,args.remark,localtime) )
    writer.add_text('args', str(args))

    device = get_device()
    if args.eps==-1: # without considering DP
        true_eps=-1
    else:
        args.noise_coef,true_eps=find_noise_multi_poi(feature_number=args.T,mixup_number=args.m,eps_re=args.eps,n=50000,delta=1e-5)
    # assert 0
    # compute different noise for X and Y
    args.noise_coefx=args.noise_coef*np.sqrt( (args.lamb*args.lamb+1 )/(args.lamb*args.lamb)  )
    args.noise_coefy=args.noise_coef*np.sqrt( (args.lamb*args.lamb+1 )  )
    writer.add_scalar('Z/noise_coef', args.noise_coefx, 0)
    writer.add_scalar('Z/noise_coefx', args.noise_coefy, 0)
    writer.add_scalar('Z/noise_coef', args.noise_coef, 0)
    print("true_noise",args.lamb,args.noise_coefx,args.noise_coefy,args.noise_coef  )
    writer.add_scalar('Z/Xnoise', (args.noise_coefx*args.Cx) / float(args.m), args.m)
    writer.add_scalar('Z/Ynoise', (args.noise_coefy*args.Cy) / float(args.m), args.m)
    writer.add_scalar('Z/true_eps', true_eps, 0)

    # load features 
    x_train = np.load(f"{feature_path}_train.npy")
    x_test = np.load(f"{feature_path}_test.npy")
    if args.dataset=="miniimagenet" or args.dataset=="imagenet":
        y_train=np.load(f"{feature_path}_train.npy_label.npy")
        y_test=np.load(f"{feature_path}_test.npy_label.npy")
    else:
        if args.dataset=="cifar10":
            train_data, test_data = get_data("cifar10", augment=False)
        if args.dataset=="cifar100":
            train_data, test_data = get_data("cifar100", augment=False)
        y_train = np.asarray(train_data.targets)
        y_test = np.asarray(test_data.targets)

    x_train=torch.from_numpy(x_train)
    x_test=torch.from_numpy(x_test)
    x_train=clipping(x_train,args.Cx,normalize=args.norm) # 
    x_test=clipping(x_test,args.Cx,normalize=args.norm)
    x_test=x_test.numpy()
    mx,my=mixup_classes_gpu(x_train,y_train)
    trainset = torch.utils.data.TensorDataset(mx, my)
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    raw_train_set = torch.utils.data.TensorDataset(x_train, torch.from_numpy(y_train))

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    raw_train_loader = torch.utils.data.DataLoader(raw_train_set, batch_size=5000, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False, num_workers=1, pin_memory=True)

    n_features = x_train.shape[-1]
    try:
        mean = np.load(f"{feature_path}_mean.npy")
        var = np.load(f"{feature_path}_var.npy")
    except FileNotFoundError:
        mean = np.zeros(n_features, dtype=np.float32)
        var = np.ones(n_features, dtype=np.float32)
    print("{feature_path}_mean.npy",mean.shape,var.shape)

    bn_stats = (torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device))
    model = nn.Sequential(StandardizeLayer(bn_stats), nn.Linear(n_features, NUMBER_OF_CLASSES)).to(device)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / len(y_train),
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
    best=0
    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)
        if best<test_acc:
            best=test_acc
            with torch.no_grad():
                test_losses, test_acc =test_for_membership(model, test_loader)
                train_losses, train_acc =test_for_membership(model, raw_train_loader)
                y_true=torch.cat([torch.zeros_like(train_losses),torch.ones_like(test_losses)]).numpy()
                y_score=torch.cat([train_losses,test_losses]).numpy()
                auc=roc_auc_score(y_true, y_score)
                gap=train_acc-test_acc
                writer.add_scalar('bestmembership/test_acc', test_acc, epoch)
                writer.add_scalar('bestmembership/auc', auc, epoch)
                writer.add_scalar('bestmembership/gap', gap, epoch)
                writer.add_scalar('bestmembership/train_acc', train_acc, epoch)
        if epoch%20==0:
            with torch.no_grad():
                test_losses, test_acc =test_for_membership(model, test_loader)
                train_losses, train_acc =test_for_membership(model, raw_train_loader)
                y_true=torch.cat([torch.zeros_like(train_losses),torch.ones_like(test_losses)]).numpy()
                y_score=torch.cat([train_losses,test_losses]).numpy()
                auc=roc_auc_score(y_true, y_score)
                gap=train_acc-test_acc
                writer.add_scalar('membership/test_acc', test_acc, epoch)
                writer.add_scalar('membership/auc', auc, epoch)
                writer.add_scalar('membership/gap', gap, epoch)
                writer.add_scalar('membership/train_acc', train_acc, epoch)
        scheduler.step()

        if noise_multiplier > 0:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_sgd)
            print(f"ε = {epsilon:.3f}")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None
        print(epoch,best)
        epsilon = None
        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)
        writer.add_scalar('Acc/test_acc', test_acc, epoch)
        writer.add_scalar('Acc/train_acc', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Loss/train', train_acc, epoch)
    writer.add_scalar('Acc/best', best, int(args.m))
    writer.close()

    logdir="record_classes/{}/eps{}/{}_C{}{}{}".format(args.dataset,args.eps,args.feature_path,args.Cx,args.Cy ,args.remark)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(logdir+".log","a") as f:
        f.writelines(['{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.m,args.lamb,args.classes,best,test_acc,auc,gap) ,])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise_multiplier', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--feature_path', default=None)
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--m', type=float, default=1)
    parser.add_argument('--T', type=int, default=50000)
    parser.add_argument('--noise_coef', type=float, default=0.0)
    parser.add_argument('--eps', type=float, default=-1)
    parser.add_argument('--Cx', type=float, default=1)
    parser.add_argument('--Cy', type=float, default=1)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--remark', type=str, default="")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--lamb', type=float, default=1) # sigma_x=sqrt( (lamb*lamb+1)/(lamb*lamb)) noise_coef
    parser.add_argument('--classes', type=float, default=3.0)
    args = parser.parse_args()
    main(**vars(args))
