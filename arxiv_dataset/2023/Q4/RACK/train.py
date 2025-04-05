import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from datasets.Getloader import Getloader
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import scipy.io as scio
import random
import math
import argparse

parser = argparse.ArgumentParser(description='Train GNN for action prediction')
parser.add_argument('--LR', dest='LR',
                  help='Learning rate',
                  default=0.00001, type=float)
parser.add_argument('--EPOCH', dest='EPOCH',
                  help='total epoches',
                  default=800, type=int)
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE',
                  help='batch size',
                  default=16, type=int)
parser.add_argument('--p_rate', dest='p_rate',
                  help='dropout ratios',
                  default=0.5, type=float)
parser.add_argument('--best_acc', dest='best_acc',
                  help='record current best acc to save the best model',
                  default=0.5, type=float)
parser.add_argument('--segments', dest='segments',
                  help='the number segments for a full video',
                  default=10, type=int)
parser.add_argument('--gpus', dest='gpus',
                  help='gpu id for training',
                  default='2', type=str)
parser.add_argument('--dims', dest='dims',
                  help='featue dimensions',
                  default=1024, type=int)
parser.add_argument('--nclass', dest='nclass',
                  help='class numbers',
                  default=51, type=int)

parser.add_argument('--save_path', dest='save_path',
                  help='path to save model',
                  default=os.path.normpath("checkpoints/hmdb51"), type=str)
parser.add_argument('--trainfile', dest='trainfile',
                  help='path for the training file',
                  default=os.path.normpath("data/hmdb51/trdata_hmdb51_split1.npy"), type=str)
parser.add_argument('--tr_labelfile', dest='tr_labelfile',
                  help='path for the training label file',
                  default=os.path.normpath("data/hmdb51/trlabel_hmdb51_split1.npy"), type=str)
parser.add_argument('--testfile', dest='testfile',
                  help='path for the test label file',
                  default=os.path.normpath("data/hmdb51/testdata_hmdb51_split1.npy"), type=str)
parser.add_argument('--te_labelfile', dest='te_labelfile',
                  help='path for the test label label file',
                  default=os.path.normpath("data/hmdb51/testlabel_hmdb51_split1.npy"), type=str)
parser.add_argument('--pretrain_model', dest='pretrain_model',
                  help='path for the pretrain_model',
                  default=os.path.normpath(""), type=str)
parser.add_argument('--save_results', dest='save_results',
                  help='path to save results',
                  default=os.path.normpath("results/hmdb51/"), type=str)


args = parser.parse_args()

print('args:', args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
setup_seed(123456)

# Hyper Parameters
EPOCH = args.EPOCH              
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR              # learning rate
p_rate=args.p_rate
best_acc=args.best_acc
save_path=args.save_path
segments = args.segments
dims=args.dims
nclass=args.nclass
pretrain_model = args.pretrain_model

# prepare training data
train_file = args.trainfile 
tr_data = np.load(train_file)  # (N,segments,D), N: samples, D: feature dimensions
tr_data=tr_data.astype(np.float32)

tr_labelfile = args.tr_labelfile
tr_labels = np.load(tr_labelfile)  # (N,1)
#tr_data = torch.from_numpy(tr_data)
#tr_labels = torch.from_numpy(tr_labels)

# prepare test data
test_data = np.load(args.testfile)
te_label = np.load(args.te_labelfile)  
test_data=test_data.astype(np.float32)
test_data = torch.from_numpy(test_data)
te_label = torch.from_numpy(te_label)

print('traindata,testdata',tr_data.shape,test_data.shape)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# dataloader
torch_data = Getloader(tr_data, tr_labels)
train_loader = DataLoader(torch_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=10)
tedata = Getloader(test_data, te_label)
test_loader = DataLoader(tedata, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=11)

# define GCN
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=10):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_n = node_n
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.ones(node_n, node_n)*0.99)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        support = torch.matmul(input, self.weight)
        mup=torch.matmul(input,torch.transpose(input,1,2))
        mdown=torch.matmul(torch.norm(input,dim=-1,keepdim=True),torch.transpose(torch.norm(input,dim=-1,keepdim=True),1,2))
        w = torch.div(mup,mdown)
        if mask == []:
            adjacent_weights = self.att
        else:
            adjacent_weights=torch.mul(self.att,mask).to(torch.float32)
        adjacent_weights = torch.mul(adjacent_weights,w)
        output = torch.matmul(adjacent_weights, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=10):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()

    def forward(self, x, mask):
        y = self.gc1(x, mask)
        b, n, f = y.shape
        #pdb.set_trace()
        #print('x,y shape:',x.shape,y.shape)
        #print('b,n,f:',b,n,f)
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y+x, mask)
        b, n, f = y.shape
        y2 = y
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y+y2)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# define Rack-RGB/Rack-flow
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, p_dropout, num_stage = 1, hidden_feature=512, node_n=10):
        super(Net, self).__init__()
        self.num_stage = num_stage
        self.gcn_t1=GraphConvolution(n_feature, hidden_feature, node_n = node_n)
        self.gcn_s1=GraphConvolution(n_feature, hidden_feature, node_n = node_n)
        self.gcn_t=[]
        self.node_n=node_n
        for i in range(num_stage):
            self.gcn_t.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.gcn_t = nn.ModuleList(self.gcn_t)

        self.gcn_s=[]
        for i in range(num_stage):
            self.gcn_s.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.gcn_s = nn.ModuleList(self.gcn_s)


        self.out = nn.Linear(hidden_feature, n_output)   # output layer
        self.dropout = nn.Dropout()


    def forward(self, x, mask):
        t_out0=[]
        t_out = self.gcn_t1(x, [])
        t_out0.append(t_out)
        for i in range(self.num_stage):
            t_out = self.gcn_t[i](t_out,[])
            t_out0.append(t_out)

        s_out0=[]
        s_out = self.gcn_s1(x, mask)
        s_out0.append(s_out)
        for i in range(self.num_stage):
            s_out = self.gcn_s[i](s_out, mask)
            s_out0.append(s_out) 

        x = self.dropout(s_out)
        xt = self.dropout(t_out)
        out=[]
        out_t=[]
        for i in range(self.node_n):
            x1 = self.out(x[:,i])
            out.append(x1)
            x2 = self.out(xt[:,i])
            out_t.append(x2)
        out = torch.stack(out,dim=1)
        out_t = torch.stack(out_t,dim=1)
        return (out,out_t,t_out0,s_out0)
# define a mask for multiple progress level
mask=np.zeros((segments,segments))
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if i>=j:
            mask[i,j]=1
print('mask:\n',mask)

mask=torch.from_numpy(mask)
cnn = Net(n_feature=dims, n_output=nclass, p_dropout = p_rate,node_n=args.segments)    
cnn = cnn.cuda()
print(cnn)  # print network architecture
total_params = sum(p.numel() for p in cnn.parameters())
print(f'{total_params:,} total parameters.')


# optimizer
lr_list = []
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,250,350], gamma=0.95, last_epoch=-1) 

loss_classifier = nn.CrossEntropyLoss()                     
loss_mse = nn.MSELoss()

# load the pretrain model
if len(pretrain_model)>1:
    cnn.load_state_dict(torch.load(pretrain_model))
    print('Finish loading the pretrained model')


for epoch in range(EPOCH):
    # training 
    cnn.train()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    for step, (b_x, b_y) in enumerate(train_loader):  
        if b_x.shape[0] < 2:
            continue
        (output,output_t, t_out0,s_out0) = cnn(b_x.cuda(),mask.cuda())              
        lc = 0.0
        for i in range(args.segments):
            lc +=  loss_classifier(output[:,i], b_y.cuda())   
            lc += loss_classifier(output_t[:,i], b_y.cuda())
        lfeat = 0 
        for i in range(len(t_out0)):
           lfeat += loss_mse(t_out0[i], s_out0[i])
           lfeat += loss_mse(torch.matmul(torch.transpose(s_out0[i],1,2),s_out0[i]),torch.matmul(torch.transpose(t_out0[i],1,2),t_out0[i]))
        loss = lc +  lfeat
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()                

    # testing
    cnn.eval()
    # test accuracy
    logits = []
    for step, (xt,yt) in enumerate(test_loader):
        test_out = cnn(xt.cuda(),mask.cuda())
        batch_logits = test_out[0].cpu()
        logits.append(batch_logits.data)
    logits = torch.cat(logits, dim=0)
    test_accuracy = []
    results=[]
    gt=[]
    np.save(args.save_results+'scores.npy',logits)
    # results of single stream
    for i in range(segments):
        test_output = F.softmax(logits[:,i], dim=1)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == te_label.data.numpy()).astype(int).sum()) / float(te_label.size(0))
        test_accuracy.append(accuracy)
        results.append(pred_y)
        gt.append(te_label.data.numpy())
        print('Obervation: %.1f'%((i+1)/segments), '| test accuracy: %.4f' % test_accuracy[i])  
        
        
        