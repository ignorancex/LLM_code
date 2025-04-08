import random
import argparse
import numpy as np
import pandas as pd
from src.utils import PickleUtils
from src.gat import GAT
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(dev)
##############################################################
# Define parameters and load data
##############################################################

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--seed', type=int, default=1986,
	                    help='global random seed number')

	parser.add_argument('--epochs', type=int, default=100,
	                    help='number of epochs of training')

	parser.add_argument('--lr', type=float, default=0.01,
	                    help='learning rate')

	parser.add_argument('--lr-factor', type=float, default=0.2,
	                    help='rate of reducing learning rate')

	parser.add_argument('--lr-patience', type=int, default=3,
	                    help='number of epochs validation loss not improving')

	parser.add_argument('--batch-size', type=int, default=512)

	parser.add_argument('--log-interval', type=int, default=20)

	parser.add_argument('--weight-decay', type=float, default=0.)

	parser.add_argument('--nb-heads', type=int, default=4, 
						help='number of attention heads')

	parser.add_argument('--dropout', type=float, default=0.6)

	parser.add_argument('--alpha', type=float, default=0.2,
						help='parameters of GAT')

	parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
	parser.set_defaults(weighted=True)

	return parser.parse_args()

##############################################################
# Define train and test functions
##############################################################

def train(epoch, model, optimizer, args, doc_emb, doc_svc, svc_emb):

    model.train()
    train_loss = 0

    idx_list = list(BatchSampler(RandomSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    #doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)
#     print("doc_emb_ts",doc_emb_ts.shape)
#     print("doc_svc_ts",doc_svc_ts.shape)
#     print("len(idx_list)",len(idx_list))
    for i in range(len(idx_list)):
        x = doc_emb_ts[idx_list[i]]
#         print("x",x.shape)
        
        adj = doc_svc_ts[idx_list[i]]
#         print("adj",adj.shape)
        
        
        
        optimizer.zero_grad()
        pred_y, _ = model(x, adj, svc_emb)
        
        
#         print("pred_y",pred_y.shape)
        
        y = doc_svc_ts[idx_list[i]]
#         print("y",y.shape)
        y=torch.tensor(y, dtype=torch.float)
        loss = F.binary_cross_entropy(torch.sigmoid(pred_y), y)
#         print("loss",loss)
        train_loss += loss.item() * len(x)
#         print("train_loss",train_loss)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * args.batch_size, len(doc_emb),
                100. * (i+1) * args.batch_size / len(doc_emb), loss.item()))

    train_loss /= len(doc_emb)
    print('Average train loss of epoch {} is {:.4f}.\n'.format(epoch, train_loss))

    return train_loss

def test(epoch, model, optimizer, args, doc_emb, doc_svc, svc_emb):
    
    model.eval()
    test_loss = 0
    correct = 0

    idx_list = list(BatchSampler(SequentialSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    #doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)
    
    with torch.no_grad():
        for i in range(len(idx_list)):
#             print("===TEST============")
            x = doc_emb_ts[idx_list[i]]
            y = doc_svc_ts[idx_list[i]]
            adj = doc_svc_ts[idx_list[i]]
            
            pred_y, _ = model(x, adj, svc_emb)
            y=torch.tensor(y, dtype=torch.float)
#             print("pred_y",pred_y.shape)
#             print("y",y.shape)
            pred = torch.sigmoid(pred_y)
            test_loss += F.binary_cross_entropy(pred, y, reduction='sum')
            #pred = F.sigmoid(pred_y)
#             print("pred",pred.shape)
            correct += accuracy(pred, y, 30).item()
#             print(correct)
            #correct += pred.eq(y.view_as(pred)).sum().item()

            if i % args.log_interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (i+1) * args.batch_size, len(doc_emb),
                    100. * (i+1) * args.batch_size / len(doc_emb)))

    test_loss /= len(doc_emb)
    accu = 100. * correct / (i+1)

    print('Average test loss of epoch {} is {:.4f}, accuracy is {:.2f}%.\n'.format(epoch, test_loss, accu))

    return test_loss, accu

def accuracy(output, target, maxk):
    #maxk = max(topk)
    #print(maxk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    #print(pred)
    ret = []
    #for k in topk:
    correct = (target * torch.zeros_like(target).scatter(1, pred[:, :maxk], 1)).float()
    #print(correct.sum() / target.sum())
        #ret.append(correct.sum() / target.sum())
        #print(ret)
    return correct.sum() / target.sum()

def precision(output, target, maxk):
    #maxk = max(topk)
    #print(maxk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    #print(pred)
    ret = []
    #for k in topk:
    correct = (target * torch.zeros_like(target).scatter(1, pred[:, :maxk], 1)).float()
    #print(correct.sum() / target.sum())
        #ret.append(correct.sum() / target.sum())
        #print(ret)
    return correct.sum() / maxk

def test_batch(model, args, doc_emb, doc_svc, svc_emb):
    
    model.eval()

    idx_list = list(BatchSampler(SequentialSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    #doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc = torch.tensor(doc_svc, dtype=torch.long, device=dev)
    with torch.no_grad():
        for i in range(len(idx_list)):
            x = doc_emb[idx_list[i]]
            adj = doc_svc[idx_list[i]]
            _, x_prime = model(x, adj, svc_emb)

            if i == 0:
                doc_emb_prime = x_prime.detach().cpu().numpy()
            else:
                doc_emb_prime = np.concatenate((doc_emb_prime, x_prime.detach().cpu().numpy()), axis=0)

            if i % args.log_interval == 0:
                print('Processed: [{}/{} ({:.0f}%)]'.format(
                    (i+1) * args.batch_size, len(doc_emb),
                    100. * (i+1) * args.batch_size / len(doc_emb)))

    return doc_emb_prime

def set_rnd_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_doc_emb(args):
    # load prepared init data for training doctor embedding
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
    print(doc_emb_data.head())
    print(doc_emb_data.shape)
    
    doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
    print(doc_emb.shape)
    #doc_spec = doc_emb_data.readmission.values
    doc_svc = doc_emb_data.svc_id.values

    # load service embedding matrix
    ppd_emb = np.loadtxt('node2vec/emb/ppd_eICU.emb', skiprows=1)
    ppd_coor = np.array([x[1:] for x in ppd_emb])
    ppd_id = [int(x[0]) for x in ppd_emb]

    # sort ppd_emb in the ascending order of node id
    svc_emb = ppd_coor[np.argsort(ppd_id), :]
    svc_emb_ts = torch.tensor(svc_emb, dtype=torch.float, device=dev)

    # prepare dataset
    num_doc = len(doc_emb_data) # len(doc_emb)
    num_svc = len(svc_emb)
    
    set_rnd_seed(args)
    rndx = np.random.permutation(range(num_doc))   
    doc_emb = doc_emb[rndx]
    #doc_spec = doc_spec[rndx]
    doc_svc = doc_svc[rndx]
    doc_emb_data=doc_emb_data[doc_emb_data.index.isin(rndx)]
    print(doc_emb_data.head())
    print(doc_emb_data.shape)
    
    adj_mat = np.zeros((num_doc, num_svc), dtype=int)
    for i in range(num_doc):
        adj_mat[i, doc_svc[i]] = 1

    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    doc_svc_ts = torch.tensor(adj_mat, dtype=torch.long, device=dev)
    
    doc_emb_test = doc_emb[round(num_doc * args.train_ratio):]
    doc_svc_test = adj_mat[round(num_doc * args.train_ratio):]
    doc_emb_data=doc_emb_data.iloc[round(num_doc * args.train_ratio):,:]
    # define and load models
    model = GAT(
        nfeat=doc_emb.shape[1], 
        nhid=int(doc_emb.shape[1] / args.nb_heads), 
        nclass=num_svc, 
        dropout=args.dropout, 
        drop_enc=True, 
        alpha=args.alpha, 
        nheads=args.nb_heads).to(dev)

    # load model parameters
    checkpoint = torch.load('saved_models/doc_emb_best')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(doc_emb_data.head())
    print(doc_emb_data.shape)
    # inference
    #doc_emb_prime = test_batch(model, args, doc_emb_ts, doc_svc_ts, svc_emb_ts)
    doc_emb_prime = test_batch(model, args, doc_emb_test, doc_svc_test, svc_emb_ts)
    print("doc_emb_prime",doc_emb_prime.shape)
    PickleUtils.saver('saved_data/doc_emb_prime_test.pkl', doc_emb_prime)
    PickleUtils.saver('saved_data/doc_emb_prime_test_labels.pkl', doc_emb_data)
    
    doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
    
    #doc_spec = doc_emb_data.readmission.values
    doc_svc = doc_emb_data.svc_id.values

#     spec_emb = np.zeros((2, 128), dtype=float)
#     for i in range(2):
#         spec_emb[i] = np.mean(doc_emb_prime[np.where(doc_spec == i)], axis=0)
#     PickleUtils.saver('saved_data/readmission_emb.pkl', spec_emb)

def main(args):
    # load prepared init data for training doctor embedding
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
#     print(doc_emb_data.shape)
#     print(doc_emb_data['patientunitstayid'].nunique())
    doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
#     print(doc_emb.shape)
    #doc_spec = doc_emb_data.readmission.values
    doc_svc = doc_emb_data.svc_id.values
#     print(doc_svc.shape)
    
    # load service embedding matrix
    ppd_emb = np.loadtxt('node2vec/emb/ppd_eICU.emb', skiprows=1)
    ppd_coor = np.array([x[1:] for x in ppd_emb])
    ppd_id = [int(x[0]) for x in ppd_emb]
    #print(doc_svc)
    # sort ppd_emb in the ascending order of node id 
    svc_emb = ppd_coor[np.argsort(ppd_id), :]
    svc_emb_ts = torch.tensor(svc_emb, dtype=torch.float, device=dev)
#     print("svc_emb_ts",svc_emb_ts.shape)
    # prepare dataset
    num_doc = len(doc_emb_data)
    num_svc = len(svc_emb)
    set_rnd_seed(args)
    rndx = np.random.permutation(range(num_doc))
#     print("num_doc ",num_doc)
#     print("num_svc",num_svc)
    
    doc_emb = doc_emb[rndx]
    #doc_spec = doc_spec[rndx]
    doc_svc = doc_svc[rndx]

    adj_mat = np.zeros((num_doc, num_svc), dtype=int)
    for i in range(num_doc):
        adj_mat[i, doc_svc[i]] = 1
#     print("adj_mat",adj_mat.shape)   
    doc_emb_train = doc_emb[:round(num_doc * args.train_ratio)]
    doc_emb_test = doc_emb[round(num_doc * args.train_ratio):]

#     doc_spec_train = doc_spec[:round(num_doc * args.train_ratio)]
#     doc_spec_test = doc_spec[round(num_doc * args.train_ratio):]

    doc_svc_train = adj_mat[:round(num_doc * args.train_ratio)]
    doc_svc_test = adj_mat[round(num_doc * args.train_ratio):]

    # define model
    set_rnd_seed(args)
    model = GAT(
        nfeat=doc_emb.shape[1], 
        nhid=int(doc_emb.shape[1] / args.nb_heads), 
        nclass=num_svc, 
        dropout=args.dropout, 
        drop_enc=True, 
        alpha=args.alpha, 
        nheads=args.nb_heads).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    # train and validation
    best_loss = 1000.

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, optimizer, args, doc_emb_train, doc_svc_train, svc_emb_ts)
        test_loss, accu = test(epoch, model, optimizer, args, doc_emb_test, doc_svc_test, svc_emb_ts)
        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
#             print("saving")
            if args.checkpoint:
                print("saving")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'accu': accu,
                    'epoch': epoch
                    }, 'saved_models/doc_emb_best')

def visualize_embeddings(args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    print("=========Visuzlization with readmission adn no readmision labels================")
    read_emb = PickleUtils.loader('saved_data/readmission_emb.pkl')
    visit_emb = PickleUtils.loader('saved_data/doc_emb_prime.pkl')
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
    print(doc_emb_data.shape)
    print(doc_emb_data.head())

    print(read_emb.shape)
    print(visit_emb.shape)

    
    #emb=emb.data.cpu().numpy()
    visit_emb=visit_emb[0:20000,:]
    print(visit_emb.shape)
#     print(emb[587,:])
#     print(emb[588,:])
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(visit_emb)
    print(tsne_proj.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    #cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(28,28))
    print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
    label = doc_emb_data['readmission'][0:20000]
    print(label)
    #for i in range(0,visit_emb.shape[0]):
    ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
        #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
    plt.show()
        
        
    return    


def visualize_embeddings_labels(args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    read_emb = PickleUtils.loader('saved_data/readmission_emb.pkl')
    visit_emb = PickleUtils.loader('saved_data/doc_emb_prime.pkl')
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
    print(doc_emb_data.shape)
    print(doc_emb_data.head())

    
    print(visit_emb.shape)

    data = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
    data = data.sort_values(by=['patientunitstayid','offset','PPD name'])

   
    
#     print(pat_table.head())
#     print(svc_dict.head())
#     print(data.head())
#     print(data['svc_id'].nunique())
    data_labels=data[['svc_id','PPD name']].drop_duplicates()
    #data[['svc_id','PPD name']].drop_duplicates().to_csv('saved_data/labels.csv',index=False)
    data_labels['PPD name']
    renal_ids=data_labels[data_labels['PPD name'].str.contains("renal")]['svc_id'].tolist()
    pulmonary_ids=data_labels[data_labels['PPD name'].str.contains("pulmonary")]['svc_id'].tolist()
    infectious_ids=data_labels[data_labels['PPD name'].str.contains("infectious")]['svc_id'].tolist()
    gastrointestinal_ids=data_labels[data_labels['PPD name'].str.contains("gastrointestinal")]['svc_id'].tolist()
    oncology_ids=data_labels[data_labels['PPD name'].str.contains("oncology")]['svc_id'].tolist()
    neurologic_ids=data_labels[data_labels['PPD name'].str.contains("neurologic")]['svc_id'].tolist()
    
#     print(len(renal_ids),renal_ids)
#     print(len(pulmonary_ids),pulmonary_ids)
#     print(len(infectious_ids),infectious_ids)
#     print(len(gastrointestinal_ids),gastrointestinal_ids)
#     print(len(oncology_ids),oncology_ids)
#     print(len(neurologic_ids),neurologic_ids)
    
    inter=set(renal_ids).intersection(set(pulmonary_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
  
    
    inter=set(renal_ids).intersection(set(infectious_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    infectious_ids=list(set(infectious_ids)-inter)
   
    inter=set(renal_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    gastrointestinal_ids=list(set(gastrointestinal_ids)-inter)
    
    inter=set(renal_ids).intersection(set(oncology_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    oncology_ids=list(set(oncology_ids)-inter)
    
    inter=set(renal_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    
    inter=set(pulmonary_ids).intersection(set(infectious_ids))
#     print(inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
    infectious_ids=list(set(infectious_ids)-inter)
    inter=set(pulmonary_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    inter=set(pulmonary_ids).intersection(set(oncology_ids))
#     print(inter)
    inter=set(pulmonary_ids).intersection(set(neurologic_ids))
#     print(inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
    neurologic_ids=list(set(neurologic_ids)-inter)
    
    inter=set(infectious_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    gastrointestinal_ids=list(set(gastrointestinal_ids)-inter)
    inter=set(infectious_ids).intersection(set(oncology_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    oncology_ids=list(set(oncology_ids)-inter)
    inter=set(infectious_ids).intersection(set(neurologic_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    neurologic_ids=list(set(neurologic_ids)-inter)
    
    
    inter=set(gastrointestinal_ids).intersection(set(oncology_ids))
#     print(inter)
    inter=set(gastrointestinal_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    inter=set(oncology_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    
    
    
   
    
   
   
    
    
    
#     print(data.shape)
#     print(data['patientunitstayid'].nunique())
    data['label']=0
    data.loc[data['svc_id'].isin(renal_ids),'label']=1
    data.loc[data['svc_id'].isin(pulmonary_ids),'label']=2
    data.loc[data['svc_id'].isin(infectious_ids),'label']=3
    data.loc[data['svc_id'].isin(gastrointestinal_ids),'label']=4
    data.loc[data['svc_id'].isin(oncology_ids),'label']=5
    data.loc[data['svc_id'].isin(neurologic_ids),'label']=6
    
    
    #data=data.groupby('patientunitstayid').max().reset_index()
#     print(data.head())
#     print(data.shape)
    data=data[['patientunitstayid','label']].drop_duplicates()
    
    visit_emb=pd.DataFrame(visit_emb)
    visit_emb= visit_emb.add_suffix('_emb')
    doc_emb_data=pd.concat([doc_emb_data,visit_emb],axis=1)
    
    doc_emb_data=doc_emb_data.merge(data,how='left',on='patientunitstayid')
    print(doc_emb_data.shape)
    
    
    print(doc_emb_data['label'].unique())
    
    
    #print(visit_emb.shape)
    
    #visit_emb = np.delete(visit_emb, neg,axis=0)
#     doc_emb_data=doc_emb_data[~doc_emb_data.index.isin(neg)]
#     print(doc_emb_data.shape)
    #print(doc_emb_data.head())
    #visit_emb=visit_emb[neg]
    #emb=emb.data.cpu().numpy()
    #visit_emb=visit_emb[0:20000,:]
    #print(visit_emb.shape)
#     print(emb[587,:])
#     print(emb[588,:])

    temp=doc_emb_data.groupby('patientunitstayid').count().reset_index()
    print("temp",temp.head())
    temp=temp[temp['label']==1]
    print("temp",temp.shape)
    temp2=doc_emb_data.groupby('patientunitstayid').first().reset_index()
    doc_emb_data=temp2[temp2['patientunitstayid'].isin(temp['patientunitstayid'].unique())]
    print("temp2",doc_emb_data.shape)
    
    
    
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(doc_emb_data.iloc[:,4:132])
    print(tsne_proj.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    #cmap = cm.get_cmap('tab20')
    #fig, ax= plt.subplots(2,3, figsize=(50, 50))
    fig, ax= plt.subplots(figsize=(5, 5))
    
    doc_emb_data=doc_emb_data.reset_index()
    renals=doc_emb_data.index[doc_emb_data['label'] == 1].tolist()
    pulmonary=doc_emb_data.index[doc_emb_data['label'] == 2].tolist()
    infectious=doc_emb_data.index[doc_emb_data['label'] == 3].tolist()
    gastrointestinal=doc_emb_data.index[doc_emb_data['label'] == 4].tolist()
    oncology=doc_emb_data.index[doc_emb_data['label'] == 5].tolist()
    neurologic=doc_emb_data.index[doc_emb_data['label'] == 6].tolist()
    neg=doc_emb_data.index[doc_emb_data['label'] == 0].tolist()
    print(len(neg))
    
    print("renals",100*(doc_emb_data[doc_emb_data['label']==1].shape[0]/doc_emb_data.shape[0]))
    print("pulmonary_ids",100*(doc_emb_data[doc_emb_data['label']==2].shape[0]/doc_emb_data.shape[0]))
    print("infectious_ids",100*(doc_emb_data[doc_emb_data['label']==3].shape[0]/doc_emb_data.shape[0]))
    print("gastrointestinal_ids",100*(doc_emb_data[doc_emb_data['label']==4].shape[0]/doc_emb_data.shape[0]))
    print("oncology_ids",100*(doc_emb_data[doc_emb_data['label']==5].shape[0]/doc_emb_data.shape[0]))
    print("neurologic_ids",100*(doc_emb_data[doc_emb_data['label']==6].shape[0]/doc_emb_data.shape[0]))
    
    ax.scatter(tsne_proj[renals,0],tsne_proj[renals,1], color = "r", s=20,label='Renals')
    ax.scatter(tsne_proj[pulmonary,0],tsne_proj[pulmonary,1], color = "b", s=20,label='pulmonary')
    ax.scatter(tsne_proj[infectious,0],tsne_proj[infectious,1], color = "g", s=20,label='infectious')
    ax.scatter(tsne_proj[gastrointestinal,0],tsne_proj[gastrointestinal,1], color = "brown", s=20,label='gastrointestinal')
    ax.scatter(tsne_proj[oncology,0],tsne_proj[oncology,1], color = "cyan", s=20,label='oncology')
    ax.scatter(tsne_proj[neurologic,0],tsne_proj[neurologic,1], color = "pink", s=20,label='neurologic')
    ax.scatter(tsne_proj[neg,0],tsne_proj[neg,1], color = "yellow", s=10,label='None')
    ax.legend(bbox_to_anchor=(1, 1))
    plt.show()
    
#     test_ids = PickleUtils.loader('saved_data/doc_emb_prime_test_labels.pkl')
    
#     doc_emb_data=pd.concat([pd.get_dummies(doc_emb_data['label'],prefix='label'),doc_emb_data],axis=1)
#     print(doc_emb_data.head())
#     doc_emb_data=doc_emb_data[doc_emb_data['patientunitstayid'].isin(list(test_ids['patientunitstayid'].unique()))]
#     del doc_emb_data['svc_id']
#     del doc_emb_data['embedding']
#     df=doc_emb_data.groupby('patientunitstayid').max().reset_index()
#     print(df.head())
#     print(df.shape)
#     print(df['patientunitstayid'].nunique())
    
#     print(doc_emb_data['label'].unique())
#     X,y=df.iloc[:,8:136],df['label_1']
    
#     print(X.shape)
#     print(y.shape)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#     model = LogisticRegression().fit(X_train, y_train) 
#     logits=model.predict_log_proba(X_test)
#     prob=model.predict_proba(X_test)
#     #print(prob.shape)
#     fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
#     auc = metrics.auc(fpr, tpr)
#     print("Logistic Regression",auc)
    
#     model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, y_train)
#     prob=model.predict_proba(X_test)
#     fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
#     auc = metrics.auc(fpr, tpr)
#     print("Xgboost",auc)
    
#     X,y=df.iloc[:,8:136],df['label_2']
    
#     print(X.shape)
#     print(y.shape)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#     model = LogisticRegression().fit(X_train, y_train) 
#     logits=model.predict_log_proba(X_test)
#     prob=model.predict_proba(X_test)
#     #print(prob.shape)
#     fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
#     auc = metrics.auc(fpr, tpr)
#     print("Logistic Regression",auc)
    
#     model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, y_train)
#     prob=model.predict_proba(X_test)
#     fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
#     auc = metrics.auc(fpr, tpr)
#     print("Xgboost",auc)
    
#     return   


def classification(args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    print("==============Classification task=================")
    read_emb = PickleUtils.loader('saved_data/readmission_emb.pkl')
    visit_emb = PickleUtils.loader('saved_data/doc_emb_prime.pkl')
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
    print(doc_emb_data.shape)
    print(doc_emb_data.head())

    
#     print(visit_emb.shape)

    data = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
    data = data.sort_values(by=['patientunitstayid','offset','PPD name'])

   
    
#     print(pat_table.head())
#     print(svc_dict.head())
#     print(data.head())
#     print(data['svc_id'].nunique())
    data_labels=data[['svc_id','PPD name']].drop_duplicates()
    #data[['svc_id','PPD name']].drop_duplicates().to_csv('saved_data/labels.csv',index=False)
    data_labels['PPD name']
    renal_ids=data_labels[data_labels['PPD name'].str.contains("renal")]['svc_id'].tolist()
    pulmonary_ids=data_labels[data_labels['PPD name'].str.contains("pulmonary")]['svc_id'].tolist()
    infectious_ids=data_labels[data_labels['PPD name'].str.contains("infectious")]['svc_id'].tolist()
    gastrointestinal_ids=data_labels[data_labels['PPD name'].str.contains("gastrointestinal")]['svc_id'].tolist()
    oncology_ids=data_labels[data_labels['PPD name'].str.contains("oncology")]['svc_id'].tolist()
    neurologic_ids=data_labels[data_labels['PPD name'].str.contains("neurologic")]['svc_id'].tolist()
    
#     print(len(renal_ids),renal_ids)
#     print(len(pulmonary_ids),pulmonary_ids)
#     print(len(infectious_ids),infectious_ids)
#     print(len(gastrointestinal_ids),gastrointestinal_ids)
#     print(len(oncology_ids),oncology_ids)
#     print(len(neurologic_ids),neurologic_ids)
    
    inter=set(renal_ids).intersection(set(pulmonary_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
  
    
    inter=set(renal_ids).intersection(set(infectious_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    infectious_ids=list(set(infectious_ids)-inter)
   
    inter=set(renal_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    gastrointestinal_ids=list(set(gastrointestinal_ids)-inter)
    
    inter=set(renal_ids).intersection(set(oncology_ids))
#     print(inter)
    renal_ids=list(set(renal_ids)-inter)
    oncology_ids=list(set(oncology_ids)-inter)
    
    inter=set(renal_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    
    inter=set(pulmonary_ids).intersection(set(infectious_ids))
#     print(inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
    infectious_ids=list(set(infectious_ids)-inter)
    inter=set(pulmonary_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    inter=set(pulmonary_ids).intersection(set(oncology_ids))
#     print(inter)
    inter=set(pulmonary_ids).intersection(set(neurologic_ids))
#     print(inter)
    pulmonary_ids=list(set(pulmonary_ids)-inter)
    neurologic_ids=list(set(neurologic_ids)-inter)
    
    inter=set(infectious_ids).intersection(set(gastrointestinal_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    gastrointestinal_ids=list(set(gastrointestinal_ids)-inter)
    inter=set(infectious_ids).intersection(set(oncology_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    oncology_ids=list(set(oncology_ids)-inter)
    inter=set(infectious_ids).intersection(set(neurologic_ids))
#     print(inter)
    infectious_ids=list(set(infectious_ids)-inter)
    neurologic_ids=list(set(neurologic_ids)-inter)
    
    
    inter=set(gastrointestinal_ids).intersection(set(oncology_ids))
#     print(inter)
    inter=set(gastrointestinal_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    inter=set(oncology_ids).intersection(set(neurologic_ids))
#     print(inter)
    
    
    
    
   
    
   
   
    
    
    
#     print(data.shape)
#     print(data['patientunitstayid'].nunique())
    data['label']=0
    data.loc[data['svc_id'].isin(renal_ids),'label']=1
    data.loc[data['svc_id'].isin(pulmonary_ids),'label']=2
    data.loc[data['svc_id'].isin(infectious_ids),'label']=3
    data.loc[data['svc_id'].isin(gastrointestinal_ids),'label']=4
    data.loc[data['svc_id'].isin(oncology_ids),'label']=5
    data.loc[data['svc_id'].isin(neurologic_ids),'label']=6
    
    
    #data=data.groupby('patientunitstayid').max().reset_index()
#     print(data.head())
#     print(data.shape)
    data=data[['patientunitstayid','label']].drop_duplicates()
    
    visit_emb=pd.DataFrame(visit_emb)
    visit_emb= visit_emb.add_suffix('_emb')
    doc_emb_data=pd.concat([doc_emb_data,visit_emb],axis=1)
    
    doc_emb_data=doc_emb_data.merge(data,how='left',on='patientunitstayid')
    print(doc_emb_data.shape)
    
    
    print(doc_emb_data['label'].unique())
    
    
    #print(visit_emb.shape)
    
    #visit_emb = np.delete(visit_emb, neg,axis=0)
#     doc_emb_data=doc_emb_data[~doc_emb_data.index.isin(neg)]
#     print(doc_emb_data.shape)
    #print(doc_emb_data.head())
    #visit_emb=visit_emb[neg]
    #emb=emb.data.cpu().numpy()
    #visit_emb=visit_emb[0:20000,:]
    #print(visit_emb.shape)
#     print(emb[587,:])
#     print(emb[588,:])

    temp=doc_emb_data.groupby('patientunitstayid').count().reset_index()
#     print("temp",temp.head())
    temp=temp[temp['label']==1]
#     print("temp",temp.shape)
    temp2=doc_emb_data.groupby('patientunitstayid').first().reset_index()
    doc_emb_data=temp2[temp2['patientunitstayid'].isin(temp['patientunitstayid'].unique())]
#     print("temp2",doc_emb_data.shape)
    
    
    test_ids = PickleUtils.loader('saved_data/doc_emb_prime_test_labels.pkl')
    
    doc_emb_data=pd.concat([pd.get_dummies(doc_emb_data['label'],prefix='label'),doc_emb_data],axis=1)
#     print(doc_emb_data.head())
#     doc_emb_data=doc_emb_data[doc_emb_data['patientunitstayid'].isin(list(test_ids['patientunitstayid'].unique()))]
#     del doc_emb_data['svc_id']
#     del doc_emb_data['embedding']
#     df=doc_emb_data.groupby('patientunitstayid').max().reset_index()
#     print(df.head())
#     print(df.shape)
#     print(df['patientunitstayid'].nunique())
    
    doc_emb_data=doc_emb_data[doc_emb_data['label']>0]
    print(doc_emb_data.shape)
    print(doc_emb_data['patientunitstayid'].nunique())
    
    print(doc_emb_data['label'].unique())
    
    print("renals",100*(doc_emb_data[doc_emb_data['label']==1].shape[0]/doc_emb_data.shape[0]))
    print("pulmonary_ids",100*(doc_emb_data[doc_emb_data['label']==2].shape[0]/doc_emb_data.shape[0]))
    print("infectious_ids",100*(doc_emb_data[doc_emb_data['label']==3].shape[0]/doc_emb_data.shape[0]))
    print("gastrointestinal_ids",100*(doc_emb_data[doc_emb_data['label']==4].shape[0]/doc_emb_data.shape[0]))
    print("oncology_ids",100*(doc_emb_data[doc_emb_data['label']==5].shape[0]/doc_emb_data.shape[0]))
    print("neurologic_ids",100*(doc_emb_data[doc_emb_data['label']==6].shape[0]/doc_emb_data.shape[0]))

    
    
    #X_train, X_test, y_train, y_test = doc_emb_data.iloc[:,8:136],df.iloc[:,8:136],doc_emb_data['label'],df['label']
    print(doc_emb_data.iloc[:,11:139].head())
    #tsne = TSNE(2, verbose=1)
    #tsne_proj = tsne.fit_transform(doc_emb_data.iloc[:,4:132])
    fp_list,fn_list,tp_list,tn_list=[],[],[],[]
    prec_list,rec_list=[],[]
    for i in [1,2,3,4,5,6]:
        col="label_"+str(i)
#         print(doc_emb_data.head())
#         print(doc_emb_data.shape)
#         pos=doc_emb_data[doc_emb_data[col]==1]
#         neg=doc_emb_data[doc_emb_data[col]==0]
#         neg=neg.iloc[10:2*(10+pos.shape[0]),:]
#         df=pd.concat([pos,neg],axis=0)
        X,y=doc_emb_data.iloc[:,11:139],doc_emb_data[col]
        print(X.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        model = LogisticRegression().fit(X_train, y_train) 
        logits=model.predict_log_proba(X_test)
        prob=model.predict_proba(X_test)
        y_pred=model.predict(X_test)
        #print(prob.shape)
    #     fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
    #     auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        tn_list.append(tn)
        prec_list.append(metrics.precision_score(y_test,y_pred))
        rec_list.append(metrics.recall_score(y_test,y_pred))
        print("macro",f1_score(y_test, y_pred, average='macro'))
        print("micro",f1_score(y_test, y_pred, average='micro'))
    print(fp_list,fn_list,tp_list,tn_list)
    prec =sum(tp_list)/(sum(tp_list)+sum(fp_list))
    recall=sum(tp_list)/(sum(tp_list)+sum(fn_list))
    print("micro", (2 *  prec * recall) / (prec+recall))
    
    prec =sum(prec_list)/6
    recall=sum(rec_list)/6
    print("macro", (2 *  prec * recall) / (prec+recall))

def ml_models():
    print("=========Readmission Prediction=================")
    pat_table_orig = pd.read_csv('saved_data/patient.csv')
    pat_table_orig=pat_table_orig[pat_table_orig['unitdischargestatus']!='unknown']
    print(pat_table_orig['unitdischargestatus'].unique())
    
    visit_emb_data = PickleUtils.loader('saved_data/doc_emb_prime.pkl')
    doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')
    print(visit_emb_data.shape)
    print(doc_emb_data.shape)
    print(doc_emb_data.head())
    print(doc_emb_data['patientunitstayid'].nunique())
    
    doc_emb_data=doc_emb_data.merge(pat_table_orig[['patientunitstayid','unitdischargestatus']],how='left',on='patientunitstayid')
    idx = doc_emb_data[doc_emb_data['unitdischargestatus'].notnull()].index.tolist()
    #print(idx)
    visit_emb_data=visit_emb_data[idx]
    doc_emb_data=doc_emb_data.dropna()
    print(doc_emb_data['unitdischargestatus'].unique())
    doc_emb_data.loc[doc_emb_data['unitdischargestatus']=='Alive','unitdischargestatus']=0
    doc_emb_data.loc[doc_emb_data['unitdischargestatus']=='Expired','unitdischargestatus']=1
    print(doc_emb_data['unitdischargestatus'].unique())
    print(doc_emb_data.shape)
    
    
    X,y=visit_emb_data,doc_emb_data['readmission']
    y=y.astype('int')
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    model = LogisticRegression().fit(X_train, y_train) 
    logits=model.predict_log_proba(X_test)
    prob=model.predict_proba(X_test)
    print(prob.shape)
    fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
    auc = metrics.auc(fpr, tpr)
    print("Logistic Regression",auc)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, prob[:,1])
    apr = metrics.auc(recall, precision)
    print("apr",apr)
    
    
    model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, y_train)
    prob=model.predict_proba(X_test)
    fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
    auc = metrics.auc(fpr, tpr)
    print("Xgboost",auc)
    
def start():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1986,
                        help='global random seed number')

    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs of training')
    parser.add_argument('--train_ratio', type=int, default=0.8,
                        help='train_ratio')
    parser.add_argument('--test_ratio', type=int, default=0.2,
                        help='test_ratio')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('--lr-factor', type=float, default=0.2,
                        help='rate of reducing learning rate')

    parser.add_argument('--lr-patience', type=int, default=3,
                        help='number of epochs validation loss not improving')

    parser.add_argument('--batch-size', type=int, default=52)

    parser.add_argument('--log-interval', type=int, default=100)

    parser.add_argument('--weight-decay', type=float, default=0.)

    parser.add_argument('--nb-heads', type=int, default=4, 
                        help='number of attention heads')

    parser.add_argument('--dropout', type=float, default=0.6)

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='parameters of GAT')

    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',default='True')
    parser.set_defaults(weighted=True)

    args=parser.parse_args(args=[])
    
    main(args)
    get_doc_emb(args)
    visualize_embeddings(args)
    visualize_embeddings_labels(args)
    ml_models()
    classification(args)
    
    

