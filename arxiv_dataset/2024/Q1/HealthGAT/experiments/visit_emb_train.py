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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
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

def train(epoch, model, optimizer, args, doc_emb, doc_svc, fut_svc, svc_emb):

    model.train()
    train_loss = 0

    idx_list = list(BatchSampler(RandomSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    #doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)
    fut_svc_ts = torch.tensor(fut_svc, dtype=torch.long, device=dev)
#     print("doc_emb_ts",doc_emb_ts.shape)
#     print("doc_svc_ts",doc_svc_ts.shape)
#     print("len(idx_list)",len(idx_list))
    for i in range(len(idx_list)):
        x = doc_emb_ts[idx_list[i]]
#         print("x",x.shape)
        
        adj = doc_svc_ts[idx_list[i]]
#         print("adj",adj.shape)
        y = doc_svc_ts[idx_list[i]]
#         print("y",y.shape)
        y=torch.tensor(y, dtype=torch.float)
#         print("y",y.shape)
#         print(y[0])
        fut = fut_svc_ts[idx_list[i]]
        fut=torch.tensor(fut, dtype=torch.float)
        
#         print("fut",fut.shape)
#         print(fut[0])
        y=torch.cat([y,fut],dim=1)
        #print("y",y.shape)
#         print(y[0])
   
        optimizer.zero_grad()
        pred_y, _ = model(x, adj, svc_emb)
        
        
#         print("pred_y",pred_y.shape)
#         print(pred_y[0])
#         print(pred_y2[0])
        
        loss = F.binary_cross_entropy(torch.sigmoid(pred_y), y)
                
#         print("y",y.shape)
        
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

def test(epoch, model, optimizer, args, doc_emb, doc_svc, fut_svc, svc_emb):
    
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    idx_list = list(BatchSampler(SequentialSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    #doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)
    fut_svc_ts = torch.tensor(fut_svc, dtype=torch.long, device=dev)
    
    with torch.no_grad():
        for i in range(len(idx_list)):
#             print("===TEST============")
            x = doc_emb_ts[idx_list[i]]
            y = doc_svc_ts[idx_list[i]]
            adj = doc_svc_ts[idx_list[i]]
            fut = fut_svc_ts[idx_list[i]]
            
            pred_y, _ = model(x, adj, svc_emb)
            y=torch.tensor(y, dtype=torch.float)
            fut=torch.tensor(fut, dtype=torch.float)
            y=torch.cat([y,fut],dim=1)
#             print("pred_y",pred_y.shape)
#             print("y",y.shape)
            pred = torch.sigmoid(pred_y)
            test_loss += F.binary_cross_entropy(pred, y, reduction='sum')
            #pred = F.sigmoid(pred_y)
#             print("pred",pred.shape)
            correct1 += accuracy(pred[:,0:1476], y[:,0:1476], 30).item()
            correct2 += accuracy(pred[:,1476:], y[:,1476:], 30).item()
        
            correct3 += precision(pred[:,0:1476], y[:,0:1476], 30).item()
            correct4 += precision(pred[:,1476:], y[:,1476:], 30).item()
#             print(correct)
            #correct += pred.eq(y.view_as(pred)).sum().item()

            if i % args.log_interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (i+1) * args.batch_size, len(doc_emb),
                    100. * (i+1) * args.batch_size / len(doc_emb)))

    test_loss /= len(doc_emb)
    accu1 = 100. * correct1 / len(doc_emb)
    accu2 = 100. * correct2 / len(doc_emb)
    prec1 = 100. * correct3 / len(doc_emb)
    prec2 = 100. * correct4 / len(doc_emb)

    print('Average test loss of epoch {} is {:.4f}, accuracy is {:.2f}%, accuracy is {:.2f}%, precision is {:.2f}%, precision is {:.2f}%.\n'.format(epoch, test_loss, accu1,accu2,prec1,prec2))

    return test_loss

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
    doc_emb_data = pd.read_parquet('saved_data/visit_init_emb.parquet')
    print(doc_emb_data.shape)
    doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
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
    
    doc_emb_data=doc_emb_data[doc_emb_data.index.isin(rndx)]
    doc_emb_data=doc_emb_data.iloc[round(num_doc * args.train_ratio):,:]
    print(doc_emb_data.head())
    PickleUtils.saver('saved_data/visit_emb_prime_test_labels.pkl', doc_emb_data)
    
    
    adj_mat = np.zeros((num_doc, num_svc), dtype=int)
    for i in range(num_doc):
        adj_mat[i, doc_svc[i]] = 1
    
#     fut_mat = np.zeros((num_doc, num_svc), dtype=int)
#     for i in range(num_doc):
#         fut_mat[i, future_svc[i]] = 1
        
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    print(doc_emb_ts.shape)
    doc_svc_ts = torch.tensor(adj_mat, dtype=torch.long, device=dev)
#     fut_ts = torch.tensor(fut_mat, dtype=torch.long, device=dev)
    
    # define and load models
    model = GAT(
        nfeat=doc_emb.shape[1], 
        nhid=int(doc_emb.shape[1] / args.nb_heads), 
        nclass=2*num_svc, 
        dropout=args.dropout, 
        drop_enc=True, 
        alpha=args.alpha, 
        nheads=args.nb_heads).to(dev)

    # load model parameters
    checkpoint = torch.load('saved_models/visit_emb_best')
    model.load_state_dict(checkpoint['model_state_dict'])

    # inference
    visit_emb_prime = test_batch(model, args, doc_emb_ts, doc_svc_ts, svc_emb_ts)
    print("doc_emb_prime",visit_emb_prime.shape)
    PickleUtils.saver('saved_data/visit_emb_prime.pkl', visit_emb_prime)

#     doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
    
#     #doc_spec = doc_emb_data.readmission.values
#     doc_svc = doc_emb_data.svc_id.values

#     spec_emb = np.zeros((2, 128), dtype=float)
#     for i in range(2):
#         spec_emb[i] = np.mean(doc_emb_prime[np.where(doc_spec == i)], axis=0)
#     PickleUtils.saver('saved_data/readmission_emb.pkl', spec_emb)

def main(args):
    # load prepared init data for training doctor embedding
    doc_emb_data = pd.read_parquet('saved_data/visit_init_emb.parquet')
#     print(doc_emb_data.shape)
#     print(doc_emb_data['patientunitstayid'].nunique())
    doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
#     print(doc_emb.shape)
    #doc_spec = doc_emb_data.readmission.values
    doc_svc = doc_emb_data.svc_id.values
    future_svc = doc_emb_data.label.values
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
    future_svc = future_svc[rndx]
    
    adj_mat = np.zeros((num_doc, num_svc), dtype=int)
    for i in range(num_doc):
        adj_mat[i, doc_svc[i]] = 1
        
    fut_mat = np.zeros((num_doc, num_svc), dtype=int)
    for i in range(num_doc):
        fut_mat[i, future_svc[i]] = 1
#     print("adj_mat",adj_mat.shape)   
    doc_emb_train = doc_emb[:round(num_doc * args.train_ratio)]
    doc_emb_test = doc_emb[round(num_doc * args.train_ratio):]

#     doc_spec_train = doc_spec[:round(num_doc * args.train_ratio)]
#     doc_spec_test = doc_spec[round(num_doc * args.train_ratio):]

    doc_svc_train = adj_mat[:round(num_doc * args.train_ratio)]
    doc_svc_test = adj_mat[round(num_doc * args.train_ratio):]
    
    fut_train = fut_mat[:round(num_doc * args.train_ratio)]
    fut_test = fut_mat[round(num_doc * args.train_ratio):]
    
    # define model
    set_rnd_seed(args)
    model = GAT(
        nfeat=doc_emb.shape[1], 
        nhid=int(doc_emb.shape[1] / args.nb_heads), 
        nclass=2*num_svc,
        dropout=args.dropout, 
        drop_enc=True, 
        alpha=args.alpha, 
        nheads=args.nb_heads).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    # train and validation
    best_loss = 10000.

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, optimizer, args, doc_emb_train, doc_svc_train, fut_train, svc_emb_ts)
        test_loss = test(epoch, model, optimizer, args, doc_emb_test, doc_svc_test, fut_test, svc_emb_ts)
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
                    'epoch': epoch
                    }, 'saved_models/visit_emb_best')

def visualize_embeddings(args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
   
    visit_emb_data = pd.read_parquet('saved_data/visit_temporal_emb.parquet')

    print(visit_emb_data.shape)

    
    #emb=emb.data.cpu().numpy()
    visit_emb=visit_emb_data.iloc[0:20000,3:]
    print(visit_emb.shape)
#     print(emb[587,:])
#     print(emb[588,:])
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(visit_emb)
    print(tsne_proj.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    #cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(28,28))
    
#     inv_featVocab = {v: k for k, v in featVocab.items()}
    label = visit_emb_data['readmission'][0:20000]
    print(label)
    #for i in range(0,visit_emb.shape[0]):
    ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
        #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
    plt.show()
        
        
    return    

def ml_models():
    visit_emb_data = pd.read_parquet('saved_data/visit_temporal_emb.parquet')
    X,y=visit_emb_data.iloc[:,3:],visit_emb_data['readmission']
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
    
    model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, y_train)
    prob=model.predict_proba(X_test)
    fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
    auc = metrics.auc(fpr, tpr)
    print("Xgboost",auc)

def ml_models_labels():
    doc_emb_data = pd.read_parquet('saved_data/visit_init_emb.parquet')
    #[doc_emb_data['label']!=None]
    print(doc_emb_data.shape)
    print(doc_emb_data.head())
    # load trained svc embedding
    X = PickleUtils.loader('saved_data/visit_emb_prime.pkl')
    print(X.shape)
    
    X=pd.DataFrame(X)
    X= X.add_suffix('_emb')
    
    doc_emb_data=pd.concat([doc_emb_data,X],axis=1)
    doc_emb_data=doc_emb_data.dropna()
    print(doc_emb_data.shape)
    print(doc_emb_data.head())
    
    
    test_ids = PickleUtils.loader('saved_data/visit_emb_prime_test_labels.pkl')
    print(test_ids.shape)
    
    doc_emb_data=doc_emb_data.merge(test_ids[['patientunitstayid','seq']],on=['patientunitstayid','seq'],how='inner')
    #doc_emb_data=doc_emb_data[doc_emb_data['patientunitstayid'].isin(test_ids[['patientunitstayid','seq']].unique())]
    print(doc_emb_data.shape)
    
    
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


    doc_emb_data['label2']=0
    for index, row in doc_emb_data.iterrows():
        if len(list(set(list(row.label)).intersection(set(renal_ids))))>0:
            #print("true")
            doc_emb_data.loc[index,'label2']=1
#         doc_emb_data.loc[idx,'label'.to_list(),'label2']=1
    #doc_emb_data.loc[len(doc_emb_data.label.to_list())>0,'label2']=1
    print('positives',doc_emb_data[doc_emb_data['label2']==1].shape)
    print('negatives',doc_emb_data[doc_emb_data['label2']==0].shape)
    
    X,y=doc_emb_data.iloc[:,6:134],doc_emb_data['label2']
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
    
    model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, y_train)
    prob=model.predict_proba(X_test)
    fpr, tpr, threshholds = metrics.roc_curve(y_test, prob[:,1])
    auc = metrics.auc(fpr, tpr)
    print("Xgboost",auc)
    
    
    doc_emb_data['label2']=0
    for index, row in doc_emb_data.iterrows():
        if len(list(set(list(row.label)).intersection(set(pulmonary_ids))))>0:
            #print("true")
            doc_emb_data.loc[index,'label2']=1
#         doc_emb_data.loc[idx,'label'.to_list(),'label2']=1
    #doc_emb_data.loc[len(doc_emb_data.label.to_list())>0,'label2']=1
    print('positives',doc_emb_data[doc_emb_data['label2']==1].shape)
    print('negatives',doc_emb_data[doc_emb_data['label2']==0].shape)
    
    X,y=doc_emb_data.iloc[:,6:134],doc_emb_data['label2']
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

    parser.add_argument('--batch-size', type=int, default=5)

    parser.add_argument('--log-interval', type=int, default=5000)

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
#     ml_models()
#     ml_models_labels()

