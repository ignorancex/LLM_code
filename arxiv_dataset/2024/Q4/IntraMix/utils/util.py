import copy
import torch
import numpy as np
import torch.nn as nn
import torch_sparse
from torch_sparse import SparseTensor
import torch.nn.functional as F 
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,Coauthor,Flickr
from ogb.nodeproppred  import PygNodePropPredDataset

from model.gcn import GCN,GCNNet
from model.gat import GAT,GAT_BN
from model.gsage import GraphSAGE,GraphSAGE_BN
from model.grand import GRAND
from model.gcnii import GCNII
from model.appnp import APPNP_Net,APPNP_BN

def load_model(args,num_node_features,num_classes,hid_num,dropout_rate1,dropout_rate2):
    if args.model_name=='gcn':
        if args.dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-proteins','Flickr']:
            model=GCNNet(num_node_features,num_classes,dropout_rate1,dropout_rate2,hid_num)
        else:
            model=GCN(num_node_features,num_classes,hid_num,dropout_rate1,dropout_rate2)
    elif args.model_name=='gat':
        if args.dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-proteins','Flickr']:
            model=GAT_BN(num_node_features,num_classes,hid_num,args.layer,args.nb_heads,args.nb_heads2,dropout_rate1,dropout_rate2,args.alpha)
        else:
            model=GAT(num_node_features,num_classes,hid_num,args.nb_heads,args.nb_heads2,dropout_rate1,dropout_rate2,args.alpha)
    elif args.model_name=='gsage':
        if args.dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-proteins','Flickr']:
            model=GraphSAGE_BN(num_node_features,num_classes,hid_num,args.layer,dropout_rate1,dropout_rate2)
        else:
            model=GraphSAGE(num_node_features,num_classes,hid_num,dropout_rate1,dropout_rate2)
    elif args.model_name=='grand':
        model=GRAND(num_node_features,hid_num,num_classes,dropout_rate1,dropout_rate2,args,args.use_bn)
    elif args.model_name=='gcnii':
        model=GCNII(num_node_features,num_classes,hid_num,args,dropout_rate1,dropout_rate2)
    elif args.model_name=='appnp':
        if args.dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-proteins','Flickr']:
            model=APPNP_BN(num_node_features,num_classes,args,args.layer,hid_num,dropout_rate1,dropout_rate2)
        else:
            model=APPNP_Net(num_node_features,num_classes,args,hid_num,dropout_rate1,dropout_rate2)
    return model

def load_data(dataset_name,root_path,device):
    if dataset_name in ['Cora','CiteSeer','Pubmed']:
        dataset=Planetoid(root_path,dataset_name,transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        x,y,train_mask,val_mask,test_mask,edge_index=copy.deepcopy(data.x).to(device),copy.deepcopy(data.y).to(device),\
            copy.deepcopy(data.train_mask),copy.deepcopy(data.val_mask),copy.deepcopy(data.test_mask),copy.deepcopy(data.edge_index)
        train_index=torch.nonzero(train_mask==True,as_tuple=True)[0]
        val_index=torch.nonzero(val_mask==True,as_tuple=True)[0]
        test_index=torch.nonzero(test_mask==True,as_tuple=True)[0]
    elif dataset_name in ['ogbn-arxiv','ogbn-products',]:  
        dataset = PygNodePropPredDataset(name = dataset_name, root = root_path,transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        x,y,train_index,val_index,test_index,edge_index=copy.deepcopy(data.x).to(device),copy.deepcopy(data.y.squeeze(1)).to(device),\
            copy.deepcopy(split_idx['train']).to(device),copy.deepcopy(split_idx['valid']).to(device),copy.deepcopy(split_idx['test']).to(device),copy.deepcopy(data.adj_t).to(device)
    elif dataset_name in ['ogbn-proteins']:
        dataset = PygNodePropPredDataset(name = dataset_name, root = root_path,transform=T.ToSparseTensor(attr='edge_attr'))
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data.x= data.adj_t.mean(dim=1)
        
        data.adj_t.set_value_(None)

        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
        split_idx = dataset.get_idx_split()
        print(data.y.size())
        print(data.x.size())
        x,y,train_index,val_index,test_index,edge_index=copy.deepcopy(data.x).to(device),copy.deepcopy(data.y).to(device),\
            copy.deepcopy(split_idx['train']).to(device),copy.deepcopy(split_idx['valid']).to(device),copy.deepcopy(split_idx['test']).to(device),copy.deepcopy(data.adj_t).to(device)
    elif dataset_name in ['Flickr']:
        dataset=Flickr(root_path,pre_transform=T.Compose([T.NormalizeFeatures()]))
        data = dataset[0].to(device)
        x,y,train_mask,val_mask,test_mask,edge_index=copy.deepcopy(data.x).to(device),copy.deepcopy(data.y).to(device),\
            copy.deepcopy(data.train_mask),copy.deepcopy(data.val_mask),copy.deepcopy(data.test_mask),copy.deepcopy(data.edge_index)
        train_index=torch.nonzero(train_mask==True,as_tuple=True)[0]
        val_index=torch.nonzero(val_mask==True,as_tuple=True)[0]
        test_index=torch.nonzero(test_mask==True,as_tuple=True)[0]
    elif dataset_name in ['CS','Physics']:
        dataset=Coauthor(root_path,dataset_name,transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        x,y,edge_index=data.x.to(device),data.y.to(device),data.edge_index.to(device)
        train_mask,val_mask,test_mask=random_splits_CS(labels=data.y,num_classes=dataset.num_classes,percls_trn=20,percls_val=30)
        train_index=torch.nonzero(train_mask==True,as_tuple=True)[0]
        val_index=torch.nonzero(val_mask==True,as_tuple=True)[0]
        test_index=torch.nonzero(test_mask==True,as_tuple=True)[0]    
    return x,y,edge_index,train_index,val_index,test_index,dataset

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits_CS(labels, num_classes, percls_trn=20, percls_val=30):
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(rest_index, size=num_nodes)

    return train_mask, val_mask, test_mask

def train_model_pyg_dynamic_data(model,optimizer,x,edge_index,y,train_mask,val_mask,test_mask,epochs,save_path=None,patient=40,test_y=None):
    best_val_acc=0.0
    cur_test_acc=0.0
    bad_epoch=0
    for epoch in range(epochs):
        model.train()
        outputs = model(x,edge_index)
        train_loss = F.cross_entropy(outputs[train_mask], y[train_mask])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if save_path!=None:
            if test_y!=None:
                train_loss_,train_acc,val_loss,val_acc,test_acc=test_model_pyg_dynamic_data(model,x,edge_index,test_y,train_mask,val_mask,test_mask)
            else:
                train_loss_,train_acc,val_loss,val_acc,test_acc=test_model_pyg_dynamic_data(model,x,edge_index,y,train_mask,val_mask,test_mask)
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                cur_test_acc=test_acc
                torch.save(model.state_dict(), save_path)
                bad_epoch=0
            else:
                bad_epoch+=1
            if bad_epoch==patient:
                break
            print("epoch:{},val_acc:{},test_acc:{}".format(epoch,val_acc.item(),test_acc.item()))
    return cur_test_acc

def test_model_pyg_dynamic_data(model,x,edge_index,y,train_mask,val_mask,test_mask):
    model.eval()  # val
    with torch.no_grad():
        outputs = model(x,edge_index)
        train_loss_ = F.cross_entropy(outputs[train_mask], y[train_mask]).item()
        train_pred = outputs[train_mask].max(dim=1)[1].type_as(y[train_mask])
        train_correct = train_pred.eq(y[train_mask]).double()
        train_correct = train_correct.sum()
        train_acc = (train_correct / len(y[train_mask])) * 100
        
        val_loss = F.cross_entropy(outputs[val_mask], y[val_mask]).item()
        val_pred = outputs[val_mask].max(dim=1)[1].type_as(y[val_mask])
        correct = val_pred.eq(y[val_mask]).double()
        correct = correct.sum()
        val_acc = (correct / len(y[val_mask])) * 100
    model.eval()
    with torch.no_grad():
        test_pred = outputs[test_mask].max(dim=1)[1].type_as(y[test_mask])
        correct = test_pred.eq(y[test_mask]).double()
        correct = correct.sum()
        test_acc = (correct / len(y[test_mask])) * 100
    return train_loss_,train_acc,val_loss,val_acc,test_acc

def generate_pseduo_label(model,x,edge_index,train_mask,train_y,label_index,score_threshold=0.5):
    with torch.no_grad():
        outputs=model(x,edge_index)
        outputs_p=nn.functional.normalize(outputs.detach(),dim=1)
        score, index = torch.max(outputs_p.data, 1)
        b=-torch.ones_like(index)
        pseduo_y=torch.where(score>=score_threshold,index,b)
        tensor_isin = torch.isin(label_index,torch.where(pseduo_y>=0)[0])
        label_index=label_index[tensor_isin]
        pseduo_y=pseduo_y[label_index]
        train_y[label_index]=pseduo_y
        train_mask[label_index]=True
    return train_mask,train_y,label_index

def generate_pseduo_label_same(model,x,edge_index,train_mask,train_y,pseduo_label_index,model_dropout_rate=0.5,score_threshold=0.0,vote_num=5):
    model.dropout_rate1=model_dropout_rate
    model.train()
    train_mask_num,train_y_num,pseduo_label_index_num=copy.deepcopy(train_mask),copy.deepcopy(train_y),copy.deepcopy(pseduo_label_index)
    train_mask_prev,train_y_prev,_=generate_pseduo_label(model,x,edge_index,train_mask_num,train_y_num,pseduo_label_index_num,score_threshold=score_threshold)
    with torch.no_grad():
        for num in range(1,vote_num):
            train_mask_num,train_y_num,pseduo_label_index_num=copy.deepcopy(train_mask),copy.deepcopy(train_y),copy.deepcopy(pseduo_label_index)
            train_mask_num,train_y_num,_=generate_pseduo_label(model,x,edge_index,train_mask_num,train_y_num,pseduo_label_index_num,score_threshold=score_threshold)
            correct=train_y_prev.eq(train_y_num)
            train_mask_prev=~(~train_mask_prev+~correct)
    label_train=torch.nonzero(train_mask_prev==True,as_tuple=True)[0]
    return train_mask_prev,train_y_prev,label_train


def generate_inclass_mixup_with_high_confidence_edge(x,edge_index,noise_y,noise_y_high,clean_y,label_index,label_index_high,train_mask,val_mask,test_mask,class_num,device):
    for class_name in range(class_num):
        class_index=torch.nonzero(noise_y==class_name).squeeze().cpu()
        class_index=class_index[np.isin(class_index.cpu(), label_index.cpu())]
        high_class_index=torch.nonzero(noise_y_high==class_name).squeeze().cpu()
        high_class_index=high_class_index[np.isin(high_class_index.cpu(), label_index_high.cpu())]
        class_len=min(len(high_class_index),len(class_index))
        class_index,high_class_index=class_index[:class_len],high_class_index[:class_len]

        end_edge_node1=high_class_index[torch.randperm(high_class_index.nelement())]
        end_edge_node2=high_class_index[torch.randperm(high_class_index.nelement())]

        lam = np.random.beta(2, 2)
        random_index = class_index[torch.randperm(class_index.nelement())]
        mixed_x = lam * x[class_index,:] + (1 - lam) * x[random_index, :]

        start_mixup_node_index=x.size()[0]
        x=torch.concat([x,mixed_x.to(device)],dim=0) 
        noise_y=torch.concat([noise_y,class_name*torch.ones(len(class_index)).to(device)])
        clean_y=torch.concat([clean_y,class_name*torch.ones(len(class_index)).to(device)])

        start_edge_node=torch.arange(start_mixup_node_index,start_mixup_node_index+len(class_index),1)
        start_edge_node=torch.concat([start_edge_node,start_edge_node],dim=0)
        end_edge_node=torch.concat([end_edge_node1,end_edge_node2],dim=0)

        if isinstance(edge_index,SparseTensor):
            add_edge = SparseTensor(row=start_edge_node, col=end_edge_node).to(device)
            edge_index=torch_sparse.add(edge_index,add_edge)
            edge_index=edge_index.to_symmetric()
        else:
            add_edge=torch.stack([start_edge_node,end_edge_node],dim=0)
            add_edge_2=torch.stack([end_edge_node,start_edge_node],dim=0)
            edge_index=torch.concat([edge_index,add_edge.to(device)],dim=1)
            edge_index=torch.concat([edge_index,add_edge_2.to(device)],dim=1)

        train_mask=torch.concat([train_mask,torch.Tensor([True]*len(class_index)).bool().to(device)])
        val_mask=torch.concat([val_mask,torch.Tensor([False]*len(class_index)).bool().to(device)])
        test_mask=torch.concat([test_mask,torch.Tensor([False]*len(class_index)).bool().to(device)])
    train_mask[label_index]=False
    noise_y=torch.tensor(noise_y,dtype=torch.int64).to(device)
    clean_y=torch.tensor(clean_y,dtype=torch.int64).to(device)
    return x,noise_y,clean_y,edge_index,train_mask,val_mask,test_mask


def generate_inclass_mixup(x,edge_index,noise_y,clean_y,label_index,train_mask,val_mask,test_mask,class_num,device):
    for class_name in range(class_num):
        class_index=torch.nonzero(noise_y==class_name).squeeze().cpu()
        class_index=class_index[np.isin(class_index.cpu(), label_index.cpu())]
        lam = np.random.beta(2, 2)
        random_index = class_index[torch.randperm(class_index.nelement())]
        mixed_x = lam * x[class_index,:] + (1 - lam) * x[random_index, :]
        
        start_mixup_node_index=x.size()[0]
        x=torch.concat([x,mixed_x.to(device)],dim=0) 
        noise_y=torch.concat([noise_y,class_name*torch.ones(len(class_index)).to(device)])
        clean_y=torch.concat([clean_y,class_name*torch.ones(len(class_index)).to(device)])
        start_edge_node=torch.arange(start_mixup_node_index,start_mixup_node_index+len(class_index),1)
        
        start_edge_node=torch.concat([start_edge_node,start_edge_node],dim=0)
        end_edge_node=torch.concat([class_index,random_index],dim=0)
    
        if isinstance(edge_index,SparseTensor):
            add_edge = SparseTensor(row=start_edge_node, col=end_edge_node).to(device)
            edge_index=torch_sparse.add(edge_index,add_edge)
            edge_index=edge_index.to_symmetric()
        else:
            add_edge=torch.stack([start_edge_node,end_edge_node],dim=0)
            add_edge_2=torch.stack([end_edge_node,start_edge_node],dim=0)
            edge_index=torch.concat([edge_index,add_edge.to(device)],dim=1)
            edge_index=torch.concat([edge_index,add_edge_2.to(device)],dim=1)

        train_mask=torch.concat([train_mask,torch.Tensor([True]*len(class_index)).bool().to(device)])
        val_mask=torch.concat([val_mask,torch.Tensor([False]*len(class_index)).bool().to(device)])
        test_mask=torch.concat([test_mask,torch.Tensor([False]*len(class_index)).bool().to(device)])
    noise_y=torch.tensor(noise_y,dtype=torch.int64).to(device)
    train_mask[label_index]=False
    clean_y=torch.tensor(clean_y,dtype=torch.int64).to(device)
    return x,noise_y,clean_y,edge_index,train_mask,val_mask,test_mask

