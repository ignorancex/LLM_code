import torch
import torch_sparse
import torch.nn.functional as F
import copy
import numpy as np
from torch_sparse import SparseTensor



def train_model_pyg_ogb(model,optimizer,x,edge_index,y,train_index,val_index,test_index,epochs,evaluator,save_path=None,patient=40,test_y=None):
    best_val_acc=0.0
    cur_test_acc=0.0
    bad_epoch=0
    for epoch in range(epochs):
        model.train()
        outputs = model(x,edge_index)
        train_loss = F.cross_entropy(outputs[train_index], y[train_index].squeeze(1))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if save_path!=None:
            if test_y!=None:
                train_acc,val_acc,test_acc=test_model_pyg_ogb(model,x,edge_index,test_y,train_index,val_index,test_index,evaluator)
            else:
                train_acc,val_acc,test_acc=test_model_pyg_ogb(model,x,edge_index,y,train_index,val_index,test_index,evaluator)
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                cur_test_acc=test_acc
                torch.save(model.state_dict(), save_path)
                bad_epoch=0
            else:
                bad_epoch+=1
            if bad_epoch==patient:
                break
    return cur_test_acc

def test_model_pyg_ogb(model,x,edge_index,y,train_index,val_index,test_index,evaluator):
    model.eval() 
    with torch.no_grad():
        outputs = model(x,edge_index)
        y_pred = outputs.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': y[train_index],
            'y_pred': y_pred[train_index],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y[val_index],
            'y_pred': y_pred[val_index],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y[test_index],
            'y_pred': y_pred[test_index],
        })['acc']

    return train_acc,val_acc,test_acc*100


def generate_pseduo_label(model,x,edge_index,train_y,train_index,label_index,score_threshold=0.5):
    with torch.no_grad():
        outputs=model(x,edge_index)
        outputs_p=F.normalize(outputs.detach(),dim=1)
        score, index = torch.max(outputs_p.data, 1)
        b=-torch.ones_like(index)
        pseduo_y=torch.where(score>=score_threshold,index,b)
        tensor_isin = torch.isin(label_index,torch.where(pseduo_y>=0)[0])
        label_index=label_index[tensor_isin]
        pseduo_y=pseduo_y[label_index]
        train_y[label_index]=pseduo_y.reshape(-1,1)
        if train_index==None:
            return train_y
        train_index=torch.concat([train_index,label_index])
    return train_y,train_index

def generate_pseduo_label_same(model,x,edge_index,train_index,train_y,pseduo_label_index,model_dropout_rate=0.5,score_threshold=0.0,vote_num=9):
    model.dropout_rate1=model_dropout_rate
    model.train()
    train_index_ori,train_y_num,pseduo_label_index_num=copy.deepcopy(train_index),copy.deepcopy(train_y),copy.deepcopy(pseduo_label_index)
    train_y_prev,train_index_prev=generate_pseduo_label(model,x,edge_index,train_y_num,train_index_ori,pseduo_label_index_num,score_threshold=score_threshold)
    with torch.no_grad():
        for num in range(1,vote_num):
            train_y_num,pseduo_label_index_num=copy.deepcopy(train_y),copy.deepcopy(pseduo_label_index)
            train_y_num=generate_pseduo_label(model,x,edge_index,train_y_num,None,pseduo_label_index_num,score_threshold=score_threshold)
            correct=train_y_prev.eq(train_y_num)
            correct_index=torch.nonzero(correct==True,as_tuple=True)[0]
            train_index_prev=train_index_prev[torch.isin(train_index_prev,correct_index)]
    return train_y_prev,train_index_prev


def generate_inclass_mixup_with_high_confidence_edge(x,edge_index,noise_y,noise_y_high,clean_y,label_index,label_index_high,train_index,class_num,device):
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
        noise_y=torch.concat([noise_y,class_name*torch.ones(len(class_index)).reshape(-1,1).to(device)])
        clean_y=torch.concat([clean_y,class_name*torch.ones(len(class_index)).reshape(-1,1).to(device)])

        add_edge_node=torch.arange(start_mixup_node_index,start_mixup_node_index+len(class_index),1)
        start_edge_node=torch.concat([add_edge_node,add_edge_node],dim=0)
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

        train_index=torch.concat([train_index,add_edge_node.to(device)])
    isin_result = torch.isin(train_index,label_index)
    noise_y=torch.tensor(noise_y,dtype=torch.int64).to(device)
    clean_y=torch.tensor(clean_y,dtype=torch.int64).to(device)
    return x,noise_y,clean_y,edge_index,train_index


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
