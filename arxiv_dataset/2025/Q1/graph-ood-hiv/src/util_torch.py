import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch.nn import Linear, ReLU
from torch_geometric.nn import GeneralConv
from torch_geometric.nn import global_max_pool
import torch.nn as nn
import multiprocessing
import torch_geometric
import random
import numpy as np
import shutil

class LightningMIX(pl.LightningModule):
    def __init__(self, lr=0.001, **kwargs):
        super().__init__()
        self.gcn = CustomGCN(**kwargs)
        self.fc = CustomFC(**kwargs)
        self.layer_out = nn.Linear(2*kwargs["num_classes"], kwargs["real_num_classes"])
        self.sigmoid = nn.Sigmoid()
        self.lr = lr

        self.loss = torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"])

    def forward(self, fc_x, graph_x, edge_index, edge_attr, batch_index):
        fc_z = self.fc(fc_x)
        gnn_z = self.gcn(graph_x, edge_index, edge_attr, batch_index)
        z = self.sigmoid(self.layer_out(torch.cat((fc_z,gnn_z),axis=-1)))
        return z

    def step(self, batch, batch_index, split = "test"):
        fc_x, graph_batch = batch
        graph_x, edge_index, edge_attr = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
        batch_index = graph_batch.batch

        x_out = self(fc_x.float(), graph_x,edge_index, edge_attr, batch_index)
        
        label = graph_batch.y
        loss_value = self.loss(x_out, label)

        # metrics here
        pred = x_out.argmax(-1)
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/"+split, loss_value)
        self.log("accuracy/"+split, accuracy)

        return loss_value, accuracy

    def training_step(self, batch, batch_index):
        return self.step(batch, batch_index, "train")[0] #0 beacuse in training just loss

    def validation_step(self, batch, batch_index):
        return self.step(batch, batch_index, "val")
      
    def test_step(self, batch, batch_index):
        return self.step(batch, batch_index, "test")
        
    def predict_step(self, batch, batch_idx):
        fc_x, graph_batch = batch
        graph_x, edge_index, edge_attr = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
        batch_index = graph_batch.batch

        return self(fc_x.float(), graph_x, edge_index, edge_attr, batch_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
        
class CustomGCN(nn.Module):
    def __init__(self, **kwargs):
        super(CustomGCN, self).__init__()

        self.num_features = kwargs["num_graph_features"] \
                    if "num_graph_features" in kwargs.keys() else 3
        self.num_classes = kwargs["num_classes"] \
                    if "num_classes" in kwargs.keys() else 2

        # hidden layer node features
        self.hidden = 4

        self.model = Sequential("x, edge_index, edge_weight, batch_index", [
                (GeneralConv(self.num_features, self.hidden, in_edge_channels=1, attention=True, heads = 5, directed_msg=True), 
                    "x, edge_index, edge_weight -> x1"),
                (ReLU(), "x1 -> x1d"),                                         
                (GeneralConv(self.hidden, self.hidden*2, in_edge_channels=1, attention=True, heads = 5, directed_msg=True), "x1d, edge_index, edge_weight -> x2"),  
                (ReLU(), "x2 -> x2d"),                                        
                (GeneralConv(self.hidden*2, self.hidden*4, in_edge_channels=1, attention=True, heads = 5, directed_msg=True), "x2d, edge_index, edge_weight -> x3"),  
                (ReLU(), "x3 -> x3d"),                                                               
                (global_max_pool, "x3d, batch_index -> x6"),                  
                (Linear(self.hidden*4, self.num_classes), "x6 -> x_out")])
         
    def forward(self, x, edge_index, edge_attr, batch_index):
        x_out = self.model(x, edge_index, edge_attr, batch_index)
        return x_out      

class LightningGCN(pl.LightningModule):
    def __init__(self, loss_weight=None, **kwargs):
        super().__init__()

        self.model = CustomGCN(**kwargs) 

        self.loss = torch.nn.CrossEntropyLoss(weight=loss_weight)
    
    def forward(self, x, edge_index, edge_attr, batch_index):
        x_out = self.model(x, edge_index, edge_attr, batch_index)
        return x_out

    def step(self, batch, batch_index, split = "test"):
        x, edge_index,edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, edge_attr, batch_index)
        
        loss_value = self.loss(x_out, batch.y)

        # metrics here
        pred = x_out.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/"+split, loss_value)
        self.log("accuracy/"+split, accuracy)

        return loss_value, accuracy

    def training_step(self, batch, batch_index):
        return self.step(batch, batch_index, "train")[0] #0 beacuse in training just loss

    def validation_step(self, batch, batch_index):
        return self.step(batch, batch_index, "val")
      
    def test_step(self, batch, batch_index):
        return self.step(batch, batch_index, "test")

    def predict_step(self, batch, batch_idx):
        x, edge_index,edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_index = batch.batch
        
        return self(x, edge_index, edge_attr, batch_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.001)


class CustomFC(nn.Module):
    def __init__(self, num_features, embedding_sizes, num_classes, **kwargs):
        super(CustomFC, self).__init__()

        self.layer_1 = nn.Linear(num_features, embedding_sizes[0])
        self.layer_2 = nn.Linear(embedding_sizes[0], embedding_sizes[1])
        self.layer_out = nn.Linear(embedding_sizes[1], num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        return x

class FullyConnectedNN(pl.LightningModule):
    def __init__(self, num_features, embedding_sizes, num_classes, lr=0.001, loss_weight=None, **kwargs):
        super().__init__()

        self.model = CustomFC(num_features, embedding_sizes, num_classes)

        self.lr = lr

        self.loss = torch.nn.CrossEntropyLoss(weight=loss_weight)

    def forward(self,x):
        return self.model(x)

    def configure_optimizers(self):
      lr = self.lr
      optimizer = torch.optim.AdamW(self.parameters(),lr=lr)
      return optimizer

    def step(self, batch, batch_idx, split):
        x, y = batch
        y_hat = self(x)
        loss_value = self.loss(y_hat, y)
        self.log('loss/'+split, loss_value)
        acc = 1-((y_hat>0.5).int() - y[:,None].int()).abs().float().mean() #[:,None] needed if y has len(shape)=1
        self.log('accuracy/'+split, acc)
        return loss_value, acc

    def training_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"train")[0] #[0] beacuse during training only loss is needed

    def validation_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"val")

    def test_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x,y = batch
        return self(x)

def check_checkpoint_path(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

def prepare_loaders_FC(data, batch_size):
    #Create loaders
    loaders = {}
    for split in ["train", "val", "test"]:
        loaders[split] = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data["x_"+split]),torch.Tensor(data["y_"+split]).long()), batch_size = batch_size, num_workers = multiprocessing.cpu_count(), pin_memory=False, persistent_workers = True, shuffle = split=="train")
    return loaders

def prepare_trainer(num_epochs,checkpoint_path):
    callbacks = []
    callbacks.append(pl.callbacks.early_stopping.EarlyStopping(patience=10, monitor="loss/val", mode="min"))

    #if checkpoint_path exists: delete folder
    check_checkpoint_path(checkpoint_path)
    
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, filename="best", save_top_k = 1, save_last = False, monitor= "loss/val", mode= "min"))

    trainer = pl.Trainer(max_epochs = num_epochs, callbacks=callbacks, accelerator = ["cpu","gpu"][torch.cuda.is_available()]) #val_check_interval=val_check_interval

    return trainer

def test_model(model,trainer,loaders):
    for split in ["train", "val", "test"]:
        trainer.test(model,loaders[split])

def reload_FC(checkpoint_path, num_features, num_classes, data, loaders, batch_size, class_weight):
    PATH_saved_model = os.path.join(checkpoint_path,"best.ckpt")
    model = FullyConnectedNN.load_from_checkpoint(PATH_saved_model,num_features=num_features, embedding_sizes=[64,256], num_classes=num_classes, loss_weight=class_weight)
    trainer = pl.Trainer()
    loaders["train"] = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data["x_train"]),torch.Tensor(data["y_train"]).long()), batch_size = batch_size, num_workers = multiprocessing.cpu_count(), pin_memory=False, persistent_workers = True)

    return model, trainer, loaders

def reload_GNN(checkpoint_path, num_graph_features, num_classes, dataset, index_traindata, num_val, loaders, batch_size, class_weight):
    PATH_saved_model = os.path.join(checkpoint_path,"best.ckpt")
    model = LightningGCN.load_from_checkpoint(PATH_saved_model, num_graph_features = num_graph_features, num_classes = num_classes, loss_weight=class_weight)
    trainer = pl.Trainer(accelerator = ["cpu","gpu"][torch.cuda.is_available()])
    loaders["train"] = torch_geometric.loader.DataLoader(dataset['train'][list(index_traindata[num_val:])], batch_size=batch_size, num_workers = multiprocessing.cpu_count()//2, pin_memory=False, persistent_workers = True)

    return model, trainer, loaders

def reload_MIX(checkpoint_path, dataset, index_traindata, num_val, loaders, batch_size, **kwargs):
    PATH_saved_model = os.path.join(checkpoint_path,"best.ckpt")
    model = LightningMIX.load_from_checkpoint(PATH_saved_model, **kwargs)
    trainer = pl.Trainer(accelerator = ["cpu","gpu"][torch.cuda.is_available()])
    loaders["train"] = torch_geometric.loader.DataLoader(dataset['train'][list(index_traindata[num_val:])], batch_size=batch_size, num_workers = multiprocessing.cpu_count()//2, pin_memory=False, persistent_workers = True)

    return model, trainer, loaders

def prepare_loaders_GNN(dataset,batch_size,val_split=0.2, random_seed=42):
    #shuffle dataset and get train/validation/test splits
    random.seed(random_seed)

    num_samples = len(dataset['train'])
    index_traindata = np.random.choice(range(num_samples),size=num_samples,replace=False)

    num_val = int(num_samples * val_split)

    loaders = {}
    loaders["train"] = torch_geometric.loader.DataLoader(dataset['train'][list(index_traindata[num_val:])], batch_size=batch_size, num_workers = multiprocessing.cpu_count()//2, pin_memory=False, persistent_workers = True, shuffle=True)
    loaders["val"] = torch_geometric.loader.DataLoader(dataset['train'][list(index_traindata[:num_val])], batch_size = batch_size, num_workers = multiprocessing.cpu_count()//2, pin_memory=False, persistent_workers = True)
    loaders["test"] = torch_geometric.loader.DataLoader(dataset['test'], batch_size=batch_size, num_workers = multiprocessing.cpu_count()//2, pin_memory=False, persistent_workers = True)

    return num_val, index_traindata, loaders