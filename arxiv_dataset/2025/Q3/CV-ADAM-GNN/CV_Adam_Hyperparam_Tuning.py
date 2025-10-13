# -*- coding: utf-8 -*-


import torch
from dataloder import AdjSampler
from VRGCN import VRGCN
import time
import json
import copy
from CV_Adam_Functions import train,test
from GraphData import GraphData

###################################################################


#datasets
#dataset_name ='ogbn-arxiv'
dataset_name ='Cora'
#dataset_name ='CiteSeer'
#dataset_name ='Reddit'
#dataset_name ='flickr'


GraphData=GraphData(dataset_name)
data,num_in_channels,num_out_channels=GraphData.data_preprocessing() #returns CogDL graph object
epochs=GraphData.epochs
weight_decay=GraphData.weight_decay
dropout=GraphData.dropout
hidden_size=GraphData.hidden_size

print(f"dataset_name wd={weight_decay} dropout={dropout} hidden dim={hidden_size}")

runs=3
num_layers=2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = VRGCN(
    num_nodes=data.x.shape[0],
    in_channels=num_in_channels,
    hidden_channels=hidden_size,
    out_channels=num_out_channels,
    dropout=dropout,
    num_layers=num_layers,
    device=device,
).to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)

#saving list of randomly generated model states to initialize for each optimizer for each run
init_model_weights_list=[]
for i in range(runs):
    model.reset_parameters() #randomly initializes model parameters
    initial_model_weights=copy.deepcopy(model.state_dict()) #saves initial model params for that run
    init_model_weights_list.append(initial_model_weights)



optimizer_list=['Adam','HeavyBall','AMSGrad','AdaGrad','SGD']    
    

#############creating lists of hyperparameters to search#############################
#[lr,n_neighbors,bsz]

parameters_list=[]

if dataset_name=="Cora" or dataset_name=="CiteSeer":
    lr_vals=[.001,.01,.05]
    sampled_neighbor_vals=[2,5]
    batch_size=[10,20,50]
if dataset_name=="ogbn-arxiv":
    lr_vals=[.001,.005,.01]
    sampled_neighbor_vals=[2,5]
    batch_size=[1000,2048,5000]
if dataset_name=="flickr":
    lr_vals=[.1,.5,.8]
    sampled_neighbor_vals=[2,5]
    batch_size=[1000,2000,5000]
if dataset_name=="Reddit":
    lr_vals=[.01,.05]
    sampled_neighbor_vals=[2]
    batch_size=[1000]

for l in lr_vals:
    for n in sampled_neighbor_vals:
        for b in batch_size:
            hyperparameters=[l,n,b]
            parameters_list.append(hyperparameters)
##################################################################

for optim in optimizer_list:
    optimizer_results_dict={
        "dataset_name":dataset_name,
        "optimizer":optim,
        "epochs":epochs,
        "runs":runs,
        "results_list":[]
        }
    experiment_num=1
    for experiment_params in parameters_list:
        print(f"{optim} experiment {experiment_num}")
        
        start=time.time()
        
        lr=experiment_params[0]
        neighbors=experiment_params[1]
        num_neighbors=[neighbors,neighbors] #hard coded for 2 layers
        batch_size=experiment_params[2]
        
        results_dict={
            "experiment_num":experiment_num,
            "hyperparameters":experiment_params,
            "optimizer":optim,
            "train_acc":[],
            "test_acc":[],
            "val_acc":[],
            "loss_list":[]
            }

       
        train_loader = AdjSampler(data, sizes=num_neighbors, batch_size=batch_size, shuffle=True)
        test_loader = AdjSampler(data, sizes=[-1], batch_size=batch_size, shuffle=False, training=False)
        
        for run in range(runs):        
           
            #using the same initial weights for each run #
            initial_model_weights=init_model_weights_list[run]
            model.load_state_dict(initial_model_weights)
            model.eval()
            model.initialize_history(x, test_loader) #initializes H bar matrices with forward pass using symmetric normalized A matrix
           
            
            #initializing accuracy and loss lists
            epoch_test_acc_list=[]
            epoch_train_acc_list=[]
            epoch_val_acc_list=[]
            epoch_loss_list=[]
           
            
            #getting initial accuracies
            train_acc0,val_acc0,test_acc0=test(model,x,y,data,test_loader)
            epoch_test_acc_list.append(test_acc0)
            epoch_train_acc_list.append(train_acc0)
            epoch_val_acc_list.append(val_acc0)
            
            
            #defining optimizers
            beta1=0.9
            beta2=0.999
            if optim=='SGD':
                optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay)
            elif optim=='HeavyBall':
                optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=beta1,dampening=beta1)
            elif optim=='Adam':
                optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(beta1,beta2),eps=1e-08,weight_decay=weight_decay,amsgrad=False)
            elif optim=='AMSGrad':
                optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(beta1,beta2),eps=1e-08,weight_decay=weight_decay,amsgrad=True)
            elif optim=='AdaGrad':
                optimizer=torch.optim.Adagrad(model.parameters(),lr=lr,weight_decay=weight_decay,eps=1e-08)
            
            
            for epoch in range(epochs):
                
                #training
                loss = train(model,optimizer,x,y,train_loader)
                epoch_loss_list.append(loss)
                
                #testing
                train_acc, val_acc, test_acc = test(model,x,y,data,test_loader)

                epoch_test_acc_list.append(test_acc)
                epoch_train_acc_list.append(train_acc)
                epoch_val_acc_list.append(val_acc)
               
            #adding accuracy list for the current run to list of accuracies in results_dict
            results_dict["train_acc"].append(epoch_train_acc_list)
            results_dict["test_acc"].append(epoch_test_acc_list)
            results_dict["val_acc"].append(epoch_val_acc_list)
            results_dict["loss_list"].append(epoch_loss_list)
            
        end=time.time()
        print(f"time for experiment {experiment_num} = {end-start}")
        results_dict["time"]=end-start
        optimizer_results_dict["results_list"].append(results_dict)
        experiment_num+=1
        
    #saving dictionary for all hyperparam experiments for one optimizer
    file_name=dataset_name+"_"+optim+"_hyperparam_experiments.json"
    with open(file_name,'w') as json_file:
        json.dump(optimizer_results_dict,json_file)
    
       
        
      
