#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as t
from preprocessing import generate_data,generate_batch
from settings import *
from network_functions import forward,optimize
def evaluate(network,epoch,test_images,nb_test_batches,results_train_test,best_result,best_epoch):
    with t.no_grad():
        network.eval()
        errors=[]
        generator=generate_data(test_images,batch_size,True)
        total_time_test=0
        for nb in range(nb_test_batches):
            test_image=generate_batch(generator,batch_size)
            inputs_placeholder=t.tensor(test_image).float().permute([0,3,1,2]).cuda()
            elapsed_time=forward(network,epoch,inputs_placeholder,nb,nb_test_batches,True)
            errors.append(network.error.cpu().data)
        if t.stack(errors).mean()<best_result.cpu().data:
            t.save({'model_state_dict':network.state_dict(),'optimizer_state_dict':network.optimizer.state_dict(),},'checktrain/checkpoint')
            best_result=t.stack(errors).mean()
            best_epoch=epoch
        results_train_test.write(('\nStep %d:\nMean Acuuracy=%5f\nbest@%5f\nbestepoch#%5d'
                  % (epoch,t.stack(errors).mean(),best_result,best_epoch)))
        results_train_test.flush()
    return best_result,best_epoch
def train(network,epoch,images,nb_batches,results_train,best_result,best_epoch):
    network.train()
    errors=[]
    generator=generate_data(images,batch_size,False)
    for nb in range(nb_batches):
        trial_image=generate_batch(generator,batch_size)
        inputs_placeholder=t.tensor(trial_image).float().permute([0,3,1,2]).cuda()
        network.zero_grad()
        elapsed_time=forward(network,epoch,inputs_placeholder,nb,nb_batches,False)
        optimize(network)
        errors.append(network.error.cpu().data)
    results_train.write(('\nStep %d:\nMean Accuracy=%5f\n best@%5f\nbestepoch#%5d'
              % (epoch,t.stack(errors).mean(),best_result,best_epoch)))
    results_train.flush()


# In[ ]:




