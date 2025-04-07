import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as np

def save_model(model,baseline = False, file_path = None):
    if file_path == None:
      timestr = time.strftime("%Y%m%d-%H%M%S")
      file_path = f'models/weights/{model.trial}_weights50k_lv{model.encoder.latent_dim}_{timestr}.pt'

    torch.save(model.state_dict(), file_path)
    #files.download(file_path) 
    return print(f'model weights downloaded: {file_path}')

def load_model(model,file_path):
    # load a trained model
    return model.load_state_dict(torch.load(file_path))


def train_model(model,dataloader,n_epochs= 1000, beta = 0.005, save = True):
    """
    Train a model
    Parameters:
    model: model to train
    dataloader: pytorch DataLoader
    n_epochs: number of epochs to train for
    beta: beta parameter on KL-div loss
    save: True or False
    """

    model = model.train()
    L2_loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr = 1e-3) #torch.optim.Adam(model.parameters(),lr = 1e-3, weight_decay=1e-4)

    for epoch in range(n_epochs): #100
      ep_loss = 0
      kldiv_loss = 0
      ep_rec_loss = 0
      for i_batch, minibatch in enumerate(dataloader):
        inputs, outputs = minibatch
        optim.zero_grad()
        dxdt_hat = model(inputs)
        dxdt = outputs

        # add beta vae loss
        beta = beta #deafult = 0.005 selecting a small beta value to balance hnn_loss 

        # train with just hnn_loss for first couple of epochs
        if epoch > 5: 
          kl_loss = beta* model.encoder.kl_divergence(model.encoder.mu, model.encoder.log_var, dim = model.encoder.latent_dim)  #model.encoder.kl_loss
        else: 
          kl_loss = 0
        rec_loss =  L2_loss(dxdt, dxdt_hat)
        loss = rec_loss + kl_loss
        loss.backward()
        optim.step()
        ep_loss += loss.detach().numpy()
        ep_rec_loss += rec_loss.detach().numpy()
        if kl_loss != 0:
          kldiv_loss += kl_loss.detach().numpy()
      print(f'epoch: {epoch} -- total_loss = {ep_loss} -- kl_loss = {kldiv_loss} -- rec_loss = {ep_rec_loss}')

    if save == True:
      save_model(model = model)
    return model.eval()
