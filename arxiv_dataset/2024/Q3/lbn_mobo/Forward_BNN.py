import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import scipy.io as sio
import argparse
import numpy as np
import time
import os
from layer_config_forward import MultiLayerPerceptron_forward


def BNN_diverse_func(net_n, iter_n):
  print(net_n)
  def weights_init(m):
      if type(m) == nn.Linear:
          m.weight.data.normal_(0.0, 1e-3)
          m.bias.data.fill_(0.)

  def update_lr(optimizer, lr):
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  def count_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)
  #--------------------------------
  # Device configuration
  #--------------------------------
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device: %s'%device)

  #--------------------------------
  # Hyper-parameters
  #--------------------------------
  input_size = 30
  hidden_size = [100, 50, 100]
  num_classes = 2
  num_epochs = 60
  batch_size = 10
  learning_rate = 5*1e-4
  learning_rate_decay = 0.95
  reg = 0.001


  mat = sio.loadmat('Dataset/dset_%d.mat' %(iter_n))
  mat = mat['dset']
  Design_data = mat['X'][0][0]
  Prformance_data = mat['Y'][0][0]

  x_train_tensor = torch.from_numpy(Design_data).float()
  y_train_tensor = torch.from_numpy(Prformance_data).float()


  dataset = TensorDataset(x_train_tensor, y_train_tensor)

  lengths = [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)]
  torch.manual_seed(net_n+1)
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)



  model_forward = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes, net_n).to(device)
  print(count_parameters(model_forward))

  model_forward.apply(weights_init)
  model_forward.to(device)

  # Loss and optimizer
  def sum_mse(yhat,y):
      return torch.sum(torch.mean((yhat - y) ** 2, dim=1))

  criterion_MSE = nn.MSELoss()
  criterion_sum_mse = sum_mse
  optimizer = torch.optim.Adam(model_forward.parameters(), lr=learning_rate)

  # Train the model_forward
  lr = learning_rate
  total_step = len(train_loader)
  time_start = time.time()
  for epoch in range(num_epochs):
      for i, (controll, state) in enumerate(train_loader):
          # Move tensors to the configured device
          controll = controll.to(device)
          state = state.to(device)
          #################################################################################
          # Implement the training code                                             #
          optimizer.zero_grad()
          outputs = model_forward(controll)
          loss = criterion_sum_mse(outputs, state)
          loss.backward()
          optimizer.step()

          if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

      # Code to update the lr
      lr *= learning_rate_decay
      update_lr(optimizer, lr)
      with torch.no_grad():
          correct = 0
          total = 0
          state_all = torch.zeros(0, num_classes).to(device)
          outputs_all = torch.zeros(0, num_classes).to(device)
          for controll, state in val_loader:
              state = state.to(device)
              state_all = torch.cat((state_all,state),0)
              ####################################################
              #evaluation #
              controll = controll.to(device)
              outputs = model_forward(controll)
              outputs_all = torch.cat((outputs_all,outputs),0)

          loss = criterion_MSE(state_all, outputs_all)
          print('Validataion MSE is: {}'.format(loss))
          one_epoch_time = time.time() - time_start


  # save the model
  model_name = 'Models/iter_%d/mu_net_%d.ckpt' %(iter_n, net_n)
  torch.save(model_forward.state_dict(), model_name)





