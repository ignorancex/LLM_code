# ``Benefits of Additive Noise in Composing Classes with Bounded Capacity''

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os, shutil
from torch.utils.data import random_split
import copy
import pickle
import math
import functools
from scipy import special
from covering_numbers import*

# In[2]:


# Import MNIST digit dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)                                         


trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=False, num_workers=0)



testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                              shuffle=False, num_workers=0)



# Load the train and test MNIST Images
dataiter = iter(trainloader)

dataiter2 = iter(testloader)


x_train, y_train = dataiter.next()
x_test, y_test = dataiter2.next()



# In[3]:


#Normalizing the dataset and loading it again. We split the data into 59000 training and 1000 validation
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(x_train[0].mean(),x_train[0].std())]))

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(x_train[0].mean(),x_train[0].std())]))                                     


trainset, validset = random_split(trainset, [59000, 1000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=False, num_workers=0)

validloader = torch.utils.data.DataLoader(validset, batch_size= 1000,
                                          shuffle = False, num_workers = 0)

testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                              shuffle=False, num_workers=0)

dataiter = iter(trainloader)

dataiter2 = iter(testloader)

dataiter3 = iter(validloader)

x_train, y_train = dataiter.next()
x_test, y_test = dataiter2.next()

x_valid, y_valid = dataiter3.next()


# In[4]:


#Moving device to GPU

if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')
  
print(device)


# In[6]:


#A function to create directories for saving models and data
def path_gen(parent_dir, layers, noise_level):
  dir = ''
  for i in range(len(layers)):
    dir = dir + str(layers[i]) + '_'
  dir = dir + 'neurons' + str(noise_level)
  path = os.path.join(parent_dir,dir)
  
  try:
    if os.path.exists(path) == False:
      os.makedirs(path)

    else:
      for f in os.listdir(path):
        os.remove(os.path.join(path, f))
        print('removed')
  except Exception as e:
        print('Failed to delete %s. Reason: %s' % (f, e))
  return path


# In[7]:


#This function computes the accuracy based on 0-1 loss
def accuracy(inputs, targets, model):
  inputs = inputs.to(device)
  targets = targets.to(device)
  
  prediction = torch.argmax(model(inputs),dim = 1) == targets
  accu = torch.sum(prediction)/inputs.size(0)
  return accu


# In[8]:


#This function calculates the 1,\infty matrix norm for the network layers
def norm_calc(model):
  norms = {}
  prod = 1
  params = model.state_dict()
  max_norm = 0
  for i in range(len(model.linears)): 
    layer_name = 'linears.' + str(i) + '.weight' #Loading model parameters from saved models
    norm = torch.linalg.norm(params[layer_name],ord = float('inf')).to(device).data.item()
    norms[layer_name] = norm
    if norm > max_norm:
        max_norm = norm
    prod = prod * norm
  norms['max_norm'] = max_norm
  return [prod, norms]


# In[9]:


#This function calculates the 2,1 matrix norm 
def norm_calc_21(model):
  norms = {}
  params = model.state_dict()
  for i in range(len(model.linears)):
    layer_name = 'linears.' + str(i) + '.weight' #Loading model parameters from saved models
    norm = 0

    for i in range(params[layer_name].size()[0]):
        norm = norm + torch.linalg.norm(params[layer_name][i]).to(device).data.item()
    norms[layer_name] = norm
  
  return norms


# In[10]:


#This function computes the spectral norm of the network layers
def norm_calc_spectral(model):
  norms = {}
  params = model.state_dict()
  for i in range(len(model.linears)): #Loading model parameters from saved models
    layer_name = 'linears.' + str(i) + '.weight'
    norm = torch.linalg.norm(params[layer_name], ord = 2).to(device).data.item()
    norms[layer_name] = norm
  return norms


# In[11]:


#This function computes the Frobenious norm of the network layers
def norm_calc_fro(model):
  norms = {}
  params = model.state_dict()
  for i in range(len(model.linears)):
    layer_name = 'linears.' + str(i) + '.weight'
    norm = torch.linalg.norm(params[layer_name], ord = 'fro').to(device).data.item()
    norms[layer_name] = norm
  return norms


# In[12]:


#This function computes the ramp loss
def ramp_loss(outputs, targets, margin):
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes = 10)
    targets_one_hot = targets_one_hot.to(device)
    outputs = outputs.to(device)
    pred_one_hot = torch.mul(outputs, targets_one_hot)
    pred = torch.max(pred_one_hot, dim = 1)[0]
    targets_bar = 1 - targets_one_hot
    max_second_one_hot = torch.mul(outputs, targets_bar)
    max_second = torch.max(max_second_one_hot, dim = 1)[0]
    ramp = 1 + (max_second - pred)/margin
    loss = torch.nn.functional.relu(ramp)
    loss[loss>1]=1
    loss = torch.mean(loss)
    return loss


# In[13]:


#This function computes the Hinge loss for classification 
def myMarginLoss(outputs, targets, margin):
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes = 10)
    targets_one_hot = targets_one_hot.to(device)
    outputs = outputs.to(device)
    pred_one_hot = torch.mul(outputs, targets_one_hot)
    pred = torch.max(pred_one_hot, dim = 1)[0]
    targets_bar = 1 - targets_one_hot
    max_second_one_hot = torch.mul(outputs, targets_bar)
    max_second = torch.max(max_second_one_hot, dim = 1)[0]
    ramp = 1 + (max_second - pred)/margin
    loss = torch.nn.functional.relu(ramp)
    loss = torch.mean(loss)
    return loss    


# In[22]:


#Training function
def train(images,labels,model,optimizer, criterion, epochs,parent_dir,n_average):

  # data set using mini-batches of size 32.
  batch_size = 32
  result_all = []
  model.train()
  # loop over the entire dataset epoch times
  for epoch in range(epochs): 
      result = {}

      # Loop over 32 random elements in trainset (#images/32) times.
      for i in range(int(images.shape[0]/batch_size)):

          indx = random.sample(range(images.shape[0]), batch_size)

          targets = labels[indx]

          inputs = images[indx,:,:,:]

          # Move model and data to GPU
          model = model.to(device)
          inputs = inputs.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()
          
          outputs = torch.empty_like(model(inputs))
          # Average the model n_average times if we want to simulate the expectation during training 
          for i in range(n_average):
              outputs += model(inputs) 
                
          outputs = torch.div(outputs, n_average)
        
          loss = criterion(outputs, targets)
        
          loss.backward()

          optimizer.step()

      inputs = images.to(device)
      targets = labels.to(device)
      #Calculating norms and losses during each epoch of training.
      norm =  norm_calc(model)
      loss = criterion(model(inputs),targets).data.item()
      margin_loss = functools.partial(myMarginLoss, margin = 0.1)
      marginLoss = margin_loss(model(inputs),targets).data.item()
      accu = accuracy(inputs,targets,model).data.item()
        
      #Saving the model
      path = os.path.join(parent_dir , "model in epoch" + str(epoch))
      torch.save(model, path)



        
      result['epoch'] = epoch
      result['train_accuracy'] = accu
      result['train_loss'] = loss
      result['path'] = path
      result['model'] = model
      result_all.append(result)
        
      print("########################################")
      print("epoch: ",epoch, " ,  train loss: ", loss, ", margin loss: ", marginLoss,
            " , train accuracy: " , accu, ", norms: ", norm[1] )

  return result_all


# In[23]:


#This function tests a trained model on an input sample and returns accuracy along with ramp and margin losses
def test(images,targets,criterion, margin, margin_func, ramp_func, model):


    test_loss = 0



    inputs = images

  # Move data to GPU
    inputs = inputs.to(device)
    targets = targets.to(device)

  # forward pass
    outputs = model(inputs)
    
    
    
  # Compute the total error on the test set
    test_loss = criterion(outputs, targets).data.item()
    test_accu = accuracy(inputs, targets, model).data.item()
    test_margin_loss = margin_func(outputs, targets, margin = margin).data.item()
    test_ramp_loss = ramp_func(outputs, targets, margin = margin).data.item()
    
    return [test_loss, test_accu, test_margin_loss, test_ramp_loss]


# In[24]:


#This class is used to instanciate neural network objects. 
#The constructor takes as input the layers and noise level
class ZeroNeuralNetRegressor(nn.Module):

    def __init__(self, layers, noise_level):
        super(ZeroNeuralNetRegressor, self).__init__()

        self.noise_level = noise_level
        self.n_neurons = copy.copy(layers)
        
        #We create linear layers based on the input list of layers 
        self.n_neurons.append(10)
        self.layers = []
        self.layers.append(nn.Linear(28*28, layers[0]))
        for i in (range(len(self.n_neurons))):
          
         if i < (len(self.n_neurons)-1):
            self.layers.append(nn.Linear(self.n_neurons[i],self.n_neurons[i+1]))


        self.linears = nn.Sequential(*self.layers)



    def forward(self, x):

        x = x.view(-1, 28*28)
        for i in range(len(self.linears)):

                
            x = torch.sigmoid(self.linears[i](x)) - 0.5 #Using sigmoid -0.5 to have an odd activation function
            x = x + self.noise_level * torch.randn(self.n_neurons[i]).to(device) #Adding Gaussian noise to output of neurons

        return x


# In[54]:


#This function gets as input a model, training data and settings and an array of noise levels
#It trains the network for each noise level in noise_levels
def main_train_fast(layers, noise_levels, decays, epochs, criterion, lr, parent_dir, save_path, n_average_train):



  models_all = {}
  
  print('done')
  for noise_lv in noise_levels:
    print('training for noise level: ', noise_lv)
    path = path_gen(parent_dir, layers, noise_lv)
    # A dictionary that stores results for a single noise level.
    # A list containing all of these results for different noise levels will be saved to drive and returned
    results = {}
    print(path)

    NeuralNet = ZeroNeuralNetRegressor(layers, noise_lv)
    #Using weight decay  
    weight_decays = [] #No weight decays used at the moment!
    
    for i in range(len(decays)):
        param_decay = {'params': NeuralNet.linears[i].parameters(), 'weight_decay': decays[i]}
        weight_decays.append(param_decay)
    

    optimizer = optim.SGD(weight_decays, lr=lr, momentum = 0.9)

    out_train = train(x_train,y_train,NeuralNet,optimizer, criterion, epochs, path, n_average_train)
    
                             
    models_all[noise_lv] = out_train
    
  f = open(save_path, 'wb')
  pickle.dump(models_all, f)
  f.close()
    
  return models_all


# In[26]:


#This function computes the accuracies and errors using test function. 
#The result are computed for several times and averaged to simulate the expectation of output in noisy network
def get_results(noise_levels, load_path, save_path, criterion, margin, margin_func, ramp_func,
                n_average_train, n_average_validation, n_average_test):
    
    f = open(load_path, 'rb') # getting the results of training
    train_results_all = pickle.load(f)
    f.close()
    
    results_all = [] # A list of dictionaries. Each dictionary corresponds to a single epoch for a single noise level
    
    
    for noise_lv in noise_levels:
        train_result = train_results_all[noise_lv] # train.pckl is a dictionary. Each key in dictionary is a noise level
        print(noise_lv)                                           # each noise level has a list of dictionaries containing trian loss, accu and epoch
        for result in train_result:
            results = {} # This will be a dictionaryk.mwith all the detail for each epoch in each noise level
            results['noise_level'] = noise_lv
            epoch = result['epoch']
            p = result['path']
            
            results['epoch'] = epoch
            results['path'] = p
   
            # If there are no averages for training just use the data while trained
            # Otherwise, compute it
            if n_average_train ==1:
                results['train_loss'] = result['train_loss']
                results['train_accuracy'] = result['train_accuracy']
            elif n_average_train >1:
                out_train = [0,0,0,0]
                for i in range(n_average_train):
                    out = test(images = x_train, targets = y_train, criterion= criterion,margin = margin, 
                               margin_func = margin_func, ramp_func = ramp_func, model = torch.load(p))
        
                   
                    out_train[0] += out[0]
                    out_train[1] += out[1]
                    out_train[2] += out[2]
                    out_train[3] += out[3]
                                    

                out_train[0] = out_train[0]/n_average_train
                out_train[1] = out_train[1]/n_average_train
                out_train[2] = out_train[2]/n_average_train
                out_train[3] = out_train[3]/n_average_train


                results['train_loss'] = out_train[0]
                results['train_accuracy'] = out_train[1]
                results['train_margin_loss'] = out_train[2]
                results['train_ramp_loss'] = out_train[3]


                
            out_test = [0,0,0,0]
 
            for i in range(n_average_test):
                out = test(images = x_test, targets = y_test, criterion= criterion,margin = margin, 
                           margin_func = margin_func, ramp_func = ramp_func, model = torch.load(p))
                out_test[0] += out[0]
                out_test[1] += out[1]
                out_test[2] += out[2]
                out_test[3] += out[3]


            out_test[0] = out_test[0]/n_average_test
            out_test[1] = out_test[1]/n_average_test
            out_test[2] = out_test[2]/n_average_test
            out_test[3] = out_test[3]/n_average_test


            results['test_loss'] = out_test[0]
            results['test_accuracy'] = out_test[1]
            results['test_margin_loss'] = out_test[2]
            results['test_ramp_loss'] = out_test[3]


            #Averaging for validation, both loss and accuracy
            out_valid = [0,0,0,0]
     
            for i in range(n_average_validation):
                out = test(images = x_valid, targets = y_valid, criterion= criterion,margin = margin, 
                           margin_func = margin_func,  ramp_func = ramp_func, model = torch.load(p))
                out_valid[0] += out[0]
                out_valid[1] += out[1]
                out_valid[2] += out[2]
                out_valid[3] += out[3]



            out_valid[0] = out_valid[0]/n_average_validation
            out_valid[1] = out_valid[1]/n_average_validation
            out_valid[2] = out_valid[2]/n_average_validation
            out_valid[3] = out_valid[3]/n_average_validation



            results['validation_loss'] = out_valid[0]
            results['validation_accuracy'] = out_valid[1]
            results['validation_margin_loss'] = out_valid[2]
            results['validation_ramp_loss'] = out_valid[3]


           
            results_all.append(results)
            print('done')
    
    f = open(save_path, 'wb')
    pickle.dump(results_all, f)
    f.close()
    
    return results_all


# In[27]:


#Find the model with best validation accuracy among epochs for each noise level and 
# record the accuracies and norms of the model.
def find_winner(noise_levels, load_path, save_path):
        
    f = open(load_path, 'rb') # getting the results of training
    input_results = pickle.load(f)
    f.close()
    
    valid_results = [] # A list of dictionaries. The keys are noise levels and each contains
                        #the accuracies and lossess for trian, validation and test of the winner epoch
    for noise_lv in noise_levels:
        results = {}
        max_validation = 0
        for result in input_results:
            if result['noise_level'] == noise_lv and result['validation_accuracy'] >= max_validation:
                max_validation = result['validation_accuracy']
                results = result
        m = torch.load(results['path']) #Needs to be commented out
        out_norm_one_infty = norm_calc(m)
        results['norm_one_infty'] = {'product_of_norms': out_norm_one_infty[0], 
                              'norms_of_layers': out_norm_one_infty[1]}
        
        out_norm_21 = norm_calc_21(m)
        results['norm_21'] = {'norms_of_layers': out_norm_21}
        
                              
        out_norm_spectral = norm_calc_spectral(m)
        results['norm_spectral'] = {'norms_of_layers': out_norm_spectral}
    
        out_norm_fro = norm_calc_fro(m)
        results['norm_fro'] = {'norms_of_layers': out_norm_fro}
        
        
       
        valid_results.append(results)
    
    f = open(save_path, 'wb')
    pickle.dump(valid_results, f)
    f.close()
    
    return valid_results




# In[46]:


# This function gets a directory and computes covering numbers and NVAC for models in that directory.
# The models have the same architecture but trained with different noise levels.
# The composition_layer decides where to break the network 
# covering_func is the covering number that is used for the first layers 
# Pass the composition_layer considering layer numbers starting from 1 not 0
# Sample == True to compute NVAC
# Sample_type == 'ous' or 'nonm'
#If you want all to be from another covering number you need to consider the output layer and use len(layer)+1 as composition
#for example, if we have layers =[500,500,64,64], new_layers would be [784,500,500,64,64,10], composition_layers = 2
# first_layers = [784,500,500], second_layers = [500,64,64,10]
def mixed_covering(parent_dir, load_path, save_path, layers, logarithmic, covering_func, composition_layer, margin, sample, sample_type,
                   n_input, n_output, bound, lipschitz_activation , lipschitz_loss, epsilons, n_samples, delta): 
    
    f = open(load_path, 'rb') # getting the results of winners, this is a dictionary
    input_results = pickle.load(f)
    f.close()  

 #  n_layers = len(layers) + 1 # We need one additional layer to account for input -> first layer
   # Constructing the layers array that include input and output layers as well
    new_layers = [n_input]
    for i in range(len(layers)):
        new_layers.append(layers[i])
    new_layers.append(n_output)
    
    # Getting arrays for the first and second parts of the composition
    first_layers = new_layers[:composition_layer+1]
    second_layers = new_layers[composition_layer:]
    # Getting epsilons for the first and second part
    first_epsilon = 0
    for i in range(0,composition_layer):
        first_epsilon += epsilons[i]
    
    second_epsilon = epsilons[composition_layer:] 
    # Getting the results for different noise levels
    noise_levels = []
    l2_covers = []
    min_samples = []
    updated_results = []
    # We incorporated margin into the loss lipschitzness = loss_lipschitz/margin
    for result in input_results:
        print('###########################################')
        print(result['noise_level'])
        if result['noise_level'] > 0:
            noise_factor = result['noise_level']
        else:
            noise_factor = 1
            
        noise_levels.append(result['noise_level'])
        out_results = result
        
        if sample == True:
            epsilon_sample = (1 - result['train_ramp_loss']) / 10
            if result['train_ramp_loss'] >= 1: #Just a sanity check, ramp loss is always less than or equal to 1
                print("High ramp loss!")
            else: #Spreading epsilon based on number of layers for each part of the network. #If all layers are from another method, it would be the total epsilon
                epsilon_per_layer = epsilon_sample / (len(layers)+1)
                first_epsilon = composition_layer * epsilon_per_layer
                second_epsilon = [epsilon_sample - first_epsilon for i in range(composition_layer, len(layers)+1)]
        # Changing epsilon for going from d_2 to d_tv
        # We do not want to lose it if we use other methods in all layers
        if result['noise_level'] == 0 and composition_layer != len(layers)+1:
            first_epsilon = (2/9) * 1 * first_epsilon
        elif composition_layer != len(layers)+1:
            first_epsilon = (2/9) * result['noise_level'] * first_epsilon
        
        # We handle the case where we have our covering bound for all layers
        if composition_layer == 0:
            first_cover = []
            first_cover.append('All ours, ')
            first_cover.append(0)
            second_cover = l2_covering_ours(result = result, layers = second_layers, bound = bound, 
                                        lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                        epsilons = second_epsilon, n_samples = n_samples, delta = delta)
            
        # We handle the case where we want all layers from another method
        elif composition_layer == (len(layers)+1):
            second_cover = []
            second_cover.append('')
            second_cover.append(0)
            first_cover = covering_func(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = n_samples, delta = delta)
        else:
            first_cover = covering_func(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = n_samples, delta = delta)
        
 
      
            second_cover = l2_covering_ours(result = result, noise_factor = noise_factor, layers = second_layers, bound = bound, logarithmic=logarithmic,
                                        lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                        epsilons = second_epsilon, n_samples = n_samples, delta = delta)
    
        l2_cover = first_cover[1] + second_cover[1]
        l2_covers.append(l2_cover) # To keep track of l2 covers for noise levels
        
       
        cover_name = first_cover[0] + ' for ' + str(composition_layer) + ' layers'
        
        out_results['covering_number_' + cover_name] = l2_cover
 #### Computing minimum samples only if we want we set sample to True       
        if sample == True:
                    
            if sample_type == 'ours': #The covering number will be computed inside min_samples function
                if result['train_ramp_loss'] < 1:
                    min_sample = get_min_samples_ours(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                                      bound = bound, noise_factor = noise_factor, margin=margin, logarithmic=logarithmic,
                                                      lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                                      first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                                      first_layers = first_layers, second_layers = second_layers, 
                                                      n_samples = n_samples, delta = delta)
                    print('Covering Number = ', l2_cover, "; Min Samples = ", "{:e}".format(min_sample[0]))

                else: 
                    min_sample = [0,0]
                    min_sample[0] = n_samples
                    min_sample[1] = 0
            #We compute NVAC for other approaches only for network with no noise
            elif sample_type == 'lipschitz' and result['noise_level'] == 0: #The covering number will be computed inside min_samples function
                if result['train_ramp_loss'] < 1:
                    min_sample = get_min_samples_lipschitz(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                                      bound = bound, margin=margin,
                                                      lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                                      first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                                      first_layers = first_layers, second_layers = second_layers, 
                                                      n_samples = n_samples, delta = delta)
                    print('Covering Number = ', l2_cover, "; Min Samples = ", "{:e}".format(min_sample[0]))

                else: 
                    min_sample = [0,0]
                    min_sample[0] = n_samples
                    min_sample[1] = 0
                    
            elif sample_type == 'pseudo' and result['noise_level'] == 0:#The covering number will be computed inside min_samples function
                if result['train_ramp_loss'] < 1:
                    min_sample = get_min_samples_pseudo(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                                      bound = bound, margin=margin,
                                                      lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                                      first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                                      first_layers = first_layers, second_layers = second_layers, 
                                                      n_samples = n_samples, delta = delta)
                    print('Covering Number = ', l2_cover, "; Min Samples = ", "{:e}".format(min_sample[0]))

                else: 
                    min_sample = [0,0]
                    min_sample[0] = n_samples
                    min_sample[1] = 0
            
            elif sample_type == 'others' and result['noise_level'] == 0:
                if result['train_ramp_loss'] < 1:
                    min_sample = get_min_samples(covering_number = l2_cover, margin = margin, 
                                         margin_loss = result['train_ramp_loss'], delta = delta)
                    
                    print('Covering Number = ', l2_cover, "; Min Samples = ", "{:e}".format(min_sample[0]))

                else: 
                    min_sample = [0,0]
                    min_sample[0] = n_samples
                    min_sample[1] = 0
            #This is dummy
            else:
                min_sample = [0,0]
                min_sample[0] = n_samples
                min_sample[1] = 0
            min_samples.append(min_sample) # To keep track of minimum samples for noise levels
            out_results['min_samples_' + cover_name] = min_sample[0]   
            out_results['gap_' + cover_name] = min_sample[1]   
         
            
        updated_results.append(out_results)
    # We save NVACS and plot   
    f = open(save_path, 'wb')
    pickle.dump(updated_results, f)
    f.close()

    

    return updated_results
    


# In[47]:


def plot_norms(load_path, save_path):
    
    f = open(load_path, 'rb') # getting the results of training
    input_results = pickle.load(f)
    f.close()    
    
    noise_levels = []
    product_norms = []
    layer_norms = []
    layer_names = [key for key in input_results[0]['norm_one_infty']['norms_of_layers'].keys()]

    for result in input_results:
        noise_levels.append(result['noise_level'])
        product_norms.append(result['norm_one_infty']['product_of_norms'])
        norms = [i for i in result['norm_one_infty']['norms_of_layers'].values()]
        layer_norms.append(norms)
    
    
    
    plt.figure(figsize=(8,8))
    plt.plot(noise_levels, product_norms)

    plt.legend(['product of norms'])
    plt.title('Product of norms of all layers Vs. Noise')
    plt.xlabel('noise levels')
    plt.ylabel('Product of norms')
    plt.grid()
    plt.savefig(os.path.join(save_path,"product_norms.png"))
    
    for i in range(len(layer_names)):
        norms_plot= []
        for ln in layer_norms:
            norms_plot.append(ln[i])
            
        plt.figure(figsize=(12,12))
        plt.plot(noise_levels, norms_plot)

        plt.legend([layer_names[i]])
        plt.title('Norms of individual layer Vs. Noise')
        plt.xlabel('noise levels')
        plt.ylabel('norm of ' + layer_names[i])
        plt.grid()
        plt.savefig(os.path.join(save_path, layer_names[i] + ".png"))
       


# In[71]:
def main():
    
    #Defining layers and noise levels
    layers = [64,64,64]
    decays = [0,0,0]
    noise_levels  = np.linspace(0,0.5,11)
    #noise_levels =[0.05,0.1]
    epochs = 40
    margin = 0.1
    lr = 0.3
    n_average_training = 1
    n_average_train = 50
    n_average_validation = 50
    n_average_test = 50
    parent_dir = os.path.join(os.getcwd(),'test')
    
    if os.path.exists(parent_dir) == False:
        os.makedirs(parent_dir)
        
    
    criterion = torch.nn.CrossEntropyLoss()
    
    
    # In[72]:
    
    
    #Training models for different noise levels and saving them
    path_train = os.path.join(parent_dir, 'train.pckl')
    
    main_train_fast(layers = layers, noise_levels = noise_levels, decays= decays, epochs = epochs, criterion = criterion, lr = lr,
                    parent_dir = parent_dir, save_path = path_train, n_average_train = n_average_training)
    
    result_path = os.path.join(parent_dir, 'results_all.pckl')
    
    res = get_results(noise_levels = noise_levels, load_path = path_train, save_path = result_path, margin = 0.1,
                      criterion = criterion, margin_func = myMarginLoss, ramp_func = ramp_loss, 
                      n_average_train = n_average_train, n_average_validation = n_average_validation, n_average_test = n_average_test)
    
    win_path = os.path.join(parent_dir, 'valid_results.pckl')
    
    
    
    find_winner(noise_levels, load_path = result_path, save_path = win_path)
    
    plot_norms(load_path = win_path, save_path = parent_dir)
    
    
    # In[56]:
    
    
    ### Computing our covering number
    
    win_path = os.path.join(parent_dir, 'valid_results.pckl')
    
    mixed_covers_path = os.path.join(parent_dir, 'mixed_cover.pckl')
    
        
    epsilons = [0.01, 0.01 ,0.01,0.01]
    delta = 0.01
    
    #Computing the covering number of Theorem 26
    mixed_covers_winners = mixed_covering(parent_dir = parent_dir, load_path = win_path, save_path = mixed_covers_path,
                                          layers = layers, logarithmic = False, covering_func = l2_covering_first,
                                          composition_layer= 1, margin = 0.1, sample = True,  sample_type = 'ours',n_input = 784, n_output = 10,
                                          bound = 0.5, lipschitz_activation=1 ,lipschitz_loss=2, epsilons=epsilons,
                                          n_samples=59000, delta=delta)
        
    
    
    # In[58]:
    
    
    ### Computing every other covering number
    #composition_layer = len(layers)+1 to get every layer from the covering_func approach
    composition_layer = 4
    delta = 0.01
    spectral_func = functools.partial(l2_covering_spectral, input_data = x_train)
    mixed_covers_winners = mixed_covering(parent_dir = parent_dir, load_path = mixed_covers_path, save_path = mixed_covers_path,
                                          layers = layers, logarithmic=False, covering_func = spectral_func,
                                          composition_layer= composition_layer, margin = 0.1, sample = True,  
                                          sample_type = 'others' ,n_input = 784, n_output = 10,
                                          bound = 0.5, lipschitz_activation=1 ,lipschitz_loss=2, epsilons=epsilons,
                                          n_samples=59000, delta=delta)
    
    mixed_covers_winners = mixed_covering(parent_dir = parent_dir, load_path = mixed_covers_path, save_path = mixed_covers_path,
                                          layers = layers, logarithmic=False, covering_func = l2_covering_lipschitz,
                                          composition_layer= composition_layer, margin = 0.1, sample = True,
                                          sample_type = 'lpischitz',n_input = 784, n_output = 10,
                                          bound = 0.5, lipschitz_activation=1 ,lipschitz_loss=2, epsilons=epsilons,
                                          n_samples=59000, delta=delta)
    
    mixed_covers_winners = mixed_covering(parent_dir = parent_dir, load_path = mixed_covers_path, save_path = mixed_covers_path,
                                          layers = layers, logarithmic=False, covering_func = l2_covering_norm,
                                          composition_layer= composition_layer, margin = 0.1, sample = True, 
                                          sample_type = 'others', n_input = 784, n_output = 10,
                                          bound = 0.5, lipschitz_activation=1 ,lipschitz_loss=2, epsilons=epsilons,
                                          n_samples=59000, delta=delta)
    
    
    
    mixed_covers_winners = mixed_covering(parent_dir = parent_dir, load_path = mixed_covers_path, save_path = mixed_covers_path,
                                         layers = layers, logarithmic=False, covering_func = l2_covering_pseudo,
                                         composition_layer= composition_layer, margin = 0.1, sample = True, 
                                         sample_type = 'pseudo',n_input = 784, n_output = 10,
                                         bound = 1, lipschitz_activation=1 ,lipschitz_loss=2, epsilons=epsilons,
                                         n_samples=59000, delta=delta)
    

# In[525]:
if __name__ == "__main__":
    main()




