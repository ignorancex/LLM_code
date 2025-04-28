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
import tikzplotlib
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')
  
print(device)

# We input a trained model here with other parameters of the network and compute the norm-based cover.
def l2_covering_norm(result, layers, bound, lipschitz_activation, lipschitz_loss, epsilon, n_samples, delta): 
    
    n_layers = len(layers) - 1
    n_input = layers[0]
    n_output = layers[-1]
    
    epsilon_2 = epsilon/ (lipschitz_loss * math.sqrt(n_output))
    constant = (1/2) * pow( 2 * bound , 2 * n_layers) * math.log(2*n_input + 2, 2)
    

    max_norm = 0
    for i in range(len(layers)-1): # We use len layers - 1 because we need the last layer to be output layer
        layer_name = 'linears.' + str(i) + '.weight'
        norm = result['norm_one_infty']['norms_of_layers'][layer_name]
        if norm > max_norm:
            max_norm = norm
    print(epsilon_2)   
    first_term = n_output * constant * pow(2 * max_norm * lipschitz_activation, n_layers * (n_layers + 1))

    l2_cover = pow(1/epsilon_2, 2*n_layers) * first_term
    
    #Change the base of log
    l_2cover = l2_cover / math.log(math.exp(1),2)
    
    out_result = [] 
    out_result.append('l2 book')
    out_result.append(l2_cover)
          
    return out_result   


# In[29]:


#Computing the Spectral bound for a given model
def l2_covering_spectral(result, input_data, layers, bound, lipschitz_activation, lipschitz_loss, epsilon, n_samples, delta): 


    n_layers = len(layers) - 1
    n_input = layers[0]
    n_output = layers[-1]
    
    input_norm = 0
    for i in range(input_data.size()[0]):
        input_norm += pow(torch.linalg.norm(input_data[i,0,:,:], ord = 'fro').to(device).data.item(),2)
    input_norm = math.sqrt(input_norm/n_samples) #Normalized input norm
    
    print("input_norm: ", input_norm)
    max_param = max(layers)
    epsilon_2 = epsilon/ lipschitz_loss
    constant = pow(input_norm, 2) * math.log(2 * max_param * max_param)
    
  
    first_term = 1
    second_term = 0
    
    for i in range(len(layers)-1): # We use len layers - 1 because we need the last layer to be output layer
        layer_name = 'linears.' + str(i) + '.weight'
        spectral_norm = result['norm_spectral']['norms_of_layers'][layer_name]
        norm_21 = result['norm_21']['norms_of_layers'][layer_name]
        
        first_term = first_term * lipschitz_activation * spectral_norm
        second_term = second_term + pow(norm_21/spectral_norm, 2/3)
     
    
    
    
    l2_cover = (1/pow(epsilon_2,2)) * constant * pow(first_term,2) * pow(second_term, 3)
    

    out_result = [] 
    out_result.append('l2 spectral')
    out_result.append(l2_cover)
          
    return out_result   


# In[30]:


#Computing the Pseudi-dim-based bound for a given model 
def l2_covering_pseudo(result, layers, bound, lipschitz_activation, lipschitz_loss, epsilon, n_samples, delta): 

    n_layers = len(layers) - 1
    n_output = layers[-1]

    epsilon_2 = epsilon / (lipschitz_loss * (math.sqrt(n_output))) 
    noise_levels = []
    l2_covers = []
    rademachers = []
    generalization_gaps =[]
    updated_results = []
    
    n_params = 0
    comp_units = 0
    for i in range(len(layers)-1):
        n_params = n_params + layers[i] * layers[i+1]
    
    n_params_minus_last = n_params - layers[-1] * layers[-2]   
    
    for i in range(1,len(layers)):
        comp_units += layers[i]
    
    comp_units_rvo = comp_units - n_output + 1
    first_term = pow((n_params_minus_last+2) * comp_units_rvo, 2)
    second_term = math.log( 18 * (n_params_minus_last+2) * pow(comp_units_rvo,2), 2)
    pdim = first_term + 11 * (n_params_minus_last+2) * comp_units_rvo * second_term
    l2_cover = n_output * pdim *  math.log((math.exp(1) * n_samples * bound) / (epsilon_2 * pdim))
    out_result = []
    out_result.append('Pdim')
    out_result.append(l2_cover)
    
    
    return out_result       


# In[31]:


#Computing the lipschitzness-based model
def l2_covering_lipschitz(result, layers, bound, lipschitz_activation, lipschitz_loss, epsilon, n_samples, delta): 

    n_layers = len(layers) - 1
    n_output = layers[-1]
    
    epsilon_2 = epsilon / (lipschitz_loss * (math.sqrt(n_output))) 
    noise_levels = []
    l2_covers = []
    rademachers = []
    generalization_gaps =[]
    updated_results = []
    n_params = 0
    for i in range(len(layers)-1):
        n_params = n_params + layers[i] * layers[i+1]
     
    n_params_minus_last = n_params - layers[-1] * layers[-2]
    
    max_norm = 0
    for i in range(len(layers)-1): # We use len layers - 1 because we need the last layer to be output layer
        layer_name = 'linears.' + str(i) + '.weight'
        norm = result['norm_one_infty']['norms_of_layers'][layer_name]
        if norm > max_norm:
            max_norm = norm      
    numerator = 4 * bound * n_samples  * math.exp(1) * (n_params_minus_last + n_output) * pow(lipschitz_activation * max_norm, n_layers)
    denumerator = epsilon_2 * (lipschitz_activation * max_norm -1)

    l2_cover = (n_params + n_params_minus_last * (n_output-1)) *  math.log(numerator/denumerator) 

    out_result = []
    out_result.append('Parameter book')
    out_result.append(l2_cover)
    
    
    return out_result       


# In[32]:


#Computing the covering number of our approach (Theorem 26) given a model
def l2_covering_ours(result, noise_factor, logarithmic, layers, bound, lipschitz_activation ,lipschitz_loss, epsilons, n_samples, delta): 
    
    
    n_input = layers[0]
    n_output = layers[-1]
    
    expect_loss = 2 * bound * math.sqrt(n_output) * lipschitz_loss # The amplification in epsilon by taking expectation and loss
    epsilons = [(1/ expect_loss)*epsilon for epsilon in epsilons] 
    constant = 4 * pow(4+bound, 3/2) * (1 / pow(2*math.pi,0.25)) # The constant in the covering number
    constant_log = 4 + bound # The constant in the logarithm.
    
  #  bound_activations = [math.log((1-epsilon)/epsilon) for epsilon in epsilons] # The value, which is the inverse of sigmoid at 1-epsilon
 
    


    l2_cover = 0
    out_results = result

    for i in range(len(layers)-1): # We use len layers - 1 because we need the last layer to be output layer
        layer_name = 'linears.' + str(i) + '.weight'
        norm = result['norm_one_infty']['norms_of_layers'][layer_name] 

        d = layers[i]
        p = layers[i+1]

        if logarithmic == False: 
            constant_epsilon_prime = noise_factor / ((4 + bound) * d) 
    
            epsilon_prime = [constant_epsilon_prime*epsilon for epsilon in epsilons] 
            bound_activations = [math.log((1-epsilon)/epsilon) for epsilon in epsilon_prime]
            
            denumerator = pow(epsilons[i], 3/2) * pow(noise_factor,2)
           
            denumerator_log = epsilons[i] * noise_factor
    
    
            numerator = lipschitz_activation * math.sqrt(bound * bound_activations[i]) * pow(d, 5/2) 
            numerator_log = bound * d
                    
            base = constant * (numerator/denumerator) * math.log(constant_log * numerator_log/denumerator_log)
            cover_layer = (d+1) * p * math.log(base)#
            l2_cover = l2_cover + cover_layer
      
        elif logarithmic == True:
    # we use the following codes if the noise_factor id given logarithmically
            bound_activations = [math.log((4+bound)*d)-math.log(epsilon)-1*noise_factor for epsilon in epsilons]
            numerator = lipschitz_activation * math.sqrt(bound * bound_activations[i]) * pow(d, 5/2)
            numerator_log = bound * d
            denumerator = pow(epsilons[i], 3/2)
            denumerator_log = epsilons[i] 
            
            
            
            first_term = math.log(constant*(numerator/denumerator)) - 2*noise_factor*math.log(10) 
            second_term =  math.log(math.log(constant_log * numerator_log/denumerator_log)-noise_factor*math.log(10))
            cover_layer = first_term + second_term
            l2_cover = l2_cover +  (d+1) * p * cover_layer   
                

    out_result = []
    out_result.append('Ours')
    out_result.append(l2_cover)
   
    return out_result   


# In[33]:


#Computing the covering number that is used for the first layer in Theorem 26
def l2_covering_first(result, layers, bound, lipschitz_activation ,lipschitz_loss, epsilon, n_samples, delta): 
    
    
    n_input = layers[0]
    n_output = layers[-1]
    
    epsilon_2 = epsilon / lipschitz_activation
    constant = 2 * math.exp(1) * bound # The constant in the covering number
    
    


    l2_cover = n_input * n_output * math.log((constant * n_samples) / epsilon_2)

    out_result = []
    out_result.append('First')
    out_result.append(l2_cover)
   
    return out_result   


# In[42]:


#Computing NVAC given a covering number
#Covering number is the log of d2 cover
def get_min_samples(covering_number, margin_loss, margin, delta):
   
    
    epsilon = (1 - margin_loss) / 10
    
    m = (36 * covering_number) / pow(epsilon,2)
    
    rademacher = 4 * epsilon + (6/math.sqrt(m)) * math.sqrt(covering_number) 
    
    probability_term = math.sqrt(math.log(2/delta) / (2 * m))
    
    gap = 2 * rademacher + 3 * probability_term
    
    print("margin_loss:", margin_loss, "; gap = ", gap, "; rademacher = ", rademacher, "samples required = ", m)
    return [m, gap]


# In[43]:


#Computing NVAC for Theorem 26 that depends on n_samples
def get_min_samples_ours(result, layers, margin_loss, margin, bound, logarithmic,noise_factor, lipschitz_activation, lipschitz_loss, first_epsilon, second_epsilon,
                         first_layers, second_layers, n_samples, delta):
   
    
    epsilon = (1 - margin_loss) / 10
    
    first_cover = l2_covering_first(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = 1, delta = delta)
   
    #Our covering number is not dependent on the number of samples
    second_cover = l2_covering_ours(result = result, layers = first_layers, bound = bound, logarithmic=logarithmic, noise_factor = noise_factor,
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilons = second_epsilon, n_samples = n_samples, delta = delta)
    
    constant = first_cover[1] + second_cover[1]
  
    print('constant = ', constant)
    n_params = layers[0] * layers[1]
   
    a = (36 * n_params) / (epsilon * epsilon)
    b = (36 * constant) / (epsilon * epsilon)
 



    m = b

    while m < a * math.log(m) + b:
        m = m * 1.00000005

    
   
    new_first_cover = l2_covering_first(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation/margin, lipschitz_loss = lipschitz_loss,
                                    epsilon = first_epsilon, n_samples = m, delta = delta)
    
    new_cover = new_first_cover[1] + second_cover[1]

    rademacher = 4 * epsilon + (6/math.sqrt(m)) * math.sqrt(new_cover) 
    
    probability_term = math.sqrt(math.log(2/delta) / (2 * m))
    
    gap = 2 * rademacher + 3 * probability_term
    
    print("margin_loss:", margin_loss, "; gap = ", gap, "; rademacher = ", rademacher, "samples required = ", m)
    return [m,gap]


# In[44]:


#Computing NVAC of Lipschitzness-based approach that depends on n_samples
def get_min_samples_lipschitz(result, layers, margin_loss, margin, bound, lipschitz_activation, lipschitz_loss, first_epsilon, second_epsilon,
                         first_layers, second_layers, n_samples, delta):
   
    n_output = layers[-1]
    epsilon = (1 - margin_loss) / 10
    first_cover = l2_covering_lipschitz(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = 1, delta = delta)
    
    
    constant = first_cover[1]
  
    print('constant = ', constant)
    n_params = 0
    for i in range(1,len(layers)):
        n_params += layers[i] * layers[i-1]
        
    n_params_minus_last = n_params - layers[-1] * layers[-2]
  
    a = (36 * n_params_minus_last * n_output) / (epsilon * epsilon)
    b = (36 * constant) / (epsilon * epsilon)
   
    m = b

    while m < a * math.log(m) + b:
        m = m * 1.00000005

    
    
   
    new_first_cover = l2_covering_lipschitz(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = m, delta = delta)
    
    new_cover = new_first_cover[1] 
    rademacher = 4 * epsilon + (6/math.sqrt(m)) * math.sqrt(new_cover) 
    
    probability_term = math.sqrt(math.log(2/delta) / (2 * m))
    
    gap = 2 * rademacher + 3 * probability_term
    
    print("margin_loss:", margin_loss, "; gap = ", gap, "; rademacher = ", rademacher, "samples required = ", m)
    return [m,gap]


# In[45]:


#Computing NVAC of Pseudo-dim-based approach that depends on n_samples
def get_min_samples_pseudo(result, layers, margin_loss, margin, bound, lipschitz_activation, lipschitz_loss, first_epsilon, second_epsilon,
                         first_layers, second_layers, n_samples, delta):
    print('bound=',bound)
    n_output = layers[-1]
    epsilon = (1 - margin_loss) / 10
    first_cover = l2_covering_pseudo(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = 1, delta = delta)
    
    
    constant = first_cover[1]
  
    print('constant = ', constant)
    n_params = 0
    comp_units = 0
    for i in range(len(layers)-1):
        n_params = n_params + layers[i] * layers[i+1]
    
    n_params_minus_last = n_params - layers[-1] * layers[-2] 
  
    for i in range(1,len(layers)):
        comp_units += layers[i]
    
    comp_units_rvo = comp_units - n_output + 1

    first_term = pow((n_params_minus_last+2) * comp_units_rvo, 2)
    second_term = math.log( 18 * (n_params_minus_last+2) * pow(comp_units_rvo,2), 2)    
    pdim = first_term + 11 * (n_params_minus_last+2) * comp_units_rvo * second_term
        
        
    a = (36 * pdim * n_output) / (epsilon * epsilon)
    b = (36 * constant) / (epsilon * epsilon)
 
 


    m = pdim
    i = 0
    j = 0

    while m > a * math.log(m) + b:
        m = m * 1.000001
        i+=1
    m = m / 1.000001

    while m > a * math.log(m) + b:
        m = m * 1.000000001
        j+=1


   
    new_first_cover = l2_covering_pseudo(result = result, layers = first_layers, bound = bound, 
                                    lipschitz_activation = lipschitz_activation, lipschitz_loss = lipschitz_loss/margin,
                                    epsilon = first_epsilon, n_samples = m, delta = delta)
    
    new_cover = new_first_cover[1] 
    
    rademacher = 4 * epsilon + (6/math.sqrt(m)) * math.sqrt(new_cover) 
    
    probability_term = math.sqrt(math.log(2/delta) / (2 * m))
    
    gap = 2 * rademacher + 3 * probability_term
    
    print("margin_loss:", margin_loss, "; gap = ", gap, "; rademacher = ", rademacher, "samples required = ", m)
    return [m,gap]
