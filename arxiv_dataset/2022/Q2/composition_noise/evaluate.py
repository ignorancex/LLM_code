# ``Benefits of Additive Noise in Composing Classes with Bounded Capacity''

# In[1]:
import os, shutil
import pickle
import tikzplotlib
import math
import numpy as np
import torch
import io
from train import get_min_samples_ours,get_min_samples_lipschitz
import matplotlib.pyplot as plt

# In[2]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


### Computing NVAC for small noise levels using the central architecture.
directory = os.path.join(os.getcwd(), 'models/250_250_250/mixed_cover.pckl')
f = open(directory, 'rb') # getting the results of training
input_results = CPU_Unpickler(f).load()
f.close()

l2_covers = []
min_samples = []

layers = [250,250,250]
new_layers = [784,250,250,250,10]
composition_layer = 1
result = input_results[0] #Getting the training results
margin = 0.1
coeffs = [10 * i for i in range(-20,0)]
#noise_factors = [pow(10,i) for i in coeffs]
noise_factors = [10 * i for i in range(-35,0)] 
logarithmic = True# Logarithmically
bound = 0.5
lipschitz_activation = 1
lipschitz_loss = 2
n_samples = 59000
delta = 0.01
epsilon_sample = (1 - result['train_ramp_loss']) / 10
epsilon_per_layer = epsilon_sample / (len(layers)+1)
first_epsilon = composition_layer * epsilon_per_layer
second_epsilon = [epsilon_sample - first_epsilon for i in range(composition_layer, len(layers)+1)]
        # Changing epsilon for going from d_2 to d_tv



# Getting arrays for the first and second parts of the composition
composition_layer = 1
first_layers = new_layers[:composition_layer+1]
second_layers = new_layers[composition_layer:]

for noise_factor in noise_factors:
    min_sample = get_min_samples_ours(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                    bound = bound, noise_factor = noise_factor, margin=margin, logarithmic=logarithmic,
                                    lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                    first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                    first_layers = first_layers, second_layers = second_layers, 
                                  n_samples = n_samples, delta = delta)

    min_samples.append(min_sample[0])
    print(noise_factor)

noise_factors.append(math.log(0.05,10))
result = input_results[1]
min_sample = get_min_samples_ours(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                    bound = bound, noise_factor = math.log(0.05,10), margin=margin,logarithmic=logarithmic,
                                    lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                    first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                    first_layers = first_layers, second_layers = second_layers, 
                                  n_samples = n_samples, delta = delta)
min_samples.append(min_sample[0])

noise_factors.append(math.log(0.1,10))
result = input_results[2]
min_sample = get_min_samples_ours(result = result, layers = new_layers, margin_loss = result['train_ramp_loss'],
                                    bound = bound, noise_factor = math.log(0.1,10), margin=margin,logarithmic=logarithmic,
                                    lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                    first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                    first_layers = first_layers, second_layers = second_layers, 
                                  n_samples = n_samples, delta = delta)
min_samples.append(min_sample[0])

# Getting arrays for the first and second parts of the composition

composition_layer = 4
first_layers = new_layers[:composition_layer+1]
second_layers = new_layers[composition_layer:]
print('#############################################')
min_sample_param = get_min_samples_lipschitz(result = input_results[0], layers = new_layers, margin_loss = result['train_ramp_loss'],
                                        bound = bound, margin=margin,
                                        lipschitz_activation = lipschitz_activation ,lipschitz_loss = lipschitz_loss/margin, 
                                        first_epsilon = first_epsilon, second_epsilon = second_epsilon,
                                        first_layers = first_layers, second_layers = second_layers, 
                                        n_samples = n_samples, delta = delta)
parameter_min_samples = [min_sample_param[0] for i in noise_factors]
plt.figure()
plt.plot(noise_factors,min_samples)
plt.plot(noise_factors, parameter_min_samples)
plt.xlabel('Noise')
plt.ylabel('NVAC')
plt.title('NVAC for Central Architecture')
plt.grid()
plt.yscale('log')

plt.legend(['Ours','Lipschitzness-based'])

plt.savefig(os.path.join(os.getcwd(),'noise_ours_lipschitz.pdf'))
#tikzplotlib.save(os.path.join(os.getcwd(), 'noise_ours_only.tex'))


# In[3]:


#Plotting NVAC vs # of hidden layers
ours = []
spectral = []
norm_based = []
lipschitz_based = []
pdim = []
directory = 'models/250'
depths = [2,3,4,5]
for i in depths:
    
    str_layer = str(i+1)

    directory = directory + '_250'
    parent_dir = os.path.join(os.getcwd(),directory)

    mixed_covers_path = os.path.join(parent_dir, 'mixed_cover.pckl') 
    f = open(mixed_covers_path, 'rb') # getting the results of training
    input_results = CPU_Unpickler(f).load()
    f.close()    

    ours.append(input_results[1]['min_samples_First for 1 layers'])

    spectral_name = 'min_samples_l2 spectral for ' +  str_layer + ' layers'
    spectral.append(input_results[0][spectral_name])

    norm_name = 'min_samples_l2 book for ' + str_layer + ' layers'
    norm_based.append(input_results[0][norm_name])

    lipschitz_based_name = 'min_samples_Parameter book for ' + str_layer + ' layers'
    lipschitz_based.append(input_results[0][lipschitz_based_name])


    pdim_name = 'min_samples_Pdim for ' + str_layer + ' layers'
    pdim.append(input_results[0][pdim_name])
    

plt.figure()
plt.title('NVAC vs. Depth')
plt.xlabel('Depth')
plt.ylabel('NVAC')

plt.plot(depths, ours, marker = "s")
plt.plot(depths, lipschitz_based, marker = "x",color='g')
plt.plot(depths, pdim, marker = 'D',color='r')
plt.plot(depths, spectral, marker = "o")
plt.plot(depths, norm_based, marker = "^",color='purple')
plt.yscale('log')
plt.grid()


plt.legend(['Ours, noise=0.05','Lipshchitzness-based','Pseudo-dim-based','Spectral','Norm-based'])
plt.savefig(os.path.join(os.getcwd(), 'NVAC_depth_with_norm.pdf'))
#tikzplotlib.save(os.path.join(os.getcwd(), 'NVAC_depth_with_norm.tex'))


# In[4]:

#Plotting NVAC vs. Width
ours = []
spectral = []
norm_based = []
lipschitz_based = []
pdim = []
widths = [64,150,250,350,500,800,1000,1500]
for i in widths:
    
    str_layer = '4'  # len(layers) + 1
    str_neurons = str(i)
    directory = 'models/' + str_neurons + '_' + str_neurons + '_' + str_neurons 
    parent_dir = os.path.join(os.getcwd(),directory)
    print(parent_dir)
    mixed_covers_path = os.path.join(parent_dir, 'mixed_cover.pckl') 
    f = open(mixed_covers_path, 'rb') # getting the results of training
    input_results = CPU_Unpickler(f).load()
    f.close()    

    ours.append(input_results[2]['min_samples_First for 1 layers'])

    spectral_name = 'min_samples_l2 spectral for ' +  str_layer + ' layers'
    spectral.append(input_results[0][spectral_name])

    norm_name = 'min_samples_l2 book for ' + str_layer + ' layers'
    norm_based.append(input_results[0][norm_name])

    lipschitz_based_name = 'min_samples_Parameter book for ' + str_layer + ' layers'
    lipschitz_based.append(input_results[0][lipschitz_based_name])
    

    pdim_name = 'min_samples_Pdim for ' + str_layer + ' layers'
    pdim.append(input_results[0][pdim_name])
        

plt.figure()
plt.title('NVAC vs. Width')
plt.xlabel('Width')
plt.ylabel('NVAC')

plt.plot(widths, ours, marker = "s")
plt.plot(widths, lipschitz_based, marker = "x",color='g')
plt.plot(widths, pdim, marker = 'D',color='r')
plt.plot(widths, spectral, marker = "o")
plt.plot(widths, norm_based, marker = "^",color='purple')

plt.yscale('log')
plt.grid()


plt.legend(['Ours, noise=0.05','Lipshchitzness-based', 'Pseudo-dim-based','Spectral','norm-based'])
plt.savefig(os.path.join(os.getcwd(), 'NVAC_width_with_norm.pdf'))
#tikzplotlib.save(os.path.join(os.getcwd(), 'NVAC_width_with_norm.tex'))


# In[5]:


#plotting the train test ramp losses for baseline architecure
parent_dir = os.path.join(os.getcwd(),'models/250_250_250/more_average/valid_results.pckl')

f = open(parent_dir, 'rb') # getting the results of training
input_results = CPU_Unpickler(f).load()
f.close()    
    
noise_levels = []
test_error=[]
train_error = []
diff = []

for result in input_results:
    noise_levels.append(result['noise_level'])
    test_error.append(1-result['test_accuracy'])
    train_error.append(1-result['train_accuracy'])
    
y_ticks = np.linspace(0,1,10)
y_ticks = np.concatenate((np.linspace(0,0.9,10),np.linspace(0.9,1,10)),axis=0)
x_ticks = np.linspace(0,0.5,20)
    

plt.figure()
plt.plot(noise_levels, train_error)
plt.plot(noise_levels, test_error)

plt.legend(['Train', 'Test'])
plt.title('0-1 Loss Vs. Noise')
plt.xlabel('Noise standard deviation')
plt.ylabel('0-1 Loss')

plt.grid()
plt.savefig(os.path.join(os.getcwd(),'error_train_test.pdf'))
#tikzplotlib.save(os.path.join(os.getcwd(), 'error_train_test.tex'))
    
