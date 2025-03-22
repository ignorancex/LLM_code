#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:32:04 2020

@author: kflouris
"""
import torch
import numpy as np
import sys

# from torchdiffeq import odeint
import matplotlib.pyplot as plt 
from gflow import time_steps_func
# sys.path.append("../torchdiffeq-master/")

sys.path.append("torchdiffeq-master-amd")
from torchdiffeq import odeint as odeint
#=================== data ==========================

def create_linear_data(n):
    dim = np.linspace(0, 1, n, dtype=np.float32)
    def func_line(x,*argv):
        return  argv[0] + argv[1]*x
    #np.random.seed(2000)
    x2_noise = 0.02 * np.random.normal(size=n)
    #np.random.seed(2001)
    x1_noise =0.02 * np.random.normal(size=n)
    data=np.array([(func_line(dim, 0,1)+x1_noise),(func_line(dim, 0,1)+x2_noise)], dtype=np.float32).T  
    return  torch.from_numpy(data)


def create_sin_data(n):
    dim = np.linspace(0, 1, n, dtype=np.float32)
    def func_sin(x,*argv):
        return  argv[0] + argv[1]*np.sin(2*np.pi*x)
    data=np.array([(dim),(func_sin(dim, 0,1))], dtype=np.float32).T
    return  torch.from_numpy(data)

def create_circular_data(BATCH_SIZE, OUTPUT_DIM):
    # in 4 out 16
    z = np.random.randn(BATCH_SIZE)*np.sqrt(np.pi)/2.0
    if OUTPUT_DIM==2: x = np.concatenate((np.sin(z)[:,None],np.cos(z)[:,None]), axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    # if OUTPUT_DIM==2: x = np.concatenate((z[:,None],z[:,None]), axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    # if OUTPUT_DIM==2: x = np.concatenate((z[:,None],z[:,None]), axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    
    # x_ = np.concatenate((z[:,None],np.cos(z)[:,None]), axis=1) + np.random.randn(BATCH_SIZE,2)*0.05
    
    # x_ = np.concatenate((z[:,None],np.sin(z)[:,None],z[:,None],np.sin(z)[:,None],
    #                      z[:,None],np.sin(z)[:,None],z[:,None],np.sin(z)[:,None],
    #                      z[:,None],np.sin(z)[:,None],z[:,None],np.sin(z)[:,None],
    #                      z[:,None],np.sin(z)[:,None], z[:,None], np.sin(z)[:,None]), 
    #                      axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    # x_ = np.concatenate((z[:,None],np.sin(z)[:,None],z[:,None],np.sin(z)[:,None],
    #                       z[:,None],np.sin(z)[:,None],z[:,None],np.sin(z)[:,None]),
    #                       axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    # x_ = np.concatenate((np.cos(z)[:,None],np.sin(z)[:,None],np.cos(z)[:,None],np.sin(z)[:,None],
    #                       np.cos(z)[:,None],np.sin(z)[:,None],np.cos(z)[:,None],np.sin(z)[:,None]),
    #                       axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05
    # t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, BATCH_SIZE))
    # x = t * np.cos(t)
    # z = t * np.sin(t)

    # X = np.concatenate((x, z))
    # X += 0.05 * np.random.randn(2, BATCH_SIZE)
    # x_ = X.T
    # t = np.squeeze(t)   
    if OUTPUT_DIM==6: x = np.concatenate((np.cos(z)[:,None],np.sin(z)[:,None],np.cos(z)[:,None],np.sin(z)[:,None],
                          np.cos(z)[:,None],np.sin(z)[:,None]),
                          axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05    
    # x_ = np.concatenate((np.cos(z)[:,None],np.sin(z)[:,None],np.cos(z)[:,None],np.sin(z)[:,None]),
    #                       axis=1) + np.random.randn(BATCH_SIZE,OUTPUT_DIM)*0.05    
    # dtype = torch.FloatTensor
    # x_ = torch.rand((BATCH_SIZE, OUTPUT_DIM)).type(dtype)
    return x


def omniglot_data(BATCH_SIZE):
    
    from torchvision import datasets, transforms
    
    transforms = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.Omniglot('./data', transform=transforms, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 
    
    return train_dataloader, train_dataset, class_names

def mnist_data(BATCH_SIZE,val_samples=8):
    
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])
    NUM_TRAIN = 50000 - val_samples
    
    allidx = np.arange(50000)
    rstate = np.random.RandomState(seed=0)
    rstate.shuffle(allidx)
    trainvalset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainvalset, allidx[:NUM_TRAIN])
    valset = torch.utils.data.Subset(trainvalset, allidx[NUM_TRAIN:])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    testset = datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader
#
def seg_mnist_data(BATCH_SIZE,val_samples=8):
    # VALIDATION DOES NOT WORK WITH THIS, need To fix indexig see above allidx
    def get_indices(dataset,class_name):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] in class_name:
                indices.append(i)
        return indices
    
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])
    NUM_TRAIN = 50000 - val_samples
    
    # allidx = np.arange(50000)
    # rstate = np.random.RandomState(seed=0)
    # rstate.shuffle(allidx)
    trainvalset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    idx = get_indices(trainvalset, [0,1,2,3,4])
    # trainvalset =  torch.utils.data.Subset(trainvalset, idx)
    
    trainset = torch.utils.data.Subset(trainvalset, idx[:NUM_TRAIN])
    valset = torch.utils.data.Subset(trainvalset, idx[NUM_TRAIN:])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    testset = datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    
    idx = get_indices(testset, [5,6,7,8,9])
    testset = torch.utils.data.Subset(testset, idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader
#

def fmnist_data(BATCH_SIZE,val_samples=8):
    
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])
    NUM_TRAIN = 50000 - val_samples
    
    allidx = np.arange(50000)
    rstate = np.random.RandomState(seed=0)
    rstate.shuffle(allidx)
    trainvalset = datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainvalset, allidx[:NUM_TRAIN])
    valset = torch.utils.data.Subset(trainvalset, allidx[NUM_TRAIN:])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    testset = datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader
    
def cifar10_data(BATCH_SIZE, val_samples=8):
    
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])
    NUM_TRAIN = 50000 - val_samples
    
    allidx = np.arange(50000)
    rstate = np.random.RandomState(seed=0)
    rstate.shuffle(allidx)
    trainvalset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainvalset, allidx[:NUM_TRAIN])
    valset = torch.utils.data.Subset(trainvalset, allidx[NUM_TRAIN:])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, valloader, testloader, classes

def celeba_data(BATCH_SIZE, val_samples=8):
    
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])
    NUM_TRAIN = 50000 - val_samples
    
    allidx = np.arange(50000)
    rstate = np.random.RandomState(seed=0)
    rstate.shuffle(allidx)
    trainvalset = datasets.CelebA(root='./data', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainvalset, allidx[:NUM_TRAIN])
    valset = torch.utils.data.Subset(trainvalset, allidx[NUM_TRAIN:])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    testset = datasets.CelebA(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, valloader, testloader, classes

#=================== plotting ==========================


def show_visual_progress(net, glfclass, test_dataloader,rows=1, flatten=True, conditional=False, title=None):
    if title:
        plt.title(title)        
    iter(test_dataloader)
    image_rows = []
    
    for idx, (batch, label) in enumerate(test_dataloader):
        if rows == idx:break
        x_data = batch.view(-1, 28 * 28)
        zs = glfclass.determine_zs(x_data)
        if flatten: batch = batch.view(batch.size(0), 28*28)
#        images=batch.numpy().reshape(batch.size(0),28,28) # use just to plot them
        images = net(zs).detach().cpu().numpy().reshape(batch.size(0),28,28)
        image_idxs = [list(label.numpy()).index(x) for x in range(3)]
        combined_images = np.concatenate([images[x].reshape(28,28) for x in image_idxs],1)
        combined_batch = np.concatenate([x_data[x].reshape(28,28) for x in image_idxs], 1)
        combined_show = np.concatenate([combined_images, combined_batch], 0)
        image_rows.append(combined_show)
        print('||Batch-Image||=',((combined_batch - combined_images)**2).sum())  
    plt.imshow(np.concatenate(image_rows))  
    if title:
         title = title.replace(" ", "_")
         plt.savefig(title)
    plt.show()


def plot_convergence_curve_gen_adj(gflnet, x_data, log_px, name, loss_function, tau, tsteps): 
    batch_size = x_data.shape[0]
    gflnet.zero_lists()
    zs = odeint(gflnet.z_model, 
        gflnet.z0.repeat(x_data.shape[0], 1), 
        torch.from_numpy(time_steps_func(tau,tsteps)), method='rk4')
    errors = []
    for zsv in zs: 
        errors.append((loss_function(gflnet.net(zsv)[0],gflnet.net(zsv)[1], gflnet.x_data)/batch_size).cpu().detach().numpy())
    fig=plt.figure() 
    plt.plot( time_steps_func(tau,tsteps), errors, 'x'), 
    plt.savefig(name)
    plt.close()
    return fig
        
   
   
def plot_convergence_curve(gflnet, x_data, name, loss_function, tau, tsteps): 
    batch_size = x_data.shape[0]
    gflnet.zero_lists()
    gflnet.x_data=x_data
    zs = gflnet.determine_zs_allsteps()  
    # print(zs[:,0,:,:].shape)
    zs=zs[:,0,:,:]
    errors = []
    for zsv in zs: 
        errors.append((loss_function(gflnet.net(zsv)[0], gflnet.x_data)/batch_size).cpu().detach().numpy())
    fig=plt.figure() 
    plt.plot( time_steps_func(tau,tsteps), errors) 
    plt.savefig(name)
    plt.close()
    np.savetxt(name[:-8]+'errors', errors)   
    return fig
        
def reshape_reconstructions(x_data, x_recon, title=None, data_shape=None): 
    from torchvision import utils
    combined_images = x_data.reshape(data_shape)
    combined_recons = x_recon.reshape(data_shape)
    combined_show = torch.cat([combined_images, combined_recons], 0)
    img=utils.make_grid(combined_show)
    return img



def plot_reconstructions(x_data,x_recon, title=None, data_shape=(1,3,32,32) ): 
    from torchvision import utils
    plt.figure() 
    # combined_images = x_data.reshape(data_shape)
    combined_recons = x_recon.reshape(data_shape)
    # combined_show = torch.cat([combined_images, combined_recons], 0)
    # img=utils.make_grid(combined_show)
    img=utils.make_grid(combined_recons)
     
    npimg = img.cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
      title = title.replace(" ", "_")
      plt.savefig(title)    
    plt.show()
    plt.close()    

def plot_images_lowdim(x_data, x_model, log_px, OUTPUT_DIM, title=None  ): 
    if OUTPUT_DIM != 2:
        fig, axs = plt.subplots(1,int(OUTPUT_DIM/2), figsize=(15, 6), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()
        
        for i in range(int(OUTPUT_DIM/2)):
            px=np.exp(log_px)
            l=i*2
            # axs[i].plot(np.random.rand(10,10),'.')
            try:
                axs[i].plot(x_data[:,l], x_data[:,l+1], '.', color='grey' ,alpha=0.5)        
                im=axs[i].scatter( x_model[:,l], x_model[:,l+1], alpha=0.6, #c=px
                          cmap='viridis', edgecolors='none')      
            except:None
        # axs[i].set_title(str(250+i))
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)      
        plt.show() 
        plt.close()    
                   
    else:   
        px=np.exp(log_px)
        plt.figure(),
        plt.plot(x_data[:,0], x_data[:,1], '.', color='grey' ,alpha=0.3)        
        plt.scatter( x_model[:,0], x_model[:,1], c=px, alpha=0.6,
                    cmap='viridis', edgecolors='none')
        plt.colorbar();  # show color scale            
        plt.axis('equal')
        plt.show() 
        plt.close()
