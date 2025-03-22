"""
Created on Fri Apr 10 15:31:50 2020

@author: kflouris
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np

from gflow import GFlow_approx, GFlow_AMD_approx, GFlow_mom, GFlow_AMD_mom, GFlow_adj, GFlow_AMD_adj, GFlow_gen_adj

from visualisation import (mnist_data, seg_mnist_data, fmnist_data, cifar10_data,
                           plot_convergence_curve,plot_convergence_curve_gen_adj, plot_reconstructions, 
                           create_circular_data, plot_images_lowdim,
                           omniglot_data, reshape_reconstructions)

from networks import (XiaoranDec_cifar, net_cifar10_conv, net_cifar10_conv_elu, 
                      XiaoranAE_cifar, net_cifar10_conv_enc, 
                      net_cifar10_conv_AE, net_mnist_enc, net_mnist_dec, net_mnist_AE, net_cifar10_conv, 
                      net_linear,  net_mnist_samedim)

from gflow_auto import  GFlow_adj_hooks, GFlow_AMD_adj_hooks,  Autograd, Autograd_static , Autograd_simple#, GFlow_autograd,  Autograd_static, Autograd_simple,

from networks_gen import net_generative 

from pathlib import Path
import argparse 
import configparser
from shutil import copyfile
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
np.random.seed(2020)
torch.manual_seed(2020)


if __name__ == "__main__": 
    
    # ================= Configuration ===============
    parser = argparse.ArgumentParser()    
    parser.add_argument('--c', type=str, default='config/config_cifar10.ini')
    args = parser.parse_args()
     
    CONFIG = args.c
    config = configparser.ConfigParser(comment_prefixes=('#', ';'), inline_comment_prefixes=('#', ';'))
    config.read(CONFIG)

    print(CONFIG)
    for section_name in config.sections():
        print(">  {}:".format(section_name) )
        for name, value in config.items(section_name):
            print('>  %s = %s' % (name, value))
    
    SAVE_DIR        = config['DIRECTORIES']['SAVE_DIR']  
    MODEL_SAVE_DIR  = config['DIRECTORIES']['MODEL_SAVE_DIR']  
    
    BATCH_SIZE      = int(config['NETWORK']['BATCH_SIZE'])
    INPUT_DIM       = int(config['NETWORK']['INPUT_DIM'] )
    OUTPUT_DIM      = int(config['NETWORK']['OUTPUT_DIM']) 
    NET             = config['NETWORK']['NET']
    NET_LAYERS      = int(config['NETWORK']['NET_LAYERS'])
    MODEL           = config['NETWORK']['MODEL']

    EPOCHS          = int(config['OPTIMIZATION']['EPOCHS'])
    OPT_LR_RATE     = float(config['OPTIMIZATION']['OPT_LR_RATE'])
    OPT_EPSILON     = float(config['OPTIMIZATION']['OPT_EPSILON'])
    OPT_ALPHA       = float(config['OPTIMIZATION']['OPT_ALPHA'])
    
    ODE_TAU         = float(config['GFLOW']['ODE_TAU'])
    ODE_ALPHA       = float(config['GFLOW']['ODE_ALPHA'])
    ODE_TSTEPS      = int(config['GFLOW']['ODE_TSTEPS'])
    ODE_METHOD      = config['GFLOW']['ODE_METHOD']
    ODE_RTOL        = float(config['GFLOW']['ODE_RTOL'])
    ODE_ATOL        = float(config['GFLOW']['ODE_ATOL'])
    ODE_APPROX_FACTOR  = int(config['GFLOW']['ODE_APPROX_FACTOR']) 
    
    AMD_DTSTEP       = float(config['AMD']['AMD_DTSTEP'])
    AMD_DTMAXSTEP    = float(config['AMD']['AMD_DTMAXSTEP'])
    AMD_DTFACTOR     = float(config['AMD']['AMD_DTFACTOR'])
    AMD_DTITER       = float(config['AMD']['AMD_DTITER'])
    AMD_CONV_GRADDT  = float(config['AMD']['AMD_CONV_GRADDT'])
    AMD_CONV_PERCTAU = float(config['AMD']['AMD_CONV_PERCTAU'])    
    AMD_CONV_DTSTEP  = float(config['AMD']['AMD_CONV_DTSTEP'])    
    AMD_ARMIJO_SIGMA = float(config['AMD']['AMD_ARMIJO_SIGMA'])    
    
    TENSORBOARD     = config.getboolean('ACTORS','TENSORBOARD')
    TNSE            = config.getboolean('ACTORS','TNSE')
    VALIDATION      = config.getboolean('ACTORS','VALIDATION')
    TRAINING        = config.getboolean('ACTORS','TRAIN')
    TESTING         = config.getboolean('ACTORS','TEST')
    SAVE_MODEL      = config.getboolean('ACTORS','SAVE_MODEL')
    RUNNAME         = config['ACTORS']['RUNNAME']

    if RUNNAME==('None' or 'none'): runname='cpu_{}_{}_{}_lr{}_inD{}_tau{}_tst{}_odeA{}_odeAtol{}/'.format(
            MODEL,NET,ODE_METHOD, OPT_LR_RATE,INPUT_DIM,ODE_TAU,ODE_TSTEPS, ODE_ALPHA, ODE_ATOL)
    else: runname = RUNNAME 
    SAVE_DIR += runname   
    model_save_path=MODEL_SAVE_DIR + runname[:-1]

    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)  
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)    
    copyfile(CONFIG, SAVE_DIR+'config_file.ini')
            
    GFE             = MODEL == 'GFE'
    AMD             = MODEL == 'AMD'
    AMD_mom         = MODEL == 'AMD_mom'
    GFE_mom         = MODEL == 'GFE_mom'
    GFE_adj         = MODEL == 'GFE_adj'
    AMD_adj         = MODEL == 'AMD_adj'
    GFE_adj_approx  = MODEL == 'GFE_adj_approx'
    GFE_adj_hooks   = MODEL == 'GFE_adj_hooks' 
    AMD_adj_hooks   = MODEL == 'AMD_adj_hooks' 
    GFE_adj_auto    = MODEL == 'GFE_adj_auto' 
    GFE_gen         = MODEL == 'GFE_gen'
    GFE_gen_adj     = MODEL == 'GFE_gen_adj'    
    GFE_GEN_adj_hooks     = MODEL == 'GFE_GEN_adj_hooks'    
    AMD_GEN_adj_hooks   = MODEL == 'AMD_GEN_adj_hooks'    
    AE              = MODEL == 'AE'
    
    FLATTEN = NET in ['mnist','mnist_samedim', 'omniglot', 'lowdim', 'fmnist', 'seg_mnist']
    if NET=='mnist_samedim':INPUT_DIM=OUTPUT_DIM
    if TENSORBOARD:
        import tensorboardX        
        tbwriter = tensorboardX.SummaryWriter(SAVE_DIR)  
        
    # ================= Preliminary ===============
    if NET in  ['mnist','mnist_samedim', 'omniglot', 'fmnist', 'seg_mnist']  :  
        def format_net(n): return torch.sigmoid(n)
    else:
        def format_net(n): return n    
    def total_loss(z_loss, log_x_prob): 
        z=z_loss
        p=log_x_prob.mean(axis=0)
        # return (z_loss + torch.sign(p)*abs((p-z)/z))/2
        return z - p*0.5          
    # ================= Train data ===============                          
    if GFE_adj_hooks:
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
            gflow = GFlow_adj_hooks(net, INPUT_DIM, loss_function, errdt=ODE_APPROX_FACTOR, tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)   
            optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )  
    if AMD_adj_hooks:
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            def amd_loss_function(y, x, eps=1e-8): return torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1)            
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
            gflow = GFlow_AMD_adj_hooks(net, INPUT_DIM, loss_function,  amd_loss_function,approx_factor=ODE_APPROX_FACTOR, tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
                                                dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT, conv_dtstep=AMD_CONV_DTSTEP,
                                                conv_percentagetau=AMD_CONV_PERCTAU, armijosigma=AMD_ARMIJO_SIGMA)   
            optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )              
            GFE_adj_hooks=True
            
    elif GFE_adj or AMD_adj or GFE_adj_approx or GFE_adj_auto:
        if NET=='mnist':
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
        elif NET=='lowdim':
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            loss_function=torch.nn.MSELoss(reduction="mean")
        if AMD_adj:
            def amd_loss_function(y, x, eps=1e-8): return torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1)            
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
            gflow = GFlow_AMD_adj(net, INPUT_DIM, loss_function, amd_loss_function, tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
                                                dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT, conv_dtstep=AMD_CONV_DTSTEP,
                                                conv_percentagetau=AMD_CONV_PERCTAU, armijosigma=AMD_ARMIJO_SIGMA)   
            GFE_adj=True
        else: gflow = GFlow_adj(net, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)      
        s=[]
        for i, phi in enumerate(net.parameters()): 
            s.append(torch.zeros_like(phi))
        s0 = torch.zeros(INPUT_DIM) 

    elif AMD or GFE or GFE_mom or AMD_mom:  
        if   NET=='mnist':     train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
        elif NET=='seg_mnist': train_iterator, val_iterator, test_iterator= seg_mnist_data(BATCH_SIZE)
        elif NET=='fmnist':    train_iterator, val_iterator, test_iterator= fmnist_data(BATCH_SIZE)
        elif NET=='omniglot':  train_iterator, test_iterator, classes= omniglot_data(BATCH_SIZE)
        elif NET=='lowdim':
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            loss_function=torch.nn.MSELoss(reduction='mean')
            def amd_loss_function(y, x, eps=1e-8): return torch.mean((y-x)**2, dim=(1))          
            optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)
            
        elif NET=='cifar10':
            train_iterator, val_iterator, test_iterator, classes= cifar10_data(BATCH_SIZE, val_samples=BATCH_SIZE*10)
            net = XiaoranDec_cifar(latent_dim=INPUT_DIM, bn=True, sigmoid=True)
            # net = net_cifar10_conv(INPUT_DIM)
            loss_function=torch.nn.MSELoss(reduction='mean')
            def amd_loss_function(y, x, eps=1e-8): return torch.sqrt(torch.mean((y-x)**2, dim=(1,2,3)))            
            optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)
            
        if NET in ['mnist','omniglot','fmnist','seg_mnist']:
            net = net_mnist_dec(INPUT_DIM,OUTPUT_DIM)
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))            
            def amd_loss_function(y, x, eps=1e-8): return torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1)            
            optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA ) 
        
        if AMD_mom: gflow = GFlow_AMD_mom(net, INPUT_DIM, loss_function,amd_loss_function,tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
                                                dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT,
                                                conv_percentagetau=AMD_CONV_PERCTAU, armijosigma=AMD_ARMIJO_SIGMA)        
        elif AMD: gflow = GFlow_AMD_approx(net, INPUT_DIM, loss_function, amd_loss_function, tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
                                                dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT,
                                                conv_percentagetau=AMD_CONV_PERCTAU, conv_dtstep=AMD_CONV_DTSTEP, armijosigma=AMD_ARMIJO_SIGMA)        
        elif GFE_mom: gflow = GFlow_mom(net, INPUT_DIM, loss_function,tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        
        elif GFE: gflow = GFlow_approx(net, INPUT_DIM, loss_function,tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        

        

            
        elif NET =='cifar10': optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)
  
        GFE=True
        
    elif AMD_GEN_adj_hooks:
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device, dimlist=[1,2,2,2])
            loss_function=torch.nn.MSELoss(reduction="mean")
            def amd_loss_function(y, x, eps=1e-8): return torch.sqrt(torch.mean((y-x)**2, dim=(1)))            
            gflow = GFlow_AMD_adj_hooks(net, INPUT_DIM, loss_function,  amd_loss_function,approx_factor=ODE_APPROX_FACTOR, tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
                                                dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT, conv_dtstep=AMD_CONV_DTSTEP,
                                                conv_percentagetau=AMD_CONV_PERCTAU, armijosigma=AMD_ARMIJO_SIGMA)             # optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )
            optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)
    elif GFE_GEN_adj_hooks:
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device, dimlist=[1,2,2,2])
            loss_function=torch.nn.MSELoss(reduction="mean")
            def amd_loss_function(y, x, eps=1e-8): return torch.sqrt(torch.mean((y-x)**2, dim=(1)))            
            gflow = GFlow_adj_hooks(net, INPUT_DIM, loss_function, errdt=ODE_APPROX_FACTOR, tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)   
            optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)   
            AMD_GEN_adj_hooks=True
    elif GFE_gen:
        if NET=='lowdim':
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device, dimlist=[1,2,2,2])
            loss_function=torch.nn.MSELoss(reduction="mean")
            gflow = GFlow_gen_adj(net, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        
            # optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )
            optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE)       
        if NET=='mnist':
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device=device)
            def loss_function(y, x, eps=1e-8): return torch.mean(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
            gflow = GFlow_approx(net, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        
            optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )     

    elif GFE_gen_adj:
        if NET=='lowdim':
            x_=create_circular_data(BATCH_SIZE, OUTPUT_DIM)
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            # net = net_mnist(INPUT_DIM,OUTPUT_DIM)
            def total_loss(z_loss, log_x_prob): 
                z=z_loss
                p=log_x_prob.mean(axis=0)
                # return (z_loss + torch.sign(p)*abs((p-z)/z))/2
                return z - p*1
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device=device, dimlist=[1,2,2,2])
            loss_function=torch.nn.MSELoss(reduction="mean")
        elif NET=='mnist':
            train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
            net = net_generative(INPUT_DIM,OUTPUT_DIM, NET_LAYERS, device=device)
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
        
        gflow = GFlow_gen_adj(net, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)      
        s=[]
        for i, phi in enumerate(net.parameters()): 
            s.append(torch.zeros_like(phi))
        s0 = torch.zeros(INPUT_DIM).to(device)       
    
    elif AE: 
        if NET=='cifar10':
            train_iterator, val_iterator, test_iterator, classes= cifar10_data(BATCH_SIZE, val_samples=BATCH_SIZE*10)
            net = XiaoranAE_cifar(latent_dim=INPUT_DIM, bn=True, sigmoid=True)    
            loss_function=torch.nn.MSELoss(reduction='mean')            
        elif NET=='mnist':    train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
        elif NET=='seg_mnist':train_iterator, val_iterator, test_iterator= seg_mnist_data(BATCH_SIZE)
        elif NET=='fmnist':   train_iterator, val_iterator, test_iterator= fmnist_data(BATCH_SIZE)
        elif NET=='omniglot': train_iterator, val_iterator, test_iterator= mnist_data(BATCH_SIZE)
        
        if NET in ['mnist','mnist_samedim', 'omniglot', 'lowdim', 'fmnist', 'seg_mnist']:
            net = net_mnist_AE(INPUT_DIM, OUTPUT_DIM)             
            def loss_function(y, x, eps=1e-8): return torch.sum(torch.mean(-x * torch.log(torch.sigmoid(y) + eps) - (1-x) * torch.log(1 - torch.sigmoid(y) + eps), axis=1))
            optimizer = torch.optim.RMSprop(net.parameters(), lr=OPT_LR_RATE, eps=OPT_EPSILON, alpha=OPT_ALPHA )     
        
        elif NET == 'cifar10': optimizer = torch.optim.Adam(net.parameters(), lr=OPT_LR_RATE) 
        net=net.to(device)
  
        # ==================== Learning ===================================================
    if TRAINING:
        tb_step=0
        loss_tb_step=0
        for k in range(EPOCHS):
            print('\n Training \n')   
            for j, (image, y) in enumerate(train_iterator):
                #=========== frequency ============
                VISUALIZE = (j % 1000 == 0)
                VALIDATE = ( VALIDATION and j % 1 == 0 )
                TENSORBOARD_IMAGE= (TENSORBOARD and j % 100 == 0)  
                PRINT = (j % 1 == 0 )  
                tb_step+=1;
                # ============ Data fromating ============
                # if NET=='lowdim':  x_data = torch.FloatTensor(x_)         
                if NET=='lowdim':  x_data = torch.autograd.Variable(torch.Tensor(x_))       
                elif FLATTEN: x_data = image.view(-1, OUTPUT_DIM).to(device)
                else: x_data=image.to(device)
                x_data = torch.autograd.Variable(x_data)
                # ============ Forward ============
                if GFE:
                    gflow.x_data=x_data
                    zs = gflow.determine_zs(z0='rand') 
                    # gflow.z0=zs
                    # import sys
                    # sys.exit()
                    x_model, log_px = gflow.net(zs)
                    loss = loss_function(x_model, x_data)   
                     
                elif GFE_adj or GFE_adj_approx or GFE_adj_hooks or AMD_GEN_adj_hooks:
                    gflow.x_data=x_data
                    gflow.forward()
                    zs=gflow.zs[-1]
                    x_model, log_px=net(zs)
                    loss = loss_function(x_model, x_data)                  
                elif GFE_adj_auto:
                    # # Autograd_static.apply(net, INPUT_DIM, loss_function, tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)
                    # auto= Autograd_static.apply(net, INPUT_DIM, loss_function,ODE_TAU,ODE_RTOL,ODE_ATOL, ODE_TSTEPS, device, x_data)
                    # # fwrd=auto(net, INPUT_DIM, loss_function,ODE_TAU,ODE_RTOL,ODE_ATOL, ODE_TSTEPS, device)
                    # # zs = gflow.determine_zs()
                    # # gflow.forward()
                    # z0=torch.zeros((BATCH_SIZE, net.decoder[0].in_features), requires_grad=True)
                    # x_model=net(z0)[0]
                    # loss = loss_function(x_model, x_data)             
                    gflow.x_data=x_data
                    gflow.forward()
                    zs=gflow.zs[-1]
                    Autograd_simple.apply(gflow,zs)
                    x_model=net(zs)[0]
                    loss = loss_function(x_model, x_data)   
                elif GFE_gen or GFE_gen_adj:         
                    zs = gflow.determine_zs(x_data, z0=None)  
                    x_model,log_px = gflow.net(zs)
                    z_loss = loss_function(x_model, x_data) 
                    loss=total_loss(z_loss,log_px)
                elif AE: 
                    x_model = net(x_data)
                    loss = loss_function(x_model, x_data)    
                
                # ============ Backward ============
                if GFE_adj_hooks or AMD_GEN_adj_hooks: 
                    
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True, create_graph=True)
                    # loss.backward(create_graph=True)
                    # # TEMP:
                    # for name, param in net.named_parameters():
                    #     print(name,torch.mean(param.grad))

                    optimizer.step() 
                elif GFE_adj_auto:
                    # optimizer.zero_grad()
                    # loss.backward(retain_graph=True, create_graph=True)
                    # optimizer.step()  
                    dlist = gflow.backward()
                    for i, p in enumerate(net.parameters()):
                        s[i]=OPT_ALPHA*s[i] + (1-OPT_ALPHA)*((dlist[i]/BATCH_SIZE)**2)    
                        p.data.sub_(dlist[i]/BATCH_SIZE*OPT_LR_RATE/torch.sqrt(s[i] + OPT_EPSILON))                                     
                elif GFE_adj  or GFE_adj_approx or GFE_gen_adj:
                    dlist = gflow.backward(x_data)
                    for i, p in enumerate(net.parameters()):
                        s[i]=OPT_ALPHA*s[i] + (1-OPT_ALPHA)*((dlist[i]/BATCH_SIZE)**2)    
                        p.data.sub_(dlist[i]/BATCH_SIZE*OPT_LR_RATE/torch.sqrt(s[i] + OPT_EPSILON))  
                    # update the initial point
                    # s0 = (OPT_ALPHA*s0 + (1-OPT_ALPHA)*((dlist[-1]/BATCH_SIZE)**2))  
                    # gflow.z0 = gflow.z0 - (dlist[-1]/BATCH_SIZE*OPT_LR_RATE/torch.sqrt(s0 + OPT_EPSILON))                
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # # TEMP:
                    # for name, param in net.named_parameters():
                    #     print(name,torch.mean(param.grad))
                        
                    optimizer.step()               
                
                if PRINT:
                    # if GFE_gen:
                    #     px=torch.exp(log_px.cpu().detach().numpy())        
                    #     txtt='E {}, it {}: dist-loss {}; log_p(x) {}; tot_loss {}'.format(k+1, j+1, z_loss, log_px.mean(axis=0), loss)
                    # else: txtt='E {}, it {}: dist-loss {}'.format(k+1, j+1,loss)
                    txtt='E {}, it {}: dist-loss {}'.format(k+1, j+1,loss/BATCH_SIZE) 
                    print(txtt)  
                           
                if TENSORBOARD: 
                    tbwriter.add_scalar('train/total_loss', loss.item()/BATCH_SIZE,  global_step=tb_step)
                    try:
                        from sklearn.metrics import mean_squared_error
                        rmse=mean_squared_error(x_data.cpu().detach().numpy(),x_model.cpu().detach().numpy())
                        tbwriter.add_scalar('train/RMSE', rmse.item(),  global_step=tb_step) 
                        tbwriter.add_scalar('train/dist_loss', dist_loss.item(),  global_step=tb_step)
                        tbwriter.add_scalar('train/log_px', log_px.item(),  global_step=tb_step)
                    except: None  
                    try:tbwriter.flush()  
                    except: None  

                # ================= Visualisation===============
                if TENSORBOARD_IMAGE:
                    if AE:  tbwriter.add_image('train/recon', reshape_reconstructions(x_data, format_net(net(x_data)), 
                          data_shape=image.shape), global_step=tb_step)
                    
                    else:
                        # fig=plot_convergence_curve(gflow, x_data, SAVE_DIR+'E_{}_it_{}_conv.png'.format(k,j),loss_function, ODE_TAU,ODE_TSTEPS)
                        # tbwriter.add_figure('train/conv',fig, global_step=tb_step)                                    
                        tbwriter.add_image('train/recon', reshape_reconstructions(x_data, format_net(gflow.net(zs)[0]), 
                          data_shape=image.shape), global_step=tb_step)                
                    tbwriter.flush()    
                    
                if (TNSE and tb_step ==120): 
                    # if AE: torch.save(net,model_save_path+'_tsne' )
                    # else : torch.save(gflow.net,model_save_path+'_tsne' )
                    # net = torch.load(model_save_path, map_location=torch.device('cpu'))
                    # net.eval()
                    # print(">>> loaded network:", model_save_path) 
                    if NET=='mnist': train_iterator, val_iterator, test_iterator= mnist_data(1000)
                    elif NET=='fmnist': train_iterator, val_iterator, test_iterator= fmnist_data(1000)
                    for j, (image, y) in enumerate(train_iterator):
                        if FLATTEN: x_data = image.view(-1, OUTPUT_DIM).to(device)
                        else: x_data=image.to(device)
                        x_data = torch.autograd.Variable(x_data)
                        if AE:
                            # gflow = GFlow_approx(net.decoder, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        
                            # gflow.x_data=x_data
                            # zs = gflow.determine_zs()  
                            zs= net.encoder(x_data)[0]
                        else:
                            gflow.net=net
                            gflow.x_data=x_data
                            zs = gflow.determine_zs()            
                        from sklearn.manifold import TSNE      
                        import seaborn as sns
                        sns.set(font_scale=10)
                        sns.set(rc={'figure.figsize':(11.7,8.27)})
                        sns.set_style("whitegrid")
        
                        palette = sns.color_palette("bright",10)
                        tsne = TSNE(perplexity=30.0, early_exaggeration=4.0, learning_rate=100.0) 
                        X_embedded = tsne.fit_transform(zs.cpu().detach().numpy())           
                        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full',  palette=palette)                
                        # sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')                
                        import sys
                        sys.exit(0)       
                        
                if VISUALIZE :              
                    if GFE_gen_adj:
                        if NET=='lowdim':  plot_images_lowdim(x_data.cpu().detach().numpy(), x_model.cpu().detach().numpy(),
                                                          log_px.cpu().detach().numpy(), OUTPUT_DIM)
                        else:
                            # fig=plot_convergence_curve_gen_adj(gflow, x_data, log_px,SAVE_DIR+'E_{}_it_{}_conv.png'.format(k,j),total_loss_adj, ODE_TAU,ODE_TSTEPS)
                            plot_reconstructions(x_data, format_net(gflow.net(zs)[0]), 
                                               SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,j), data_shape=image.shape)                                       
                    elif AMD or GFE or GFE_adj or GFE_adj_hooks or GFE_adj_auto or AMD_GEN_adj_hooks or GFE_gen:
                        if NET=='lowdim':  plot_images_lowdim(x_data.cpu().detach().numpy(), x_model.cpu().detach().numpy(),
                                                          log_px.cpu().detach().numpy(), OUTPUT_DIM)
                        else:
                            # fig=plot_convergence_curve(gflow, x_data, SAVE_DIR+'E_{}_it_{}_conv.pdf'.format(k,j),loss_function, ODE_TAU,ODE_TSTEPS)
                            plot_reconstructions(x_data, format_net(gflow.net(zs)[0]), 
                                                    SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,j), data_shape=image.shape) 
                    elif AE:
                        plot_reconstructions(x_data,format_net(net(x_data)) , 
                                                SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,j), data_shape=image.shape)    
    
                # ================= Validation ===============
                if VALIDATE:
                    nval = 0
                    valloss = 0
                    for valj, (image, _) in enumerate(val_iterator): 
                        nval += 1
                        if FLATTEN: xval = image.view(-1, OUTPUT_DIM).to(device)
                        else: xval=image.to(device)
                    
                        xval = torch.autograd.Variable(xval)
                        # ============ Forward ============
                        if AE:
                            x_model = net(xval)   

                        else:
                            gflow.x_data=xval
                            zsval =gflow.determine_zs(z0='rand')
                            x_model = gflow.net(zsval)[0]                                 
                            
                        loss = loss_function(x_model, xval)
                        valloss += loss.item()
                    if TENSORBOARD: 
                        tbwriter.add_scalar('val/data_loss', valloss/nval,  global_step=tb_step)
                        tbwriter.flush()  
                    if TENSORBOARD_IMAGE:            
                        
                        if AE:
                            tbwriter.add_image('val/recon', reshape_reconstructions(xval, net(xval), 
                                  data_shape=image.shape), global_step=tb_step)                             
                            
                        else:                
                            # fig = plot_convergence_curve(gflow, xval, SAVE_DIR+'E_{}_it_{}_conv_val.png'.format(k,j),loss_function, ODE_TAU,ODE_TSTEPS)
                             # tbwriter.add_figure('val_convergence',fig, global_step=tb_step)          
                            # plot_reconstructions(xval, format_net(gflow.net(zsval)[0]), 
                            #                             SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,j), data_shape=image.shape) 
                            tbwriter.add_image('val/recon', reshape_reconstructions(xval, format_net(gflow.net(zsval)[0]), 
                                  data_shape=image.shape), global_step=tb_step)
                        tbwriter.flush()  

                # if j == 60: break
   
    if SAVE_MODEL:
        if AE: torch.save(net,model_save_path )
        else : torch.save(gflow.net,model_save_path )

    #============================= test ===============================================

        
    if TESTING:
        print('\n Test \n')  
        if not TRAINING: 
            net = torch.load(model_save_path, map_location=torch.device('cpu'))
            net.eval()
            print(">>> loaded network:", model_save_path )
        
        testloss_arr=[]
        loss_tb_step = 0
        testloss = 0
        tb_step = 0

        # gflow = GFlow_AMD_approx(net, INPUT_DIM, loss_function,tau=ODE_TAU, alpha=ODE_ALPHA,  aTOL=ODE_ATOL,rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device,
        #                                         dtstep=AMD_DTSTEP, dtmaxstep=AMD_DTMAXSTEP, dtfactor=AMD_DTFACTOR, dtiter=AMD_DTITER, conv_graddt=AMD_CONV_GRADDT,
        #                                         conv_percentagetau=AMD_CONV_PERCTAU)
        
        for k in range(1):

            for testj, (image, _) in enumerate(test_iterator): 
                loss_tb_step+=1
                VISUALISATION = ( testj % 100==0  )
                TENSORBOARD_IMAGE= (TENSORBOARD and testj % 50 == 0)
                PRINT = (testj % 1 == 0 )  
    
                if FLATTEN: xtest = image.view(-1, OUTPUT_DIM).to(device)
                else: xtest=image.to(device)
            
                xtest = torch.autograd.Variable(xtest) 
                # ============ Forward ============
                if AE:
                    x_model = net(xtest)        
                else:
                    # gflow.net=net.decoder
                    gflow.net=net
                    gflow.x_data=xtest
                    zstest = gflow.determine_zs(z0='rand')            
                    x_model = gflow.net(zstest)[0]
                    
                loss = loss_function(x_model, xtest)
                testloss = loss.item()/BATCH_SIZE
                testloss_arr.append(testloss)
                print('\n current mean=', np.mean(testloss_arr))
                if PRINT:
                    txtt='It {}: dist-loss {}'.format( testj+1, testloss) 
                    print(txtt)
                if VISUALISATION:
                    if AE:           plot_reconstructions(xtest,format_net(net(xtest)) , 
                                                SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,testj), data_shape=image.shape)                     
                    else:        plot_reconstructions(xtest, format_net(gflow.net(zstest)[0]), 
                                         SAVE_DIR+'test_E_{}_it_{}_examples.png'.format(k,testj), data_shape=image.shape)                   
                    
                if TENSORBOARD: 
                    tbwriter.add_scalar('test/data_loss', testloss,  global_step=loss_tb_step)
                    tbwriter.flush()  

                if TENSORBOARD_IMAGE:
                    tb_step+=1
                    if AE:
                        tbwriter.add_image('test/recon', reshape_reconstructions(xtest, net(xtest), 
                              data_shape=image.shape), global_step=tb_step)  
                        plot_reconstructions(xtest,format_net(net(xtest)) , 
                                                SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,testj), data_shape=image.shape)                     
                    else:                
                        # fig = plot_convergence_curve(gflow, xtest, SAVE_DIR+'E_{}_it_{}_conv_test.png'.format(k,j),loss_function, ODE_TAU,ODE_TSTEPS)
                        # tbwriter.add_figure('test_convergence',fig, global_step=tb_step)          
                        plot_reconstructions(xtest, format_net(gflow.net(zstest)[0]), 
                                                    SAVE_DIR+'E_{}_it_{}_examples.png'.format(k,testj), data_shape=image.shape)                   
                        tbwriter.add_image('test/recon', reshape_reconstructions(xtest, format_net(gflow.net(zstest)[0]), 
                              data_shape=image.shape), global_step=tb_step)
                    tbwriter.flush()  
            
        print('\n Test mean=', np.mean(testloss_arr))
