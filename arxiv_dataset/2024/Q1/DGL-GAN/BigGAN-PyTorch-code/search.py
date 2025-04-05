""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from tensorboardX import SummaryWriter
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns, search_fns
from sync_batchnorm import patch_replication_callback

#CANDIDATES_NORMAL = ['conv_1x1', 'conv_3x3', 'conv_5x5']
CANDIDATES_NORMAL = ['conv_1x1', 'conv_3x3', 'conv_5x5', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7']
#CANDIDATES_UP =['deconv', 'nearest', 'bilinear']
CANDIDATES_UP =['nearest', 'bilinear']

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    # config['skip_init'] = True
    config['skip_init'] = False    # because of flexible tasks, we need to initialize the network.
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator_search(**config).to(device)
  D = model.Discriminator(**config).to(device)
  #G_bar = model.Generator_search(**config).to(device)
  G_bar = model.Generator(**config, pretrain=True).to(device)
  D_bar = model.Discriminator(**config, pretrain=True).to(device) 

  # Load D from pretrained models, and initialize the weights of embedding.
  #D.load_state_dict(torch.load(config['pretrain_path']))
  #for module in D.modules():
  #    if isinstance(module, nn.Embedding):
  #        init.orthogonal_(module.weight)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator_search(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  G_Dbar = model.G_D(G, D_bar)
  Gbar_D = model.G_D(G_bar, D)
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'weight_itr': 0, 'arch_itr': 0, 'worst_itr': 0, 'epoch': 0,
                'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights_search(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    G_Dbar = nn.DataParallel(G_Dbar)
    Gbar_D = nn.DataParallel(Gbar_D)
    if config['cross_replica']:
      patch_replication_callback(GD)
      patch_replication_callback(G_Dbar)
      patch_replication_callback(Gbar_D)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_arch_log.txt' % (config['logs_root'],
                                            experiment_name)
  print('Architecture will be saved to {}'.format(test_metrics_fname))
  arch_log = open(test_metrics_fname, 'w')
  arch_log.close()
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders_search(**{**config, 'batch_size': D_batch_size,
                                      'start_weight_itr': state_dict['weight_itr'],
                                      'start_worst_itr': state_dict['worst_itr']})
  train_loader, val_loader = iter(loaders[0]), iter(loaders[1])

  # Prepare tensorboard writer
  writer = SummaryWriter(os.path.join(config['logs_root'], experiment_name))
  global_steps, arch_global_steps = state_dict['weight_itr'], state_dict['arch_itr']
  writer_dict = {'writer': writer,
                 'global_steps': global_steps,
                 'arch_global_steps': arch_global_steps}


  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train_param, train_arch, find_worst = search_fns.GAN_searching_function(G, D, GD, G_bar, D_bar, G_Dbar, Gbar_D, z_, y_, 
                                            ema, state_dict, config, writer_dict)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)

  print('Beginning training at epoch %d...' % state_dict['epoch'])
  print(f'debug: length of data_loaders is {len(loaders[0])}')
  # Which progressbar to use? TQDM or my own?
  if config['pbar'] == 'mine':
    train_pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    val_pbar = utils.progress(loaders[1],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
  else:
    train_pbar = tqdm(loaders[0])
    val_pbar = tqdm(loaders[1])
  for i in range(config['num_epochs']):
    # Open architecture log file
    arch_log = open(test_metrics_fname, 'a')
    
    count = 0
    while count < config['inner_steps']:
      x, y = next(train_loader)
      # Increment the iteration counter
      state_dict['weight_itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train_param(x, y)
      train_log.log(itr=int(state_dict['weight_itr']), **metrics)
      count += 1
    
    # Find worst G and worst D
    """
    G_bar.train()
    D_bar.train()
    count = 0
    while count < config['worst_steps']:
      try:
        x, y = next(val_loader)
      except:
        val_loader = iter(loaders[1])
        x, y = next(val_loader)
      state_dict['worst_itr'] += 1
      metrics = find_worst(x, y)
      train_log.log(itr=int(state_dict['worst_itr']), **metrics)
      count += 1
    """
    
    # Update architecture parameters
    G.train()
    for j in range(config['outer_steps']): 
      state_dict['arch_itr'] += 1
      metrics = train_arch()
      train_log.log(itr=int(state_dict['arch_itr']), **metrics)

    # Print and record the architecture
    for k, v in G.named_parameters():
      if "alphas_up" in k:
        alphas_up = v.detach().cpu().numpy()
        up_ind = np.argmax(alphas_up, axis=1)
      elif "alphas_normal" in k:
        alphas_normal = v.detach().cpu().numpy()
        normal_ind = np.argmax(alphas_normal, axis=1)
    arch_dict = {}
    for j in range(up_ind.shape[0] // 2):
      cell = 'cell_{}'.format(j)
      arch_dict[cell] = {}
      arch_dict[cell]['up_0'] = CANDIDATES_UP[up_ind[j*2]]
      arch_dict[cell]['up_1'] = CANDIDATES_UP[up_ind[j*2+1]]
      arch_dict[cell]['normal_0'] = CANDIDATES_NORMAL[normal_ind[j*2]]
      arch_dict[cell]['normal_1'] = CANDIDATES_NORMAL[normal_ind[j*2+1]]
    print(f'search arch, arch is {arch_dict}')
    arch_log.write(f'Iteration {state_dict["arch_itr"]}, search arch, arch is {arch_dict}\n')
    arch_log.close()

    # If using my progbar, print metrics.
    if config['pbar'] == 'mine':
      print(', '.join(['itr: %d' % state_dict['weight_itr']] 
                       + ['%s : %+4.3f' % (key, metrics[key])
                       for key in metrics]), end=' ')

    # Save weights and copies as configured at specified interval
    if not (state_dict['weight_itr'] % config['save_every']):
      if config['G_eval_mode']:
        print('Switchin G to eval mode...')
        G.eval()
        if config['ema']:
          G_ema.eval()
      search_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                state_dict, config, experiment_name)



def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()
