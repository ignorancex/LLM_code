''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, writer_dict):
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    writer = writer_dict['writer']
    global_steps = writer_dict['global_steps']
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_pair, vanilla_pair, cls_pair = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
        D_fake, D_real = D_pair[0], D_pair[1]
        vanilla_fake, vanilla_real = vanilla_pair[0], vanilla_pair[1]
        cls_fake, cls_real = cls_pair[0], cls_pair[1]
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
      
      # Update tensorboard
      summary = {'D/D_loss_real': torch.mean(D_loss_real),
               'D/D_loss_fake': torch.mean(D_loss_fake),
               'D/vanilla_real': torch.mean(vanilla_real),
               'D/vanilla_fake': torch.mean(vanilla_fake),
               'D/cls_real': torch.mean(cls_real),
               'D/cls_fake': torch.mean(cls_fake),
               'D/real_scores': torch.mean(D_real),
               'D/fake_scores': torch.mean(D_fake)}
      writer.add_scalars('D', summary, global_steps)

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, vanilla_fake, cls_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()
    summary = {'G_loss': torch.mean(G_loss),
               'vanilla_fake': torch.mean(vanilla_fake),
               'cls_fake': torch.mean(cls_fake),
               'fake_scores': torch.mean(D_fake)}
    writer.add_scalars('G', summary, global_steps)

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    global_steps += 1
    writer_dict['writer'] = writer
    writer_dict['global_steps'] = global_steps
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train

def GAN_searching_function(G, D, GD, G_bar, D_bar, G_Dbar, Gbar_D, z_, y_, ema, state_dict, config, writer_dict):
  def find_worst(x, y):
    G_bar.load_state_dict(G.state_dict())
    D_bar.load_state_dict(D.state_dict())
    G_bar.weight_optim.zero_grad()
    D_bar.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D_bar, True)
      utils.toggle_grad(G_bar, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D_bar.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_pair, vanilla_pair, cls_pair = G_Dbar(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
        D_fake, D_real = D_pair[0], D_pair[1]
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1

      # Optionally apply ortho reg ineD
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D_bar, config['D_ortho'])
      
      D_bar.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D_bar, False)
      utils.toggle_grad(G_bar, True)
      
    # Zero G's gradients by default before training G, for safety
    G_bar.weight_optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, vanilla_fake, cls_fake = Gbar_D(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G_bar, config['G_ortho'], 
                  blacklist=[param for param in G_bar.shared.parameters()])
    G_bar.weight_optim.step()
    
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return Gbar's loss and the components of Dbar's loss.
    return out  

  def train_arch():
    writer = writer_dict['writer']
    arch_global_steps = writer_dict['arch_global_steps']
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.arch_optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake_GDbar, _, _ = G_Dbar(z_, y_, train_G=True, split_D=config['split_D'])
      GDbar_loss = losses.generator_loss(D_fake_GDbar) / float(config['num_G_accumulations'])
      D_fake_GbarD, _, _ = Gbar_D(z_, y_, train_G=True, split_D=config['split_D'])
      GbarD_loss = losses.generator_loss(D_fake_GbarD) / float(config['num_G_accumulations'])
      duality_gap = GDbar_loss - GbarD_loss
      duality_gap.backward()
    summary = {'GDbar': torch.mean(GDbar_loss),
               'GbarD': torch.mean(GbarD_loss),
               'duality_gap': torch.mean(duality_gap)}
    writer.add_scalars('G', summary, arch_global_steps)

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.arch_optim.step()

    arch_global_steps += 1
    writer_dict['writer'] = writer
    writer_dict['arch_global_steps'] = arch_global_steps
    
    out = {'duality_gap': float(duality_gap.item()), 
            'GDbar': float(GDbar_loss.item()),
            'GbarD': float(GbarD_loss.item())}
    # Return duality gap and its components.
    return out

  def train_param(x, y):
    G.weight_optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    writer = writer_dict['writer']
    global_steps = writer_dict['global_steps']
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_pair, vanilla_pair, cls_pair = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
        D_fake, D_real = D_pair[0], D_pair[1]
        vanilla_fake, vanilla_real = vanilla_pair[0], vanilla_pair[1]
        cls_fake, cls_real = cls_pair[0], cls_pair[1]
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
      
      # Update tensorboard
      summary = {'D/D_loss_real': torch.mean(D_loss_real),
               'D/D_loss_fake': torch.mean(D_loss_fake),
               'D/vanilla_real': torch.mean(vanilla_real),
               'D/vanilla_fake': torch.mean(vanilla_fake),
               'D/cls_real': torch.mean(cls_real),
               'D/cls_fake': torch.mean(cls_fake),
               'D/real_scores': torch.mean(D_real),
               'D/fake_scores': torch.mean(D_fake)}
      writer.add_scalars('D', summary, global_steps)

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.weight_optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, vanilla_fake, cls_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()
    summary = {'G_loss': torch.mean(G_loss),
               'vanilla_fake': torch.mean(vanilla_fake),
               'cls_fake': torch.mean(cls_fake),
               'fake_scores': torch.mean(D_fake)}
    writer.add_scalars('G', summary, global_steps)

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.weight_optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['weight_itr'])

    global_steps += 1
    writer_dict['writer'] = writer
    writer_dict['global_steps'] = global_steps
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train_param, train_arch, find_worst

def GAN_training_compress_L1_function(G, D, G_bar, GD, z_, y_, ema, state_dict, config, writer_dict):
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    writer = writer_dict['writer']
    global_steps = writer_dict['global_steps']
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_pair, vanilla_pair, cls_pair = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
        D_fake, D_real = D_pair[0], D_pair[1]
        vanilla_fake, vanilla_real = vanilla_pair[0], vanilla_pair[1]
        cls_fake, cls_real = cls_pair[0], cls_pair[1]
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
      
      # Update tensorboard
      summary = {'D/D_loss_real': torch.mean(D_loss_real),
               'D/D_loss_fake': torch.mean(D_loss_fake),
               'D/vanilla_real': torch.mean(vanilla_real),
               'D/vanilla_fake': torch.mean(vanilla_fake),
               'D/cls_real': torch.mean(cls_real),
               'D/cls_fake': torch.mean(cls_fake),
               'D/real_scores': torch.mean(D_real),
               'D/fake_scores': torch.mean(D_fake)}
      writer.add_scalars('D', summary, global_steps)

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, vanilla_fake, cls_fake, s_feas, t_feas = GD(z_, y_, train_G=True, split_D=config['split_D'])
      #G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss, normal_loss, l1_term = eval('losses.' + config['g_loss'])(D_fake, t_feas, s_feas, lamina=0.1) 
      G_loss.backward()
    summary = {'G_loss': torch.mean(G_loss),
               'vanilla_fake': torch.mean(vanilla_fake),
               'cls_fake': torch.mean(cls_fake),
               'fake_scores': torch.mean(D_fake),
               'normal_loss': torch.mean(normal_loss),
               'l1_loss': torch.mean(l1_term)}
    writer.add_scalars('G', summary, global_steps)

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    global_steps += 1
    writer_dict['writer'] = writer
    writer_dict['global_steps'] = global_steps
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train

def GAN_training_compression_function(G, D, GD, z_, y_, ema, state_dict, config, writer_dict):
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    writer = writer_dict['writer']
    global_steps = writer_dict['global_steps']
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        #D_pair, vanilla_pair, cls_pair, D_bar_pair, vanilla_bar_pair, cls_bar_pair = GD(z_[:config['batch_size']], y_[:config['batch_size']], x[counter], y[counter], train_G=False, split_D=config['split_D'], with_Dbar=True)
        #D_fake, D_real, vanilla_fake, vanilla_real, cls_fake, cls_real = D_pair[0], D_pair[1], vanilla_pair[0], vanilla_pair[1], cls_pair[0], cls_pair[1]
        #D_fake_bar, D_real_bar, vanilla_fake_bar, vanilla_real_bar, cls_fake_bar, cls_real_bar = D_bar_pair[0], D_bar_pair[1], vanilla_bar_pair[0], vanilla_bar_pair[1], cls_bar_pair[0], cls_bar_pair[1]
        D_pair, vanilla_pair, cls_pair = GD(z_[:config['batch_size']], y_[:config['batch_size']], x[counter], y[counter], train_G=False, split_D=config['split_D'], with_Dbar=False)
        D_fake, D_real, vanilla_fake, vanilla_real, cls_fake, cls_real = D_pair[0], D_pair[1], vanilla_pair[0], vanilla_pair[1], cls_pair[0], cls_pair[1]

        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        #D_loss_real, D_loss_fake = eval('losses.' + config['d_loss'])(D_fake, D_real, D_fake_bar, D_real_bar)
        D_loss_real, D_loss_fake = eval('losses.' + config['d_loss'])(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
    

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
     
      # Update tensorboard
      '''
      summary = {'D/D_bar/p_real': torch.mean(D_real_bar),
               'D/D_bar/p_fake': torch.mean(D_fake_bar),
               'D/D_bar/vanilla_real': torch.mean(vanilla_real_bar),
               'D/D_bar/vanilla_fake': torch.mean(vanilla_fake_bar),
               'D/D_bar/cls_real': torch.mean(cls_real_bar),
               'D/D_bar/cls_fake': torch.mean(cls_fake_bar),
               'D/D_loss_real': torch.mean(D_loss_real),
               'D/D_loss_fake': torch.mean(D_loss_fake),
               'D/vanilla_real': torch.mean(vanilla_real),
               'D/vanilla_fake': torch.mean(vanilla_fake),
               'D/cls_real': torch.mean(cls_real),
               'D/cls_fake':torch.mean(cls_fake),
               'D/real_scores': torch.mean(D_real),
               'D/fake_scores': torch.mean(D_fake)}
      '''
      summary = {'D/D_loss_real': torch.mean(D_loss_real),
               'D/D_loss_fake': torch.mean(D_loss_fake),
               'D/vanilla_real': torch.mean(vanilla_real),
               'D/vanilla_fake': torch.mean(vanilla_fake),
               'D/cls_real': torch.mean(cls_real),
               'D/cls_fake':torch.mean(cls_fake),
               'D/real_scores': torch.mean(D_real),
               'D/fake_scores': torch.mean(D_fake)}
      writer.add_scalars('D', summary, global_steps)
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, vanilla_fake, cls_fake, D_fake_bar, vanilla_fake_bar, cls_fake_bar = GD(z_, y_, train_G=True, split_D=config['split_D'], with_Dbar=True)
      #D_fake, vanilla_fake, cls_fake = GD(z_, y_, train_G=True, split_D=config['split_D'], with_Dbar=False)
      G_loss, student_D, teacher_D = eval('losses.' + config['g_loss'])(D_fake, D_fake_bar) 
      #G_loss = eval('losses.' + config['g_loss'])(D_fake)
      G_loss_accu = G_loss / float(config['num_G_accumulations'])
      G_loss_accu.backward()
    
    summary = {'G_loss': torch.mean(G_loss),
               'fake_scores': torch.mean(D_fake),
               'vanilla_fake': torch.mean(vanilla_fake),
               'cls_fake': torch.mean(cls_fake),
               'fake_scores_bar': torch.mean(D_fake_bar),
               'cls_fake_bar': torch.mean(cls_fake_bar),
               'vanilla_fake_bar': torch.mean(vanilla_fake_bar),
               'student_D': torch.mean(student_D),
               'teacher_D': torch.mean(teacher_D)}
    '''
    summary = {'G_loss': torch.mean(G_loss),
               'fake_scores': torch.mean(D_fake),
               'vanilla_fake': torch.mean(vanilla_fake),
               'cls_fake': torch.mean(cls_fake)}
    '''
    writer.add_scalars('G', summary, global_steps)
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    global_steps += 1
    writer_dict['global_steps'] = global_steps
    writer_dict['writer'] = writer

    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out

  return train

''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils.save_search_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_search_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['weight_itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['weight_itr'],
                     z_=z_)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['weight_itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


  
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))
