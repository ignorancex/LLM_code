"""
Script used to:
  1) Train single models to TEST performance
  2) Do hyperparameter tuning (TUNE) for the models
  3) TRAIN the models to be attacked
  
It should be called from the parent folder.
"""

import os
import sys
import getopt
import copy
import shutil
import json
from importlib import util
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchaudio

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

torch.set_num_threads(1)

cwd = os.getcwd()
p_english_list = [0.005, 0.01, 0.05,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.95, 0.99, 0.995]
out_rate = 1000


def import_file(full_name, path):
    """Returns a python module given its path"""
    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#################
# Load Datasets #
#################
# MARK: Load Datasets

class CustomCommonVoice(Dataset):
    """
    Class for the (imbalanced) datasets created.

    Parameters
    ----------
    p_english : float (p_english_list)
        Proportion of audios of people with english accent in the dataset.
    idx : int [0, 9]
        Index of the annotations to be used. For each ``p_english`` there are 10
        datasets.
    set : {"train", "val", "test"}
        Indicates which dataset is to be loaded.
    transform : torchvision.transforms
        Transformations of the dataset (data augmentation, normalization, etc.)
    target_transform : func
        Transformation of the target attribute (not used).
    """
    
    def __init__(self,
                 p_english,
                 idx,
                 set="train",
                 transform=None):
        
        global p_english_list
        if (p_english not in p_english_list) or ((idx < 0) or (idx > 9)):
            raise ValueError(
                f'`p_english` can only take values within {p_english_list}, '
                f'and `idx` should be between 0 and 9')
        
        if set not in ["train", "val", "test"]:
            raise ValueError('`set` should be one of "train", "val" or "test"')
        
        global cwd
        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=os.path.join(cwd, 'CommonVoice'),
            tsv=os.path.join('datasets', str(p_english), str(idx), f'{set}_df.tsv'))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y, z = self.dataset[index]
        if self.transform:
            x = self.transform((x, y))
        return x, int(z['sex'])
    

def resample(x):
    global out_rate
    x, in_rate = x
    resample_trans = torchaudio.transforms.Resample(in_rate, out_rate)
    return resample_trans(x)

def crop(x):
    global out_rate
    llimit = (x.size(1) // 2 - out_rate // 2)
    rlimit = (x.size(1) // 2 + out_rate // 2)
    x = x[:, llimit:rlimit].flatten()
    if x.size(0) < out_rate:
        return None
    return x

def rfft(x):
    if x is None:
        return None
    return torch.fft.rfft(x)[:-1].abs()

def normalize(x):
    x = x / 200
    x = torch.where(x <= 0, 1e-5, x)
    x = torch.where(x >= 1, 1 - 1e-5, x)
    return x

transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resample),
    torchvision.transforms.Lambda(crop),
    torchvision.transforms.Lambda(rfft),
    torchvision.transforms.Lambda(normalize)
    ])

def none_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_data(p_english, idx, batch_size):
    """Loads dataset performing the required transformations for train or test."""
    
    # Load datasets
    global transform
    train_dataset = CustomCommonVoice(p_english,
                                      idx,
                                      set="train",
                                      transform=transform)
    val_dataset = CustomCommonVoice(p_english,
                                    idx,
                                    set="val",
                                    transform=transform)
    test_dataset = CustomCommonVoice(p_english,
                                     idx,
                                     set="test",
                                     transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=none_collate,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            collate_fn=none_collate,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=none_collate,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader


###############################################################################
###############################################################################

####################
# Train/Val epoch #
####################
# MARK: Train/Val epoch

def training_epoch(device, model, criterion, optimizer, scaler,
                   train_loader, batch_size, logs, verbose=False):
    running_loss = 0
    running_acc = 0
    print_each = len(train_loader) // 10
    
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Mixed precission training
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            scores = model(data)
            loss = criterion(scores, labels)
            
            _, preds = torch.max(scores, 1)
            
            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / batch_size
                running_loss += loss.item()
                running_acc += accuracy
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
        
        if verbose and ((batch_idx + 1) % print_each == 0):
            print(f'\tBatch: {batch_idx + 1}/{len(train_loader)}, '
                  f'Last Train Loss: {loss.item():.3f}, '
                  f'Last Train Acc: {accuracy:.3f}')
    
    logs['train_losses'].append(running_loss / len(train_loader))
    logs['train_accs'].append(running_acc / len(train_loader))
    
    return model, optimizer, logs


def validate(device, model, criterion, val_loader, batch_size, logs):
    """Computes loss and accuracy on validation set."""
    running_loss = 0
    running_acc = 0
    
    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            scores = model(data)
            _, preds = torch.max(scores, 1)
            loss = criterion(scores, labels)
            
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss.item()
            running_acc += accuracy
            
    logs['val_losses'].append(running_loss / len(val_loader))
    logs['val_accs'].append(running_acc / len(val_loader))
    
    return logs


###############################################################################
###############################################################################

#############
# Test Mode #
#############
# MARK: Test Mode

# MAIN for "--test" mode
def main_test(model_class, config, n_epochs, p_english, idx):
    """Does training based on test config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config is None:
        # Check tuned config of balanced model
        results_dir = os.path.join(cwd, 'results', '0_train_nns',
                                   model_class.name, str(p_english))
        tuning_done = os.path.exists(
            os.path.join(results_dir, 'tuned_config.json'))
        
        if not tuning_done:
            raise ValueError(
                'Hyperparameters should be optimized before starting'
                ' training (use --tune)')
        
        with open(os.path.join(results_dir, 'tuned_config.json'), 'r') as f:
            config = json.load(f)
    
    model = model_class(config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')

    train_loader, val_loader, test_loader = load_data(p_english,
                                                      idx,
                                                      config['batch_size'])
    
    early_stop_counter = n_epochs #10
    logs = {'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []}
    counter = 0
    best_val_acc = -1.
    
    print('@ TRAINING MODEL @')

    for epoch in range(n_epochs):
        model, optimizer, logs = training_epoch(device, model, criterion,
                                                optimizer, scaler, train_loader,
                                                config['batch_size'], logs,
                                                verbose=True)
        
        logs = validate(device, model, criterion, val_loader,
                        config['batch_size'], logs)

        print(f'*({p_english}, {idx}) => '
              f'Epoch: {epoch + 1}/{n_epochs}, '
              f'Train Loss: {logs["train_losses"][-1]:.3f}, '
              f'Val Loss: {logs["val_losses"][-1]:.3f}, '
              f'Train Acc: {logs["train_accs"][-1]:.3f}, '
              f'Val Acc: {logs["val_accs"][-1]:.3f}')

        # Keep track of best model
        if logs['val_accs'][-1] > best_val_acc:
            best_val_acc = logs['val_accs'][-1]
            counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            print(f'Val. acc. has not improved since: {best_val_acc:.3f}, '
                  f'Count: {counter}')
            if counter >= early_stop_counter:
                print('#############  Early Stopping Now  #############')
                break
    
    model.load_state_dict(best_model_state_dict)
    logs = validate(device, model, criterion, test_loader,
                    config['batch_size'], logs)
    test_acc = logs['val_accs'][-1]
            
    # Save best model
    torch.save(best_model_state_dict,
               os.path.join(
                   cwd,
                   'results',
                   '0_train_nns',
                   f'test_{model_class.name}_{p_english}_{idx}_{test_acc:.4f}.pt'))
    

###############################################################################
###############################################################################

#############
# Tune Mode #
#############
# MARK: Tune Mode

def full_training_tune(config, model_class, n_epochs, p_english, idx):
    """Does full training given a configuration for tuning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, _ = load_data(p_english, idx, config['batch_size'])
    
    logs = {'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []}
    
    model = model_class(config)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
        
    for _ in range(n_epochs):
        model, optimizer, logs = training_epoch(device, model, criterion,
                                                optimizer, scaler, train_loader,
                                                config['batch_size'], logs)
            
        logs = validate(device, model, criterion, val_loader,
                        config['batch_size'], logs)
        
        last_logs = {'train_loss': logs['train_losses'][-1],
                     'val_loss': logs['val_losses'][-1],
                     'train_acc': logs['train_accs'][-1],
                     'val_acc': logs['val_accs'][-1]}
        session.report(last_logs)
    

# MAIN for "--tune" mode
def main_tune(model_class, config, n_epochs, n_samples, p_english,
              override, restore):
    """Does hyperpatameter tuning given a tuner configuration."""
    global p_english_list
    idx = 0
    
    if p_english is not None:
        p_english_list_aux = [p_english]
    else:
        p_english_list_aux = p_english_list
        
    for p_english in p_english_list_aux:
        results_dir = os.path.join(cwd, 'results', '0_train_nns',
                                   model_class.name, str(p_english))
        tuning_done = os.path.exists(os.path.join(results_dir,
                                                  'tuned_config.json'))
        
        if (tuning_done and override) or not tuning_done:
            tuner = None
            if restore:
                try:
                    tuner = tune.Tuner.restore(
                        path=os.path.join(results_dir, 'tuner_results'),
                        trainable=tune.with_resources(
                            tune.with_parameters(
                                partial(full_training_tune,
                                        model_class=model_class,
                                        n_epochs=n_epochs,
                                        p_english=p_english,
                                        idx=idx)
                                ),
                            resources={'cpu': 1, 'gpu': torch.cuda.device_count()}
                            ))
                except:
                    pass
                
            if tuner is None:
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir, ignore_errors=True)
                os.makedirs(os.path.join(results_dir, 'tuner_checkpoint'),
                            exist_ok=True)
                
                tune_config = tune.TuneConfig(metric='val_acc',
                                              mode='max',
                                              scheduler=ASHAScheduler(
                                                  max_t=n_epochs,
                                                  grace_period=min(5, n_epochs)),
                                              num_samples=n_samples,
                                              max_concurrent_trials=5)
                
                run_config = air.RunConfig(name='tuner_results',
                                           storage_path=os.path.join(results_dir))
                    
                tuner = tune.Tuner(
                    trainable=tune.with_resources(
                        tune.with_parameters(
                            partial(full_training_tune,
                                    model_class=model_class,
                                    n_epochs=n_epochs,
                                    p_english=p_english,
                                    idx=idx)
                            ),
                        resources={'cpu': 1, 'gpu': torch.cuda.device_count()}
                        ),
                    tune_config=tune_config,
                    run_config=run_config,
                    param_space=config)
            results = tuner.fit()
            best_result = results.get_best_result()
            with open(os.path.join(results_dir, 'tuned_config.json'), 'w+') as f:
                json.dump(best_result.config, f, indent=4)
            with open(os.path.join(results_dir, 'tuned_metrics.json'), 'w+') as f:
                json.dump(best_result.metrics, f, indent=4)
                

###############################################################################
###############################################################################

#########################
# Standard Mode (Train) #
#########################
# MARK: Standard Mode (Train)

# MAIN for standar mode (train all the models)
def main_train(model_class, n_epochs, n_samples, p_english, start_idx, end_idx,
               expected_accuracy, gpu):
    """Trains ``n_samples`` models  with the same dataset and tuned configuration."""
    global p_english_list
    
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    if p_english is not None:
        p_english_list_aux = [p_english]
    else:
        p_english_list_aux = p_english_list
        
    if start_idx is not None:
        idx_range_aux = range(start_idx, end_idx)
    else:
        idx_range_aux = range(10)
        
    for p_english in p_english_list_aux:
        for idx in idx_range_aux:
            
            current_dir = os.path.join(cwd, 'results', '0_train_nns',
                                       model_class.name, str(p_english), str(idx))
            os.makedirs(current_dir, exist_ok=True)
            
            files = os.listdir(current_dir)
            n_models = len(files)
            
            checkpoint = None
            if os.path.exists(os.path.join(current_dir, 'checkpoint.pt')):
                n_models -= 1
                checkpoint = torch.load(os.path.join(current_dir,
                                                     'checkpoint.pt'),
                                        weights_only=False)
            
            for sample in range(n_models, n_samples):
                good_enough_model = False
                while not good_enough_model:
                    
                    # NOTE: we will tune only balanced models and reuse
                    # those hyperparameters for all p_english
                    
                    # Check tuned config of balanced model
                    results_dir = os.path.join(cwd, 'results', '0_train_nns',
                                               model_class.name, '0.5')
                    tuning_done = os.path.exists(
                        os.path.join(results_dir, 'tuned_config.json'))
        
                    if not tuning_done:
                        raise ValueError(
                            'Hyperparameters should be optimized before starting'
                            ' training (use --tune)')
                        
                    with open(os.path.join(results_dir,
                                           'tuned_config.json'), 'r') as f:
                        config = json.load(f)
                            
                    # Initialize model with balanced config
                    model = model_class(config)
                    model.to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=config['lr'],
                        weight_decay=config['weight_decay'])
                    
                    scaler = torch.amp.GradScaler('cuda')
                    
                    train_loader, val_loader, test_loader = load_data(
                        p_english, idx, config['batch_size'])
                    
                    early_stop_counter = n_epochs #10
                    
                    if checkpoint is None:
                        init_epoch = 0
                        logs = {'train_losses': [],
                                'val_losses': [],
                                'train_accs': [],
                                'val_accs': []}
                        counter = 0
                        best_val_acc = -1.
                        
                    else:
                        init_epoch = checkpoint['epoch'] + 1
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        
                        best_val_acc = checkpoint['best_val_acc']
                        best_model_state_dict = checkpoint['best_model_state_dict']
                        
                        logs = checkpoint['logs']
                        counter = checkpoint['counter']
                        good_enough_model = checkpoint['good_enough_model']

                    for epoch in range(init_epoch, n_epochs):
                        
                        model, optimizer, logs = training_epoch(
                            device, model, criterion, optimizer, scaler,
                            train_loader, config['batch_size'], logs)
                    
                        logs = validate(device, model, criterion, val_loader,
                                        config['batch_size'], logs)

                        print(f'**{model_class.name}** (p: {p_english}, i: {idx},'
                              f' s: {sample}) => '
                              f'Epoch: {epoch + 1}/{n_epochs}, '
                              f'Train Loss: {logs["train_losses"][-1]:.3f}, '
                              f'Val Loss: {logs["val_losses"][-1]:.3f}, '
                              f'Train Acc: {logs["train_accs"][-1]:.3f}, '
                              f'Val Acc: {logs["val_accs"][-1]:.3f}')

                        # Keep track of best model
                        if logs['val_accs'][-1] > best_val_acc:
                            best_val_acc = logs['val_accs'][-1]
                            # Valid training only if we get enough good accuracy
                            if best_val_acc >= expected_accuracy:
                                good_enough_model = True
                            counter = 0
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                        else:
                            counter += 1
                            print(f'Val. acc. has not improved since: '
                                  f'{best_val_acc:.3f}, Count: {counter}')
                            if counter >= early_stop_counter:
                                print('#############  Early Stopping Now  #############')
                                break
                            
                        # Save checkpoint after each epoch
                        torch.save({'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'best_val_acc': best_val_acc,
                                    'best_model_state_dict': best_model_state_dict,
                                    'good_enough_model': good_enough_model,
                                    'logs': logs,
                                    'counter': counter},
                                os.path.join(current_dir, 'checkpoint.pt'))
                    
                    model.load_state_dict(best_model_state_dict)
                    logs = validate(device, model, criterion, test_loader,
                                    config['batch_size'], logs)
                    best_model_test_acc = logs['val_accs'][-1]
                            
                # Save best model after training
                torch.save(best_model_state_dict,
                           os.path.join(current_dir,
                                        f'{sample}_{best_model_test_acc:.4f}.pt'))
                os.remove(os.path.join(current_dir, 'checkpoint.pt'))


###############################################################################
###############################################################################

###############
# System args #
###############
# MARK: System args

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--test\n'
              '\t--tune\n'
              '\t--override\n'
              '\t--restore\n'
              '\t--gpu\n'
              '\t<model name>\n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h',
                                   ['help', 'test', 'tune',
                                    'override', 'restore', 'gpu='])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--test\n'
              '\t--tune\n'
              '\t--override\n'
              '\t--restore\n'
              '\t--gpu\n')
        sys.exit(2)
        
    # Save selected options
    options = {'test': False,
               'tune': False,
               'override': False,
               'restore': False,
               'gpu': 0}
        
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--test\n'
                  '\t--tune\n'
                  '\t--override\n'
                  '\t--restore\n'
                  '\t--gpu\n'
                  '\t<model name>\n')
            sys.exit()
        elif opt == '--test':
            options['test'] = True
        elif opt == '--tune':
            options['tune'] = True
        elif opt == '--override':
            options['override'] = True
        elif opt == '--restore':
            options['restore'] = True
        elif opt == '--gpu':
            options['gpu'] = int(arg)
        
    # Check if selected options are compatible
    if options['test'] and options['tune']:
        print('Options "test" and "tune" are incompatible')
        sys.exit()
        
    if options['override'] and not options['tune']:
        print('Option "override" can only be used if "tune" is also used')
        sys.exit()
        
    if options['restore'] and not options['tune']:
        print('Option "restore" can only be used if "tune" is also used')
        sys.exit()
        
    if options['override'] and options['restore']:
        print('Options "override" and "restore" are incompatible')
        sys.exit()
        
        
    # TEST
    if options['test']:
        model_name = None
        n_epochs = None
        p_english = 0.5
        idx = 0
        if len(args) == 2:
            model_name = args[0]
            n_epochs = int(args[1])
        elif len(args) == 3:
            model_name = args[0]
            n_epochs = int(args[1])
            if args[2].isdigit():
                idx = int(args[2])
            else:
                p_english = float(args[2])
        elif len(args) == 4:
            model_name = args[0]
            n_epochs = int(args[1])
            p_english = float(args[2])
            idx = int(args[3])
        else:
            print('In "test" mode the following arguments can be passed:\n'
                  '\t1) model name\n'
                  '\t2) number of epochs\n'
                  f'\t3) (optional) proportion of imbalance (p in {p_english_list})\n'
                  '\t4) (optional) index of dataset (0 <= i <= 9)')
            sys.exit()
          
        aux_mod = import_file('model', os.path.join(cwd, 'models',
                                                    f'{model_name}.py'))
        model_class = aux_mod.Model
        config = None
        main_test(model_class=model_class, config=config,
                  n_epochs=n_epochs, p_english=p_english, idx=idx)
            
    # TUNE
    elif options['tune']:
        model_name = None
        n_epochs = None
        n_samples = None
        p_english = None
        override = options['override']
        restore = options['restore']
        if len(args) == 3:
            model_name = args[0]
            n_epochs = int(args[1])
            n_samples = int(args[2])
        elif len(args) == 4:
            model_name = args[0]
            n_epochs = int(args[1])
            n_samples = int(args[2])
            p_english = float(args[3])
        else:
            print('In "tune" mode the following arguments can be passed:\n'
                  '\t1) model name\n'
                  '\t2) number of epochs\n'
                  '\t3) number of samples\n'
                  f'\t4) (optional) proportion of imbalance (p in {p_english_list})\n')
            sys.exit()
            
        aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
        model_class = aux_mod.Model
        config = aux_mod.tuner_config
        main_tune(model_class=model_class, config=config,
                  n_epochs=n_epochs, n_samples=n_samples, p_english=p_english,
                  override=override, restore=restore)
        
    # STANDARD (TRAIN)
    else:
        model_name = None
        n_epochs = None
        n_samples = None
        p_english = None
        start_idx = None
        end_idx = None
        gpu = options['gpu']
        if len(args) == 3:
            model_name = args[0]
            n_epochs = int(args[1])
            n_samples = int(args[2])
        elif len(args) == 4:
            model_name = args[0]
            n_epochs = int(args[1])
            n_samples = int(args[2])
            if args[3].isdigit():
                start_idx = int(args[3])
            else:
                p_english = float(args[3])
        elif len(args) == 6:
            model_name = args[0]
            n_epochs = int(args[1])
            n_samples = int(args[2])
            p_english = float(args[3])
            start_idx = int(args[4])
            end_idx = int(args[5])
        else:
            print('In standard mode the following arguments can be passed:\n'
                  '\t1) model name\n'
                  '\t2) number of epochs\n'
                  '\t3) number of samples\n'
                  f'\t4) (optional) proportion of imbalance (p in {p_english_list})\n'
                  '\t5) (optional) first index of dataset (0 <= i <= 9)\t'
                  '\t5) (optional) last index of dataset (0 <= i <= 9)')
            sys.exit()
            
        aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
        model_class = aux_mod.Model
        expected_accuracy = aux_mod.expected_accuracy
        main_train(model_class=model_class,
                   n_epochs=n_epochs,
                   n_samples=n_samples,
                   p_english=p_english,
                   start_idx=start_idx,
                   end_idx=end_idx,
                   expected_accuracy=expected_accuracy,
                   gpu=gpu)
