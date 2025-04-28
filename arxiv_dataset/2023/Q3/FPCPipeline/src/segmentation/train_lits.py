"""

Training LiTS dataset of abdominal CT scans to detect lesions

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'unet' ))
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import shutil
from utils.utils import *
from unet.model import UNet
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from helper_functions import *
from random import sample, shuffle
from dice_loss import binary_dice_loss
from lits_dataset.dataset import LITSDataset
import pickle
import pandas as pd
from torchearlystopping.pytorchtools import EarlyStopping
# Hyper-parameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import joblib

NUM_PATIENTS_IN_GROUP = 5
NUM_PATIENTS_IN_DATASET = 131
ALPHA = 1.0

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', default='checkpoints/m_unet', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/ishaan/lesion_segmentation/data')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--training_seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--renew', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--use_tune', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--grace_period', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=1)
    args = parser.parse_args()
    return args

def train(config, args):

    if torch.cuda.is_available():
        device='cuda:0'
    else:
        raise RuntimeError('GPU not available!')

    cpu_device = torch.device('cpu')
    torch.manual_seed(args['training_seed'])
    np.random.seed(args['training_seed'])

    print('Enable dropout -- {}'.format(args['dropout']))
    print('Enable fp16 -- {}'.format(args['fp16']))
    print('Early stopping -- {}'.format(args['early_stop']))

    if args['dropout'] is True:
        T_MCD = 20
        dropout_rate = config['dropout_rate']
        structured_dropout = config['structured_dropout']
    else:
        T_MCD = 1
        dropout_rate = 0.0
        structured_dropout=False

    # Get the patients without tumor that we would like to skip
    with open(os.path.join(args['data_dir'], 'patients_without_tumor.pkl'), 'rb') as f:
        skip_patients = pickle.load(f)

    # Get the dataset fingerprints that will be used for pre-processing (windowing + z-score normalization)
    fingerprint_df = pd.read_csv(os.path.join(args['data_dir'], 'dataset_fingerprints.csv'))


    # Define the model
    model = UNet(n_channels=1,
                 num_classes=2,
                 base_filter_num=64,
                 num_blocks=config['num_blocks'],
                 dropout=args['dropout'],
                 dropout_rate=dropout_rate,
                 structured_dropout=structured_dropout,
                 use_pooling=True)

    # Define the optimizer and scaler (for fp16 computations)
    wd = 5e-5
    if args['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args['lr'],
                               weight_decay=wd)
    elif args['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args['lr'],
                              weight_decay=wd,
                              momentum=0.9,
                              nesterov=True)


    scaler = torch.cuda.amp.GradScaler(enabled=args['fp16'])

    if args['use_tune'] is True:
        trial_dir = tune.get_trial_dir()
        log_dir = os.path.join(trial_dir, 'logs')
        os.makedirs(log_dir)
        ckpt_dir = os.path.join(trial_dir, 'checkpoints')
        os.makedirs(ckpt_dir)
        print('Logs directory created at {}'.format(os.path.join(trial_dir, 'logs')))
    else:
        checkpoint_dir = os.path.join(args['checkpoint_dir'], 'best_trial')
        log_dir = os.path.join(checkpoint_dir, 'logs')
        ckpt_dir = os.path.join(checkpoint_dir, 'checkpoints')
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        if os.path.exists(ckpt_dir) is False:
            os.makedirs(ckpt_dir)

        print('Checkpoint directory: {}'.format(ckpt_dir))
        print('Logs directory: {}'.format(log_dir))
        print('Training config: {}'.format(config))


    train_patients, val_patients, test_patients = create_split_for_lits(seed=args['data_seed'],
                                                                        num_patients=NUM_PATIENTS_IN_DATASET,
                                                                        skip_patients=skip_patients)


    # Save data-split
    with open(os.path.join(ckpt_dir, 'train_patients.pkl'), 'wb') as f:
        pickle.dump(train_patients, f)
    with open(os.path.join(ckpt_dir, 'val_patients.pkl'), 'wb') as f:
        pickle.dump(val_patients, f)
    with open(os.path.join(ckpt_dir, 'test_patients.pkl'), 'wb') as f:
        pickle.dump(test_patients, f)

    model.train()

    # Move optimizer to the GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


#    if args.renew is True:
#        print('Removing stale directories')
#        try:
#            shutil.rmtree(log_dir)
#        except FileNotFoundError:
#            pass
#
#        try:
#            shutil.rmtree(args['checkpoint_dir'])
#        except FileNotFoundError:
#            pass
#
#        os.makedirs(args['checkpoint_dir'])
#        os.makedirs(log_dir)
#        n_iter = 0
#        epoch_saved = -1
#        n_iter_val = 0
#        train_patients, val_patients, test_patients = create_split_for_lits(seed=args['data_seed'],
#                                                                            num_patients=NUM_PATIENTS_IN_DATASET,
#                                                                            skip_patients=skip_patients)
#
#
#        # Save data-split
#        with open(os.path.join(args['checkpoint_dir'], 'train_patients.pkl'), 'wb') as f:
#            pickle.dump(train_patients, f)
#        with open(os.path.join(args['checkpoint_dir'], 'val_patients.pkl'), 'wb') as f:
#            pickle.dump(val_patients, f)
#        with open(os.path.join(args['checkpoint_dir'], 'test_patients.pkl'), 'wb') as f:
#            pickle.dump(test_patients, f)
#
#        model.train()
#
#        # Move optimizer to the GPU
#        for state in optimizer.state.values():
#            for k, v in state.items():
#                if torch.is_tensor(v):
#                    state[k] = v.to(device)
#
#    else:
#        if os.path.exists(args['checkpoint_dir']) is True:
#            load_dict = load_model(model=model,
#                                   optimizer=optimizer,
#                                   scaler=scaler,
#                                   checkpoint_dir=args['checkpoint_dir'],
#                                   training=True)
#
#            n_iter = load_dict['n_iter']
#            n_iter_val = load_dict['n_iter_val']
#            optimizer = load_dict['optimizer']
#            model = load_dict['model']
#            epoch_saved = load_dict['epoch']
#
#            # Get train and val patients
#            with open(os.path.join(args['checkpoint_dir'], 'train_patients.pkl'), 'rb') as f:
#                train_patients = pickle.load(f)
#            with open(os.path.join(args['checkpoint_dir'], 'val_patients.pkl'), 'rb') as f:
#                val_patients = pickle.load(f)
#
#            # Move optimizer to the GPU
#            for state in optimizer.state.values():
#                for k, v in state.items():
#                    if torch.is_tensor(v):
#                        state[k] = v.to(device)
#
#            print('Loading model and optimizer state. Last saved epoch = {}, iter = {}'.format(epoch_saved, n_iter))
#        else:
#            print('New training loop starts')
#            os.makedirs(args['checkpoint_dir'])
#            os.makedirs(log_dir)
#            epoch_saved = -1
#            n_iter = 0
#            n_iter_val = 0
#            model.train()
#            train_patients, val_patients, test_patients = create_split_for_lits(seed=args.seed,
#                                                                                num_patients=131,
#                                                                                skip_patients=skip_patients)
#
#            # Get train and val patients
#            with open(os.path.join(args['checkpoint_dir'], 'train_patients.pkl'), 'rb') as f:
#                train_patients = pickle.load(f)
#            with open(os.path.join(args['checkpoint_dir'], 'val_patients.pkl'), 'rb') as f:
#                val_patients = pickle.load(f)
#
#            # Move optimizer to the GPU
#            for state in optimizer.state.values():
#                for k, v in state.items():
#                    if torch.is_tensor(v):
#                        state[k] = v.to(device)

    num_training_patients = len(train_patients)
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    model.train()

    early_stopper = EarlyStopping(patience=args['patience'],
                                  checkpoint_dir=ckpt_dir,
                                  delta=1e-5)

    # Static class-weights for the CE Loss
    class_weights = torch.Tensor([1.0, 3.0])

    epoch_saved = -1
    n_iter = 0
    n_iter_val = 0

    for epoch in range(epoch_saved+1, args['epochs']):
        start = 0
        end = start + NUM_PATIENTS_IN_GROUP

        # Switch the order the patients appear in every epoch
        shuffle(train_patients)

        while(start != end): # Cycle through all patients in the training set in groups of 5 (so as not to crash the script due to excessive RAM usage)

            curr_training_patients = train_patients[start:end]

            datasets_to_concat = []
            # Concatenate datasets in the group
            for pat_id in curr_training_patients:
                datasets_to_concat.append(LITSDataset(data_dir=args['data_dir'],
                                                      patient_id=pat_id,
                                                      fingerprint_df=fingerprint_df,
                                                      mode='train'))

            merged_datasets = ConcatDataset(datasets_to_concat)
            print('Training set start = {}, end = {}, slices = {}'.format(start, end, merged_datasets.__len__()))

            # Define the dataloader
            dataloader = DataLoader(dataset=merged_datasets,
                                    batch_size=args['batch_size'],
                                    num_workers=4,
                                    shuffle=True)

            for idx, data in enumerate(dataloader):
                images, lesion_labels, liver_mask = data['image'], data['lesion_labels'].float(), data['liver_mask'].float()
                images = images.unsqueeze(dim=1) # Add a fake channel axis

                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                with torch.cuda.amp.autocast(enabled=args['fp16']):

                    preds = model(images.to(device))

                    ce_loss = F.cross_entropy(input=preds,
                                              target=torch.argmax(lesion_labels, dim=1).to(device),
                                              weight=class_weights.to(device))

                    dice_loss = binary_dice_loss(logits=preds, gt=lesion_labels.to(device))
                    loss = ALPHA*ce_loss + (1-ALPHA)*dice_loss

                # Backprop
                scaler.scale(loss).backward()
#                # Plot gradient histogram for each parameter
#                for name, param in model.named_parameters():
#                    if param.requires_grad is True:
#                        writer.add_histogram(name, param.grad, n_iter)

                scaler.step(optimizer)
                scaler.update()
                # Log the loss(es)
                writer.add_scalar('train/loss', loss.item(), n_iter)
                norm_preds = F.softmax(preds, dim=1).detach()
                train_dice = calculate_dice_score(norm_preds[:, 1, :, :], lesion_labels[:, 1, :, :])
                writer.add_scalar('train/dice', np.mean(train_dice), n_iter)
                n_iter += 1

            # We're done with one patient group, update the start and end pointers
            start = end
            end = min(end+NUM_PATIENTS_IN_GROUP, num_training_patients)

        # Save the model at the end of each epoch
        print('EPOCH {} ends'.format(epoch))

        if args['early_stop'] is False:
            if (epoch + 1)%10 == 0:
                save_model(model=model,
                           optimizer=optimizer,
                           scheduler=None,
                           scaler=None,
                           n_iter=n_iter,
                           n_iter_val=n_iter_val,
                           epoch=epoch,
                           checkpoint_dir=args['checkpoint_dir'],
                           suffix=epoch)

       # Test the model on the validation patients
        print('Testing model on validation patients')

        with torch.no_grad():
            datasets_to_concat = []
            model.eval()
            set_mcd_eval_mode(model)

            # Concatenate datasets in the group
            for pat_id in val_patients:
                datasets_to_concat.append(LITSDataset(data_dir=args['data_dir'],
                                                      patient_id=pat_id,
                                                      fingerprint_df=fingerprint_df,
                                                      mode='val'))

            merged_datasets = ConcatDataset(datasets_to_concat)

            # Define the dataloader
            dataloader = DataLoader(dataset=merged_datasets,
                                    batch_size=args['batch_size']//4,
                                    num_workers=4,
                                    shuffle=False)

            val_loss = []
            for idx, data in enumerate(dataloader):
                images, lesion_labels, liver_mask = data['image'], data['lesion_labels'].float(), data['liver_mask'].float()
                images = images.unsqueeze(dim=1) # Add a fake channel axis

                preds = model(images.to(device))

                ce_loss = F.cross_entropy(input=preds,
                                          target=torch.argmax(lesion_labels, dim=1).to(device),
                                          weight=class_weights.to(device))

                dice_loss = binary_dice_loss(logits=preds, gt=lesion_labels.to(device))
                loss = ALPHA*ce_loss + (1-ALPHA)*dice_loss
                val_loss.append(loss.item())

                # Log the loss(es)
                writer.add_scalar('val/loss', loss.item(), n_iter_val)

                norm_preds = F.softmax(preds, dim=1)
                val_dice = calculate_dice_score(norm_preds[:, 1, :, :], lesion_labels[:, 1, :, :])
                writer.add_scalar('val/dice', np.mean(val_dice), n_iter_val)
                n_iter_val += 1

            mean_val_loss = np.mean(np.array(val_loss))
            writer.add_scalar('val/mean_loss', mean_val_loss, epoch)

            if args['use_tune'] is True:
                # FIXED: Reporting the validation loss to tune ONLY if it improves
                # renders the scheduler useless because it will never terminate
                # "bad" trials. So we report the mean validation loss for every epoch
                # but save the checkpoint only when it improves. When comparing trials
                # we use the scope='all' so the best trial is not chosen based on the
                # results from last epoch but across all epochs!
                tune.report(mean_val_loss=mean_val_loss)

            # Check if early stopping conditions are met
            early_stop, best_epoch = early_stopper(val_loss=mean_val_loss,
                                                   curr_epoch=epoch,
                                                   model=model,
                                                   optimizer=optimizer,
                                                   scheduler=None,
                                                   scaler=scaler,
                                                   n_iter=n_iter,
                                                   n_iter_val=n_iter_val)
            if args['early_stop'] is True:
                if early_stop is True:
                    print('Early stopping condition has reached with the best epoch = {}'.format(best_epoch))
                    with open(os.path.join(args['checkpoint_dir'], 'best_epoch.txt'), 'w') as f:
                        f.write('The best epoch found was {}'.format(best_epoch))
                        f.close()
                    return


if __name__ == '__main__':

    args = build_parser()

    # Configure GPU visibility (at the top!)
    assert(len(args.gpus) <= 4)

    gpu_str = ''

    for idx, gpu in enumerate(args.gpus):
        gpu_str += '{}'.format(gpu)
        if idx != len(args.gpus)-1:
            gpu_str += ','

    print('GPU string: {}'.format(gpu_str))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    if args.use_tune is True:
        assert(torch.cuda.device_count() == len(args.gpus))

        if os.path.exists(args.checkpoint_dir) is True:
            print('Removing stale checkpoint directory')
            shutil.rmtree(args.checkpoint_dir)

        scheduler = ASHAScheduler(max_t=10000,
                                  grace_period=args.grace_period,
                                  reduction_factor=2)

        config = {'num_blocks' : tune.grid_search([3, 4])}

        if args.dropout is True:
            config['dropout_rate'] = tune.grid_search([0.1, 0.3, 0.5])
            config['structured_dropout'] = tune.grid_search([True, False])


        result = tune.run(tune.with_parameters(train,
                                               args=vars(args)),
                          resources_per_trial={'gpu':1},
                          config=config,
                          metric='mean_val_loss',
                          mode='min',
                          scheduler=scheduler,
                          num_samples=args.num_trials,
                          local_dir=args.checkpoint_dir,
                          keep_checkpoints_num=1,
                          raise_on_failed_trial=False)

        # We set scope='all' so that the trial with overall min. val loss is chosen
        # This is in line with how we checkpoint the model
        best_config = result.get_best_config(metric='mean_val_loss',
                                             mode='min',
                                             scope='all')

        best_logdir =  result.get_best_logdir(metric='mean_val_loss',
                                              mode='min',
                                              scope='all')


        joblib.dump(best_config, os.path.join(args.checkpoint_dir, 'best_config.pkl'))

        # Clean-up : delete all sub-optimal trials so that we don't run out of disk space
        clean_up_ray_trials(exp_dir=args.checkpoint_dir,
                            best_trial_dir=best_logdir,
                            new_dir=os.path.join(args.checkpoint_dir, 'best_trial'))


    else:
        config = joblib.load(os.path.join(args.checkpoint_dir, 'best_config.pkl'))
        train(config=config,
              args=vars(args))
