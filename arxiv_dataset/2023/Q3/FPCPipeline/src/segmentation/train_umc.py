"""
Script to train multiple-input U-Net (mi-unet) for lesion segmentation using UMC dataset
omprising of annotated DCE and DWI MR images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'unet' ))
from lesionsegdataset.dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import shutil
from utils.utils import *
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from helper_functions import *
from random import sample, shuffle
from dice_loss import binary_dice_loss
from torch.optim.lr_scheduler import LambdaLR
from unet.model import UNet
import pickle
from torchearlystopping.pytorchtools import EarlyStopping
# Hyper-parameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import joblib

N_CLASSES = 2
VAL_IMAGES_DIR = 'val_viz'
NUM_SLICES_IN_VOLUME = 110
ALPHA = 1.0
PATCH_SIZE = 256

if PATCH_SIZE == 256:
    NUM_UNET_BLOCKS = 6
elif PATCH_SIZE == 128:
    NUM_UNET_BLOCKS = 5

STRIDE = 32
T=10
MAX_ITER = 10000
groups = [0, 1, 2, 3, 4, 5, 6, 7]

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', default='checkpoints/m_unet', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/ishaan/lesion_segmentation/data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--training_seed', type=int, default=1234)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--model',type=str, help='NN model to be used: pnet or unet', default='unet')
    parser.add_argument('--n_orientations', type=int, help='Discretization of SE(2) group', default=1)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--use_tune', action='store_true')
    parser.add_argument('--grace_period', type=int, default=100)

    args = parser.parse_args()

    return args


def train(config, args):
    """
    Main training loop

    Args:
        args: cmd-line arguments

    Returns:
        None

    """
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

    if args['model'].lower() == 'pnet':
        model = PNet()

    elif args['model'].lower() == 'unet':
        model = UNet(dropout=args['dropout'],
                     dropout_rate=dropout_rate,
                     structured_dropout=structured_dropout,
                     num_blocks=config['num_blocks'],
                     n_channels=9,
                     num_classes=2,
                     use_pooling=True)

    elif args['model'].lower() == 'gunet':
        model = GroupLesionSegNet(num_unet_blocks=4,
                                  n_classes=2,
                                  n_orientations=args.n_orientations,
                                  dropout=args.dropout,
                                  dropout_rate=args.dropout_rate)
    else:
        raise ValueError('Invalid model {} provided'.format(args['model'].lower()))


    wd = 5e-5 # For the VI math to work out, weight decay is required [Gal and Gahramani (2016)]

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
    else:
        raise RuntimeError('{} is an invalid choice for optimizer'.format(args.optim))

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
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)

        print('Checkpoint directory: {}'.format(ckpt_dir))
        print('Logs directory: {}'.format(log_dir))
        print('Training config: {}'.format(config))

#    if args.renew is True:
#        print('Removing stale directories')
#        try:
#            shutil.rmtree(log_dir)
#        except FileNotFoundError:
#            pass
#
#        try:
#            shutil.rmtree(args.checkpoint_dir)
#        except FileNotFoundError:
#            pass
#
#        os.makedirs(args.checkpoint_dir)
#        os.makedirs(log_dir)
#        n_iter = 0
#        epoch_saved = -1
#        n_iter_val = 0
#
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
#        if os.path.exists(args.checkpoint_dir) is True:
#            load_dict = load_model(model=model,
#                                   optimizer=optimizer,
#                                   scaler=scaler,
#                                   checkpoint_dir=args.checkpoint_dir,
#                                   training=True)
#            n_iter = load_dict['n_iter']
#            n_iter_val = load_dict['n_iter_val']
#            optimizer = load_dict['optimizer']
#            model = load_dict['model']
#            epoch_saved = load_dict['epoch']
#
#           # Move optimizer to the GPU
#            for state in optimizer.state.values():
#                for k, v in state.items():
#                    if torch.is_tensor(v):
#                        state[k] = v.to(device)
#
#            print('Loading model and optimizer state. Last saved epoch = {}, iter = {}'.format(epoch_saved, n_iter))
#        else:
#            print('New training loop starts')
#            os.makedirs(args.checkpoint_dir)
#            os.makedirs(log_dir)
#            epoch_saved = -1
#            n_iter = 0
#            n_iter_val = 0
#
#            train_dirs, val_dirs = get_train_and_val_dirs(args.data_dir)
#
#            model.train()
#
    # Move optimizer to the GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    model.train()

    early_stopper = EarlyStopping(patience=args['patience'],
                                  checkpoint_dir=ckpt_dir,
                                  delta=1e-5)

    print('Starting training loop')
    class_weights = torch.Tensor([1.0, 3.0])
    n_iter = 0
    epoch_saved = -1
    n_iter_val = 0

    for epoch in range(epoch_saved+1, args['epochs']):


        for group in groups:

            dataset = LesionSegMR(pat_dir=os.path.join(args['data_dir'], 'train'),
                                  mode='train',
                                  patch_size=PATCH_SIZE,
                                  stride=STRIDE,
                                  group=group)

            num_slices = dataset.get_num_valid_slices()
            num_patches = dataset.__len__()
            num_patches_per_slice = dataset.get_num_patches_per_slice()

            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args['batch_size'],
                                    shuffle=True,
                                    num_workers=4)

            print('Training group {}, slices ={}, patches = {}, patches per slice = {}'.format(group, num_slices, num_patches, num_patches_per_slice))

            for i, data in enumerate(dataloader):
                dce_patches, dwi_patches, labels = data['dce'].float(), data['dwi'].float(), data['label'].float()
                aug_dce_patches, aug_dwi_patches, aug_labels = data['aug_dce'].float(), data['aug_dwi'].float(), data['aug_label'].float()

                if args['n_orientations'] == 1:

                    # Augmented image pass
                    model.train()
                    optimizer.zero_grad()
                    model.zero_grad()

                    with torch.cuda.amp.autocast(enabled=args['fp16']):
                        aug_images = torch.cat([aug_dce_patches, aug_dwi_patches], dim=1) # Concatenate over the channel axis
                        preds = model(aug_images.to(device))
                        ce_loss = F.cross_entropy(input=preds,
                                                  target=torch.argmax(aug_labels, dim=1).to(device),
                                                  weight=class_weights.to(device))
                        dice_loss = binary_dice_loss(logits=preds, gt=aug_labels.to(device))
                        loss = ALPHA*ce_loss + (1-ALPHA)*dice_loss


                    # Backprop
                    scaler.scale(loss).backward()

#                    # Plot gradients
#                    for name, param in model.named_parameters():
#                        if param.requires_grad is True and param.grad is not None:
#                            writer.add_histogram(name, param.grad, n_iter)
#
                    scaler.step(optimizer)
                    scaler.update()

                    # Log the loss(es)
                    writer.add_scalar('train_loss/loss', loss.item(), n_iter)
                    writer.add_scalar('train_loss/ce_loss', ce_loss.item(), n_iter)
                    writer.add_scalar('train_loss/dice_loss', dice_loss.item(), n_iter)
                    n_iter += 1

                    # Calculate dice score
                    norm_preds = F.softmax(preds, dim=1)
                    train_dice = calculate_dice_score(norm_preds[:, 1, :, :].detach(), aug_labels[:, 1, :, :].detach())
                    writer.add_scalar('dice/train', np.mean(train_dice), n_iter)
                else: # THIS PART OF THE CODE IS USED ONLY FOR SE(2) GCNNS
                    # Calculate per-class weights for current batch -- weights based on voxel statistics should remain unchanged
                    # even with data augmentation => Same set of weights used in the clean and augmented pass
                    # Clear all gradient buffers
                    optimizer.zero_grad()
                    model.zero_grad()
                    model.train()
                    # Clean image pass
                    preds = model(dce_patches.to(device), dwi_patches.to(device))

                    ce_loss = F.cross_entropy(input=preds,
                                              target=torch.argmax(labels, dim=1).to(device),
                                              weight=class_weights.to(device))

                    dice_loss = binary_dice_loss(logits=preds, gt=labels.to(device))

                    # Average of CE loss and Dice Loss -- suggested as best practice in Issense et al. (2020)
                    loss = ALPHA*ce_loss + (1-ALPHA)*dice_loss

                    # Log the loss
                    writer.add_scalar('train_loss/loss', loss.item(), n_iter)
                    writer.add_scalar('train_loss/ce_loss', ce_loss.item(), n_iter)
                    writer.add_scalar('train_loss/dice_loss', dice_loss.item(), n_iter)

                    # Backprop
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    n_iter += 1

                    # Calculate dice score
                    norm_preds = F.softmax(preds, dim=1)
                    train_dice = calculate_dice_score(preds=norm_preds[:, 1, :, :].detach(),
                                                      gt=labels[:, 1, :, :].detach())
                    writer.add_scalar('dice/train', train_dice, n_iter)

        print('\n\n EPOCH {} DONE. \n\n'.format(epoch))
        if args['early_stop'] is False:
            if (epoch+1)%50 == 0:
                save_model(model=model,
                           optimizer=optimizer,
                           scheduler=None,
                           scaler=scaler,
                           n_iter=n_iter,
                           n_iter_val=n_iter_val,
                           epoch=epoch,
                           checkpoint_dir=args.checkpoint_dir,
                           suffix=epoch)

        # Evalute model after epoch done
        model.eval()
        set_mcd_eval_mode(model)

        with torch.no_grad():
            val_loss = []
            for group in groups:
                dataset = LesionSegMR(pat_dir=os.path.join(args['data_dir'], 'val'),
                                      patch_size=256,
                                      stride=1,
                                      group=group,
                                      mode='val')

                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args['batch_size']//4,
                                        shuffle=True,
                                        num_workers=4)

                num_slices = dataset.get_num_valid_slices()
                num_patches = dataset.__len__()
                num_patches_per_slice = dataset.get_num_patches_per_slice()

                print('Validating group {}, slices ={}, patches = {}, patches per slice = {}'.format(group, num_slices, num_patches, num_patches_per_slice))

                for i, data in enumerate(dataloader):
                    dce_patches, dwi_patches, labels = data['dce'].float(), data['dwi'].float(), data['label'].float()
                    slices, patch_idxs = data['slice'], data['patch']
                    class_weights = calculate_weights_per_class(torch.argmax(labels, dim=1), num_classes=2)

                    images = torch.cat([dce_patches, dwi_patches], dim=1)

                    for mcd_iter in range(T_MCD):
                        if mcd_iter == 0:
                            preds = model(images.to(device))
                        else:
                            preds += model(images.to(device))

                    preds = preds/T_MCD

                    ce_loss = F.cross_entropy(input=preds,
                                              target=torch.argmax(labels, dim=1).to(device),
                                              weight=class_weights.to(device))

                    dice_loss = binary_dice_loss(logits=preds, gt=labels.to(device))

                    loss = ALPHA*ce_loss + (1-ALPHA)*dice_loss
                    val_loss.append(loss.item())

                    n_iter_val += 1
                    norm_preds = F.softmax(preds, dim=1).cpu()

                    val_dice = calculate_dice_score(preds=norm_preds[:, 1, :, :],
                                                    gt=labels[:, 1, :, :])

                    writer.add_scalar('dice/val', np.mean(val_dice), n_iter_val)
                    writer.add_scalar('val_loss/loss', loss.item(), n_iter_val)
                    writer.add_scalar('val_loss/ce_loss', ce_loss.item(), n_iter_val)
                    writer.add_scalar('val_loss/dice_loss', dice_loss.item(), n_iter_val)

            # Validation done, now check early stopping condition if applicable
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

            if args['early_stop'] is True:
                # Check if early stopping conditions are met
                early_stop_condition, best_epoch = early_stopper(val_loss=mean_val_loss,
                                                                 curr_epoch=epoch,
                                                                 model=model,
                                                                 optimizer=optimizer,
                                                                 scheduler=None,
                                                                 scaler=scaler,
                                                                 n_iter=n_iter,
                                                                 n_iter_val=n_iter_val)

                if early_stop_condition is True:
                    print('Early stopping condition has reached with the best epoch = {}'.format(best_epoch))
                    return


    writer.close()


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
