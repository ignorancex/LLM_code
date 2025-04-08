import json
import os
import torch
import torch.optim as optim
import shutil
import time
from tqdm import tqdm
import lib.utils as utils
import visdom
import numpy as np 
import torch.optim as optim
from functools import partial
from datetime import datetime
from utils.eval import score_dataset
from utils.schedulers.delayed_sched import *
from utils.schedulers.cosine_annealing_with_warmup import *

def init_model_params(args, dataset):
    return {
        'pose_shape': dataset["test"][0][0].shape if args.model_confidence else dataset["test"][0][0][:2].shape,
        'hidden_dim': args.model_latent_dim,
        'actnorm_scale': 1.0,
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'device': args.device,
        'model_dist': 'normal'
    }


def dump_args(args, ckpt_dir):
    path = os.path.join(ckpt_dir, "args.json")
    data = vars(args)
    with open(path, 'w') as fp:
        json.dump(data, fp)


def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def get_fn_suffix(args):
    fn_suffix = args.dataset + args.conv_oper
    return fn_suffix




                                                                
class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  # For checkpoint filename
        # Loss, Optimizer and Scheduler
        
        
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr,)
        
    def adjust_lr(self, epoch, lr=None):
        if self.scheduler is not None:
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
        elif (lr is not None) and (self.args.lr_decay is not None):
            new_lr = lr * (self.args.lr_decay ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            raise ValueError('Missing parameters for LR adjustment')
        return new_lr
    
    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args
        if not os.path.exists(self.args.save_dir):
        # Create the directory
            os.makedirs(self.args.save_dir)
        
        current_time = datetime.now()
        # path_join = os.path.join(self.args.ckpt_dir, filename)
        # path_join = os.path.join(self.args.save_dir, filename + '_' +  current_time.strftime("%Y-%m-%d_%H-%M-%S")+".pth.tar")
        path_join = os.path.join(self.args.save_dir, filename + '_' + str(epoch) + ".pth.tar")
        torch.save(state, path_join)
        if is_best:
            # shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))
            shutil.copy(path_join, os.path.join(self.args.save_dir, 'checkpoint_best.pth.tar'))
    
    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))
    
    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state
    
    
    def anneal_kl(self, iteration):
       
        warmup_iter = 300

        if self.args.lambda_anneal:
            self.model.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
        else:
            self.model.lamb = 0
        if self.args.beta_anneal:
            self.model.beta = min(self.args.beta, self.args.beta / warmup_iter * iteration)  # 0 --> 1
        else:
            self.model.beta = self.args.beta
            

    def plot_elbo(train_elbo, vis):
        global win_train_elbo
        win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)
            
    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        best_loss = 0
        train_elbo = []
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'
        if num_epochs is None:  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = args.epochs
        else:
            start_epoch = 0
            
        self.model = self.model.to(args.device)
        dataset_size = len(self.train_loader.dataset)
        elbo_running_mean = utils.RunningAverageMeter()
        if args.visdom:
            vis = visdom.Visdom(env=args.save, port=4500)
        it = 0 
        for epoch in range(start_epoch, num_epochs):
            print("Started epoch {}".format(epoch))
            self.model.train()
            loss = []
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                it = it + 1
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:,0:2, :, :]
                self.anneal_kl(it)
                self.optimizer.zero_grad()
                obj, elbo = self.model.elbo(data, dataset_size)
                
                if utils.isnan(obj).any():
                    raise ValueError('NaN spotted in objective.')
                obj.mean().mul(-1).backward()
                elbo_running_mean.update(elbo.mean())
                self.optimizer.step()
                loss.append(elbo_running_mean.avg)
                
            print('[Epoch %03d] \tbeta %.2f \tlambda %.2f training ELBO: %.4f ' % (
                epoch, self.model.beta, self.model.lamb,
                torch.stack(loss).mean()))
            new_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self.adjust_lr(epoch, new_lr)
            print('lr: {0:.3e}'.format(new_lr))
            train_elbo.append(torch.stack(loss).mean)
            
            if torch.stack(loss).mean()> best_loss:
                best_loss = torch.stack(loss).mean()
                self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
                print("Model saved!")
                eval_loss = []
                dataset_size = len(self.test_loader.dataset)
                self.model.eval()
                with torch.no_grad():
                    for i, data_batch in enumerate(tqdm(self.test_loader)):
                        data = data_batch[0].to(args.device, non_blocking=True)
                        data = data[:,0:2, :, :]
                        obj, elbo = self.model.elbo(data, dataset_size)
                        eval_loss.extend(elbo.cpu().numpy())
                auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(args.mask_root, np.array(eval_loss), self.test_loader.dataset.metadata, save_results=False, seg_len=args.seg_len)
                print('AUC ROC: {}'.format(auc_roc))
                print('AUC PR: {}'.format(auc_pr))
                print('EER: {}'.format(eer))
                print('EER TH: {}'.format(eer_th))
            
        if args.visdom:
            self.plot_elbo(train_elbo, vis)
        
        return checkpoint_filename


def init_optimizer(type_str, **kwargs):
    if type_str.lower() == 'adam':
        opt_f = optim.Adam
    else:
        return None

    return partial(opt_f, **kwargs)


def init_scheduler(type_str, lr, epochs, warmup=3):
    sched_f = None
    if type_str.lower() == 'exp_decay':
        sched_f = None
    elif type_str.lower() == 'cosine':
        sched_f = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=epochs)
    elif type_str.lower() == 'cosine_warmup':
        sched_f = partial(CosineAnnealingWarmUpRestarts, T_0=epochs, T_up=warmup)
    elif type_str.lower() == 'cosine_delayed':
        sched_f = partial(DelayedCosineAnnealingLR, delay_epochs=warmup,
                          cosine_annealing_epochs=epochs)
    elif (type_str.lower() == 'tri') and (epochs >= 8):
        sched_f = partial(optim.lr_scheduler.CyclicLR,
                          base_lr=lr, max_lr=lr,
                          step_size_up=epochs//8,
                          mode='triangular2',
                          cycle_momentum=False)
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return sched_f

