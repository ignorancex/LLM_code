import time, os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from losses.loss import loss_mossformer2_ss

class Solver(object):
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.device = self.args.device

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            self.writer = SummaryWriter('%s/tensorboard/' % args.checkpoint_dir)

        self.model = model
        self.optimizer=optimizer
        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.args.local_rank],find_unused_parameters=True)
        self._init()
 
        if self.args.network in ['MossFormer2_SS_16K','MossFormer2_SS_8K'] :
            self._run_one_epoch = self._run_one_epoch_mossformer2_ss
        else:
            print(f'_run_one_epoch is not implemented for {self.args.network}!')


    def _init(self):
        self.halving = False
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.val_no_impv = 0

        if self.args.train_from_last_checkpoint:
            flag = self._load_model()
            if flag != 2:
                ckpt_name = os.path.join(self.args.checkpoint_dir, 'last_checkpoint')
                if not os.path.isfile(ckpt_name):
                    print('[!] Last checkpoints are not found. Start new training ...')
                else:
                    with open(ckpt_name, 'r') as f:
                        model_name = f.readline().strip()
                    checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                    if flag == 0:
                        self._load_pretrained_model(checkpoint_path, load_training_stat=True)
                    if flag ==1:
                        self._load_pretrained_model(checkpoint_path)
     
        elif self.args.init_checkpoint_path != 'None':
            self._load_pretrained_model(self.args.init_checkpoint_path)
        else:
            if self.print: print('Start new training')

        self.model.to(self.device)

    def _load_model(self, mode='last_checkpoint', use_cuda=True, strict=True):
        ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
        if not os.path.isfile(ckpt_name):
            mode = 'last_best_checkpoint'
            ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
            if not os.path.isfile(ckpt_name):
                print('[!] Last checkpoints are not found. Start new training ...')
                self.epoch = 0
                self.step = 0
            return 1
        else:
            if self.print: print(f'Loading checkpoint: {ckpt_name}')
            with open(ckpt_name, 'r') as f:
                model_name = f.readline().strip()
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
            #checkpoint = self.load_checkpoint(checkpoint_path, use_cuda)
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)
            if 'model' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model'], strict=strict)
                except Exception as e:
                    print(f"The model is not sucessfully loaded, will try _load_pretrained_model")
                    return 0
            else:
                print('Checkpoint {ckpt_name} has no model key, will try _load_pretrained_model')
                return 1
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"self.optimizer is not sucessfully loaded: {e}")

            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            print('=> Reloaded previous model and optimizer. Continue training ...')
            return 2

    def load_checkpoint(self, checkpoint_path, use_cuda):
        if use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
        
    def _load_pretrained_model(self, checkpoint_path, load_optimizer=False, load_training_stat=False):
        if os.path.isfile(checkpoint_path):
            if self.print: print(f'Loading checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # load model weights
            if 'model' in checkpoint:
                pretrained_model = checkpoint['model']
            else:
                pretrained_model = checkpoint
            state = self.model.state_dict()
            for key in state.keys():
                if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                    state[key] = pretrained_model[key]
                elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                     state[key] = pretrained_model[key.replace('module.', '')]
                elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                     state[key] = pretrained_model['module.'+key]
                elif self.print: print(f'{key} not loaded')
            self.model.load_state_dict(state)
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.finetune_learning_rate 
            
            # load optimizer only
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.args.learning_rate
            # load the training states
            elif load_training_stat:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epoch=checkpoint['epoch']
                self.step = checkpoint['step']
                if self.print: print("Resume training from epoch: {}".format(self.epoch))

            else:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.args.finetune_learning_rate
                print(f'==> Done model init from {self.args.init_checkpoint_path}. Start finetune training ...')
        else:
            print(f'{checkpoint_path} is not found. Start new training ...')
            self.epoch = 0
            self.step = 0
        
    def save_checkpoint(self, mode='last_checkpoint'):
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(self.epoch, self.step))
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    'step': self.step}, checkpoint_path)

        with open(os.path.join(self.args.checkpoint_dir, mode), 'w') as f:
            f.write('model.ckpt-{}-{}.pt'.format(self.epoch, self.step))
        print("=> Save checkpoint:", checkpoint_path)

    def train(self):
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.args.max_epoch+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            start = time.time()            
            tr_loss = self._run_one_epoch(data_loader = self.train_data)
            if self.args.distributed: tr_loss = self._reduce_tensor(tr_loss)
            if self.print: print(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Train Loss {tr_loss:2.4f}')

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                if self.args.distributed: val_loss = self._reduce_tensor(val_loss)
            if self.print: print(f'Valid Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Valid Loss {val_loss:2.4f}')
          
            if self.args.tt_list is not None:
                # Test
                self.model.eval()
                start = time.time()
                with torch.no_grad():
                    test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                    if self.args.distributed: test_loss = self._reduce_tensor(test_loss)
                if self.print: print(f'Test Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Test Loss {test_loss:2.4f}')

            # Check whether to early stop and to reduce learning rate
            find_best_model = False
            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv == 5:
                    self.halving = True
                elif self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = val_loss
                find_best_model=True

            # Halfing the learning rate
            if self.halving:
                self.halving = False
                self._load_model(mode='last_best_checkpoint')
                if self.print: print('reload from last best checkpoint')

                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] *= 0.5
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                

            if self.print:
                # Tensorboard logging
                self.writer.add_scalar('Train_loss', tr_loss, epoch)
                self.writer.add_scalar('Validation_loss', val_loss, epoch)
                if self.args.tt_list is not None:
                    self.writer.add_scalar('Test_loss', test_loss, epoch)

            # Save model
            self.save_checkpoint()
            if find_best_model:
                self.save_checkpoint(mode='last_best_checkpoint')
                print("Found new best model, dict saved")
            self.epoch = self.epoch + 1

    def _run_one_epoch_mossformer2_ss(self, data_loader, state='train'):
        num_batch = len(data_loader)
        mix_loss_print = 0.0

        total_loss = 0.0
        self.accu_count = 0
        self.optimizer.zero_grad()
        stime = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            Out_List = self.model(inputs)
            loss = loss_mossformer2_ss(self.args, inputs, labels, Out_List, self.device)

            if state=='train':
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_bw = loss[loss > self.args.loss_threshold]
                    if loss_bw.nelement() > 0:
                        loss_bw = loss_bw.mean()
                        if loss_bw < 999999:
                            loss_scaled = loss_bw/(self.args.effec_batch_size / self.args.batch_size)
                            loss_scaled.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
          
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0

                else:
                    loss_bw = loss[loss > self.args.loss_threshold]
                    if loss_bw.nelement() > 0:
                        loss_bw = loss_bw.mean()
                        if loss_bw < 999999:
                            loss_bw.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                self.step += 1
                ##cal losses for printing
                mix_loss_print += loss_bw.data.cpu()
                if (i + 1) % self.args.print_freq == 0:
                    eplashed = time.time() - stime
                    speed_avg = eplashed / (i+1)
                    mix_loss_print_avg = mix_loss_print / self.args.print_freq
                    if self.print: print('Train Epoch: {}/{} Step: {}/{} | {:2.3f}s/batch | lr {:1.4e} |'
                      ' Total_Loss {:2.4f}'
                      .format(self.epoch, self.args.max_epoch,
                          i+1, num_batch, speed_avg, self.optimizer.param_groups[0]["lr"],
                          mix_loss_print_avg,
                        ))
                    mix_loss_print = 0.0
                if (i + 1) % self.args.checkpoint_save_freq == 0:
                    self.save_checkpoint()
            else:
                loss_bw = loss
                loss_bw = loss_bw.mean()
            total_loss += loss_bw.detach() #data.cpu() #clone().detach().cpu()
        return total_loss / (i+1)
    
    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
