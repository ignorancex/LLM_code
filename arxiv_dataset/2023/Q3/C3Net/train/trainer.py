import os
import sys
import torch

class Trainer:
    def __init__(self, model_path, model, loss_fn, optimizer, train_loader, validation_loader, device, keep_n_checkpoints=30, checkpoint_interval=1, validation_interval=1, loss_is_normalized=True):
        self.model_path = model_path
        self.device = device
        self.checkpoint_path = model_path
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.loss_is_normalized = loss_is_normalized
        self._model = model
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        
        self.restore_checkpoint()
        
    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    @property
    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'step': self.step,
            'optimizer': self.optimizer.state_dict(),
        }
        if self._check_is_parallel():
            state_dict['model'] = self._model.module.state_dict()
        else:
            state_dict['model'] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._load_model_state_dict(state_dict['model'])

    def store_checkpoint(self):
        chkpt = os.path.join(
            self.checkpoint_path, 'checkpoint-' + str(self.epoch) + '.pth.tar'
        )
        torch.save(self.state_dict, chkpt)

    def restore_checkpoint(self, epoch=None):
        if os.path.isfile('%s/checkpoint-1.pth.tar'%self.checkpoint_path):
            if epoch is None:
                epoch = max(
                    [
                        int(f.split('.')[0].split('-')[-1])
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith('checkpoint')
                    ]
                )
  
            chkpt = os.path.join(self.checkpoint_path, 'checkpoint-' + str(epoch) + '.pth.tar')

            if self.device == 'cuda':
                self.state_dict = torch.load(chkpt)
            else:
                self.state_dict = torch.load(chkpt, map_location=torch.device('cpu'))
        else:
            self.epoch = 0
            self.step = 0
            self.best_loss = float('inf')

    def train(self, device, n_epochs=sys.maxsize,c=0.00001):
        self._model.to(device)

        for _ in range(n_epochs):
            self.epoch += 1
            with open('%s/error_epoch%s.txt'%(self.checkpoint_path, self.epoch),'w') as error:
                # perform training epoch
                n_b = 1
                
                for train in self.train_loader:
                    self.optimizer.zero_grad()

                    # move input to gpu, if needed
                    train_batch = {}; Index = ''
                    for k, v in train.items():
                        train_batch[k] = v.to(device) 
                    
                    result = self._model(train_batch)
                    
                    error.write('Epoch : %s\tBatch: %s\n'%(self.epoch, n_b))
                    y = result['y'].tolist(); dG = train_batch['dG'].tolist()

                    for i in range(len(y)):
                        error.write('%s\t%s\t\t%s\t%s\n'%(train_batch['Solvent_ID'][i].tolist()[0], train_batch['Solute_ID'][i].tolist()[0], y[i][0], dG[i][0]))

                    loss = self.loss_fn(train_batch, result)
                    #print (train_batch['Solvent_ID'], train_batch['Solute_ID'],y,dG)
                    print ('Epoch: %d, Batch: %d, loss: %f'%(self.epoch,n_b,float(loss)))
                    n_b +=1

                    #loss.backward()
                    #self.optimizer.step()
                    self.step += 1

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()
                
                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    error.write('Validation\n')

                    val_loss = 0.0; n_val = 0
                    for val in self.validation_loader:
                        #append batch_size
                        vsize = list(val.values())[0].size(0)

                        n_val += vsize

                        # move input to gpu, if needed
                        val_batch = {}; Index = ''
                        for k, v in val.items():
                            val_batch[k] = v.to(device) 

                        val_result = self._model(val_batch)

                        val_batch_loss = (self.loss_fn(val_batch, val_result).data.cpu().numpy())
                        
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        y = val_result['y'].tolist(); dG = val_batch['dG'].tolist()
                        error.write('%s\t%s\t\t%s\t%s\n'%(val_batch['Solvent_ID'][0].tolist()[0], val_batch['Solute_ID'][0].tolist()[0], y[0][0], dG[0][0]))

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val
 
                    print ('val_loss: %s'%val_loss)
                    error.write('val_loss\t%s\n'%(val_loss))
        self.store_checkpoint()
