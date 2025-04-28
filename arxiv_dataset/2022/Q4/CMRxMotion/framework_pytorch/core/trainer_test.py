import os
import time
import numpy as np
from tqdm import tqdm

import torch

from ..utils import utils as U
from ..utils import LogWriter


tqdm_setting = {'unit': ' data', 
                'ascii': ' >=', 
                'bar_format': '{l_bar}{bar:10}{r_bar}'}


class Trainer:
    def __init__(self, model):
        self.model = model
        self.cur_epoch = 0

    def train(self, 
            train_loader, 
            validation_loader, 
            epochs,   
            output_path,
            eval_frequency=1,
            save_frequency=10,
            log_train_image=False,
            log_validation_image=False,
            log_graph_input=None):

        
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # log_path = f'{output_path}/log/{timestr}'
        log_path = f'{output_path}/log'
        logwriter_train = LogWriter(f'{log_path}/train')
        logwriter_valid = LogWriter(f'{log_path}/valid')

        if log_graph_input is not None:
            nets = self.model.net if type(self.model.net) is list else [self.model.net]
            for i, net in enumerate(nets):
                logwriter_model = LogWriter(f'{log_path}/net{i}')
                logwriter_model.write_graph(net, log_graph_input)
            logwriter_model.writer.close()

        best_loss = None
        start_ep = self.cur_epoch
        for ep in range(start_ep, epochs):
            with tqdm(total=len(train_loader.dataset), desc=f'Epoch: {ep}/{epochs} Training  ', **tqdm_setting) as pbar:
                self.cur_epoch = ep
                # train and evaluation on training dataset
                ep_train_loss = []
                for batch in train_loader:
                    train_loss = self.model.train_step(batch, ep)
                    ep_train_loss.append(train_loss)
                    pbar.update(train_loader.batch_size)
                    pbar.set_postfix_str(f'\tbatch - loss: {train_loss:.4f}')
                pbar.total = pbar.n
                pbar.set_postfix_str(f'\ttotal - loss: {np.mean(ep_train_loss):.4f}')
                pbar.set_description(desc=f'Epoch: {ep}/{epochs} Training  ')
            
            # evaluation on validation dataset
            if eval_frequency < 1 or ep % eval_frequency == 0 or ep == epochs - 1:
                eval_train_results = self.eval(train_loader, desc=f'    Evaluation: training data   ', log_image=log_train_image)
                eval_valid_results = self.eval(validation_loader, desc=f'    Evaluation: evaluation data ', log_image=log_validation_image)

                if log_train_image:
                    eval_train_image = eval_train_results.pop('image')
                    logwriter_train.write_image(eval_train_image, ep)

                if log_validation_image:
                    eval_valid_image = eval_valid_results.pop('image')
                    logwriter_valid.write_image(eval_valid_image, ep)

                logwriter_train.write_scalar(U.dict_mean(eval_train_results, axis=None), ep)
                logwriter_valid.write_scalar(U.dict_mean(eval_valid_results, axis=None), ep)

                cur_valid_loss = eval_valid_results['loss'].mean()
                if not best_loss or best_loss > cur_valid_loss:
                    best_loss = cur_valid_loss
                    self.save(f'{output_path}/ckpt/model_best.pt')

            # save checkpoint
            if save_frequency < 1 or ep % save_frequency == 0:
                self.save(f'{output_path}/ckpt/model_{ep}.pt')
        self.save(f'{output_path}/ckpt/model_final.pt')

    def eval(self, data_loader, **kwargs):
        desc = kwargs.get('desc', 'Evaluation: ')
        log_image = kwargs.get('log_image', False)

        with tqdm(total=len(data_loader.dataset), desc=desc, **tqdm_setting) as pbar:
            all_results = {}
            all_imgs = {}
            for batch in data_loader:
                results = self.model.eval_step(batch, **kwargs)
                if log_image:
                    imgs = results.pop('image')
                    all_imgs = U.dict_concat(all_imgs, imgs)
                all_results = U.dict_concat(all_results, results)
                pbar.update(data_loader.batch_size)
                pbar.set_postfix_str('\tbatch - ' + U.dict_to_str(results))
            pbar.total = pbar.n
            pbar.set_postfix_str('\ttotal - ' + U.dict_to_str(all_results))

        if log_image:
            all_results.update({'image': all_imgs})
        return all_results
        
    def save(self, ckpt_path):
        cur_state = {'epoch': self.cur_epoch}
        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        [cur_state.update({f'net{i}': nets[i].state_dict()}) for i in range(len(nets))]

        optimizers = self.model.optimizer if type(self.model.optimizer) is list else [self.model.optimizer]
        [cur_state.update({f'optimizer{i}': optimizers[i].state_dict()}) for i in range(len(optimizers))]

        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(cur_state, ckpt_path)

    def restore(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.cur_epoch = ckpt['epoch']

        if self.model.optimizer is not None:
            optimizers = self.model.optimizer if type(self.model.optimizer) is list else [self.model.optimizer]
            [optimizers[i].load_state_dict(ckpt[f'optimizer{i}']) for i in range(len(optimizers))]

        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        [nets[i].load_state_dict(ckpt[f'net{i}']) for i in range(len(nets))]

    def test(self, data_loader, output_path, log_image=True, log_text=True, logsuffix=''):
        test_results = self.eval(data_loader, desc=f'Evaluation: test data ', log_image=log_image)

        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # logwriter_test = LogWriter(f'{output_path}/log/{timestr}/test{logsuffix}')
        logwriter_test = LogWriter(f'{output_path}/log/test{logsuffix}')

        if log_image:
            test_image = test_results['image']
            logwriter_test.write_image(test_image, self.cur_epoch)
        if log_text:
            logwriter_test.write_text(U.dict_to_str(test_results), self.cur_epoch)


        return test_results

        

