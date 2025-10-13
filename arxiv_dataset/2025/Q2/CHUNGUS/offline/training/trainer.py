import tqdm
import torch
import logging
import numpy as np


class Trainer:
    def __init__(self,
                 network,
                 train_data,
                 valid_data,
                 loss_fn,
                 optimizer,
                 scheduler,
                 evaluator,
                 target_metric,
                 best_save_path,
                 last_save_path,
                 device,
                 use_loader=False):
        """ Trainer object

        :param network: network to train
        :param train_data: training data
        :param valid_data: validation data
        :param loss_fn: loss function
        :param optimizer: optimizer to use
        :param scheduler: scheduler for optimizer
        :param evaluator: evaluator to use
        :param target_metric: target evaluation metric for early stopping
        :param best_save_path: save path for best model
        :param last_save_path: save path for last model
        :param device: device to use
        :param use_loader: whether to use loader (better to leave false, unless you can't fit all data in VRAM)
        """
        
        # Note: use_loader=True is slower, because it doesn't do batch GD (it uses mini-batch GD).
        super().__init__()
        self._network = network
        self._train_data = train_data
        self._valid_data = valid_data
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._evaluator = evaluator
        self._target_metric = target_metric
        self._device = device
        self._best_model_path = best_save_path
        self._last_model_path = last_save_path
        self._use_loader = use_loader
        
    def train_step(self, data):
        """ Perform a single training step and return the detached loss item """
        embeddings = data['embeddings'].to(self._device)
        labels = data['labels'].to(self._device)

        predictionsA = self._network(embeddings[:,0].unsqueeze(-1).unsqueeze(-1))
        predictionsA = {k: v[:,:,0,0] for k,v in predictionsA.items()}
        predictionsB = self._network(embeddings[:,1].unsqueeze(-1).unsqueeze(-1))
        predictionsB = {k: v[:,:,0,0] for k,v in predictionsB.items()}
        step_loss = self._loss_fn(predictionsA, predictionsB, labels)

        self._optimizer.zero_grad()
        step_loss.backward()
        self._optimizer.step()

        return step_loss.detach().cpu().item()
        
    def train_epoch(self):
        """ Train for an epoch """
        self._network.to(self._device)
        self._network.train()
        total_loss = []
        if self._use_loader:
            for data in self._train_data:
                total_loss.append(self.train_step(data))
        else:
            total_loss = [self.train_step(self._train_data.get_all())]
        return np.array(total_loss).mean()
    
    def train(self, epochs, patience=-1):
        """ Perform a full training
        
        :param epochs: number of epochs to train for
        :param patience: patience to use for training
        :returns: training results
        """
        best_metric = None
        best_metric_epoch = None
        patience_tries = 0
        
        training_results = {
            'epoch': [],
            'train_loss': [],
            'validation': []
        }
        
        for epoch in tqdm.tqdm(range(1, epochs + 1)):
            train_loss = self.train_epoch()
            training_results['epoch'].append(epoch)
            training_results['train_loss'].append(train_loss)
            if self._evaluator:
                validation_results = self._evaluator.evaluate(self._network, self._valid_data,
                                                              device=self._device, use_loader=self._use_loader)
                training_results['validation'].append(validation_results)
                if best_metric is None or best_metric > validation_results[self._target_metric]:
                    best_metric = validation_results[self._target_metric]
                    best_metric_epoch = epoch
                    logging.info("New best metric = {:.3f} at epoch {}".format(best_metric, best_metric_epoch))
                    torch.save(self._network.state_dict(), self._best_model_path)
                    patience_tries = 0
                else:
                    patience_tries += 1
                    if patience > 0 and patience_tries >= patience:
                        logging.info("Patience exceeded")
                        break
            
            if self._scheduler is not None:
                self._scheduler.step()
                
        if best_metric is not None:
            logging.info("Training finished... Best epoch = {} with metric {} = {:.3f}".format(best_metric_epoch, self._target_metric, best_metric))
        torch.save(self._network.state_dict(), self._last_model_path)
        
        return training_results
