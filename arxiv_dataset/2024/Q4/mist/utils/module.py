import torch
import pytorch_lightning as pl
import torchmetrics
        
        
class BaseModule(pl.LightningModule):
    """
    Base class for PyTorch Lightning modules.

    Args:
        model: The neural network model to be trained.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_scheduler_name (str): Name of the learning rate scheduler to use.
        lr_scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        checkpoint_filename (str): Format string for checkpoint filenames.
    """
    def __init__(
        self,
        model,
        learning_rate: float = 8e-3,
        lr_scheduler_name: str = 'ReduceLROnPlateau',
        lr_scheduler_params: dict = None,
        early_stopping_patience: int = 7,
        checkpoint_filename: str = '{epoch}-{val_loss:.2f}',
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lr_scheduler_name = lr_scheduler_name
        
        # Default parameters for each scheduler
        default_scheduler_params = {
            'ReduceLROnPlateau': {'factor': 0.5, 'patience': 5, 'verbose': True},
            'OneCycleLR': {'max_lr': learning_rate, 'div_factor': 25.0, 'final_div_factor': 1e4},
            'CyclicLR': {'base_lr': learning_rate / 10 if learning_rate is not None else 0.001, 'max_lr': learning_rate, 'mode': 'triangular'},
            'ExponentialLR': {'gamma': 0.9},
            'CosineAnnealingLR': {'T_max': 50, 'eta_min': 0},
        }

        # Merge defaults with user-provided parameters
        self.lr_scheduler_params = {
            **default_scheduler_params.get(lr_scheduler_name, {}),
            **(lr_scheduler_params or {})
        }
        
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_filename = checkpoint_filename

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate) 
        

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict or optimizer: Configuration for the optimizer and scheduler.
        """
        
        optimizer = self.optimizer()
        # Define scheduler based on the name
        scheduler = None
        if self.lr_scheduler_name == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **self.lr_scheduler_params
                ),
                'monitor': 'val_loss',
            }
        elif self.lr_scheduler_name == 'OneCycleLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr_scheduler_params.get('max_lr', self.learning_rate),
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=len(self.train_dataloader()),
                    **self.lr_scheduler_params
                ),
                'interval': 'step',
            }
        elif self.lr_scheduler_name == 'CyclicLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.lr_scheduler_params.get('base_lr', self.learning_rate / 10),
                    max_lr=self.lr_scheduler_params.get('max_lr', self.learning_rate),
                    **self.lr_scheduler_params
                ),
                'interval': 'step',
            }
        elif self.lr_scheduler_name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                **self.lr_scheduler_params
            )
        elif self.lr_scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **self.lr_scheduler_params
            )

        if scheduler:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer

    def configure_callbacks(self):
        """
        Configures callbacks for early stopping, model checkpointing, and learning rate monitoring.

        Returns:
            list: A list of callbacks.
        """
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            verbose=True,
            mode='min',
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename=self.checkpoint_filename,
            save_top_k=1,
            mode='min',
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        return [early_stop_callback, checkpoint_callback, lr_monitor]

    def on_fit_end(self):
        """
        Called at the end of training to reload the best model checkpoint.
        """
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        if self.best_model_path:
            checkpoint = torch.load(self.best_model_path)
            print("Reloading best model:", self.best_model_path)
            self.load_state_dict(checkpoint["state_dict"])
            
            
class BCELossModule(BaseModule):
    """
    LightningModule for training models using Binary Cross-Entropy Loss.

    Args:
        model: The neural network model to be trained.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_scheduler_name (str): Name of the learning rate scheduler to use.
        lr_scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        checkpoint_filename (str): Format string for checkpoint filenames.
    """
    def __init__(
        self,
        model,
        learning_rate: float = 8e-3,
        lr_scheduler_name: str = 'ReduceLROnPlateau',
        lr_scheduler_params: dict = None,
        early_stopping_patience: int = 7,
        checkpoint_filename: str = '{epoch}-{val_loss:.2f}',
    ):
        super().__init__(
            model,
            learning_rate, 
            lr_scheduler_name, 
            lr_scheduler_params,
            early_stopping_patience,
            checkpoint_filename
        )

    def shared_step(self, batch, batch_idx, prefix):
        input = self.model(batch)
        target = (batch['ni'] != 0).float()
        num_pos = target.sum(dim=-1, keepdim=True) + 1e-6  # Avoid division by zero
        num_neg = target.shape[-1] - num_pos
        pos_weight = num_neg / num_pos
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=input, target=target, reduction="mean", pos_weight=pos_weight
        )
        accuracy = torchmetrics.functional.accuracy(
            preds=input, target=target.long(), task='binary'
        )
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"{prefix}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, prefix='test')
            
            
class CustomLossModule(BaseModule):
    """
    LightningModule for training models using a custom loss function defined in the model.

    Args:
        model: The neural network model to be trained, which computes its own loss.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_scheduler_name (str): Name of the learning rate scheduler to use.
        lr_scheduler_params (dict): Parameters for the learning rate scheduler.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        checkpoint_filename (str): Format string for checkpoint filenames.
    """
    def __init__(
        self,
        model,
        learning_rate: float = None,
        lr_scheduler_name: str = 'ReduceLROnPlateau',
        lr_scheduler_params: dict = None,
        early_stopping_patience: int = 7,
        checkpoint_filename: str = '{epoch}-{val_loss:.2f}',
    ):
        super().__init__(
            model,
            learning_rate, 
            lr_scheduler_name, 
            lr_scheduler_params,
            early_stopping_patience,
            checkpoint_filename
        )

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("test_loss", loss, logger=True)
        return loss
    
    def optimizer(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.logvariance, 'lr': 3e-3 if self.learning_rate is None else self.learning_rate},
            {'params': self.model.net.parameters(), 'lr': 5e-3 if self.learning_rate is None else self.learning_rate},
        ])
        return optimizer
    
    
class CustomLossModule_withBounds(CustomLossModule):
    def __init__(
        self,
        model,
        learning_rate: float = None,
        lr_scheduler_name: str = 'ReduceLROnPlateau',
        lr_scheduler_params: dict = None,
        early_stopping_patience: int = 7,
        checkpoint_filename: str = '{epoch}-{val_loss:.2f}',
    ):
        super().__init__(
            model,
            learning_rate, 
            lr_scheduler_name, 
            lr_scheduler_params,
            early_stopping_patience,
            checkpoint_filename
        )
        
        self.train_loss_history = []
        self.bounds_history = []

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        bounds = self.model.bounds()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if isinstance(bounds, float) or bounds.numel() == 1:
            self.log(f"bounds", bounds, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            for i_b in range(len(bounds)):
                self.log(f"bounds{i_b}", bounds[i_b], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outputs):
        bounds = self.model.bounds()
        # Append to history
        self.train_loss_history.append(
            self.trainer.callback_metrics['train_loss']
        )
        if isinstance(bounds, float) or bounds.numel() == 1:
            self.bounds_history.append(self.trainer.callback_metrics[f'bounds'])
        else:
            self.bounds_history.append(
                torch.stack([self.trainer.callback_metrics[f'bounds{i_b}'] for i_b in range(len(bounds))], dim=0)
            )
        