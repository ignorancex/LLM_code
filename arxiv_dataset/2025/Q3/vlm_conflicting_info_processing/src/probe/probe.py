import torch
import lightning as L
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.loggers import WandbLogger
import wandb
import os 
import json

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim, is_image_probe=True):
        super(LinearProbe, self).__init__()

        self.is_image_probe = is_image_probe
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc1(x.to(torch.float32)) # make sure that the float format are the same

class PL_LinearProbe(L.LightningModule):
    def __init__(self, model, self_args):
        super().__init__()
        self.model = model
        self.is_image_probe = self.model.is_image_probe
        
        self.args = self_args

        self.save_hyperparameters(ignore=["model"])

        self.loss_fn = nn.CrossEntropyLoss()

        self.test_outputs = []
    

    def forward(self, x):
        return self.model(x)
    
    def wandb_define_metrics(self):
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("validation/loss", summary="min")
        wandb.define_metric("validation/accuracy", summary="max")
        wandb.define_metric("test/correct", summary="max")
        wandb.define_metric("test/misled", summary="min")
        wandb.define_metric("test/incorrect", summary="min")

    def training_step(self, batch, batch_idx):

        im_labels, cap_labels, data = batch

        labels = im_labels if self.is_image_probe else cap_labels

        outputs = self.model(data)
        loss = self._get_loss(outputs, labels)
        self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.wandb_define_metrics()

        im_labels, cap_labels, data = batch

        labels = im_labels if self.is_image_probe else cap_labels

        outputs = self.model(data)
        loss = self._get_loss(outputs, labels)

        self.log("validation/loss", loss, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        prediction = self._get_preds(outputs)

        accuracy = self._get_accuracy(prediction, labels)

        self.log("validation/accuracy", accuracy, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):

        im_labels, cap_labels, data = batch

        outputs = self.model(data)
        prediction = self._get_preds(outputs)

        confusion_matrix = self._get_confusion_matrix(prediction, im_labels, cap_labels)

        self.test_outputs.append(confusion_matrix)

        return confusion_matrix

    def on_test_epoch_end(self):

        accumulated_confusion_matrix = {k:0 for k in self.test_outputs[0]}

        for confusion_matrix in self.test_outputs:
            for k in confusion_matrix:
                accumulated_confusion_matrix[k] += confusion_matrix[k]
        
        self.log("test/correct", accumulated_confusion_matrix["n_correct"], on_epoch=True, sync_dist=True, logger=True)
        self.log("test/misled", accumulated_confusion_matrix["n_misled"], on_epoch=True, sync_dist=True, logger=True)
        self.log("test/incorrect", accumulated_confusion_matrix["n_incorrect"], on_epoch=True, sync_dist=True, logger=True)

        directory = os.path.dirname(self.args.result_save_path)
    
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the JSON data to the specified path
        with open(f"{self.args.result_save_path}.json", 'w') as json_file:
            json.dump(accumulated_confusion_matrix, json_file, indent=4)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _get_preds(self, logits):
        return torch.argmax(logits, dim=-1)

    def _get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)
    
    def _get_accuracy(self, preds, labels):
        correct = torch.sum(preds == labels).item()
        return correct / len(labels)
    
    def _get_confusion_matrix(self, preds, image_labels, caption_labels):

        n_correct, n_misled, n_incorrect = 0,0,0

        if self.is_image_probe:
            correct_labels = image_labels
            misled_labels  = caption_labels
        else:
            correct_labels = caption_labels
            misled_labels  = image_labels

        for pred, true_label, misled_label in zip(preds, correct_labels, misled_labels):
            if pred == true_label:
                n_correct += 1
            elif pred == misled_label:
                n_misled += 1
            else:
                n_incorrect += 1
        
        return {
            "n_correct"   : n_correct,
            "n_misled"    : n_misled,
            "n_incorrect" : n_incorrect
        }




class LinearProbe_consistency(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe_consistency, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc1(x.to(torch.float32)) # make sure that the float format are the same

class PL_LinearProbe_consistency(L.LightningModule):
    def __init__(self, model, self_args, test_total):
        super().__init__()
        self.model = model

        self.args = self_args

        self.test_total = test_total

        self.save_hyperparameters(ignore=["model"])

        self.loss_fn = nn.CrossEntropyLoss()

        self.test_outputs = []
    

    def forward(self, x):
        return self.model(x)
    
    def wandb_define_metrics(self):
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("validation/loss", summary="min")
        wandb.define_metric("validation/accuracy", summary="max")
        wandb.define_metric("test/correct", summary="max")
        wandb.define_metric("test/incorrect", summary="min")

    def training_step(self, batch, batch_idx):

        consistency_labels, data = batch

        outputs = self.model(data)
        loss = self._get_loss(outputs, consistency_labels)
        self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.wandb_define_metrics()

        consistency_labels, data = batch

        outputs = self.model(data)
        loss = self._get_loss(outputs, consistency_labels)

        self.log("validation/loss", loss, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        prediction = self._get_preds(outputs)

        accuracy = self._get_accuracy(prediction, consistency_labels)

        self.log("validation/accuracy", accuracy, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):

        consistency_labels, data = batch

        outputs = self.model(data)
        prediction = self._get_preds(outputs)

        n_correct = self._get_ncorrect(prediction, consistency_labels)

        self.test_outputs.append(n_correct)

        return n_correct

    def on_test_epoch_end(self):

        n_correct = 0

        for n_correct_batch in self.test_outputs:
            n_correct += n_correct_batch
        
        self.log("test/n_correct", n_correct, on_epoch=True, sync_dist=True, logger=True)

        directory = os.path.dirname(self.args.result_save_path)
    
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the JSON data to the specified path
        with open(f"{self.args.result_save_path}.json", 'w') as json_file:
            json.dump({
                "n_correct": n_correct,
                "n_total": self.test_total
            }, json_file, indent=4)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _get_preds(self, logits):
        return torch.argmax(logits, dim=-1)

    def _get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)
    
    def _get_accuracy(self, preds, labels):
        correct = torch.sum(preds == labels).item()
        return correct / len(labels)
    
    def _get_ncorrect(self, preds, labels):
        correct = torch.sum(preds == labels).item()
        return correct





test_data_names = ["in_distribution", "semi_in_distribution", "out_of_distribution"]

class PL_LinearProbe_consistency_gen(L.LightningModule):
    def __init__(self, model, self_args, test_total):
        super().__init__()
        self.model = model

        self.args = self_args

        self.save_hyperparameters(ignore=["model"])

        self.loss_fn = nn.CrossEntropyLoss()

        self.test_outputs = [[], [], []]

        self.test_total   = test_total
    

    def forward(self, x):
        return self.model(x)
    
    def wandb_define_metrics(self):
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("validation/loss", summary="min")
        wandb.define_metric("validation/accuracy", summary="max")


    def training_step(self, batch, batch_idx):

        consistency_labels, data = batch

        outputs = self.model(data)
        loss = self._get_loss(outputs, consistency_labels)
        self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.wandb_define_metrics()

        consistency_labels, data = batch

        outputs = self.model(data)
        loss = self._get_loss(outputs, consistency_labels)

        self.log("validation/loss", loss, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        prediction = self._get_preds(outputs)

        accuracy = self._get_accuracy(prediction, consistency_labels)

        self.log("validation/accuracy", accuracy, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        consistency_labels, data = batch

        outputs = self.model(data)
        prediction = self._get_preds(outputs)

        n_correct = self._get_ncorrect(prediction, consistency_labels)

        self.test_outputs[dataloader_idx].append(n_correct)

        return n_correct

    def on_test_epoch_end(self):

        all_results = []

        for dataloader_idx in range(3):

            n_correct = 0

            for n_correct_batch in self.test_outputs[dataloader_idx]:
                n_correct += n_correct_batch
            
            self.log(f"test/n_correct[{test_data_names[dataloader_idx]}]", n_correct, on_epoch=True, sync_dist=True, logger=True)

            all_results.append(n_correct)

        directory = os.path.dirname(self.args.result_save_path)
    
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        dict_to_report = {}

        for idx, name in enumerate(test_data_names):
            dict_to_report.update({f"{name}_ncorrect": all_results[idx]})
            dict_to_report.update({f"{name}_total": self.test_total[idx]})
        
        # Save the JSON data to the specified path
        with open(f"{self.args.result_save_path}.json", 'w') as json_file:
            json.dump(dict_to_report, json_file, indent=4)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _get_preds(self, logits):
        return torch.argmax(logits, dim=-1)

    def _get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)
    
    def _get_accuracy(self, preds, labels):
        correct = torch.sum(preds == labels).item()
        return correct / len(labels)
    
    def _get_ncorrect(self, preds, labels):
        correct = torch.sum(preds == labels).item()
        return correct
