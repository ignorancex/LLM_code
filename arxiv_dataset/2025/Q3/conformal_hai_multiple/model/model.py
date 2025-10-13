from sklearn.linear_model import LogisticRegression
from config import conf
import numpy as np
import torch
import torch.nn as nn

import copy
from tqdm import tqdm
import time
import pickle
import logging
import torch.nn.functional as F
import sklearn.metrics

import torch
import torchvision.models as models



import datetime
import os

eps_cst = 1e-8

class Model:
    """Base class of the classifier model"""
    def __init__(self) -> None:
        pass

    def train(self, x,y):
        pass

    def predict(self, input):
        pass
    
    def predict_prob(self, input):
        pass
    
    def test(self, x, y):
        pass

class ModelReal(Model):
    """Model used in real data experiments"""
    def __init__(self, m_name, model_laplace=False, model_lapl_param=None) -> None:
        super().__init__()
        # 'm_name' specifies the classifier name, i.e., DenseNet, PreResNet-110 or ResNet-110.
        self.model_lapl = model_laplace
        self.model_lapl_param = model_lapl_param
        if m_name == 'r_low_acc':
            data_path = os.path.join(conf.ROOT_DIR, 'data/human_model_truth_cifar10h.csv')
            data = np.genfromtxt(data_path, delimiter=',')

            self.model_logits = data[:, 10:20]        # 10000x10
        else:
            with open(f"{conf.ROOT_DIR}/data/{m_name}.csv", "r") as f:
                csv = np.loadtxt(f, delimiter=',')      # 10000x21
                self.model_logits = csv[:, 11:]         # 10000x10
                # Models keep stored the softmax outputs for each sample in test set,
                # so we only need the index of the correspondent sample to get the softmax output

    def predict(self, input, return_tensor=False):
        self.model_logits_t = torch.tensor(self.model_logits[input], device=conf.device)        # 8000x10
        if self.model_lapl: # add to config   # to make the model inaccurate
            self.model_logits_t += self.model_lapl_param
            self.model_logits_t /= self.model_logits_t.sum(1).reshape(-1,1)
        y_hat = self.model_logits_t.multinomial(1, replacement=True, generator=conf.torch_rng)  # 8000x1 drawing sample from multinomial dist, drawing 1 from each row
        if not return_tensor:
            y_hat = y_hat.detach().cpu().numpy().flatten() 
        return y_hat
    
    def predict_prob(self, input):
        return self.model_logits[input]

    def test(self, x, y):
        y_hat = self.predict(input=x)       # 8000
        return np.mean(y == y_hat)          # accuracy  (np.argmax(self.model_logits[x],1) == y).mean()
    
class ModelReal(Model):
    """Model used in real data experiments"""
    def __init__(self, m_name, model_laplace=False, model_lapl_param=None) -> None:
        super().__init__()
        # 'm_name' specifies the classifier name, i.e., DenseNet, PreResNet-110 or ResNet-110.
        self.model_lapl = model_laplace
        self.model_lapl_param = model_lapl_param
        if m_name == 'r_low_acc':
            data_path = os.path.join(conf.ROOT_DIR, 'data/human_model_truth_cifar10h.csv')
            data = np.genfromtxt(data_path, delimiter=',')

            self.model_logits = data[:, 10:20]        # 10000x10
        else:
            with open(f"{conf.ROOT_DIR}/data/{m_name}.csv", "r") as f:
                csv = np.loadtxt(f, delimiter=',')      # 10000x21
                self.model_logits = csv[:, 11:]         # 10000x10   # Models keep stored the softmax outputs for each sample in test set, so we only need the index of the correspondent sample to get the softmax output

    def predict(self, input, return_tensor=False):
        self.model_logits_t = torch.tensor(self.model_logits[input], device=conf.device)        # 8000x10
        if self.model_lapl: # add to config   # to make the model inaccurate
            self.model_logits_t += self.model_lapl_param
            self.model_logits_t /= self.model_logits_t.sum(1).reshape(-1,1)
        y_hat = self.model_logits_t.multinomial(1, replacement=True, generator=conf.torch_rng)  # 8000x1 drawing sample from multinomial dist, drawing 1 from each row
        if not return_tensor:
            y_hat = y_hat.detach().cpu().numpy().flatten() 
        return y_hat
    
    def predict_prob(self, input):
        return self.model_logits[input]

    def test(self, x, y):
        y_hat = self.predict(input=x)       # 8000
        return np.mean(y == y_hat)          # accuracy  (np.argmax(self.model_logits[x],1) == y).mean()
    

class ModelImageNet16H(Model):
    """Model used in real data experiments"""
    def __init__(self, m_name, model_laplace=False, model_lapl_param=None) -> None:
        super().__init__()
        # 'm_name' specifies the classifier name, i.e., DenseNet, PreResNet-110 or ResNet-110.
        self.model_lapl = model_laplace
        self.model_lapl_param = model_lapl_param
        with open(f"{conf.ROOT_DIR}/data/imagenet_080.csv", "r") as f:
            csv = np.genfromtxt(f, delimiter=',', dtype=str, filling_values='')     
            self.model_logits = csv[:, 7:23].astype(np.float64)         
            # Models keep stored the softmax outputs for each sample in test set,
            # so we only need the index of the correspondent sample to get the softmax output

    def predict(self, input, return_tensor=False):
        self.model_logits_t = torch.tensor(self.model_logits[input], device=conf.device)        # 8000x10
        if self.model_lapl: # add to config   # to make the model inaccurate
            self.model_logits_t += self.model_lapl_param
            self.model_logits_t /= self.model_logits_t.sum(1).reshape(-1,1)
        y_hat = self.model_logits_t.multinomial(1, replacement=True, generator=conf.torch_rng)  # 8000x1 drawing sample from multinomial dist, drawing 1 from each row
        if not return_tensor:
            y_hat = y_hat.detach().cpu().numpy().flatten() 
        return y_hat
    
    def predict_prob(self, input):
        return self.model_logits[input]

    def test(self, x, y):
        y_hat = self.predict(input=x)       
        return np.mean(y == y_hat)          # accuracy # TODO (np.argmax(self.model_logits[x],1) == y).mean()
    
class ModelSynthetic(Model):
    """Model used in synthetic data experiments"""
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(random_state=0,n_jobs=-1, max_iter=1000, multi_class='ovr')
        self.missing_classes = []
        
    def predict(self, input):
        return self.model.predict(input)

    def predict_prob(self, input):
        ret = self.model.predict_proba(input)
        # Fix model output to return 0 probabilty for unknown classes
        for missing_class in self.missing_classes:
            ret = np.insert(ret,missing_class,0.,axis=1)
        return ret

    def train(self, x,y):
        self.model = self.model.fit(x,y)
        # Find which classes the model did not learn at all 
        # (Needed to fix tensors later)
        sorted_classes = np.sort(self.model.classes_)       # ex 98
        all_classes = np.arange(conf.n_labels)
        if self.model.classes_.shape[0] < conf.n_labels:
            i = 0
            for j in all_classes:
                if j == sorted_classes[i]:
                    i+=1
                else:
                    self.missing_classes.append(j)          # including the j in missing_classes

    def test(self, x, y):
        return self.model.score(x,y)

class ModelHateSpeech(nn.Module):
    """Model used for HateSpeech data experiments"""
    def __init__(self, input_dim, output_dim) -> None:
        super(ModelHateSpeech, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim//2).to('cuda')    # Input layer to hidden layer
        self.fc2 = nn.Linear(in_features=input_dim//2, out_features=input_dim//4).to('cuda')  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(in_features=input_dim//4, out_features=3).to('cuda')            # Hidden layer to output layer
        self.missing_classes = []

    def forward(self, x):
        x = F.relu(self.fc1(x))             # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))             # Apply ReLU activation to the second hidden layer
        x = F.softmax(self.fc3(x))          # Output layer
        return x

class ModelImageNet16H_trainable(nn.Module):
    """Model used for ImageNet16H data experiments"""
    def __init__(self, input_dim, output_dim) -> None:
        super(ModelImageNet16H, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim//2).to('cuda')                   # Input layer to hidden layer
        self.fc2 = nn.Linear(in_features=input_dim//2, out_features=input_dim//4).to('cuda')                # Hidden layer to hidden layer
        self.fc3 = nn.Linear(in_features=input_dim//4, out_features=output_dim).to('cuda')                  # Hidden layer to output layer
        # self.vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).to(conf.device)
        self.missing_classes = []

    def forward(self, x):
        x = F.relu(self.fc1(x))             # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))             # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)                     # Output layer  # no normalization via softmax
        return x

class ModelChestXray(nn.Module):
    """Model used for NIH Chest Xray data experiments"""
    def __init__(self, input_dim, output_dim) -> None:
        super(ModelChestXray, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim//2).to('cuda')       # Input layer to hidden layer
        self.fc2 = nn.Linear(in_features=input_dim//2, out_features=input_dim//4).to('cuda')    # Hidden layer to hidden layer
        self.fc3 = nn.Linear(in_features=input_dim//4, out_features=output_dim).to('cuda')               # Hidden layer to output layer
        self.missing_classes = []

    def forward(self, x):
        x = F.relu(self.fc1(x))             # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))             # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)                     # Output layer  # no normalization via softmax
        return x

class ModelCompass(nn.Module):
    """Model used for COMPASS data experiments"""
    def __init__(self, input_dim, output_dim) -> None:
        super(ModelCompass, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim//2).to('cuda')       # Input layer to hidden layer
        self.fc2 = nn.Linear(in_features=input_dim//2, out_features=input_dim//4).to('cuda')    # Hidden layer to hidden layer
        self.fc3 = nn.Linear(in_features=input_dim//4, out_features=output_dim).to('cuda')               # Hidden layer to output layer
        self.missing_classes = []

    def forward(self, x):
        x = F.relu(self.fc1(x))             # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))             # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)                     # Output layer  # no normalization via softmax
        return x
    
def compute_deferral_metrics(data_test):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )           # 0.858757
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"], data_test["labels"]
    )           # 0.900322
    results["coverage"] = 1 - np.mean(data_test["defers"])  # 0.5355125
    # get classifier accuracy when defers is 0
    results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"][data_test["defers"] == 0],
        data_test["labels"][data_test["defers"] == 0],
    )       # For those that did not defer, what is the model accuracy? ex 0.971363
    # get human accuracy when defers is 1
    results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )       # For those that deferred, what is the accuracy of the human pred   # 0.86359687
    # get system accuracy # parang overall? # combining human pred + model
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["hum_preds"] * (data_test["defers"]),
        data_test["labels"],
    )               # 0.921307506 <- quite inaccurate
    return results

class BaseTrainerMethod():
    """Abstract method for learning to defer methods based on a surrogate model"""

    def __init__(self, alpha, plotting_interval, model, device, learnable_threshold_rej = False):
        '''
        alpha: hyperparameter for surrogate loss 
        plotting_interval (int): used for plotting model training in fit_epoch
        model (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the reject score (applicable to RealizableSurrogate only)
        '''
        self.alpha = alpha      # 1
        self.plotting_interval = plotting_interval  #300
        self.model = model
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej      # True

    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """surrogate loss function"""
        pass

    def fit_epoch(self, dataloader, optimizer, verbose=False, epoch=1):     # typical torch training of model using training data
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)     # 1000x384
            data_y = data_y.to(self.device)     # 1000
            hum_preds = hum_preds.to(self.device)       # 1000
            outputs = self.model(data_x)                # 1000x4
            loss = self.surrogate_loss_function(outputs, hum_preds, data_y)     # SURROGATE LOSS FUNCTION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]        # ex 35.1
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:     # if want to print or log
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):       # 50 only
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1 :       # more like validation
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)       # note that it uses the val
                data_test = self.test(dataloader_val)  # <- this was calculated in self.fit_treshold_rej (redundant?) # "defers", "labels", "hum_preds", "preds", "rej_score", "class_probs"
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:           # system_acc is the final basis
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())     # copy affine linear model weights
                if verbose:
                    logging.info(compute_deferral_metrics(data_test))
            if scheduler is not None:
                scheduler.step()

        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def predict_prob(self, input):
        # Assuming 'output' is the raw prediction logits from your model
        input = torch.from_numpy(input).to(conf.device)
        probabilities = F.softmax(self.model(input), dim=1)
        return probabilities.detach().cpu().numpy()

class ModelTrainer(BaseTrainerMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """ Implementation of our RealizableSurrogate loss function
        """ 
        human_correct = (hum_preds == data_y).float()       # 1000
        human_correct = torch.tensor(human_correct).to(self.device)
        batch_size = outputs.size()[0]                      # batch_size        # 1000
        outputs_exp = torch.exp(outputs)                    # 1000
        new_loss = -torch.log2(
            (
                human_correct * outputs_exp[range(batch_size), -1]
                + outputs_exp[range(batch_size), data_y]
            )       # you weigh the human correct + the outputs values corresponding to GT y
            / (torch.sum(outputs_exp, dim=1) + eps_cst) # eps_cst is 1e-08 (very small for precision issues) <- for normalizing
        )  # pick the values corresponding to the labels        # size 1000
        # ce_loss = -torch.log2(
        #     (outputs_exp[range(batch_size), data_y])
        #     / (torch.sum(outputs_exp[range(batch_size), :-1], dim=1) + eps_cst)
        # )                              # output_exp for that gt label <- normalized by values corresponding to labels only     (no human factor)
        ce_loss = torch.nn.CrossEntropyLoss()(outputs_exp, data_y)      # changed
        loss = self.alpha * new_loss + (1 - self.alpha) * ce_loss       # 1000   # new loss and ce loss weighted by alpha
        return torch.sum(loss) / batch_size                             # batch mean

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
        alpha_grid=[0, 0.1, 0.3, 0.5, 0.9, 1],
    ):
        # np.linspace(0,1,11)
        best_alpha = 0
        best_acc = 0
        model_dict = copy.deepcopy(self.model.state_dict())     # NOTE Why copy?
        # for alpha in tqdm(alpha_grid):  #[0, 0.1, 0.3, 0.5, 0.9, 1] # Iterating for different alphas
        #     self.alpha = alpha  # for surrogate?
        #     self.model.load_state_dict(model_dict)
        #     self.fit(
        #         dataloader_train,
        #         dataloader_val,
        #         dataloader_test,
        #         epochs = epochs,
        #         optimizer = optimizer,
        #         lr = lr,
        #         verbose = verbose,
        #         test_interval = test_interval,
        #         scheduler = scheduler,
        #     )["system_acc"]
        #     accuracy = compute_deferral_metrics(self.test(dataloader_val))["system_acc"]
        #     logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
        #     if accuracy > best_acc:
        #         best_acc = accuracy
        #         best_alpha = alpha
        self.alpha = best_alpha
        # self.model.load_state_dict(model_dict)
        fit = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs = epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))

        return test_metrics

    def test(self, dataloader):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []     # values of 0 or 1 whether to defer, greater than threshold
        truths_all = []     # data_y
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)             # 1000x384
                data_y = data_y.to(self.device)             # 1000
                hum_preds = hum_preds.to(self.device)       # 1000

                outputs = self.model(data_x)            # 1000x4
                outputs_class = F.softmax(outputs[:, :-1], dim=1)       # Consider first 3 columns and perform softmax
                outputs = F.softmax(outputs, dim=1)     # consider human pred (4th column)
                _, predicted = torch.max(outputs.data, 1)   # predicted: 1000 (indices)     # predicted not used?
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)  # 1000, 1000   # consider only first 3
                predictions_all.extend(predicted_class.cpu().numpy())
                
                defer_scores = [ outputs.data[i][-1].item() - outputs.data[i][predicted_class[i]].item() for i in range(len(outputs.data))]     #1000 # the difference of the 4th column and the column where predicted = defer score?
                # if threshold is -0.70, then the output score must be bigger than by 0.70 than the final column to NOT defer
                defer_binary = [int(defer_score >= self.threshold_rej) for defer_score in defer_scores]   # 1000 # True if defer_score greater than threshold?
                defers_all.extend(defer_binary)      # meaning 0 or 1 if greater than theshold
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                for i in range(len(outputs.data)):
                    rej_score_all.append(
                        outputs.data[i][-1].item()
                        - outputs.data[i][predicted_class[i]].item()
                    )       # just the same as defer_scores?
                class_probs_all.extend(outputs_class.cpu().numpy())

        # convert to numpy
        defers_all = np.array(defers_all)       # 2478
        truths_all = np.array(truths_all)       # 2478
        hum_preds_all = np.array(hum_preds_all)     # 2478
        predictions_all = np.array(predictions_all)     # softmax of all but get max of 3
        rej_score_all = np.array(rej_score_all)         # differences
        class_probs_all = np.array(class_probs_all)     # softmax of only first 3
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """_summary_: Updates the average meter with the new value and the number of samples
        Args:
            val (_type_): value
            n (int, optional):  Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """_summary_

    Args:
        output (tensor): output of the model
        target (_type_): target
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        float: accuracy
    """
    maxk = max(topk)        # ex 1
    batch_size = target.size(0)     # 1000

    _, pred = output.topk(maxk, 1, True, True)      # returns val and indices; pred is indices; pred is 1000x1
    pred = pred.t()         # 1x1000
    correct = pred.eq(target.view(1, -1).expand_as(pred))       # 1x1000 of True's and False's

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)         # 351       # accuracy (can be low for initial train)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

