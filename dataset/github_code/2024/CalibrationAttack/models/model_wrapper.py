import torch
import numpy as np
import math
import utils
from torchvision import models as torch_models
from torch.nn import DataParallel
import utils
from torch.utils.data import TensorDataset
from scipy.special import softmax
from transformers import ViTFeatureExtractor, ViTForImageClassification

class Model:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def predict(self, x):
        raise NotImplementedError('ModelPT')

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        elif loss_type == 'margin_loss_calib': # for underconfidence
            n_cls = len(y[0])
            predicted_class =  np.argmax(logits, axis=1)
            predicted_onehot = utils.one_hot_encode_v2(predicted_class, n_cls=n_cls)
            
            preds_predicted_class = (logits * predicted_onehot).sum(1, keepdims=True)
            
            diff = preds_predicted_class - logits  
            diff[predicted_onehot] = np.inf  
            margin = diff.min(1, keepdims=True)
            loss = margin
            
        elif loss_type == 'margin_loss_rand_underconf': 
            n_cls = len(y[0])
            
            softmaxes = softmax(logits, axis=1)
            
            predicted_class =  np.argmax(softmaxes, axis=1)
            predicted_onehot = utils.one_hot_encode_v2(predicted_class, n_cls=n_cls)
            
            preds_predicted_class = (softmaxes * predicted_onehot).sum(1, keepdims=True)
            
            diff = preds_predicted_class - softmaxes  
            diff[predicted_onehot] = np.inf  
            margin = diff.min(1, keepdims=True)
            loss = margin
            
        elif loss_type == 'margin_loss_overconf': # for overconfidence
            loss = np.max(softmax(logits, axis=1), axis=1, keepdims=True)           
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


class ModelPT(Model):
    """
    Wrapper class around PyTorch models.
    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model, batch_size, normalize=True, model_type="resnet"):
        super().__init__(batch_size)
      
        self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
        self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])

        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)
        self.normalize = normalize
        model.eval()
        self.model = model

        self.vit_extractor = None
        if model_type == "vit" or model_type == "vitL" or model_type == "SwinTiny" or model_type == "SwinBase" :
            self.vit_extractor =  ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)
        self.model_type = model_type

    def predict(self, x, vit=False):        
        if self.normalize == True:
            x = (x - self.mean) / self.std
        x = x.astype(np.float32)  
        n_batches = math.ceil(x.shape[0] / self.batch_size)

        logits_list = []

        with torch.no_grad(): 
            for i in range(n_batches):
                if self.vit_extractor is not None:
                    x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                    x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                    if self.model_type=="vit_aaa":
                        logits = self.model(x_batch_torch).cpu().numpy()
                    else:
                        logits = self.model(x_batch_torch).logits.cpu().numpy()
                else:
                    x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                    x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                    logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits
