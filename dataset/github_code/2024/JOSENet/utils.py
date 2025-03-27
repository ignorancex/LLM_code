import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import os
import torch
import torch.nn as nn
from collections import OrderedDict


def visualize(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)

def plot_loss_function (args, paths, train_loss_tot, valid_loss_tot, epoch, save=True):
    x = []
    for i in range(epoch+1):
        x.append(i)

    #plt.title(args.model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim(bottom=0)
    plt.plot(x, train_loss_tot, color='red', label='train_loss')
    plt.plot(x, valid_loss_tot, color='blue', label='valid_loss')
    plt.legend(loc="upper right")
    #plt.plot()
    if (save):
        path = os.path.join("results","loss")
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(os.path.join(path, args.model_name+'.jpg'), format='jpg', dpi=600)

def plot_confusion_matrix (args, paths, test_true, test_pred, save=True):
    print (test_true)
    print (test_pred)
    cm = confusion_matrix(test_true, test_pred, normalize='true')
    df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Fight', 'Fight']],
          columns = [i for i in ['Non-Fight', 'Fight']])
    plt.figure(figsize = (3,3))
    svm = sn.heatmap(df_cm, annot=True)
    figure = svm.get_figure() 
    if (save):
        path = os.path.join("results","confusion")
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(os.path.join(path, args.model_name+'.jpg'), format='jpg', dpi=600)

def plot_roc_curve (args, paths, test_true, test_probs, save=True):
    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(test_true, test_probs)
    auc = metrics.auc(fpr, tpr)
    print ("AUC: ", auc)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    #plt.plot()
    # Save to local and then to drive
    if (save):
        path = os.path.join("results","roc")
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(os.path.join(path, args.model_name+'.jpg'), format='jpg', dpi=600)
    
    
def weighted_loss(pred, gold, lab0_count, lab1_count, device):

    gold = gold.contiguous().view(-1)

    w_lab1 = lab0_count / (lab1_count+lab0_count)
    w_lab0 = lab1_count / (lab1_count+lab0_count)

    weights = []
    for g in gold:
        if (g == 0):
            weights.append(w_lab0)
        else:
            weights.append(w_lab1)
    
    w = torch.Tensor(weights).to(device)
    loss = F.binary_cross_entropy_with_logits(pred, gold, weight=w, reduction='mean')
    return loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=1.0, gamma=1.5):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        #at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        #F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
#focal_loss = WeightedFocalLoss()



def load_sub_network (model, args, paths, input_type, model_name):
    pretrained_dict = torch.load(os.path.join(paths.models_self_supervised, args.model_self_supervised[input_type]+".pt"))
    new_dict = OrderedDict()
    # 1. filter out unnecessary keys
    for k,v in list(pretrained_dict.items()):
        if (model_name in k):
            k = k.split(".")
            k.remove(model_name)
            k = '.'.join(map(str, k))
            new_dict[k] = v
        if ("classifier" in k or "fc" in k):
            del pretrained_dict[k]
        
    pretrained_dict = new_dict
    model.load_state_dict(pretrained_dict, strict=False)
    print ("Successfully loaded weights for "+model_name)
    return model

