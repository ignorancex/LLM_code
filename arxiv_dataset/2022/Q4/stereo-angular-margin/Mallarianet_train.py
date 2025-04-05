import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from losses import AngularPenaltySMLoss, BroadFaceArcFace, ArcFace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Models import Projection

from Dataloader import Loadmalariadata

CUDA_VISIBLE_DEVICES = 1


def plot_acc(history):
    plt.plot([x["train_acc"] for x in history],"-bx")
    plt.plot([x["val_acc"] for x in history],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train acc","val acc"])
    plt.show()

def plot_loss(history):
    plt.plot([x.get("train_loss") for x in history], "-bx")
    plt.plot([x["val_loss"] for x in history],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])
    plt.show()
    

def get_default_device():
    """Pick GPU if available, else CPU"""
    
    gpu_name = torch.cuda.get_device_name()
    print(gpu_name)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# This is Our model With Projection included with Loss cross entropy
class CCE_Model(nn.Module):
    def __init__(self, num_classes=2, projection=True):
        super(CCE_Model, self).__init__()
        self.projection = projection
        self.project = Projection()
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv = nn.Sequential(
            *list(self.res50_model.children())[:-1])
        for param in self.res50_conv.parameters():
            param.requires_grad = True
        self.fatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512, 256)
        if self.projection:
            self.fc2 = nn.Linear(257, num_classes)
        else:
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, labels):
        x = self.res50_conv(x)
        x = self.fatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        if self.projection:
            x = self.dropout(x)
            x = self.project(x)
            out = self.fc2(x)
        else:
            x = self.dropout(x)
            out = self.fc2(x)

        loss = F.cross_entropy(out, labels)
        return loss,out


# This base Model with Projection Function For all Loss Types
class Backbone_Net(nn.Module):
    def __init__(self, projection=True):
        super(Backbone_Net, self).__init__()
        self.projection = projection
        self.project = Projection()
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv = nn.Sequential(
            *list(self.res50_model.children())[:-1])
        for param in self.res50_conv.parameters():
            param.requires_grad = True
        self.fatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512, 256)
        
    def forward(self, x, embed=False):
        x = self.res50_conv(x)
        x = self.fatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if self.projection:
            x = self.dropout(x)
            x = self.project(x)
        return x

# This Model with Projection For loss  ['sphereface', 'cosface','broadface','arcface']
class Model(nn.Module):
    def __init__(self, num_classes=2, loss_type='sphereface', projection=True):
        super(Model, self).__init__()
        if projection:
            print("Running With Projection")
            self.backbone = Backbone_Net(projection=True)
            self.features = 257
        else:
            print("Running Without Projection")
            self.backbone = Backbone_Net(projection=False)
            self.features = 256

        if loss_type == 'sphereface':
            self.loss = AngularPenaltySMLoss(
                self.features, num_classes, loss_type='sphereface')
        elif loss_type == 'cosface':
            self.loss = AngularPenaltySMLoss(
                self.features, num_classes, loss_type='cosface')
        elif loss_type == 'broadface':
            self.loss = BroadFaceArcFace(
                self.features, num_classes, compensate=False)
        elif loss_type == 'arcface':
            self.loss = ArcFace(self.features, num_classes)
        else:
            raise ValueError(
                "Enter The Valid Loss: ['sphereface', 'cosface','broadface','arcface'] ")

    def forward(self, x, labels, embed=False):
        x = self.backbone(x)
        if embed:
            return x
        L = self.loss(x, labels)
        return L


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def top1_accuracy(output, labels,):
    _, preds = torch.max(output, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return acc


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    for batch in val_loader:
        feats, labels = batch
        loss, out = model(feats, labels=labels)
        acc = top1_accuracy(out, labels)
    return loss, acc

# This is Trning Loop for class OurModelConvAngularPen and class ConvAngularPen
def fit(epochs, train_dl, test_dl, model, optimizer, max_lr, weight_decay, scheduler, grad_clip=None):

    history = []

    optimizer = optimizer(model.parameters(), max_lr,
                          weight_decay=weight_decay, momentum=0.92)

    scheduler = scheduler(optimizer, max_lr, epochs=epochs,
                          steps_per_epoch=len(train_dl))

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        lrs = []
        train_losses = []
        train_acc = []
        Val_loss = []
        Val_acc = []
        result = {}
        #loop =  tqdm(train_loader,leave=False)
        loop = tqdm(train_dl, total=len(train_dl))
        for batch in loop:
            images, labels = batch
            loss, out = model(images, labels)
            acc = top1_accuracy(out, labels)
    
            train_acc.append(acc.cpu().detach().numpy())
            
            train_losses.append(loss.cpu().detach().numpy())
            
            # print("loss",loss)
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(train_loss=np.average(train_losses), train_acc=np.average(train_acc))
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            lrs.append(get_lr(optimizer))

        vloss, vacc = evaluate(model, test_dl)
        Val_loss.append(vloss.cpu().detach().numpy())
        Val_acc.append(vacc.cpu().detach().numpy())
        # Validation phase
        #result["lrs"] = lrs
        result['train_loss'] = np.average(train_losses)
        result['train_acc'] = np.average(train_acc)
        result['val_loss'] = np.average(Val_loss)
        result['val_acc'] = np.average(Val_acc)
        print('Epoch', epoch, result)
        history.append(result)
    return history


epochs = 17
optimizer = torch.optim.SGD
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-5
scheduler = torch.optim.lr_scheduler.OneCycleLR


train_dl, test_dl = Loadmalariadata()

device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)

test_dl = DeviceDataLoader(test_dl, device)

#our_model = Model(loss_type='sphereface', projection=True).cuda()

our_model = CCE_Model(projection=True).cuda()

history = fit(epochs=epochs,
              train_dl=train_dl,
              test_dl=test_dl,
              model=our_model,
              optimizer=optimizer,
              max_lr=max_lr,
              grad_clip=grad_clip,
              weight_decay=weight_decay,
              scheduler=torch.optim.lr_scheduler.OneCycleLR)

df = pd.DataFrame.from_dict(history)

print('Saved Csv for.....')

df.to_csv('CCE_Model_Malarianet_with_pro_no_arug_1e-4.csv', index=False)

torch.save(our_model,'CCE_Model_Malarianet_with_pro_no_arug_1e-4.pth')

print('Saved Model For...')
print('....{} Model Executed.....')


plot_loss(history)

plot_acc(history)

# losses = ['sphereface', 'cosface', 'broadface', 'arcface']

# for i in losses[2:]:
#     print("....Running Model For {}....".format(i))
#     our_model = Conv_Net_Without_Projection(loss_type=i).cuda()
#     history = fit(epochs=epochs,
#                   train_dl=train_dl,
#                   test_dl=test_dl,
#                   model=our_model,
#                   optimizer=optimizer,
#                   max_lr=max_lr,
#                   grad_clip=grad_clip,
#                   weight_decay=weight_decay,
#                   scheduler=torch.optim.lr_scheduler.OneCycleLR)

#     df = pd.DataFrame.from_dict(history)

#     print('Saved Csv for.....', i)

#     df.to_csv(i+'100 org.csv', index=False)

#     torch.save(our_model, i+'100 org.pth')

#     print('Saved Model For...', i)
#     print('....{} Model Executed.....'.format(i))
