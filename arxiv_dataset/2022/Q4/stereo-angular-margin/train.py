import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

CUDA_VISIBLE_DEVICES = 1


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
            loop.set_postfix(train_loss=np.average(
                train_losses), train_acc=np.average(train_acc))
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
