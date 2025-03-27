import torch
import torch.nn.functional as F
from tqdm import trange
from utils import get_network


def pretrain_mse(
    train_model,
    channel,
    num_classes,
    img_size,
    device, 
    dl_syn,
    pre_opt,
    pre_lr,
    pre_wd,
    pre_epoch,
):  

    # model and opt
    model = get_network(train_model, channel, num_classes, img_size).to(device)
    if pre_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=pre_lr, momentum=0.9, weight_decay=pre_wd)
    elif pre_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=pre_lr, weight_decay=pre_wd)
    else:
        raise NotImplementedError
    
    # pretrain        
    model.train()
    for _ in trange(1, pre_epoch+1):
        for x_syn, y_syn in dl_syn:
            # loss
            x_syn = x_syn.to(device)
            y_syn = y_syn.to(device)
            loss = F.mse_loss(model(x_syn), y_syn)
            # update
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model