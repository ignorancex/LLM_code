import random
import numpy as np
import torch
import torch.nn.functional as F
import subprocess


def seed_everything(seed=2000):
	random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union-inter + 1)

    return (wbce + wiou).mean()


def get_exp_name():
    try:
        result = subprocess.run(['tmux', 'display-message', '-p', '#S'], capture_output=True, text=True, check=True)
        tmux_session_name = result.stdout.strip()
        return tmux_session_name
    except:
        return 'debug'