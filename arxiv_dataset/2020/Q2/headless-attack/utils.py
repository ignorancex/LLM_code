import torch
from models import *

def load_pretrained_net(net_name, chk_name, model_chk_path, test_dp=0):
    """
    Load the pre-trained models. CUDA only :)
    """
    net = eval(net_name)(test_dp=test_dp)
    net = nn.DataParallel(net).cuda()
    net.eval()
    print('==> Resuming from checkpoint for %s..' % net_name)
    checkpoint = torch.load('./{}/{}'.format(model_chk_path, chk_name) % net_name)
    if 'module' not in list(checkpoint['net'].keys())[0]:
        # to be compatible with DataParallel
        net.module.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint['net'])

    return net