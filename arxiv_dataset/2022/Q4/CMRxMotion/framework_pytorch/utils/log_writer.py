import os

import torch
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter


class LogWriter():
    def __init__(self, log_path):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalar(self, eval_dict, epoch, tag=''):
        for key in eval_dict:
            tpath = os.path.join(key, tag) if tag else key
            self.writer.add_scalar(tpath, eval_dict[key], epoch)



    def write_image(self, img_dict, epoch=0, tag=''):
        for key in img_dict:
            tpath = os.path.join(key, tag) if tag else key
            self.writer.add_image(tpath, img_dict[key], epoch, dataformats='HWC')

    def write_text(self, str, epoch=0, tag=''):
        self.writer.add_text(tag, str, epoch)

    def write_graph(self, net, input):
        self.writer.add_graph(net ,input)