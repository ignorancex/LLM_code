import copy

from tqdm import tqdm

from lib.utils import bcsr_Training

import numpy as np
import quadprog
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal

from lib.utils import compute_loss_all_batches
from lib.gnn_models import set_model_task
from distutils.util import strtobool


class Bare(nn.Module):
    def __init__(self, args):
        super().__init__()

    def observe(self, args, model, task_id, optimizer, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, n_traj_samples=3, kl_coef= 1., dataloader=None, n_epoch=0):
        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id=task_id, n_traj_samples=n_traj_samples,
                                             kl_coef=kl_coef, save_pred_traj=True, learn_task_id = args.mask_select, update_proto=n_epoch==args.nepos
                                             )
        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        return train_res




cl_dict = {'bare':Bare} #