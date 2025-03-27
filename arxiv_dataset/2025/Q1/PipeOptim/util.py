'''''''''''''''''''''
// Implementation of PipeOptim based on the framework of Pytorch
// Author: Lei Guan
// Email: guanleics@gmail.com
// Time: Jan.25, 2019
// Copyright 2019 All rights reserved.
// Redistribution and use the source code for commercial purpose are forbidden.
// Redistribution of the source code must be under permission of the author.
// Redistribution of the source code must retain the above copyright notice.
'''''''''''''''''''''
import time
import torch
import torch.optim as optim
import torch.distributed as dist
import random
#from basic_util import *



def save_image_results(model_name, runtime_list, val_loss_list, val_top1_list, val_top5_list):
    filename = 'experiment_results/' + model_name + \
               '_result_' + time.strftime("%Y%m%d", time.localtime())
    file = open(filename, 'w')
    file.write(str(runtime_list))
    file.write('\r\n')
    file.write(str(val_loss_list))
    file.write('\r\n')
    file.write(str(val_top1_list))
    file.write('\r\n')
    file.write(str(val_top5_list))
    file.write('\r\n')
    file.close()

def save_translation_results(model_name, runtime_list):
    filename = 'experiment_result/' + model_name + \
               '_result_' + time.strftime("%Y%m%d", time.localtime())
    file = open(filename, 'w')
    file.write(str(runtime_list))
    file.write('\r\n')
    file.close()


def show_running_time(epoch, lr, seconds):
    rank = dist.get_rank()
    num_process = dist.get_world_size()
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if rank == num_process-1:
        print("Epoch: %d; lr:%f; Current time: %d:%02d:%02d" % (epoch, lr, h, m, s))




