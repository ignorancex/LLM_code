# !/usr/bin/env python3

import torch
from torch import nn

import utils
import models
import data
import argument

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    if len(args.order_list) > args.num_tasks:
        args.order_list = list(range(args.num_tasks))
    ### device confirm
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ### data preparation 
    data_train_list, data_test_list, data_train, data_test = data.mkdataset(args)
    if args.learn_noise_before == True:
        args.order_list.insert(0, 10)
        randimg_train = data.noise_generate(round(len(data_train) * args.argmt_perc), 255)
        randimg_test = data.noise_generate(round(len(data_test) * args.argmt_perc), 255)
        data_train_list.append(randimg_train)
        data_test_list.append(randimg_test)
        args.num_tasks = 11

    # print all args
    argument.print_args(args)
    print('Device is', device)
    
    ### build network
    if args.model_name in ['owm_fc']:
        net = models.create(name = args.model_name, num_hidden = args.num_hidden, num_classes = args.num_classes)
    else:
        raise ValueError("Network is not supported!")
    loss = nn.CrossEntropyLoss()
    print(net)

    ### optimizer
    optimizer = utils.get_optimizer(net, args.optimizer_type, args.learning_rate, args.momentum, args.model_name)

    ### train
    if args.CL_method != 'joint':
        for i in args.order_list:
            print('Training task ', i)
            dataset_i = data_train_list[i]
            utils.train(net, loss, dataset_i, data_test, optimizer, device, args)
            for i in range(args.num_tasks): 
                if args.list_eval:
                    acc, statlist = utils.evaluate_accuracy(data_test_list[i], net, stat = True, batch_size = args.batch_size)
                    print('test acc of task '+ str(i) + ' : ' + str(acc) + ' ' + str(statlist))
                    print('test acc of task %d on it\'s training set: %.4f' % (i, utils.evaluate_accuracy(data_train_list[i], net, batch_size = args.batch_size)))
                # else: print('test acc of task %d : %.4f' % (i, utils.evaluate_accuracy(data_test_list[i], net, batch_size = args.batch_size)))
                else: 
                    print('test acc of task %d : %.4f' % (i, utils.evaluate_accuracy(data_test_list[i], net, batch_size = args.batch_size)))
                    print('test acc of task %d on it\'s training set: %.4f' % (i, utils.evaluate_accuracy(data_train_list[i], net, batch_size = args.batch_size)))
    else:
        print('Training in joint scenario')
        utils.train(net, loss, data_train, data_test, optimizer, device, args)
        for i in range(args.num_tasks): print('test acc of task %d : %.4f' 
            % (i, utils.evaluate_accuracy(data_test_list[i], net, batch_size = args.batch_size)))


if __name__ == '__main__':
    
    args = argument.parser()
    main(args)