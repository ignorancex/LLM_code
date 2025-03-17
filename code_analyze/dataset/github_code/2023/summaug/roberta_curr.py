from pathlib import Path
from turtle import setup
from transformers.utils import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.optim as optim
import random
from torch import nn
import evaluate
import datasets
import datetime
from data import *
import argparse
import os
import torch.distributed as dist

logging.set_verbosity_error()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def get_lambda(step, all_steps, warmup_steps, type):
    if type == 'formula':
        return min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
    elif type == 'linear':
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0, float(all_steps - step) / float(max(1, all_steps - warmup_steps))
        )


def train(args,trainset,valset,testset = None):
    setup_seed(args.seed)
    PATH = args.ptr_model_path
    model_file_name = PATH + '/best_model_{}.pth'.format(args.gpu_id)
    if testset is None:
        model = RobertaForSequenceClassification.from_pretrained(args.initial_point)
    else:
        config = RobertaConfig.from_pretrained(args.initial_point)
        config.num_labels = 10
        model = RobertaForSequenceClassification.from_pretrained(args.initial_point)
        para = torch.load(model_file_name,'cpu')
        model.load_state_dict(para)
        model.classifier = RobertaClassificationHead(config)
        model.num_labels = 10
        # para = torch.load(model_file_name)
        # keys_to_remove = list()
        # for k,v in para.items():
        #     if 'classifier' in k:
        #         keys_to_remove.append(k)
        # for key in keys_to_remove:
        #     del para[key]
        # model.load_state_dict(para, strict = False)
        # del para
    optimizer = optim.Adam(model.parameters(), lr = 0)
    print(len(trainset),  len(valset))
    model = model.to(args.device)
    if args.multi_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bz,shuffle=False, sampler = train_sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bz,shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.val_bz, shuffle=False, num_workers=0)
    print(len(trainloader),len(valloader))
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.val_bz, shuffle=False, num_workers=0)
        print(len(testset),len(testloader))
    n_epoches = args.s_epoch if testset is None else args.epoch
    accumulate_step = args.acc_step
    total_steps = math.ceil(n_epoches*len(trainloader)/(args.train_bz * accumulate_step))
    wm_steps = int(total_steps*args.wm_ratio)
    print('***********do training***********')
    batch_counter = 0
    step_counter = 0
    log_step = args.log_step
    log_file = open('{}.txt'.format(args.logfile_name),'w')
    print_loss = True
    max_acc = 0.0
    max_lr = args.lr_s if testset is None else args.lr_g
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    train_loss = 0.0   
    for epoch in range(n_epoches):
        model.train()
        train_sampler.set_epoch(epoch)
        for id,item in tqdm(enumerate(trainloader)):
            data,label = item[0],item[1]
            data['input_ids'] = data['input_ids'].to(args.device)
            data['attention_mask'] = data['attention_mask'].to(args.device)
            label = label.to(args.device)
            outputs = model(**data, labels=label)
            loss = outputs[0]
            loss = loss / accumulate_step
            batch_counter += 1
            if batch_counter % accumulate_step == (accumulate_step - 1):
                loss.backward()
                if args.schedule != 'constant':
                    lr = max_lr * get_lambda((step_counter/accumulate_step), total_steps, wm_steps, args.schedule)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                step_counter += 1
                print_loss = True
            train_loss = train_loss + loss.item()
            if (step_counter % log_step == log_step - 1) and print_loss:
                    print('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step))
                    log_file.write('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step)+'\n')
                    train_loss = 0.0
                    print_loss = False
        if (not args.multi_gpu) or args.local_rank == 0:
            val_loss = 0.0
            val_counter = 0
            correct_num = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                for id,item in tqdm(enumerate(valloader)):
                    data,label = item[0],item[1]
                    data['input_ids'] = data['input_ids'].to(args.device)
                    data['attention_mask'] = data['attention_mask'].to(args.device)
                    label = label.to(args.device)
                    outputs = model(**data, labels=label)
                    preds = outputs.logits.argmax(dim=1)
                    loss = outputs.loss
                    val_loss = val_loss + loss.item()
                    val_counter += 1
                    total_num += outputs.logits.size(0)
                    correct_num += (preds == label).sum().item()
                val_loss = val_loss / val_counter
            
            print('epoch{}, valid_loss:{}, acc:{}'.format(epoch+1,val_loss,correct_num/total_num))
            if correct_num/total_num > max_acc:
                max_acc = correct_num/total_num
                if args.multi_gpu:
                    torch.save(model.module.state_dict(), model_file_name)
                else:
                    torch.save(model.state_dict(), model_file_name)
                print('*******best model updated*******')
    if ((not args.multi_gpu) or args.local_rank == 0) and testset is not None:
        correct_num = 0
        total_num = 0
        if args.multi_gpu:
            model.module.load_state_dict(torch.load(model_file_name))
        else:
            model.load_state_dict(torch.load(model_file_name))
        model.eval()
        with torch.no_grad():
            for id,item in tqdm(enumerate(testloader)):
                data,label = item[0],item[1]
                data['input_ids'] = data['input_ids'].to(args.device)
                data['attention_mask'] = data['attention_mask'].to(args.device)
                outputs = model(**data)
                preds = outputs.logits.argmax(dim=1)
                # print(preds)
                total_num += outputs.logits.size(0)
                correct_num += (preds.cpu() == label).sum().item()
        print('acc:{}'.format(correct_num/total_num))
        with open('results_{}.txt'.format(args.gpu_id), 'a') as f:
            f.write(str(args)+'\n'+str(correct_num/total_num)+'\n')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default = '0')
    parser.add_argument('--initial_point', type=str, default = 'roberta-large')
    parser.add_argument('--ptr_model_path', type=str, default = './models')
    parser.add_argument('--label_pool', action='store_true', default = False)
    parser.add_argument('--silver', action='store_true', default = False)
    parser.add_argument('--schedule', type=str, help='type of scheduler',default = 'linear')
    parser.add_argument('--wm_ratio', type =float, help='ratio of warmup steps', default = 0.05)
    parser.add_argument('--max_len', type=int, help='max length when tokenization', default=512)
    parser.add_argument('--seed', type=int, help='random seed', default=123)
    parser.add_argument('--lr_s', type=float ,help='maximum learning rate for silver data', default=2e-5)
    parser.add_argument('--lr_g', type=float ,help='maximum learning rate for gold data', default=2e-5)
    parser.add_argument('--test_size', type=float, help='size of validation set', default=0.1)
    parser.add_argument('--n_sample', type=int, help='training samples under low resource settings', default=None)
    parser.add_argument('--train_bz', type=int, help='training batch size', default=4)
    parser.add_argument('--val_bz', type=int, help='val/test batch size', default=10)
    parser.add_argument('--acc_step', type=int, help='accumulation step', default=4)
    parser.add_argument('--epoch', type=int, help='training epoch', default=6)
    parser.add_argument('--s_epoch', type=int, help='training epoch', default=2)
    parser.add_argument('--log_step', type=int, help='step interval to log loss information', default=200)
    parser.add_argument('--logfile_name', type=str, default = 'training_log')
    parser.add_argument('--augfile', type=str, help='augmentation file, summ_aug or aeda_aug', default = 'summ_aug_multiclass')
    args = parser.parse_args()
    train_data,val_data,test_data = get_10class_data()
    shuffle_id_train = shuffle(np.arange(len(train_data['text'])), random_state=123)
    shuffle_id_val = shuffle(np.arange(len(val_data['text'])), random_state=123)
    data_id = None
    if args.n_sample is not None:
        data_id = shuffle_id_train[:args.n_sample]
        pos_r, train_data = sample(train_data, data_id)
        val_id = shuffle_id_val[:int(args.n_sample*args.test_size/(1-args.test_size))]
        _, val_data = sample(val_data, val_id)
        # print(pos_r)
    tokenizer = AutoTokenizer.from_pretrained(args.initial_point) 
    aug_data = get_aug_data(args.augfile, train_data, data_id,separate=True, label_pool=args.label_pool)
    aug_train_x, aug_val_x,aug_train_y,aug_val_y = train_test_split(aug_data['text'],aug_data['label'], test_size = args.test_size, random_state = 123)
    aug_train = dict()
    aug_train['text'] = aug_train_x
    aug_train['label'] = aug_train_y
    aug_val = dict()
    aug_val['text'] = aug_val_x
    aug_val['label'] = aug_val_y
    testset = IMDBDataset(tokenizer,  test_data, args.max_len)
    trainset_s = IMDBDataset(tokenizer,  aug_train, args.max_len)
    trainset_g = IMDBDataset(tokenizer,  train_data, args.max_len)
    valset_s = IMDBDataset(tokenizer,  aug_val, args.max_len)
    valset_g = IMDBDataset(tokenizer,  val_data, args.max_len)    
    gpu_ids = [int(x) for x in args.gpu_id.split(',')]
    args.multi_gpu = (len(gpu_ids) > 1)
    if args.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        dist.init_process_group("nccl", world_size = len(gpu_ids),timeout = datetime.timedelta(seconds=3600))
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    else:
        args.device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    if args.silver:
        train(args,trainset_s,valset_s)
    else:
        train(args,trainset_g,valset_g,testset)
    

if __name__=='__main__':
    main()