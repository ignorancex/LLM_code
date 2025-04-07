import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from smp_model import *
from smp_data import *
from tqdm import tqdm
import csv
from tools import *
from transformers import logging
import warnings
from cb import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--warm_start_epoch', type=int, default=0) # Set to 10 for testing the trained model.
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', default=1e-3, type=float)

parser.add_argument('--images_dir', type=str, default='img_yt')
parser.add_argument('--gt_path', type=str, default="pop_time_series/popularity.csv")

data_dir = ''
data_filename = '0711_yt_seq1'
# parser.add_argument('--train_files', type=str, default=data_dir+'train/train'+data_filename+'.csv')
# parser.add_argument('--val_files', type=str, default=data_dir+'val/val'+data_filename+'.csv')
# parser.add_argument('--test_files', type=str, default=data_dir+'test/test'+data_filename+'.csv')
parser.add_argument('--train_files', type=str, default=data_dir+'train/train'+data_filename+'.csv')
parser.add_argument('--val_files', type=str, default=data_dir+'test/test'+data_filename+'.csv')
parser.add_argument('--test_files', type=str, default=data_dir+'val/val'+data_filename+'.csv')

parser.add_argument('--seq_len', type=int, default=30)

parser.add_argument('--model_choose', type=int, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--ckpt_name', type=str, default='best-39-1.6959.pth')
parser.add_argument('--result_file', type=str, default='all_result.csv')
parser.add_argument('--write', type=bool, default=False)

parser.add_argument('--continue_model', type=bool, default=False)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--ablation', type=bool, default=False)


def load_data(args):
    # data loading
    if args.model_choose == 0:
        trainset = bili_data_lstm(csv_file=args.train_files, root_dir=args.images_dir)
        valset = bili_data_lstm(csv_file=args.val_files, root_dir=args.images_dir)
        testset = bili_data_lstm(csv_file=args.test_files, root_dir=args.images_dir)
    elif args.model_choose == 1:
        trainset = youtube_data_lstm(args.train_files, args.images_dir, args.gt_path)
        valset = youtube_data_lstm(args.val_files, args.images_dir, args.gt_path)
        testset = youtube_data_lstm(args.test_files, args.images_dir, args.gt_path)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=True)


    return train_loader, val_loader, test_loader


# train
def train(args, models, train_loader, val_loader):
    model, cb_model = models
    loss_fn = nn.MSELoss()
    lr = args.lr
    weight_decay = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1,
                                                           verbose=True, min_lr=0)
    min_mae = 10
    for epoch in range(args.epochs):
        batch_train_losses = []
        model.train()
        # training
        preds = []
        labels = []
        out_f_list = []
        for num, data in enumerate(tqdm(train_loader)):

            img = data['img'].to(device)
            text = data['text']
            meta = data['meta'].to(device)

            label = data['label'].to(device)
            # if isinstance(model, bili_transformer):
            #     label_tensor = torch.tensor(label)
            #     out = model(img, text, meta, label_tensor)
            # else:
            out, out_f = model(img, text, meta)

            train_loss = loss_fn(out, label)
            batch_train_losses.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1.)
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()

            for i in range(out.shape[0]):
                preds.append(out[i].cpu().detach().numpy().tolist())
                labels.append(label[i].cpu().detach().numpy().tolist())
                out_f_list.append(out_f[i].cpu().detach().numpy().tolist())

        avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses), 5)
        print('=====Epoch %d averaged training loss: %.6f=====' % (epoch + 1,  avg_train_loss))
        print('=====Epoch %d train result=====' % (epoch + 1))
        print_output_seq(labels, preds)

        # valid
        model.eval()
        batch_val_losses = []
        preds = []
        val_labels = []
        val_out_f_list = []
        for num, data in enumerate(tqdm(val_loader)):

            img = data['img'].to(device)
            text = data['text']
            meta = data['meta'].to(device)

            label = data['label'].to(device)

            out, out_f = model(img, text, meta)

            val_loss = loss_fn(out, label)
            batch_val_losses.append(val_loss.item())

            for i in range(out.shape[0]):
                preds.append(out[i].cpu().detach().numpy().tolist())
                val_labels.append(label[i].cpu().detach().numpy().tolist())
                val_out_f_list.append(out_f[i].cpu().detach().numpy().tolist())

        avg_val_loss = round(sum(batch_val_losses) / len(batch_val_losses), 5)
        mae = mean_absolute_error(val_labels, preds)
        print('=====Epoch %d averaged val loss: %.6f=====' % (epoch + 1, avg_val_loss))
        print('=====Epoch %d val result=====' % (epoch + 1))
        print_output_seq(val_labels, preds)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler.step(avg_val_loss)
        # print("recent learning rate:%.4f" % lr)
        # lr = min(0.001, lr * 0.8)

        if mae < min_mae + 0.01:
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'epoch-%d-%.4f.pth' % (epoch + 1, mae)))  # .state_dict()
            # wandb.watch(model,log = 'all')
            print('Saved model. Testing...')  # .state_dict()

        if mae < min_mae:
            min_mae = mae

        # if epoch % 3 == 2:
        #     cb_reg_fuc(args, cb_model, (out_f_list, labels), (val_out_f_list, val_labels))


# test
def test(args, model, test_loader):
    model.eval()

    # save result
    if args.write:
        with open(args.result_file, 'w') as f:
            pass

    preds = []
    labels = []
    count = 0

    for num, data in enumerate(tqdm(test_loader)):

        img = data['img'].to(device)
        text = data['text']
        meta = data['meta'].to(device)

        label = data['label'].to(device)

        out = model(img, text, meta)

        count += 1
        print(out)
        for i in range(out.shape[0]):
            preds.append(out[i].cpu().detach().numpy().tolist())
            labels.append(label[i].cpu().detach().numpy().tolist())

        # write result
        if args.write:
            with open(args.result_file, 'a+', newline='', encoding='UTF-8-sig') as f:
                for i in range(len(out)):
                    new_lines = [data['id'][i], out[i].cpu().detach().numpy().tolist(), label[i].cpu().detach().numpy().tolist()]
                    writer = csv.writer(f)
                    writer.writerow(new_lines)

    return print_output_seq(labels, preds)


# ablation study
def ablation_study(args, model, loader):
    ablation_result = []
    # get column length
    data = next(iter(loader))
    text_length = len(data['text'])
    meta_length = (data['meta']).size(1)

    # discard img
    pre_list = []
    label_list = []
    for num, data in enumerate(tqdm(loader)):
        img = torch.zeros(args.batch_size, 3, 224, 224).to(device)
        text = data['text']
        meta = data['meta'].to(device)
        label = data['label']

        out = model(img, text, meta)

        out = out.cpu().detach().squeeze(1).numpy().tolist()
        label = label.squeeze(1).numpy().tolist()
        pre_list.extend(out)
        label_list.extend(label)

    print("===discard img result===")
    ablation_result.append(print_output_part(label_list, pre_list))

    # discard text
    for i in range(text_length):
        pre_list = []
        label_list = []
        for num, data in enumerate(tqdm(loader)):
            batch_size = (data['meta']).size(0)
            img = data['img'].to(device)
            text = data['text']
            text[i] = tuple(["0" for i in range(args.batch_size)])
            meta = data['meta'].to(device)
            label = data['label']

            out = model(img, text, meta)

            out = out.cpu().detach().squeeze(1).numpy().tolist()
            label = label.squeeze(1).numpy().tolist()
            pre_list.extend(out)
            label_list.extend(label)

        print("===discard text %d result===" % i)
        ablation_result.append(print_output_part(label_list, pre_list))

    # discard meta
    for i in range(meta_length):
        pre_list = []
        label_list = []
        for num, data in enumerate(tqdm(loader)):
            img = data['img'].to(device)
            text = data['text']
            meta = data['meta'].to(device)
            meta[:, i] = 0
            label = data['label']

            out = model(img, text, meta)

            out = out.cpu().detach().squeeze(1).numpy().tolist()
            label = label.squeeze(1).numpy().tolist()
            pre_list.extend(out)
            label_list.extend(label)

        print("===discard meta %d result===" % i)
        ablation_result.append(print_output_part(label_list, pre_list))

    # get model out
    get_output = True
    if get_output:
        model_out = test(args, model, loader)
        ablation_result.append(model_out)
    ablation_result = np.array(ablation_result)
    write_to_csv("ablation_result.csv", ablation_result)


# def get_label(args, data):
#     if args.label_type =
#         data['label']


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_data(args)

    if args.model_choose == 0:
        model = bili_lstm1(args.seq_len)

    elif args.model_choose == 1:

        model = youtube_lstm3(args.seq_len, args.batch_size)
        cb_model = build_reg_model(args, "train")

    if args.test or args.ablation:
        args.continue_model = True

    if args.continue_model:
        # args.ckpt_name = os.listdir(args.ckpt_path)[0]
        # model = torch.load(os.path.join(args.ckpt_path, args.ckpt_name))
        model_dict = torch.load(os.path.join(args.ckpt_path, args.ckpt_name))
        model.load_state_dict(model_dict)
        print('Loaded model')

    model = model.to(device)


    if args.train:
        train(args, (model, cb_model), train_loader, val_loader)
    elif args.test:
        test(args, model, test_loader)
    elif args.ablation:
        ablation_study(args, model, test_loader)
    else:
        print("please choose the mode")
