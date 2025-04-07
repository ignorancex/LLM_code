import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from smp_model import *
from smp_data import *
from tqdm import tqdm
import csv
from tools import *
from transformers import logging
import warnings
import random
from cb import *
import logging
import builtins

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--warm_start_epoch', type=int, default=0)  # Set to 10 for testing the trained model.
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', default=1e-3, type=float)

parser.add_argument('--images_dir', type=str, default='img_bili')
parser.add_argument('--gt_path', type=str, default="0")
parser.add_argument('--train_files', type=str, default="train/train1230_30seq.csv")
parser.add_argument('--test_files', type=str, default="test/test1230_30seq.csv")

parser.add_argument('--seq_len', type=int, default=30)

parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--ckpt_name', type=str, default='best-39-1.6959.pth')
parser.add_argument('--result_file', type=str, default='all_result.csv')
parser.add_argument('--write', type=bool, default=False)

parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--K_fold', type=int)


def load_data(args):
    train_set = bili_data_lstm(args.train_files, args.images_dir)
    test_set = bili_data_lstm(args.test_files, args.images_dir)

    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    return train_loader, val_loader, test_loader

# train
def train(args, model, train_loader, val_loader):
    logging.basicConfig(filename=os.path.join(args.ckpt_path, f'train_{args.K_fold}.log'), level=logging.INFO)
    loss_fn = nn.MSELoss()
    lr = args.lr
    weight_decay = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1, verbose=True, min_lr=0)
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

            out = model(img, text, meta)

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

        avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses), 5)
        print('=====Epoch %d averaged training loss: %.6f=====' % (epoch + 1,  avg_train_loss))
        print('=====Epoch %d train result=====' % (epoch + 1))
        out_print = print_output_seq(labels, preds)

        logging.info('=====Epoch %d averaged training loss: %.6f=====' % (epoch + 1, avg_train_loss))
        logging.info('=====Epoch %d train result=====' % (epoch + 1))
        logging.info(out_print)

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
            out = model(img, text, meta)

            val_loss = loss_fn(out, label)
            batch_val_losses.append(val_loss.item())

            for i in range(out.shape[0]):
                preds.append(out[i].cpu().detach().numpy().tolist())
                val_labels.append(label[i].cpu().detach().numpy().tolist())

        avg_val_loss = round(sum(batch_val_losses) / len(batch_val_losses), 5)
        scheduler.step(avg_val_loss)
        mae = mean_absolute_error(val_labels, preds)
        print('=====Epoch %d averaged val loss: %.6f=====' % (epoch + 1, avg_val_loss))
        print('=====Epoch %d val result=====' % (epoch + 1))
        out_print = print_output_seq(val_labels, preds)

        logging.info('=====Epoch %d averaged training loss: %.6f=====' % (epoch + 1, avg_train_loss))
        logging.info('=====Epoch %d val result=====' % (epoch + 1))
        logging.info(out_print)

        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler.step(avg_val_loss)
        # print("recent learning rate:%.4f" % lr)
        # lr = min(0.001, lr * 0.8)

        if mae < min_mae + 0.01:
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'{args.K_fold}-%d-%.4f.pth' % (epoch + 1, mae)))  # .state_dict()
            print('Saved model. Testing...')  # .state_dict()

        if mae < min_mae:
            min_mae = mae


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

        out, _ = model(img, text, meta)

        count += 1
        # print(out)
        for i in range(out.shape[0]):
            preds.append(out[i].cpu().detach().numpy().tolist())
            labels.append(label[i].cpu().detach().numpy().tolist())
        # write result
        # if args.write:
        #     with open(args.result_file, 'a+', newline='', encoding='UTF-8-sig') as f:
        #         for i in range(len(out)):
        #             new_lines = [data['id'][i], out[i].cpu().detach().numpy().tolist(), label[i].cpu().detach().numpy().tolist()]
        #             writer = csv.writer(f)
        #             writer.writerow(new_lines)
    print_output_seq(labels, preds)

    # return print_output_seq(labels, preds)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_data(args)

    model = bili_lstm1(args.seq_len, args.batch_size)

    if args.test:
        # model_dict = torch.load(os.path.join(args.ckpt_path, args.ckpt_name))
        import glob
        model_files = glob.glob(os.path.join(args.ckpt_path, str(args.K_fold) + "*.pth"))
        model_dict = torch.load(model_files[0])

        model.load_state_dict(model_dict)
        print('Loaded model')

    model = model.to(device)

    if args.train:
        train(args, model, train_loader, val_loader)
    elif args.test:
        test(args, model, test_loader)
    else:
        print(r"please choose 'train' or 'test'")
