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
import logging
import builtins
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--warm_start_epoch', type=int, default=0)  # Set to 10 for testing the trained model.
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', default=1e-2, type=float)

parser.add_argument('--images_dir', type=str, default='../data_source/img_yt')      # Set the path of images.
parser.add_argument('--gt_path', type=str, default="0")
parser.add_argument('--data_files', type=str, default="../data_source/basic_view_pn.csv")  # Set the path of data set.

parser.add_argument('--seq_len', type=int, default=29) # Set the count of days you want to predict.

parser.add_argument('--ckpt_path', type=str, default="ckpt_with_lai")

parser.add_argument('--result_file', type=str, default='all_result.csv') #Set the file to save the results
parser.add_argument('--write', type=bool, default=True)

parser.add_argument('--train', type=bool, default=False)  
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--K_fold', type=int,default=0)
parser.add_argument('--use_mlp', type=bool, default=False) 

class CustomLoss(nn.Module):
    def __init__(self, initial_lambda1=1.0, initial_lambda2=1.0, initial_weight=1, epsilon=1e-6, base_loss=nn.SmoothL1Loss(0.1)):
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss
        self.initial_lambda1 = initial_lambda1
        self.initial_lambda2 = initial_lambda2
        self.lambda1 = initial_lambda1
        self.lambda2 = initial_lambda2
        self.epsilon = epsilon
        self.initial_weight = initial_weight
        self.peak_weight = initial_weight

    def update_weights(self, current_step, total_steps):
        # Update weights using cosine annealing strategy
        self.lambda1 = self.initial_lambda1 * (0.5 * (1 + torch.cos(torch.tensor(current_step / total_steps * 3.141592653589793))))
        self.lambda2 = self.initial_lambda2 * (0.5 * (1 + torch.cos(torch.tensor(current_step / total_steps * 3.141592653589793))))
        self.peak_weight = self.initial_weight * (0.5 * (1 + torch.cos(torch.tensor(current_step / total_steps * 3.141592653589793))))

    def forward(self, outputs, targets, current_step, total_steps):
        # Update weights
        self.update_weights(current_step, total_steps)

        # basical loss
        base_loss = self.base_loss(outputs, targets)

        # Detect peaks and create one-hot vectors
        peak_indices_outputs = torch.argmax(outputs, dim=1)
        peak_indices_targets = torch.argmax(targets, dim=1)
        
        one_hot_outputs = F.one_hot(peak_indices_outputs, num_classes=outputs.size(1)).float()
        one_hot_targets = F.one_hot(peak_indices_targets, num_classes=targets.size(1)).float()

        # Calculating L1 loss
        peak_loss = F.l1_loss(one_hot_outputs, one_hot_targets)

        # Calculate the first derivative
        first_derivative = outputs[:, 1:] - outputs[:, :-1]
        target_first_derivative = targets[:, 1:] - targets[:, :-1]
        first_derivative_loss = self.base_loss(first_derivative, target_first_derivative)

        # Calculate the second derivative
        second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
        target_second_derivative = target_first_derivative[:, 1:] - target_first_derivative[:, :-1]
        second_derivative_loss = self.base_loss(second_derivative, target_second_derivative)

        # Adding Laplace smoothing
        laplacian_smoothing = self.epsilon * (
            torch.sum(torch.abs(first_derivative)) + torch.sum(torch.abs(second_derivative))
        )

        l2_reg = torch.tensor(0.0).to(outputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        # Calculate the total loss
        total_loss = base_loss + self.peak_weight * peak_loss + \
                     self.lambda1 * first_derivative_loss + \
                     self.lambda2 * second_derivative_loss + \
                     laplacian_smoothing

        return total_loss

    def get_lambda_values(self):
        return self.lambda1, self.lambda2

def load_data(args, K, n):
    random_seed = 23
    random_generator = random.Random(random_seed)
    # data_files = args.data_files.rstrip("_for_early.csv") + f"_by_lai{n}.csv"
    data_files = args.data_files
    print(data_files)
    # data loading
    data_set = youtube_data_lstm(data_files, args.images_dir, args.gt_path)
    batch_size = args.batch_size

    # Calculate the size of each fold
    fold_size = len(data_set) // K

    # Create a list of random indices
    indices = list(range(len(data_set)))
    
    random_generator.shuffle(indices)

    # Calculate the index range of the validation set and test set of the current fold
    val_start = n * fold_size
    train_start = (n + 1) * fold_size

    # Select the index of the validation set
    val_indices = indices[val_start:train_start]
    test_indices = val_indices

    # The index of the training set is the rest
    train_indices = [i for i in indices if i not in val_indices]

    # Create a subset based on an index
    train_set = Subset(data_set, train_indices)
    val_set = Subset(data_set, val_indices)
    test_set = Subset(data_set, test_indices)

    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    return train_loader, val_loader, test_loader


# train
def train(args, model, train_loader, val_loader):
    logging.basicConfig(filename=os.path.join(args.ckpt_path, f'train_{args.K_fold}.log'), level=logging.INFO)
    loss_fn = CustomLoss()
    lr = args.lr
    weight_decay = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1, verbose=True, min_lr=0)
    min_mae = 10
    total_steps = len(train_loader) * args.epochs
    for epoch in range(args.epochs):
        batch_train_losses = []
        model.train()
        # training
        preds = []
        labels = []
        out_f_list = []
        for num, data in enumerate(tqdm(train_loader)):

            img = data['img'].to(device)
            text = data['text'].to(device) if isinstance(data['text'], torch.Tensor) else data['text']
            meta = data['meta'].to(device)
            cat = data['cat'].to(device) if isinstance(data['cat'], torch.Tensor) else data['cat']
            label = data['label'].to(device)

            optimizer.zero_grad()

            out, _ = model(img, text, meta, cat)
            current_step = epoch * len(train_loader) + num

            train_loss = loss_fn(out, label,current_step,total_steps)
            batch_train_losses.append(train_loss.item())


            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            current_step = epoch * len(train_loader) + num

            img = data['img'].to(device)
            text = data['text']
            meta = data['meta'].to(device)
            cat = data['cat']

            label = data['label'].to(device)

            out, _ = model(img, text, meta, cat)

            val_loss = loss_fn(out, label,current_step,total_steps)
            batch_val_losses.append(val_loss.item())

            for i in range(out.shape[0]):
                preds.append(out[i].cpu().detach().numpy().tolist())
                val_labels.append(label[i].cpu().detach().numpy().tolist())
            # if args.write:
            #     with open(args.result_file, 'w', newline='', encoding='UTF-8-sig') as f:
            #         for i in range(len(out)):
            #             new_lines = [data['id'][i], out[i].cpu().detach().numpy().tolist(), label[i].cpu().detach().numpy().tolist()]
            #             writer = csv.writer(f)
            #             writer.writerow(new_lines)

        avg_val_loss = round(sum(batch_val_losses) / len(batch_val_losses), 5)
        scheduler.step(avg_val_loss)
        mae = mean_absolute_error(val_labels, preds)
        print('=====Epoch %d averaged val loss: %.6f=====' % (epoch + 1, avg_val_loss))
        print('=====Epoch %d val result=====' % (epoch + 1))
        out_print = print_output_seq(val_labels, preds)

        torch.cuda.empty_cache()

        logging.info('=====Epoch %d averaged training loss: %.6f=====' % (epoch + 1, avg_train_loss))
        logging.info('=====Epoch %d val result=====' % (epoch + 1))
        logging.info(out_print)

        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler.step(avg_val_loss)
        # print("recent learning rate:%.4f" % lr)
        # lr = min(0.001, lr * 0.8)
        # write result

        count=0
        if mae < min_mae:
            min_mae=mae
            count=0
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'{args.K_fold}-%d-%.4f.pth' % (epoch + 1, mae)))  # .state_dict()
            print('Saved model. Testing...')  # .state_dict()
        else:
            count+=1
            if count>=5:
                break

# test
def test(args, model, test_loader):
    model.eval()
    output_path = args.ckpt_path + '/' + args.result_file

    # save result
    if args.write:
        with open(output_path, 'w') as f:
            pass

    preds = []
    labels = []
    count = 0

    for num, data in enumerate(tqdm(test_loader)):

        img = data['img'].to(device)
        text = data['text']
        meta = data['meta'].to(device)
        cat = data['cat']

        label = data['label'].to(device)

        with torch.no_grad():
             out, _ = model(img, text, meta, cat)

        count += 1
        # print(out)
        for i in range(out.shape[0]):
            preds.append(out[i].cpu().detach().numpy().tolist())
            labels.append(label[i].cpu().detach().numpy().tolist())
        # write result
        if args.write:
            with open(output_path, 'a+', newline='', encoding='UTF-8-sig') as f:
                for i in range(out.shape[0]):
                    new_lines = [data['id'][i], out[i].cpu().detach().numpy().tolist(), label[i].cpu().detach().numpy().tolist()]
                    writer = csv.writer(f)
                    writer.writerow(new_lines)
    print_output_seq(labels, preds)
    # return print_output_seq(labels, preds)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_data(args, 5, args.K_fold)

    if args.use_mlp is True:
        model = youtube_MLP(args.seq_len, args.batch_size)
    else:
        model = youtube_lstm3(args.seq_len, args.batch_size)

    if args.test:
        import glob
        # model_files = glob.glob(os.path.join(args.ckpt_path, str(args.K_fold) + "*.pth"))[0]
        # model_dict = torch.load(model_files)
        model_files = os.path.join(args.ckpt_path, "../ckpt_with_lai/K-epoch-mae.pth")   #set the path of the trained model you want to test.
        model_dict = torch.load(model_files)
        model.load_state_dict(model_dict)
        print('Loaded model ' + model_files)

    model = model.to(device)

    if args.train:
        train(args, model, train_loader, val_loader)
    elif args.test:
        test(args, model, test_loader)
    else:
        print(r"please choose 'train' or 'test'")
