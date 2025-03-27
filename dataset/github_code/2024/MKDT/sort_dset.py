import argparse
import torch
import os
from tqdm import tqdm
from utils import get_dataset, get_network, build_trainset
from tqdm import tqdm
import pickle
from reparam_module import ReparamModule
import numpy as np


def main(args):
    channel, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)
    train_labels_path = f"/home/jennyni/MKDT/target_rep/{args.ssl_algo}/{args.dataset}_target_rep_train.pt"

    trainloader, labels_all = build_trainset(dataset=args.dataset,
                                 dst_train=dst_train,
                                 train_labels_path=train_labels_path, 
                                 channel=channel, 
                                 batch_train=args.batch_train,
                                 shuffle=False
                                )
    epoch_losses = {}

    num_examples = len(trainloader.dataset)
    print(num_examples)
    # [each example's loss in the first epoch, averaged over all the buffers]
    epoch_losses = np.zeros(num_examples)

    for num_buffer in range(1, args.num_buffers + 1):
        checkpoint_path = f'/home/jennyni/MKDT/buffers_{args.ssl_algo}/{args.dataset}/{args.model}/replay_buffer_{num_buffer-1}.pt'
        param_all = torch.load(checkpoint_path)
        first_epoch_checkpoint = param_all[0][1] # Load the checkpoint from the first epoch
        first_epoch_checkpoint = torch.cat([p.data.to(args.device).reshape(-1) for p in first_epoch_checkpoint], 0)

        teacher_net = get_network(args.model, channel, labels_all.shape[1], im_size).to(args.device)
        teacher_net = ReparamModule(teacher_net)

        for i, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            with torch.no_grad():
                outputs = teacher_net(inputs, flat_param=first_epoch_checkpoint)
                mse_losses = torch.mean((outputs - targets) ** 2, dim=1)
                individual_losses = mse_losses.detach().cpu().numpy()
                epoch_losses[i * args.batch_train:(i + 1) * args.batch_train] += individual_losses

    epoch_losses /= args.num_buffers

    sorted_indices = np.argsort(epoch_losses)[::-1]

    print("Sorted indices of high loss subset:", sorted_indices)

    dset_name = (args.dataset).lower() if args.dataset != "Tiny" else "tiny_imagenet"

    if args.dataset != "ImageNet":
        top_2_percent_idx = sorted_indices[:int(0.02 * num_examples)]
        top_5_percent_idx = sorted_indices[:int(0.05 * num_examples)]
        
        output_dir = f'/home/jennyni/MKDT/init/{dset_name}'
        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/{args.dataset}_{args.ssl_algo}_2_high_loss_indices.pkl', 'wb') as f:
            pickle.dump(top_2_percent_idx, f)
        with open(f'{output_dir}/{args.dataset}_{args.ssl_algo}_5_high_loss_indices.pkl', 'wb') as f:
            pickle.dump(top_5_percent_idx, f)
        
    else:
        top_1000 = sorted_indices[:1000]
        
        # Create the directory if it doesn't exist
        output_dir = f'/home/jennyni/MKDT/init/{dset_name}'
        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/{args.dataset}_{args.ssl_algo}_1000_high_loss_indices.pkl', 'wb') as f:
            pickle.dump(top_1000, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sort Dataset Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_buffers', type=int, default=100, help='number of buffers')
    parser.add_argument('--ssl_algo', type=str, default="barlow_twins", choices=['barlow_twins', 'simclr'], help='Algorithm to train the SSL')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    args = parser.parse_args()
    main(args)