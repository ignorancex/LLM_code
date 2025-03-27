import os
import argparse
import numpy as np
import pprint
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_time
import wandb
import copy
import pickle 
import random
from reparam_module import ReparamModule
from datetime import datetime

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.iters + 1, args.eval_it).tolist()

    # Load the dataset
    channel, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)
    args.im_size = im_size
    print(f"Image size: {im_size}")

    wandb.init(sync_tensorboard=False,
               project="MKDT",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)

    # Organize the Real Dataset
    ''' organize the real dataset '''
    
    images_all = []
    
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    images_all = torch.cat(images_all, dim=0).to("cpu").detach()
    labels_all = torch.load(args.train_labels_path, map_location="cpu").detach()
    
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(idx):  # get random n images from class c
        return images_all[idx]

    ''' initialize the synthetic data '''
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    with open(args.image_init_idx_path, "rb") as f:
        img_init_idx = pickle.load(f)
    image_syn = get_images(img_init_idx)
    print("Shape of the synthetic images: ", image_syn.shape)
    label_syn = copy.deepcopy(labels_all[img_init_idx])
    print("Shape of the synthetic image labels: ", label_syn.shape)

    if args.batch_syn is None:
        args.batch_syn = len(image_syn)
        
    ''' training '''
    # The image, the labels, and the learning rate are all differentiable
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    label_syn = label_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_labels = torch.optim.SGD([label_syn], lr=args.lr_labels, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = None
    if args.criterion == "mse":
        criterion = nn.MSELoss().to(args.device)
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)

    # Extract the buffers
    expert_dir = args.expert_dir
    print("Expert Dir: {}".format(expert_dir))

    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    file_idx = 0
    expert_idx = 0
    random.shuffle(expert_files)
    buffer = torch.load(expert_files[file_idx], map_location="cpu")
    random.shuffle(buffer) # Only necessary when max_experts (as in MTT is used). We do not use max_experts here. 

    # Get the current date and time
    current_datetime = datetime.now()
    RUN_ID = current_datetime.strftime("%Y-%m-%d_%H:%M:%S") + str(args.run_id)
    save_dir = os.path.join(".", "logged_files", args.dataset, RUN_ID)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "hparams.txt"), "w") as f:
        f.write(pprint.pformat(vars(args), indent=4))
        
    print('%s training begins'%get_time())
    pbar = tqdm(range(0, args.iters), desc="Distilling")
    for it in pbar:
        log_it = it + 1
        if (it == 0) or (log_it in eval_it_pool):
            with torch.no_grad():
                image_save = image_syn.cuda()
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(log_it)))
                # visualize
                # grid = make_grid(image_save.clone(), nrow=100)
                # wandb.log({"images": wandb.Image(grid.detach().cpu())}, step=it)
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(log_it)))
                torch.save(syn_lr.cpu(), os.path.join(save_dir, "synlr_{}.pt".format(log_it)))
                print("Synthetic LR", syn_lr.item())
        
        student_net = get_network(args.model, channel, label_syn.shape[1], im_size, dist=False).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
        expert_trajectory = buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(expert_files):
                file_idx = 0
                random.shuffle(expert_files)
            # print("loading file {}".format(expert_files[file_idx]))
            random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(image_syn))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = image_syn[these_indices]
            this_y = y_hat[these_indices]

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            mse_loss = criterion(x, this_y)

            grad = torch.autograd.grad(mse_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss
        optimizer_img.zero_grad()
        optimizer_labels.zero_grad()
        optimizer_lr.zero_grad()
        
        grand_loss.backward()
        
        avg_grad = torch.mean(torch.abs(image_syn.grad)).item()
        optimizer_img.step()
        optimizer_labels.step()
        optimizer_lr.step()

        pbar.set_postfix({"Grand_Loss": grand_loss.detach().cpu().item(),
                   "Start_Epoch": start_epoch, 
                   "Avg Grad": avg_grad})
        for _ in student_params:
            del _
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_it', type=int, default=1000, help='how often to evaluate')

    parser.add_argument('--iters', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-04, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=1e-1, help='initialization for synthetic learning rate')

    parser.add_argument('--batch_syn', type=int, default=256, help='should only use this if you run out of VRAM')

    parser.add_argument('--data_path', type=str, default='/data', help='dataset path')

    parser.add_argument('--expert_epochs', type=int, default=5, help='how many expert epochs the target params are') # M but in epochs from paper
    parser.add_argument('--syn_steps', type=int, default=40, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=5, help='max epoch we can start at')
    
    parser.add_argument('--lr_labels', type=float, default=1, help='learning rate for updating labels')
    parser.add_argument('--image_init_idx_path', type=str, help='path to the initial images')
    parser.add_argument('--train_labels_path', type=str, required=True, help='path to the target representation')
    parser.add_argument('--run_id', type=str, default=None, help='id for run')
    parser.add_argument('--expert_dir', type=str, help='dir for expert trajectories')
    parser.add_argument('--criterion', type=str, default="mse") # Should always use MSE

    args = parser.parse_args()

    main(args)

