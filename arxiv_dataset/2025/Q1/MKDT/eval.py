import os
import random
import argparse
import pickle
import wandb
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
from utils import get_dataset, get_network, SoftLabelDataset
from pretrain_methods import pretrain_mse
from evaluation.linear_evaluation import le_run

def aggregate_results(results):
    aggregated_results = {}
    for dataset in results[0].keys():
        dataset_results = np.array([result[dataset] for result in results])
        mean = np.mean(dataset_results, axis=0)
        std = np.std(dataset_results, axis=0)
        aggregated_results[dataset] = {'mean': mean.tolist(), 'std': std.tolist()}
            
    return aggregated_results

def main(args):
    target_datasets = ["CIFAR100", "CIFAR10", "aircraft", "cub2011", "dogs", "flowers", "Tiny"]

    wandb.init(
        project="mkdt_data_distillation_evaluation",
        config=args
    )

    final_res = {}
    for td in target_datasets:
        args.test_dataset = td
        res_list = []
        # device
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)

        # seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # data
        train_img_channel, train_img_size, num_pretrain_classes, dst_train_pretrain, _ = get_dataset(dataset=args.train_dataset, data_path=args.data_path)
        eval_img_channel, eval_img_size, eval_num_classes, eval_dst_train, eval_dst_test = get_dataset(args.test_dataset, args.data_path, subset_size=args.subset_frac, epc=args.epc)
        
        # Used for test (linear evaluation and finetuning)
        dl_tr = torch.utils.data.DataLoader(eval_dst_train, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        dl_te = torch.utils.data.DataLoader(eval_dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("Test training data: ", len(eval_dst_train))

        labels_all = torch.load(args.label_path, map_location="cpu")

        if args.result_dir is None: # Undistilled images baseline
            if args.subset_path is not None:
                with open(args.subset_path, "rb") as f:
                    subset_idx = pickle.load(f)
            
            else:
                subset_idx = None # Full dataset
                
            dst_train_pretrain = SoftLabelDataset(dst_train_pretrain, labels_all, subset_idx)
            print(f"Number of images for pretrain: {len(dst_train_pretrain)}")
            dl_syn = torch.utils.data.DataLoader(dst_train_pretrain, batch_size=args.pre_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            # visualize
            all_images = torch.stack([img for img, _ in dst_train_pretrain], dim=0)
            grid = make_grid(all_images.clone(), nrow=100)
            wandb.log({"undistilled images": wandb.Image(grid.detach().cpu())})
            del all_images
        

        else: # Distilled images
            # Synthetic images
            try:
                if args.use_krrst:
                    x_syn = torch.load(f"{args.result_dir}/x_syn.pt") 
                    x_syn = x_syn.detach().cpu()
                else: # MKDT
                    x_syn = torch.load(f"{args.result_dir}/images_{args.distilled_steps}.pt")
                    x_syn = x_syn.detach().cpu()
            except:
                raise ValueError("No synthetic images found. ")
            # Synthetic LR
            try:
                syn_lr = torch.load(f"{args.result_dir}/synlr_{args.distilled_steps}.pt") 
                args.pre_lr = syn_lr.detach().cpu()
            except:
                print("No syn lr found, fall back to original pre_lr")
            # Synthetic representations
            try:
                if args.use_krrst:
                    y_syn = torch.load(f"{args.result_dir}/y_syn.pt")
                    y_syn = y_syn.detach().cpu()
                else: # MKDT
                    y_syn = torch.load(f"{args.result_dir}/labels_{args.distilled_steps}.pt")
                    y_syn = y_syn.detach().cpu()
            except:
                raise ValueError("No synthetic representations found. ")
            
            assert x_syn.requires_grad == False and y_syn.requires_grad == False
            assert x_syn.grad is None and y_syn.grad is None
            assert y_syn.shape[-1] == labels_all.shape[1]

            x_syn, y_syn = x_syn.detach(), y_syn.detach()
            x_syn, y_syn = x_syn.to(device), y_syn.to(device)

            dst_train_pretrain = TensorDataset(x_syn, y_syn)
            print(f"Number of images for pretrain: {len(dst_train_pretrain)}")
            dl_syn = DataLoader(dst_train_pretrain, batch_size=args.pre_batch_size, shuffle=True, num_workers=0)

            # visualize
            grid = make_grid(x_syn.clone(), nrow=100)
            wandb.log({"distilled images": wandb.Image(grid.detach().cpu())})

        num_target_features = labels_all.shape[1]
        print("Pretraining LR: ", args.pre_lr)

        # Random encoder perf
        if args.no_pretrain:
            random_model = get_network(args.train_model, eval_img_channel, num_target_features, train_img_size, fix_net=True).to(device)
            rd_acc = le_run(
                        train_model=args.train_model, 
                        channel=eval_img_channel, 
                        num_classes=eval_num_classes,
                        img_size=eval_img_size,
                        device=device, 
                        init_model=random_model, 
                        dl_tr=dl_tr, 
                        dl_te=dl_te,
                        le_iters=args.le_iters,
                        seed=args.seed
                    )
            wandb.log({f"{td}_random_accuracy": rd_acc})
            del random_model
            res_list.append(rd_acc)
        
        # With pretraining
        else:
            final_model = pretrain_mse(
                                    train_model=args.train_model,
                                    channel=train_img_channel, 
                                    num_classes=num_target_features,
                                    img_size=train_img_size,
                                    device=device, 
                                    dl_syn=dl_syn,
                                    pre_opt=args.pre_opt,
                                    pre_lr=args.pre_lr,
                                    pre_wd=args.pre_wd,
                                    pre_epoch=args.pre_epoch,
                                )
            final_acc = le_run(
                                train_model=args.train_model, 
                                channel=eval_img_channel, 
                                num_classes=eval_num_classes,
                                img_size=eval_img_size,
                                device=device, 
                                init_model=final_model, 
                                dl_tr=dl_tr, 
                                dl_te=dl_te,
                                le_iters=args.le_iters,
                                seed=args.seed
                            )
        
            del final_model
            wandb.log({f"{td}_subset_final_accuracy": final_acc})
            res_list.append(final_acc)

        final_res[td] = res_list

    print(final_res)
    wandb.log(final_res)
    wandb.finish(quiet=True)

    return final_res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Parameter Processing')

    # Config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    # Data
    parser.add_argument('--data_path', type=str, default="/data")
    parser.add_argument('--train_dataset', type=str, default="CIFAR100")
    parser.add_argument('--test_dataset', type=str, default="CIFAR100")

    # Label
    parser.add_argument('--label_path', type=str, required=True)

    # Path for the subset / distilled data
    parser.add_argument('--subset_path', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)

    # Model
    parser.add_argument('--train_model', type=str, default="ConvNet")

    # Args for logging 
    parser.add_argument("--results_csv", default="mkdt_results.csv")
    
    # Hparms for pretrain
    parser.add_argument('--pre_opt', type=str, default="sgd") 
    parser.add_argument('--pre_epoch', type=int, default=20)
    parser.add_argument('--pre_batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.1)
    parser.add_argument('--pre_wd', type=float, default=1e-4)
    parser.add_argument('--distilled_steps', type=int, default=1000) 
    parser.add_argument('--use_krrst', action="store_true")
    parser.add_argument('--no_pretrain', action="store_true")

    # Hparms for test
    parser.add_argument('--le_iters', type=int, default=20) 
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--subset_frac', type=float, default=None) # fraction of the subset
    parser.add_argument('--epc', type=int, default=None) # examples per class

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        args = parser.parse_args()
        args.seed = seed
        results.append(main(args))

    final_aggregated_results = aggregate_results(results)
    print(final_aggregated_results)

    curr_results_df = pd.DataFrame(index=[0])
    curr_results_df["timestamp"] = pd.Timestamp.now()
    args_dict = vars(args)
    for key in args_dict:
        try:
            curr_results_df[key] = args_dict[key]
        except:
            print(f"Couldn't save {key}")
        
    for dataset, stats in final_aggregated_results.items():
        mean = stats['mean'][0]
        std = stats['std'][0]
        print(f"{dataset}: {mean:.2f} ± {std:.2f}")
        curr_results_df[dataset] = "$" + f"{mean:.2f}" + "\scriptstyle{ \pm " + f"{std:.2f}" + "}" + "$"
    
    print(curr_results_df)  
    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
    else:
        results_df = pd.DataFrame()

    results_df = pd.concat([results_df, curr_results_df], ignore_index=True)
    results_df.to_csv(args.results_csv, index=False)

    # wandb.log(final_aggregated_results)
    wandb.finish(quiet=True)