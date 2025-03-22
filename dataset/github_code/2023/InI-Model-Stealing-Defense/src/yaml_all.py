import argparse
import yaml
import os

dataset_dict = {
    "cifar10": "../data/cifar10",
    "cifar100": '../data/cifar100',
    "mnist": "../data/",
    "kmnist": "../data/",
    "emnist": "../data/",
    "emnistletters": "../data/",
    "fashionmnist": "../data",
    "svhn": "../data/svhn",
    "imagenet_tiny": "../data/imagenet_tiny"
}

def main():

    parser = argparse.ArgumentParser(description='write_yaml')
    parser.add_argument("--path", type=str, required=True)

    # basic
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    # defense
    parser.add_argument('--defense', type=str, required=True)
    ## nd
    parser.add_argument('--nd_model', type=str, default='resnet18')
    parser.add_argument('--nd_epochs', type=int, default=150)
    parser.add_argument('--nd_lr', type=float, default=0.1)
    parser.add_argument('--nd_step_size', type=int, default=50)
    parser.add_argument('--nd_lr_gamma', type=float, default=0.1)
    ## ini
    parser.add_argument('--ini_model', type=str, default='resnet18')
    parser.add_argument('--ini_dataset_ood', type=str, default='imagenet_tiny')
    parser.add_argument('--ini_student_iter', type=int, default=0)
    parser.add_argument('--ini_victim_iter', type=int, default=1)
    parser.add_argument('--ini_student_lr', type=float, default=0.1)
    parser.add_argument('--ini_victim_lr', type=float, default=0.1)
    parser.add_argument('--ini_epochs', type=int, default=150)
    parser.add_argument('--ini_lr_step', type=int, default=50)
    parser.add_argument('--ini_momentum', type=float, default=0.5)
    parser.add_argument('--ini_weight_decay', type=float, default=0.001)
    parser.add_argument('--ini_scheduler', type=str, default='steplr')
    parser.add_argument('--ini_clip_grad', default=True)
    parser.add_argument('--ini_max_grad_norm', type=int, default=10)
    parser.add_argument('--ini_loss', type=str, default='l1')
    parser.add_argument('--ini_grad_loss', type=str, default='kl')
    parser.add_argument('--ini_use_ie', default=False, action='store_true')
    parser.add_argument('--ini_use_pcgrad', default=False, action='store_true')
    parser.add_argument('--ini_train_student', default=False, action='store_true')
    parser.add_argument('--ini_beta', type=float, default=0)
    parser.add_argument('--ini_gamma', type=float, default=0)
    parser.add_argument('--ini_delta', type=float, default=0)
    parser.add_argument('--ini_epsilon', type=float, default=0)
    parser.add_argument('--ini_save_interval', type=int, default=10)
    parser.add_argument('--ini_student_path', type=str, default=None)

    # attack
    parser.add_argument('--attack', type=str, required=True)
    parser.add_argument('--attack_checkpoint_path', type=str, default=None)
    parser.add_argument('--pred_type', type=str, default='soft')

    ## knockoff
    parser.add_argument('--knockoff_model', type=str, default='resnet18')
    parser.add_argument('--knockoff_budget', type=int, default=50000)
    parser.add_argument('--knockoff_lr_S', type=float, default=0.1)
    parser.add_argument('--knockoff_scheduler', type=str, default='cosine')
    parser.add_argument('--knockoff_weight_decay', type=float, default=0.0005)
    parser.add_argument('--knockoff_momentum', type=float, default=0.9)
    parser.add_argument('--knockoff_epochs', type=int, default=50)
    parser.add_argument('--knockoff_disable_pbar', action='store_true', default=True)
    parser.add_argument('--knockoff_adaptive_mode', type=str, default='none')
    parser.add_argument('--knockoff_n_adaptive_queries', type=int, default=5)
    parser.add_argument('--knockoff_sur_dataset', type=str, default='cifar100')
    ## jbda jbda-tr
    parser.add_argument('--jbda_model', type=str, default='resnet18')
    parser.add_argument('--jbda_lr_S', type=float, default=0.1)
    parser.add_argument('--jbda_scheduler', type=str, default='cosine')
    parser.add_argument('--jbda_steps', type=list, default=[0.1, 0.3, 0.5])
    parser.add_argument('--jbda_scale', type=float, default=0.3)
    parser.add_argument('--jbda_weight_decay', type=float, default=0.0005)
    parser.add_argument('--jbda_momentum', type=float, default=0.9)
    parser.add_argument('--jbda_num_seed', type=int, default=150)
    parser.add_argument('--jbda_aug_rounds', type=int, default=6)
    parser.add_argument('--jbda_epochs', type=int, default=10)
    parser.add_argument('--jbda_mode', type=str, default='jbda')
    parser.add_argument('--jbda_lmbda', type=float, default=0.1)

    args = parser.parse_args()

    basic = {
        "dataset": args.dataset,
        "dataset_root": dataset_dict[args.dataset],
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    if args.defense == 'nd':
        defense = {
            "defense": "nd",
            "defense_args": {
                "model": args.nd_model,
                "exp_path": args.path,
                "epochs": args.nd_epochs,
                "lr": args.nd_lr,
                "step_size": args.nd_step_size,
                "lr_gamma": args.nd_lr_gamma
            }
        }
    elif args.defense == 'ini':
        defense = {
            "defense": "ini",
            "defense_args": {
                "model": args.ini_model,
                "dataset_ood": args.ini_dataset_ood,
                "dataset_ood_root": dataset_dict[args.ini_dataset_ood],
                "student_iter": args.ini_student_iter,
                "victim_iter": args.ini_victim_iter,
                "student_lr": args.ini_student_lr,
                "victim_lr": args.ini_victim_lr,
                "epochs": args.ini_epochs,
                "lr_step": args.ini_lr_step,
                "momentum": args.ini_momentum,
                "weight_decay": args.ini_weight_decay,
                "scheduler": args.ini_scheduler,
                "clip_grad": args.ini_clip_grad,
                "max_grad_norm": args.ini_max_grad_norm,
                "loss": args.ini_loss,
                "grad_loss": args.ini_grad_loss,
                "use_ie": args.ini_use_ie,
                "use_pcgrad": args.ini_use_pcgrad,
                "train_student": args.ini_train_student,
                "beta": args.ini_beta,
                "gamma": args.ini_gamma,
                "delta": args.ini_delta,
                "epsilon": args.ini_epsilon,
                "save_interval": args.ini_save_interval,
                "exp_path": args.path,
                "student_path": args.ini_student_path
            }
        }

    if args.attack == 'knockoff':
        attack = {
            "attack": "knockoff",
            "attack_args": {
                "checkpoint_path": args.attack_checkpoint_path,
                "model": args.knockoff_model,
                "budget": args.knockoff_budget,
                "lr_S": args.knockoff_lr_S,
                "scheduler": args.knockoff_scheduler,
                "weight_decay": args.knockoff_weight_decay,
                "momentum": args.knockoff_momentum,
                "epochs": args.knockoff_epochs,
                "pred_type": args.pred_type,
                "disable_pbar": args.knockoff_disable_pbar,
                "adaptive_mode": args.knockoff_adaptive_mode,
                "n_adaptive_queries": args.knockoff_n_adaptive_queries,
                "sur_dataset": args.knockoff_sur_dataset,
                "sur_dataset_root": dataset_dict[args.knockoff_sur_dataset]
            }
        }
    elif args.attack == 'jbda':
        attack = {
            "attack": "jbda",
            "attack_args": {
                "checkpoint_path": args.attack_checkpoint_path,
                "model": args.jbda_model,
                "lr_S": args.jbda_lr_S,
                "scheduler": args.jbda_scheduler,
                "steps": args.jbda_steps,
                "scale": args.jbda_scale,
                "weight_decay": args.jbda_weight_decay,
                "momentum": args.jbda_momentum,
                "num_seed": args.jbda_num_seed,
                "aug_rounds": args.jbda_aug_rounds,
                "epochs": args.jbda_epochs,
                "mode": args.jbda_mode,
                "lmbda": args.jbda_lmbda,
                "pred_type": args.pred_type,
                "train_dataset": args.dataset,
                "train_dataset_root": dataset_dict[args.dataset]
            }
        }
    
    data = basic
    data.update(defense)
    data.update(attack)
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    with open(f"{args.path}/config.yaml","w") as f:
        yaml.safe_dump(data=data, stream=f, allow_unicode=True)



if __name__ == '__main__':
    main()
