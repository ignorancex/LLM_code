from __future__ import print_function
import argparse
import random
import torch
import torchvision
import sparselearning
import models.vanilla_cnn
from models.vanilla_cnn import *
from models import cifar_resnet, initializers, vgg, convolution
from models.mlps import MLP_CIFAR10, MLP_CIFAR10_DROPOUT, STUPID_MLP_CIFAR10
from data_handling.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, str2bool, get_ImageNet_loaders, get_tinyimagenet_dataloaders, get_higgs_dataloaders
from models.conv_cifar10 import SmallConvNet_CIFAR10
from models.mlps import MLP_Higgs
from models.lenet import LeNet_300_100
from data_handling.logger import *
from trainer import run_testing, run_training, resume, run_eval, run_pruning
from setup_utils import get_mask, get_optimizer
from models.imagenet_resnet import build_resnet
from models.vgg_plain import VGG16
from sparselearning.weight_preprocesser import get_weight_processer
from models.mlps import MLP


def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # training_and_eval
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=str2bool, default="false",
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=["adam", "sgd"],
                        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='mnist', choices=["mnist", "cifar10", "higgs", "tiny"])
    parser.add_argument('--fp16', type=str2bool, default="false", help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    # parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='', help='model to use. Options: mlp_cifar10, conv_cifar10, mlp_higgs, ...')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    # other
    parser.add_argument('--bench', type=str2bool, default="true",
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max_threads', type=int, default=10, help='How many threads to use for data loading.')
    # saving and logging
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to store the logs')
    parser.add_argument('--save_dir', type=str, default='./save', help='where to store other results')
    parser.add_argument('--verbose', type=str2bool, default="true", help="toggle the verbose mode")
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_features', type=str2bool, default="false",
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--save_checkpoint', type=str2bool, default="false")

    # sparse
    parser.add_argument('--scaled', type=str2bool, default="false", help='scale the initialization by 1/density')
    parser.add_argument('--use_wandb', type=str2bool, default="true")
    parser.add_argument('--save_locally', type=str2bool, default="true")
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--tag', type=str, help="experiment type. Used only for wandb")
    parser.add_argument('--opt_order', type=str, choices=["before", "after"], default="before")
    parser.add_argument('--manual_stop', type=str2bool, default="false", help="if true, will automatically stop the "
                                                                              "training after first pruning")
    parser.add_argument("--distributed", type=str2bool, default="false")
    parser.add_argument('--init_type', type=str, default="default",
                        choices=['binary', 'kaiming_normal', 'scaled_kaiming_normal', 'kaiming_uniform', 'orthogonal',
                                 'conv_orthogonal', 'delta_orthogonal', 'bimodal_kaiming_normal', 'default'])
    parser.add_argument('--step1', type=int)
    parser.add_argument('--step2', type=int)
    parser.add_argument('--end_pruning', type=str2bool, default="false")
    parser.add_argument('--double_precision', type=str2bool, default="false")

    # mlp
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--activation', type=str, default="tanh", choices=["linear", "tanh", "relu", "selu", "hard_tanh"])

    #cnn
    parser.add_argument('--channel_width', type=int, default=128)

    # ortho and DI
    parser.add_argument('--weight_processer', type=str, default="none", choices=["AI", "sao", "sparse_orthogonal", "sparse_fan_in", "none"])
    parser.add_argument('--log_preprocessing', type=str2bool, default='false')
    parser.add_argument('--record_jacobian', type=str2bool, default='false')
    parser.add_argument('--AI_iters', type=int, default=10000)
    parser.add_argument('--sigma_w', type=float, default=1)
    parser.add_argument('--sigma_b', type=float, default=0)
    parser.add_argument('--q_star', type=float, default=1)
    parser.add_argument('--more_nonzeros', type=str2bool, default='false')
    parser.add_argument('--log_every_iter', type=str2bool, default="false")

    #sao
    parser.add_argument('--degree', type=int)
    parser.add_argument('--same_mask', type=str2bool, default='false')

    parser.add_argument('--log-dpl-and-exit', type=str2bool, default='false')

    sparselearning.core.add_sparse_args(parser)
    return parser


def main(args):

    if args.sparse_init.endswith("AI"):
        args.weight_processer = "AI"
    if args.sparse_init.endswith("EI"):
        args.weight_processer = "sparse_orthogonal"
    if args.sparse_init.endswith("SAO"):
        args.weight_processer = "sao"
    if args.sparse_init.endswith("EIS"):
        args.weight_processer = "structured_sparse_orthogonal"

    logger = setup_logger(args)
    if args.verbose:
        print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.no_cuda:
        assert torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.verbose:
        print_and_log('\n\n')
        print_and_log('=' * 80)

    if args.death not in ["magnitude", "Random", "SET", "SETFixed", "RunningMagnitude"]:
        args.opt_order = "after"
    if args.manual_stop:
        args.opt_order = "after"

    fix_seeds(args)

    output, test_loader, train_loader, valid_loader = get_data(args)
    model = get_model(args, device, output)

    if args.weight_processer == "sao":
        from sao.init_calls import Delta_Init, Linear_Init
        if not args.sparse:
            assert args.degree == args.channel_width, "Asking for a dense model but performing sao with degree < channel_width (so a sparse model). Collision - changed one of the variables to either perform dense or sparse training" 
        sparsity = 1-args.density if args.degree is None else None
        print("Calling SAO with degree {} and sparsity {}".format(args.degree, sparsity))

        if "mlp" in args.model:
            model = Linear_Init(model, method="SAO", gain=args.sigma_w, sigma_b=args.sigma_b, sparsity=sparsity, degree=args.degree,
                activation=args.activation, in_channels_0 = 3, num_classes=output)
        else:
            model = Delta_Init(model, method="SAO", gain=args.sigma_w, sigma_b=args.sigma_b, sparsity=sparsity, degree=args.degree,
                activation=args.activation, in_channels_0 = 3, num_classes=output)
                
        model.sao_called=True
        args.sparse=False # To prevent calling the masks
        for name, burren in model.named_buffers():
            print(name, torch.count_nonzero(burren)/torch.numel(burren))
    else:
        model.sao_called=False

    if args.verbose:
        info_beginning(args, model)
        print_and_log(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = get_optimizer(args, model)
    if args.step1 is not None and args.step2 is None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[args.step1],
                                                            last_epoch=-1)
    elif args.step1 is not None and args.step2 is not None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[args.step1, args.step2],
                                                            last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(args.epochs / 2) * args.multiplier,
                                                                        int(args.epochs * 3 / 4) * args.multiplier],
                                                            last_epoch=-1)
    if args.resume:
        resume(args, device, model, optimizer, test_loader)
    if args.fp16:
        model, optimizer = setup_fp16(args, model, optimizer)

    mask = get_mask(args, model, optimizer, train_loader, device)

    # create output file
    save_subfolder = get_output_file(args)

    print('Densities after initialization')
    if mask is not None:
        densities = mask.print_density()
        if args.log_dpl_and_exit:
            metrics = {"layer_densities":densities}
            logger.log(metrics)
            logger.save()
            logger.finish()
            exit(0)

    if args.double_precision:
        #go back to 32 precison, since "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Double'
        model = model.to(torch.float32)
        mask.double_precision = False
    best_acc = 0.0


    preprocesser = get_weight_processer(args, model, mask, logger, output, device)
    preprocesser()
    print('Densities after preprocessing')
    if mask is not None:
        if args.degree is None:
            mask.print_density(check=True)  # also verifies if there's no target / actual density mismatch
        else:
            mask.print_density(check=False)

    run_training(args, best_acc, device, lr_scheduler, mask, model, optimizer, save_subfolder, train_loader,
                 valid_loader, logger)
    run_eval(args, device, model, save_subfolder, test_loader, logger)
    # run_testing(args, device, model, save_subfolder, test_loader, logger)

    if args.end_pruning and not args.sparse:
        densities = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        args.sparse_init = "global_magnitude"
        args.sparse = True
        metric_dict={}
        for d in densities:
            d_model = get_model(args, device, output)
            d_model.load_state_dict(model.state_dict())
            args.density = d
            mask = get_mask(args, d_model, optimizer, train_loader, device)
            mask.apply_mask()
            test_loss, test_acc = run_pruning(args, device, d_model, test_loader)
            metric_dict["p_{}_eval_loss".format(args.density)] = test_loss
            metric_dict["p_{}_eval_acc".format(args.density)] = test_acc
        logger.log_no_step_dict(metric_dict)



def setup_fp16(args, model, optimizer):
    if args.verbose:
        print('FP16')
    optimizer = FP16_Optimizer(optimizer,
                               static_loss_scale=None,
                               dynamic_loss_scale=True,
                               dynamic_loss_args={'init_scale': 2 ** 16})
    model = model.half()
    return model, optimizer


def get_output_file(args):
    # save_path = os.path.join(args.save_dir,
    #                          os.path.join(str(args.model),
    #                                       os.path.join(str(args.data),
    #                                                        os.path.join(str(args.sparse_init), str(args.seed)))))
    # if args.sparse:
    #     save_subfolder = os.path.join(save_path, 'sparsity' + str(1 - args.density))
    # else:
    #     save_subfolder = os.path.join(save_path, 'dense')
    save_subfolder = args.save_dir
    if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)
    return save_subfolder

def get_model(args, device, output):

    last = "logits" if "GraSP" in args.death else "logsoftmax"
    if args.model == 'resnet50':
        last = 'logits'
    if args.data == "higgs":
        last = 'logits'
        print("Last layer output type (data=higgs always uses logits):", last)
    if args.record_jacobian:
        last = 'logits'
        print("If recording jacobian, always use:", last)
    if args.scaled:
        init_type = 'scaled_kaiming_normal'
    else:
        init_type = args.init_type
    if "vgg-like" == args.model:
        model = VGG16("like", num_classes=10, last=last, actv_fn=args.activation,
                      init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed)).to(device)
    elif "vgg-C" == args.model:
        model = VGG16("C", num_classes=10, last=last, actv_fn=args.activation,
                      init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed)).to(device)
    elif "vgg-16-pytorch" == args.model:
        model = torchvision.models.vgg16_bn(num_classes=10).to(device)
        model.last = "logits"
    elif 'vgg' in args.model:
        #model = vgg.VGG(depth=int(args.model[-2:]), dataset=args.data, batchnorm=True, last=last).to(device)
        model = vgg.VGG(depth=int(args.model[-2:]), dataset=args.data, batchnorm=True, actv_fn=args.activation, last=last,
                        init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed)).to(device)
    elif 'mlp_cifar10' == args.model:
        model = MLP_CIFAR10(last=last, init_weights=initializers.initializations(init_type,args.density, args, seed=args.seed)).to(device)
    elif 'mlp_cifar10_dropout' == args.model:
        model = MLP_CIFAR10_DROPOUT(last=last, density=args.density).to(device)
    elif 'stupid_mlp' == args.model:
        model = STUPID_MLP_CIFAR10(last=last, init_weights=initializers.initializations(init_type,args.density, args, seed=args.seed)).to(device)
    elif 'mlp' == args.model:
        model = MLP(output=output, depth=args.depth, width=args.width, actv_fn=args.activation, last="logits",
                    init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed)).to(device)
    elif 'resnet50' == args.model:
        model = build_resnet('resnet50', 'classic')
    elif 'conv_cifar10' in args.model:
        model = SmallConvNet_CIFAR10(last=last).to(device)
    elif args.model == 'mlp_higgs':
        model = MLP_Higgs().to(device)
    elif 'cifar_resnet' in args.model:
        model = cifar_resnet.Model.get_model_from_name(args.model,
                                                       initializers.initializations(init_type, args.density, args, seed=args.seed),
                                                       outputs=output, actv_fn=args.activation, last=last).to(device)
    elif 'conv' in args.model:
        model = convolution.CifarConv(output=output, depth=args.depth, actv_fn=args.activation, last="logits",
            init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed), width=args.channel_width).to(device)
    elif 'van' in args.model:
        model = models.vanilla_cnn.__dict__[args.model](
            c=args.channel_width, num_classes=output, activation=args.activation, last="logits",
            init_weights=initializers.initializations(init_type, args.density, args, seed=args.seed)
        ).to(device)
    elif 'efficientnet-b0' in args.model:
        model = torchvision.models.efficientnet_b0(num_classes=output)
        model = model.to(device)
        model.last = "logits"
        init_weights = initializers.initializations(init_type, args.density, args, seed=args.seed)
        model.apply(init_weights)
    elif 'efficientnet-b1' in args.model:
        model = torchvision.models.efficientnet_b1(num_classes=output)
        model = model.to(device)
        model.last = "logits"
        init_weights = initializers.initializations(init_type, args.density, args, seed=args.seed)
        model.apply(init_weights)
    elif 'efficientnet-b3' in args.model:
        model = torchvision.models.efficientnet_b3(num_classes=output)
        model = model.to(device)
        model.last = "logits"
        init_weights = initializers.initializations(init_type, args.density, args, seed=args.seed)
        model.apply(init_weights)
    elif 'efficientnet-v2-s' in args.model:
        model = torchvision.models.efficientnet_v2_s(num_classes=output)
        model = model.to(device)
        model.last = "logits"
        init_weights = initializers.initializations(init_type, args.density, args, seed=args.seed)
        model.apply(init_weights)
    else:
        raise ValueError("Unknown model {}".format(args.model))
    model.output_dim = output

    if args.double_precision:
        model.to(torch.float64)
        print("Using double precision")

    return model



def get_data(args):
    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        output = 10
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                          max_threads=args.max_threads)
        output = 10
    elif args.data == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                           max_threads=args.max_threads)
        output = 100
    elif args.data == 'imagenet':
        print ("WARNING: valid and test are the same dataset in this implementation")
        train_loader, valid_loader, test_loader = get_ImageNet_loaders(args, distributed=False)
        output = 1000
    elif args.data == 'tiny':
        print ("WARNING: valid and test are the same dataset in this implementation")
        train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(args)
        output = 200
    elif args.data == 'higgs':
        train_loader, valid_loader, test_loader = get_higgs_dataloaders(args)
        output = 1  # binary classification, just one output is needed
    else:
        raise ValueError("Unknown dataset")
    return output, test_loader, train_loader, valid_loader


def fix_seeds(args):
    # fix random seed for Reproducibility
    torch.backends.cudnn.benchmark = args.bench
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def parse_args_default(args=None):
    parser = get_parser()
    return parser.parse_args(args=args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
