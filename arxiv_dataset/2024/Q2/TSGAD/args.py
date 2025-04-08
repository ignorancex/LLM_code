import os
import pickle
import time
import argparse


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)


def init_sub_args(args):
    # dataset = "UBnormal" if args.dataset == "UBnormal" else "ShanghaiTech"
    dataset = args.dataset
    if args.vid_path_train and args.vid_path_test and args.pose_path_train and args.pose_path_test:
        args.vid_path = {'train': args.vid_path_train,
                         'test': args.vid_path_test}

        args.pose_path = {'train': args.pose_path_train,
                          'test': args.pose_path_test}
    else:
        args.vid_path = {'train': os.path.join(args.data_dir, dataset, 'train/images/'),
                         'test':  os.path.join(args.data_dir, dataset, 'test/frames/')}

        args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'train/'),
                          'test':  os.path.join(args.data_dir, dataset, 'pose', 'test/')}
    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None
    model_args = args_rm_prefix(args, 'model_')
    return args, model_args


def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser(prog="STG-NF")
    # General Args
    parser.add_argument('--vid_path_train', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train_abnormal', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train', type=str, default=None, help='Path to training pose')
    parser.add_argument('--vid_path_test', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path_test', type=str, default=None, help='Path to test pose')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal', 'c1', 'c2', 'c3', 'c4', 'combined', 'corridor', 'Avenue', 'CPCC', 'CPCC_X_setup',
                                 'CPCC0','CPCC1','CPCC2','CPCC3','CPCC4','CPCC5','CPCC6'], help='Dataset for Eval')
    parser.add_argument('--vid_res', type=str, default=None, help='Video Res')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999, help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=32, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vid', type=int, default=0, help='Plot test videos')
    parser.add_argument('--only_test', action='store_true', help='Visualize train/test data')
    parser.add_argument('--mask_root', type=str, help='Path to the gt masks.')

    # Data Params
    parser.add_argument('--num_transform', type=int, default=2, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1], help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=1, metavar='PNS', choices=[0, 1], help='Scale keeping proportions [1/0] (default: 1)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=24, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=12, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')

    # Model Params
    parser.add_argument('--model_ckpt_dir', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--model_save_dir', type=str, default= './results', metavar='model', help="Path to a pretrained model")
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--score_save_dir', type=str, default= './results/scores', metavar='score', help="Where to save the scores?")
    parser.add_argument('--batch_size', type=int, default=256,  metavar='B', help='Batch size for train')
    parser.add_argument('--epochs', '-model_e', type=int, default=20, metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--model_optimizer', '-model_o', type=str, default='adamx', metavar='model_OPT', help = "Optimizer")
    parser.add_argument('--model_lr', type=float, default=5e-10, metavar='LR', help='Optimizer Learning Rate Parameter')
    parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--model_latent_dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--model_dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('--model_confidence', action='store_true', help='Create Figs')
    parser.add_argument('--model_beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--model_lambda', default=0, type=float, help='ELBO penalty term')
    parser.add_argument('--model_alpha', default=1, type=float, help='MSE multiplier')
    parser.add_argument('--model_gamma', default=1, type=float, help='ELBO multiplier')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--model_beta-anneal', action='store_true')
    parser.add_argument('--model_lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--encoder_decoder_type', type=str, default='convolutional', choices=['convolutional', 'transformer_based'])
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DROPOUT', help='Dropout training Parameter (default: 0.3)')
    parser.add_argument('--conv_oper', type=str, default='sagc', metavar='CONV_OPER',help="Convolutional Operator to be used [sagc, gcn] (default: 'sagc')")
    parser.add_argument('--act', type=str, default='relu', metavar='ACT_TYPE', help="Activation used in ST-GCN [relu, mish] (default: 'relu')")
    parser.add_argument('--sched', type=str, default='tri', metavar='AE_SCH', help="Optimization LR scheduler (default: 'tri')")
    parser.add_argument('--mse', action='store_true', help='Adds MSE loss to the elbo loss')

    return parser


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            model_key = key[len(prefix):]
            wp_args_dict[model_key] = value

    return wp_args


def create_exp_dirs(experiment_dir, dirmap=''):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


def save_dataset(dataset, fname):
    with open(fname, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)

