import argparse
import os
import torch
from exp.exp_supervised import Exp_Supervised
from exp.exp_finetune import Exp_Finetune
from exp.exp_pretrain.exp_pretrain_lead import Exp_Pretrain_LEAD
from exp.exp_pretrain.exp_pretrain_moco import Exp_Pretrain_MOCO
from exp.exp_pretrain.exp_pretrain_ts2vec import Exp_Pretrain_TS2Vec
from exp.exp_pretrain.exp_pretrain_biot import Exp_Pretrain_BIOT
from exp.exp_pretrain.exp_pretrain_eeg2rep import Exp_Pretrain_EEG2Rep
import random
import numpy as np
from utils.tools import compute_avg_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LEAD')

    # basic config
    parser.add_argument('--method', type=str, required=True, default='LEAD',
                        help='Overall method (combinations of task_name, model, model_id) name, '
                             'options: [LEAD, MOCO, Transformer, TCN]')
    parser.add_argument('--task_name', type=str, required=True, default='supervised',
                        help='task name, options:[supervised, pretrain_lead, pretrain_moco, finetune]')
    parser.add_argument('--model', type=str, required=True, default='LEAD',
                        help='backbone model name, options: [Transformer, TCN, LEAD]')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Single-Dataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of all dataset folders')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument("--pretraining_datasets", type=str, default="ADSZ,APAVA-19,ADFSU,AD-Auditory,TDBRAIN-19,TUEP,REEG-PD-19",
                        help="List of datasets folder names for pretraining (No overlapping with downstream datasets).")
    parser.add_argument("--training_datasets", type=str, default="ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19",
                        help="List of datasets folder names for pretraining linear probe, supervised, and finetune training.")
    parser.add_argument("--testing_datasets", type=str, default="ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19",
                        help="List of datasets folder names for pretraining linear probe, supervised, and finetune validation and test.")
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/',
                        help='location of pre-trained model checkpoints')

    # model define for baselines
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=32, help='patch_len used in PatchTST, BIOT')
    parser.add_argument('--stride', type=int, default=8, help='stride used in PatchTST')

    # ADformer/LEAD params
    parser.add_argument("--patch_len_list", type=str, default="4", help="a list of patch len used in Medformer, ADformer")
    parser.add_argument("--up_dim_list", type=str, default="76",
                        help="a list of up dimension factor used in ADformer")
    parser.add_argument("--augmentations", type=str, default="flip,frequency,jitter,mask,channel,drop",
                        help="a comma-seperated list of augmentation types (none, jitter or scale). "
                             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",)
    parser.add_argument("--no_inter_attn", action="store_true",
                        help="whether to use inter-attention in encoder, "
                             "using this argument means not using inter-attention", default=False)
    parser.add_argument("--no_temporal_block", action="store_true",
                        help="whether to use temporal block in encoder", default=False)
    parser.add_argument("--no_channel_block", action="store_true",
                        help="whether to use channel block in encoder", default=False)

    # MOCO params
    parser.add_argument('--K', type=int, default=65536, help='Size of the queue in MOCO method')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum for updating the key encoder')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for the softmax in MOCO method')

    # EEG2Rpe params
    parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")

    # LEAD params
    parser.add_argument('--contrastive_loss', type=str, default='all',
                        help='contrastive loss modules enabled, options: [sample,subject,all]')


    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument("--swa", action="store_true", help="use stochastic weight averaging", default=False)
    parser.add_argument('--no_normalize', action='store_true',
                        help='do not normalize data in data loader', default=False)
    parser.add_argument('--sampling_rate', type=int, default=128, help='frequency sampling rate')
    parser.add_argument('--low_cut', type=float, default=0.5, help='low cut for bandpass filter')
    parser.add_argument('--high_cut', type=float, default=45, help='high cut for bandpass filter')
    # fixed: fixed split, mccv: monte carlo cross validation, loso: leave-one-subject-out
    parser.add_argument('--cross_val', type=str, default='fixed',
                        help='cross validation methods, options: [fixed, mccv, loso]')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'supervised':
        print("Supervised learning")
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain_lead':
        print("Pretraining with LEAD method")
        Exp = Exp_Pretrain_LEAD
    elif args.task_name == 'pretrain_moco':
        print("Pretraining with MoCo method")
        Exp = Exp_Pretrain_MOCO
    elif args.task_name == 'pretrain_ts2vec':
        print("Pretraining with TS2Vec method")
        Exp = Exp_Pretrain_TS2Vec
    elif args.task_name == 'pretrain_biot':
        print("Pretraining with BIOT method")
        Exp = Exp_Pretrain_BIOT
    elif args.task_name == 'pretrain_eeg2rep':
        print("Pretraining with EEG2Rep method")
        Exp = Exp_Pretrain_EEG2Rep
    elif args.task_name == 'finetune':
        print("Finetune")
        Exp = Exp_Finetune
    else:
        raise ValueError('task_name unknown, should be supervised, pretrain_lead, pretrain_moco, '
                         'pretrain_ts2vec, pretrain_biot, pretrain_eeg2rep or finetune.')

    total_params = 0
    sample_val_metrics_dict_list = []
    subject_val_metrics_dict_list = []
    sample_test_metrics_dict_list = []
    subject_test_metrics_dict_list = []
    if args.is_training == 1:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            # setting record of experiments
            args.seed = seed
            setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                # args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
        compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                        sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    elif args.is_training == 0:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                # args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting, test=1)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
        compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                        sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    else:
        raise ValueError('is_training should be 1 or 0, representing training or testing.')
