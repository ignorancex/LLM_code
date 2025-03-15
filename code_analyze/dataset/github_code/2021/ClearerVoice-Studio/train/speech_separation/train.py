import yamlargparse, os, random
import numpy as np

import torch
from dataloader.dataloader import get_dataloader
from solver import Solver

import sys
sys.path.append('../../')

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, init_method='env://', world_size=args.world_size)

    from networks import network_wrapper
    model = network_wrapper(args).ss_network
    model = model.to(device)

    if (args.distributed and args.local_rank ==0) or args.distributed == False:
        print("started on " + args.checkpoint_dir + '\n')
        print(args)
        #print(model)
        #print("\nTotal number of model parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print("\nTotal number of model parameters: {} \n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
    
    if args.network in ['MossFormer2_SS_16K','MossFormer2_SS_8K']:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)
    else:
        print(f'in Main, {args.network} is not implemented!')
        return

    train_sampler, train_generator = get_dataloader(args,'train')
    _, val_generator = get_dataloader(args, 'val')
    if args.tt_list is not None:
        _, test_generator = get_dataloader(args, 'test')
    else:
        test_generator = None
    args.train_sampler=train_sampler

    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator
                ) 
    solver.train()


if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser("Settings")
    
    # Log and Visulization
    parser.add_argument('--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile) 

    # experiment setting
    parser.add_argument('--mode', type=str, default='train', help='run train or inference')
    parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/MossFormer2_SS_16K',help='the checkpoint dir')
    parser.add_argument('--network', type=str, default='frcrn', help='the model network types to be loaded for speech enhancment: MossFormer2_SS_16K, MossFormer2_SS_8K')
    parser.add_argument('--train_from_last_checkpoint', type=int, help='0 or 1, whether to train from a pre-trained checkpoint, includes model weight, optimizer settings')
    parser.add_argument('--init_checkpoint_path', type=str, default = None, help='pre-trained model path for initilizing the model weights for a new training')
    parser.add_argument('--print_freq', type=int, default=10, help='No. steps waited for printing info')
    parser.add_argument('--checkpoint_save_freq', type=int, default=50, help='No. steps waited for saving new checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size')

    # dataset settings
    parser.add_argument('--load-type', dest='load_type', type=str, help='training data format: one_input_one_output, one_input_multi_outputs')
    parser.add_argument('--tr-list', dest='tr_list', type=str, help='the train data list')
    parser.add_argument('--cv-list', dest='cv_list',type=str, help='the cross-validation data list')
    parser.add_argument('--tt-list', dest='tt_list',type=str, default=None, help='optional, the test data list')
    parser.add_argument('--accu_grad', type=int, help='whether to accumulate grad')
    parser.add_argument('--max_length', type=int, help='max_length of mixture in training')
    parser.add_argument('--num_workers', type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000)
    parser.add_argument('--load_fbank', type=int, default=None, help='calculate and load fbanks for inputs')
    # model
    parser.add_argument('--num-spks', dest='num_spks', type=int, default=2)
    parser.add_argument('--encoder_kernel-size', dest='encoder_kernel_size', type=int, default=16,
                        help='the Conv1D kernel size of encoder ')
    parser.add_argument('--encoder-embedding-dim', dest='encoder_embedding_dim', type=int, default=512,
                        help='the encoder output embedding size')
    parser.add_argument('--mossformer-squence-dim', dest='mossformer_sequence_dim', type=int, default=512,
                        help='the feature dimension used in MossFormer block')
    parser.add_argument('--num-mossformer_layer', dest='num_mossformer_layer', type=int, default='24',
                        help='the number of mosssformer layers used for sequence processing') 

    # optimizer
    parser.add_argument('--effec_batch_size', type=int, help='effective Batch size')
    parser.add_argument('--max-epoch', dest='max_epoch',type=int,default=20,help='the max epochs')
    parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='the num gpus to use')
    parser.add_argument('--init_learning_rate',  type=float, help='Init learning rate')
    parser.add_argument('--finetune_learning_rate',  type=float, help='Finetune learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument('--clip-grad-norm', dest='clip_grad_norm', type=float, default=10.)
    parser.add_argument(
        '--loss-threshold', dest='loss_threshold', type=float, default=-9999.0, help='the mimum loss threshold') 
    # Distributed training
    parser.add_argument("--local-rank", dest='local_rank', type=int, default=0)

    args, _ = parser.parse_known_args()

    # check for single- or multi-GPU training
    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
    assert torch.backends.cudnn.enabled, "cudnn needs to be enabled"
    main(args)




