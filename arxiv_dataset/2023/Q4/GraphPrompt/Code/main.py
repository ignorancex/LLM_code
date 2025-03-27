from dataset.dataset import *
import logging
import argparse
from trainer import SimpleTrainer


# set up seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    if not os.path.exists(name):
        os.makedirs(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--filename', type=str, default='data/datasets/cl.obo')
    parser.add_argument('--use_text_preprocesser', action='store_true', default=False)
    parser.add_argument('--is_unseen', action='store_true', default=False)
    parser.add_argument('--pretrained_model', type=str, default='dmis-lab/biobert-base-cased-v1.2')
    parser.add_argument('--exp_path', type=str, default='../exp/simple/')

    parser.add_argument('--max_seq_len', type=int, default=60)

    parser.add_argument('--eval_k', type=int, default=10)

    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--syn_ratio', type=float, default=0.5)
    parser.add_argument('--ent_ratio', type=float, default=0.25)
    parser.add_argument('--hrt_ratio', type=float, default=0.25)
    parser.add_argument('--cst_ratio', type=float, default=0.25)
    parser.add_argument('--path_depth', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_scheduler', action='store_true', default=False)
    parser.add_argument('--emb_init_std', type=float, default=1e-3)
    parser.add_argument('--use_get_ent_emb', action='store_true', default=False)
    parser.add_argument('--pretrain_emb_iter', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=0)  # default: no max grad norm

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-model', action='store_true', default=False)

    args = parser.parse_args()
    args = args.__dict__

    torch.cuda.set_device(eval(args['gpu_ids']))

    logger = setup_logger(name=args['exp_path'], log_file=os.path.join(args['exp_path'], 'log.log'))
    args['logger'] = logger
    print(args)
    import sys;

    sys.stdout.flush()

    assert args['hrt_ratio'] > 0 or args['path_depth'] == 1 and args['hrt_ratio'] == 0

    setup_seed(args['seed'])

    b = SimpleTrainer(args)
    ###################
    # b.load()
    ###################
    b.eval(b.valid_dataset, epoch=0)
    b.train()
    accu_1, accu_k, pack = b.eval(b.test_dataset, epoch=-1, return_output=True)

    path = args['exp_path']
    acc_file = os.path.join(path, 'test_acc1.txt')
    if os.path.exists(acc_file):
        last_acc1 = float(open(acc_file).readline())
        if accu_1 < last_acc1:
            print('not best acc')
            exit(2)

    print(float(accu_1), file=open(acc_file, 'w'))
    if args['save_model']:
        b.save()
        torch.save(pack, os.path.join(path, 'pack.bin'))
