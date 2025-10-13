# import warnings
# warnings.simplefilter("error")
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
# from visualizer import get_local

from helpers import init_helper, data_helper, vsumm_helper
from model.transfomer_with_window import Transformer
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
logger = logging.getLogger()
def get_encoder_decoder( num_feature, **kwargs):
    num_head = kwargs['num_head']
    enlayer= kwargs['enlayers']
    delayer = kwargs['delayers']

    wid_size = kwargs['window_size']
    stride = kwargs['stride']
    length = kwargs['length']
    attention_mode  = kwargs['attention_mode']

    dff = kwargs['dff']
    return Transformer(T = 0,
                       dim_in=num_feature,
                       heads=num_head,
                         enlayers=enlayer,
                        delayers=delayer,
                         dim_mid=64,
                         length =length,
                         window_size=wid_size,
                         attention_mode = attention_mode,
                         stride=stride,
                         dff = dff)

def get_model(model_type, **kwargs):
    if model_type == 'encoder-decoder':
        return get_encoder_decoder(**kwargs)
    else:
        raise ValueError(f'Invalid model type {model_type}')
def compute_ovp_youtube_fscore(gt_score,pre_score):
    overlap = (pre_score & gt_score).sum()
    precision = overlap / pre_score.sum()
    recall = overlap / gt_score.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

def evaluate(args, model, val_loader):

    model.eval()
    stats = data_helper.AverageMeter('fscore')

    with torch.no_grad():

        for test_key, seq,seqdiff, gt_score, cps, n_frames, nfps, picks, user_summary,gt_summary in val_loader:

            seq = torch.as_tensor(seq,dtype=torch.float32).unsqueeze(0).to(args.device)


            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gt_score, cps, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            target1 = seq.squeeze(0)[target]

            global_idxa = cps[:, 0]  # GA 3
            global_idxb = cps[:, 1]
            idx_mid = (global_idxa + global_idxb) // 2
            global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
            global_idx = np.concatenate((global_idx, idx_mid))


            out, _, _, _, = model(seq,target1, global_idx)


            pred_summ1 = torch.zeros(len(target))

            a, b = out.shape
            for j in range(b):
                column = out[:, j]
                min_value = torch.min(column)
                max_value = torch.max(column)
                for i in range(a):
                    if column[i] == max_value and max_value == torch.max(out[i, :]):
                        pred_summ1[j] = max_value
                        break
                else:
                    pred_summ1[j] = min_value


            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            keyshot_summ_pred = vsumm_helper.get_keyshot_summ(pred_summ1, cps, n_frames, nfps, picks)
            fscore = vsumm_helper.get_summ_f1score(keyshot_summ_pred,user_summary,eval_metric)


            stats.update(fscore=fscore)


    return stats.fscore


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(args)
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)

            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)

            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)



            fscore = evaluate(args, model, val_loader)
            stats.update(fscore=fscore)

            logger.info(f'{split_path.stem} split {split_idx}: , F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: '
                    f'F-score: {stats.fscore:.4f}')

if __name__ == '__main__':
    main()
