import logging
import torch
import numpy as np

from model.transfomer_with_window  import Transformer
from model.losses import cul_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper

import time

logger = logging.getLogger()

def train(args, split, save_path, spt_idx):

    model = Transformer(
        T=0,
        dim_in=args.num_feature,
        heads=args.num_head,
        enlayers=args.enlayers,
        delayers=args.delayers,
        dim_mid=args.dim_mid,

        length=args.length,
        window_size=args.window_size,
        stride=args.stride,
        attention_mode=args.attention_mode,
        dff=args.dff)

    print('FullTransNet mode total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = model.to(args.device)

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay,)

    max_val_fscore = -1

    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)



    for epoch in range(args.max_epoch):
        model.train()

        for train_keys, seq, seqdiff, gtscore, change_points, n_frames, nfps, picks, user_sum, gt_summary in train_loader:

            video_loss = []
            # print(train_keys)
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks)

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            target = vsumm_helper.downsample_summ(keyshot_summ)
            summ_feature = seq.squeeze(0)[target]

            if not target.any():
                continue


            global_idxa = change_points[:, 0]  # GA 3
            global_idxb = change_points[:, 1]
            idx_mid = (global_idxa + global_idxb) // 2
            global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
            global_idx = np.concatenate((global_idx, idx_mid))



            pred_summ,  enc_self_attns, dec_self_attns, dec_enc_attns= model(seq,summ_feature,global_idx)

            loss = cul_loss(pred_summ, target, args.loss, args.smooth)
            video_loss.append(loss.item())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        val_fscore = evaluate(args, model, val_loader)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))



        dataset = str(args.splits).split('/')[-1].split('.')[0]
        logger.info(
            f' {dataset}:{spt_idx} '
            f'Epoch: {epoch}/{args.max_epoch} '

            f'test F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')


    return max_val_fscore, model