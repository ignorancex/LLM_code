import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import sys

from src.evaluation.evaluator import convert_to_text
from .utils import to_cuda

def _evaluate_batch_while_generate_order_beam(
        model, order_model, params, batch, lengths, positions, langs, src_lens, trg_lens, gen_type,
        separate_order_model, beam_size,
        iter_mult=1, decode_order_strategy='topk', mode='train',
):
    """Run on one example"""
    max_len, batch_size = batch.size(0), batch.size(1)

    # log probabilities of previously present tokens at each position
    #log_probs_tokens = torch.zeros(dec_len).cuda()

    #mask_tok = dico.word2id[MASK_WORD]

    selected_pos = []

    # predict all tokens depending on src/trg
    order_pred_mask = torch.zeros_like(batch).byte()
    for batch_idx in range(batch_size):
        if gen_type == "src2trg":
            order_pred_mask[src_lens[batch_idx]+1:-1,batch_idx] = 1
        elif gen_type == "trg2src":
            order_pred_mask[1:src_lens[batch_idx]-1,batch_idx] = 1

    dec_len = trg_lens.max() - 2 if gen_type == 'src2trg' else src_lens.max() - 2
    range_tensor = torch.arange(dec_len).unsqueeze(0)
    range_tensor = range_tensor.expand(batch_size, range_tensor.size(1))
    full_range_tensor = torch.arange(max_len).unsqueeze(0)
    full_range_tensor = full_range_tensor.expand(batch_size, full_range_tensor.size(1))
    if gen_type == 'src2trg':
        mask_tensor = (range_tensor < (trg_lens - 2).unsqueeze(1))
        full_mask_tensor_1 = (full_range_tensor > (src_lens).unsqueeze(1))
        full_mask_tensor_2 = (full_range_tensor < (src_lens + trg_lens - 1).unsqueeze(1))
        full_mask_tensor = full_mask_tensor_1 & full_mask_tensor_2
    else:
        mask_tensor = (range_tensor < (src_lens - 2)).unsqueeze(1)
        full_mask_tensor = (full_range_tensor < trg_lens - 1).unsqueeze(1)
        full_mask_tensor[:, 0] = False

    batch = batch.unsqueeze(2).repeat((1, 1, beam_size)).reshape(max_len, -1)
    lengths = lengths.unsqueeze(1).repeat((1, beam_size)).reshape(-1)
    positions = positions.unsqueeze(2).repeat((1, 1, beam_size)).reshape(max_len, -1)
    langs = langs.unsqueeze(2).repeat((1, 1, beam_size)).reshape(max_len, -1)
    #order_pred_mask = order_pred_mask.unsqueeze(2).repeat((1, 1, beam_size)).reshape(max_len, -1)
    full_mask_tensor = full_mask_tensor.unsqueeze(1).repeat((1, beam_size, 1)).reshape(-1, max_len)

    total_score_argmax_toks = torch.zeros(batch_size * beam_size).cuda()
    total_topbeam_scores = torch.zeros(batch_size * beam_size).cuda()

    all_order_logits, all_order_loss_masks = [], []
    topbeam_all_orders = - torch.ones((dec_len, batch_size * beam_size), dtype=torch.int).cuda() * 100
    topbeam_all_order_logits = torch.zeros((dec_len, batch_size * beam_size, max_len)).cuda()
    # do iter_mult passes over entire sentence

    #order_model.eval()

    # dec_iter_random = list(range(dec_len))
    # #np.random.shuffle(dec_iter_random)
    # dec_iter_random_1 = [4, 3, 0, 1, 2]
    # dec_iter_random_2 = [0, 4, 3, 2, 1]
    # dec_iter_random_3 = [1, 3, 0, 2, 4]
    # dec_iter_randoms = [dec_iter_random, dec_iter_random_1, dec_iter_random_2, dec_iter_random_3]

    # all_dec_iter_random = []
    # for i in range(batch_size * beam_size):
    #     all_dec_iter_random.append(torch.tensor(dec_iter_randoms[i % 4]).unsqueeze(1))
        #all_dec_iter_random.append(torch.tensor(dec_iter_random).unsqueeze(1))
        #np.random.shuffle(dec_iter_random)

    #all_dec_iter_random = torch.cat(all_dec_iter_random, dim=1)

    for dec_iter in range(iter_mult*dec_len):
        #order_logits_pred = (torch.zeros(dec_len, batch_size) - 1e30).cuda()
        tensor = order_model('fwd', x=batch.clone(), lengths=lengths, positions=positions, langs=langs, causal=False)
        _, order_logits_pred_full = order_model('predict_wo_targets', tensor=tensor, pred_mask=torch.t(full_mask_tensor.int().byte().cuda()))
        order_logits_pred_full = order_logits_pred_full - (1 - full_mask_tensor.int().cuda()) * 1e30

        #order_logits_pred = torch.t(order_logits_pred.masked_scatter_(torch.t(mask_tensor.cuda()), order_logits_pred_squashed))
        # m = Categorical(logits=order_logits_pred - (1 - mask_tensor.int().cuda()) * 1e30)
        if decode_order_strategy == 'sample':
            m = Categorical(logits=order_logits_pred_full)
            sampled_order = m.sample()
            # all_sampled_order.append(sampled_order.unsqueeze(1))
        elif decode_order_strategy == 'topk':
            topbeam_scores, topbeam_orders = torch.topk(order_logits_pred_full, beam_size) # [beam_size * batch_size, beam_size]
            #print(order_logits_pred_full)
            # all_sampled_order.append(sampled_order)
        else:
            raise NotImplementedError

        if dec_iter == 0:
            topbeam_scores_batch_view = topbeam_scores.view(batch_size, beam_size, beam_size)
            topbeam_orders_batch_view = topbeam_orders.view(batch_size, beam_size, beam_size)
            total_topbeam_scores = total_topbeam_scores + torch.diagonal(topbeam_scores_batch_view, dim1=1, dim2=2).reshape(-1)# + selected_pos_scores
            topbeam_orders = torch.diagonal(topbeam_orders_batch_view, dim1=1, dim2=2).reshape(-1)
            topbeam_all_orders[0, :] = topbeam_orders
            topbeam_all_order_logits[0, :, :] = order_logits_pred_full

        else:
            topbeam_orders_batch_view = topbeam_orders.view(batch_size, beam_size, beam_size)
            topbeam_scores_batch_view = topbeam_scores.view(batch_size, beam_size, beam_size)

            new_total_topbeam_scores = total_topbeam_scores.unsqueeze(1) + topbeam_scores # [beam_size * batch_size, beam_size]

            # sort and take beam_size highest
            new_topbeam_order_flat = new_total_topbeam_scores.reshape((batch_size, -1)).argsort()[:, -beam_size:]
            topbeam_orders = torch.zeros((batch_size * beam_size)).long().cuda()

            # create clones of the tokens and scores so far so we don't overwrite when updating
            topbeam_all_orders_clone = topbeam_all_orders.clone()
            topbeam_all_order_logits_clone = topbeam_all_order_logits.clone()
            total_topbeam_scores_clone = total_topbeam_scores.clone()
            total_score_argmax_toks_clone = total_score_argmax_toks.clone()

            batch_clone = batch.clone()
            #order_pred_mask_clone = order_pred_mask.clone()
            full_mask_tensor_clone = full_mask_tensor.clone()
            #print(new_topbeam_order_flat.reshape(-1)[:10])

            # iterate over the highest scoring beams
            for i_beam, topbeam_order_flat in enumerate(new_topbeam_order_flat.reshape(-1)):
                batch_idx = int(np.floor(i_beam / beam_size))
                topbeam_order_row = int(torch.floor(topbeam_order_flat.cpu().float() / beam_size))
                topbeam_order_col = int(topbeam_order_flat % beam_size)
                original_topbeam_pos = batch_idx * beam_size + topbeam_order_row
                #print(batch_idx, original_topbeam_pos, topbeam_order_col)

                topbeam_order = topbeam_orders_batch_view[batch_idx][topbeam_order_row][topbeam_order_col]
                topbeam_orders[i_beam] = topbeam_order
                if full_mask_tensor_clone[original_topbeam_pos][topbeam_order] == torch.tensor(False):
                    continue

                topbeam_order_logits = order_logits_pred_full[original_topbeam_pos]

                topbeam_all_orders[:, i_beam] = topbeam_all_orders_clone[:, original_topbeam_pos]
                topbeam_all_order_logits[:, i_beam] = topbeam_all_order_logits_clone[:, original_topbeam_pos]
                batch[:, i_beam] = batch_clone[:, original_topbeam_pos]
                total_topbeam_scores[i_beam] = total_topbeam_scores_clone[original_topbeam_pos] + \
                                               topbeam_scores_batch_view[batch_idx][topbeam_order_row][topbeam_order_col]
                total_score_argmax_toks[i_beam] = total_score_argmax_toks_clone[original_topbeam_pos]
                #order_pred_mask[:, i_beam] = order_pred_mask_clone[:, original_topbeam_pos]
                full_mask_tensor[i_beam, :] = full_mask_tensor_clone[original_topbeam_pos]

                if topbeam_order in topbeam_all_orders[:dec_iter, i_beam]:
                    # print(topbeam_orders_batch_view[batch_idx][topbeam_order_row])
                    print(dec_iter)
                    print(topbeam_orders[original_topbeam_pos])
                    print(topbeam_order)
                    print(full_mask_tensor_clone[original_topbeam_pos])
                    #print(torch.t(order_pred_mask_clone)[original_topbeam_pos])
                    print(order_logits_pred_full[original_topbeam_pos])
                    assert False

                topbeam_all_orders[dec_iter, i_beam] = topbeam_order
                topbeam_all_order_logits[dec_iter, i_beam, :] = topbeam_order_logits
            #topbeam_orders = topbeam_all_orders[dec_iter]

        # topbeam_orders = torch.tensor([dec_iter + 1] * (beam_size * batch_size)) \
        #                  + src_lens.unsqueeze(1).repeat((1, beam_size)).reshape(-1)
        # topbeam_orders = torch.tensor([dec_iter_random[dec_iter] + 1] * (beam_size * batch_size)) \
        #                  + src_lens.unsqueeze(1).repeat((1, beam_size)).reshape(-1)
        #topbeam_orders = all_dec_iter_random[dec_iter] + src_lens.unsqueeze(1).repeat((1, beam_size)).reshape(-1) + 1

        # create a prediction mask just for argmax pos
        pred_mask = torch.zeros_like(batch).byte()
        for beam_idx in range(batch_size * beam_size):
            batch_idx = int(np.floor(beam_idx / beam_size))
            if gen_type == "src2trg":
                # if src_lens[batch_idx]+ argmax_pos[batch_idx] + 1 < pred_mask.size(0) - 1:
                if topbeam_orders[beam_idx] < trg_lens[batch_idx] + src_lens[batch_idx] - 1:
                    assert topbeam_orders[beam_idx] < pred_mask.size(0) - 1
                    pred_mask[topbeam_orders[beam_idx], beam_idx] = 1
                else:
                    pred_mask[-1, beam_idx] = 1
            elif gen_type == "trg2src":
                if topbeam_orders[beam_idx] < src_lens[beam_idx] - 1:
                    pred_mask[topbeam_orders[beam_idx], beam_idx] = 1
                else:
                    pred_mask[src_lens[beam_idx] + 1] = 1
            else:
                sys.exit("something is wrong")

        # re-run the model on the masked batch
        if params.generation_model_cpu:
            tensor = model('fwd', x=batch.cpu(), lengths=lengths.cpu(), positions=positions.cpu(),
                           langs=langs.cpu(),
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask).cuda()
        else:
            # print(topbeam_orders)
            # print(torch.t(pred_mask)[0:10])
            # print(torch.t(batch))
            tensor = model('fwd', x=batch, lengths=lengths, positions=positions, langs=langs,
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask)
        #all_order_logits.append(order_logits_pred_full.unsqueeze(1))

        # if dec_iter < dec_len:
        #     selected_pos.append(sampled_order)
        # else:
        #     sampled_order = selected_pos[dec_iter % dec_len.numpy()]
        """
        # create a prediction mask just for argmax pos
        pred_mask = torch.zeros_like(batch).byte()
        for batch_idx in range(batch_size):
            if gen_type == "src2trg":
                #if src_lens[batch_idx]+ argmax_pos[batch_idx] + 1 < pred_mask.size(0) - 1:
                if sampled_order[batch_idx] < trg_lens[batch_idx] + src_lens[batch_idx] - 1:
                    #assert src_lens[batch_idx]+ sampled_order[batch_idx] + 1 < pred_mask.size(0) - 1
                    #pred_mask[src_lens[batch_idx]+ sampled_order[batch_idx] + 1, batch_idx] = 1
                    assert sampled_order[batch_idx] < pred_mask.size(0) - 1
                    pred_mask[sampled_order[batch_idx], batch_idx] = 1
                else:
                    pred_mask[-1, batch_idx] = 1
                #     pred_mask[src_lens[batch_idx] + argmax_pos[batch_idx] + 1, batch_idx] = 1
            elif gen_type == "trg2src":
                if sampled_order[batch_idx] < src_lens[batch_idx] - 1:
                    # pred_mask[sampled_order[batch_idx] + 1,batch_idx] = 1
                    pred_mask[sampled_order[batch_idx], batch_idx] = 1
                else:
                    pred_mask[src_lens[batch_idx] + 1] = 1
            else:
                sys.exit("something is wrong")

        # re-run the model on the masked batch
        if params.generation_model_cpu:
            tensor = model('fwd', x=batch.cpu(), lengths=lengths.cpu(), positions=positions.cpu(), langs=langs.cpu(),
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask).cuda()
        else:
            tensor = model('fwd', x=batch, lengths=lengths, positions=positions, langs=langs,
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask)
        """

        if not separate_order_model:
            scores_pred = scores_pred[0]

        # select "best" token (argmax) given that best position
        # score_argmax_tok, argmax_tok = torch.topk(scores_pred, 1, dim=-1)
        # score_argmax_tok = torch.log_softmax(score_argmax_tok, 1)
        # argmax_tok = argmax_tok.squeeze(1)
        # score_argmax_tok = score_argmax_tok.squeeze(1)

        scores_pred = torch.log_softmax(scores_pred, -1)
        score_argmax_tok, argmax_tok = torch.topk(scores_pred, 1, dim=-1)
        argmax_tok = argmax_tok.squeeze(1)
        score_argmax_tok = score_argmax_tok.squeeze(1)

        # argmax_tok = scores_pred.argmax(dim=-1)
        # score_argmax_tok = torch.log_softmax(scores_pred, 1)[:, argmax_tok]

        total_score_argmax_toks += score_argmax_tok # + score_argmax_pos)
        # substitute that token in
        order_pred_loss_mask = torch.zeros(batch_size * beam_size, 1)

        # print(torch.t(order_pred_mask)[0])
        # print(torch.sum(torch.t(order_pred_mask)[0]))
        # print(topbeam_orders)
        for beam_idx in range(batch_size * beam_size):
            batch_idx = int(np.floor(beam_idx / beam_size))
            if gen_type == "src2trg":
                if pred_mask[-1, beam_idx] != 1 and full_mask_tensor[beam_idx, topbeam_orders[beam_idx]] == torch.tensor(True):
                    batch[topbeam_orders[beam_idx], beam_idx] = argmax_tok[beam_idx]
                    #order_pred_mask[topbeam_orders[beam_idx], beam_idx] = 0
                    full_mask_tensor[beam_idx, topbeam_orders[beam_idx]] = False
                    if torch.sum(full_mask_tensor[beam_idx].int()) == 0:
                        if mode == 'train':
                            topbeam_orders[beam_idx] = -100
                    else:
                        order_pred_loss_mask[beam_idx, 0] = 1
                else:
                    topbeam_orders[beam_idx] = -100

                curr_tokens = batch[src_lens[0]+1:-1]
            elif gen_type == "trg2src":
                if pred_mask[src_lens[batch_idx] + 1] != 1 and full_mask_tensor[beam_idx, topbeam_orders[beam_idx]] == torch.tensor(True):
                    batch[topbeam_orders[beam_idx], beam_idx] = argmax_tok[beam_idx]
                    #order_pred_mask[topbeam_orders[beam_idx], beam_idx] = 0
                    full_mask_tensor[beam_idx, topbeam_orders[beam_idx]] = False
                    if torch.sum(full_mask_tensor[beam_idx].int()) == 0:
                        if mode == 'train':
                            topbeam_orders[beam_idx] = -100
                    else:
                        order_pred_loss_mask[beam_idx, 0] = 1
                else:
                    topbeam_orders[beam_idx] = -100
                curr_tokens = batch[1:src_lens[0]-1]
            else:
                sys.exit("something is wrong")

        all_order_loss_masks.append(order_pred_loss_mask)

    # print(torch.t(topbeam_all_orders)[0])
    # print(torch.t(topbeam_all_orders)[1])
    # print(torch.t(topbeam_all_orders)[2])
    # print(torch.t(topbeam_all_orders)[3])
    # print(torch.t(topbeam_all_orders)[4])

    #all_order_logits = torch.cat(all_order_logits, dim=1)
    total_score_argmax_toks = total_score_argmax_toks / ((iter_mult * dec_len))
    all_order_loss_masks = torch.cat(all_order_loss_masks, dim=1)
    #print(all_sampled_order)
    return batch, topbeam_all_orders, topbeam_all_order_logits, all_order_loss_masks, total_score_argmax_toks


def _evaluate_order_model_on_batches(
        order_model, ob, params, full_mask_tensor, lengths, positions, langs, src_lens, eval_old_policy=False
):
    order_model.train()
    #print('start')
    with torch.autograd.detect_anomaly():
        total_len, dec_len = ob.size(0), ob.size(1)
        batch = torch.transpose(ob, 1, 2).reshape(total_len, -1)
        full_mask_tensor = full_mask_tensor.reshape(-1, total_len)

        src_lens = src_lens.unsqueeze(1).repeat(1, dec_len).reshape(-1)
        lengths = lengths.unsqueeze(1).repeat(1, dec_len).reshape(-1)
        positions = positions.unsqueeze(2).repeat(1, 1, dec_len).reshape(total_len, -1)
        langs = langs.unsqueeze(2).repeat(1, 1, dec_len).reshape(total_len, -1)

        tensor = order_model(
            'fwd', x=batch, lengths=lengths, positions=positions, langs=langs, causal=False
        )
        # print(batch[:, 0])
        # print(batch[:, 1])
        # print(batch[:, 2])
        # print(full_mask_tensor[0])
        # print(full_mask_tensor[1])
        # print(full_mask_tensor[2])
        # print(positions[:, 0])
        # print(positions[:, 1])
        # print(positions[:, 2])
        # print(langs[:, 0])
        # print(langs[:, 1])
        # print(langs[:, 2])
        # exit()
        # print(batch)
        # print(full_mask_tensor)
        _, order_logits_pred_full = order_model(
            'predict_wo_targets', tensor=tensor, pred_mask=torch.t(full_mask_tensor).int().byte()
        )
        order_logits_pred_full = order_logits_pred_full - (1 - full_mask_tensor.int()) * 1e30

        state_value = order_model("predict_baseline_wo_targets", tensor=tensor, src_lens=src_lens)
        # for i in range(dec_len):
            # print(i)
            # print(order_logits_pred_full[i])
            # print(order_logits_pred_full[i+dec_len])

    if eval_old_policy:
        order_model.eval()
        tensor = order_model(
            'fwd', x=batch, lengths=lengths, positions=positions, langs=langs, causal=False
        )
        _, old_order_logits_pred_full = order_model(
            'predict_wo_targets', tensor=tensor, pred_mask=torch.t(full_mask_tensor).int().byte(),
            use_old_policy=True
        )
        old_order_logits_pred_full = old_order_logits_pred_full - (1 - full_mask_tensor.int()) * 1e30
    else:
        old_order_logits_pred_full = None
    return order_logits_pred_full, old_order_logits_pred_full, state_value


def _evaluate_batch_while_generate_order(
        model, order_model, params, batch, lengths, positions, langs, src_lens, trg_lens, gen_type,
        separate_order_model, rl_baseline,
        iter_mult=1, decode_order_strategy='sample', mode='train', predefined_order=None, restrain_action_space=False,
        top_k=1
):
    """Run on one example"""
    max_len, batch_size = batch.size(0), batch.size(1)

    total_score_argmax_toks = torch.zeros(batch_size).cuda()
    selected_pos = []

    # predict all tokens depending on src/trg
    order_pred_mask = torch.zeros_like(batch).byte()
    for batch_idx in range(batch_size):
        if gen_type == "src2trg":
            order_pred_mask[src_lens[batch_idx]+1:-1,batch_idx] = 1
        elif gen_type == "trg2src":
            order_pred_mask[1:src_lens[batch_idx]-1,batch_idx] = 1

    dec_len = trg_lens.max() - 2 if gen_type == 'src2trg' else src_lens.max() - 2
    full_range_tensor = torch.arange(max_len).unsqueeze(0)
    full_range_tensor = full_range_tensor.expand(batch_size, full_range_tensor.size(1))
    if gen_type == 'src2trg':
        full_mask_tensor_1 = (full_range_tensor > (src_lens).unsqueeze(1))
        full_mask_tensor_2 = (full_range_tensor < (src_lens + trg_lens - 1).unsqueeze(1))
        full_mask_tensor = full_mask_tensor_1 & full_mask_tensor_2
    else:
        full_mask_tensor = (full_range_tensor < (src_lens - 1).unsqueeze(1))
        full_mask_tensor[:, 0] = False

    all_sampled_order, all_order_logits, all_order_loss_masks = [], [], []
    all_state_values, all_ob, all_full_mask_tensors = [], [], []

    # Used for restrain_action_space
    log_probs_tokens = torch.zeros(max_len, batch_size).cuda()
    if gen_type == 'src2trg':
        sampled_order = src_lens.clone().cuda()
    else:
        sampled_order = torch.ones_like(src_lens).cuda()

    # do iter_mult passes over entire sentence
    for dec_iter in range(iter_mult*dec_len):
        all_ob.append(batch.clone().unsqueeze(1))
        all_full_mask_tensors.append(full_mask_tensor.clone().unsqueeze(1))

        if restrain_action_space:
            model.eval()
            # gets hidden representations
            tensor = model('fwd', x=batch.clone(), lengths=lengths, positions=positions, langs=langs, causal=False)

            # gets the predictions
            # size: trg_len x bsz x |V|
            scores_pred = model('predict_wo_targets_all_pos', tensor=tensor, pred_mask=torch.t(full_mask_tensor).int().byte().cuda())
            # if not separate_order_model:
            #     scores_pred = scores_pred[0]

            # calculate log prob and prob for entropy
            log_probs_pred = torch.log_softmax(scores_pred, dim=-1)
            probs_pred = torch.softmax(scores_pred, dim=-1)

            # calculate entropies and include normalization to put entropy and probability terms on same scale
            entropies = -(probs_pred * log_probs_pred).sum(dim=-1)
            vocab_size = scores_pred.size(-1)

            # left to right bias
            ltor_bias = torch.log((torch.abs(torch.arange(max_len).unsqueeze(1) - src_lens.unsqueeze(0)).float() / dec_len.float()).cuda())
            ltor_bias[torch.isinf(ltor_bias)] = 0

            # left to right following last sampled order
            follow_bias = dec_len.float() / (torch.arange(max_len).unsqueeze(1).cuda() - sampled_order.unsqueeze(0).float()).float()
            follow_bias[torch.isinf(follow_bias)] = 0

            # get probability distribution over positions to choose from
            l2r_positions_logits = -1 * ltor_bias
            l2r_positions_logits = torch.t(l2r_positions_logits) - (1 - full_mask_tensor.int().cuda()) * 1e30  # [bsz, seq_len]
            follow_positions_logits = 1 * follow_bias
            follow_positions_logits = torch.t(follow_positions_logits) - (1 - full_mask_tensor.int().cuda()) * 1e30  # [bsz, seq_len]
            ent_positions_logits = - 1 * entropies
            ent_positions_logits = torch.t(ent_positions_logits) - (1 - full_mask_tensor.int().cuda()) * 1e30  # [bsz, seq_len]
            prob_positions_logits = - 1 * log_probs_tokens
            prob_positions_logits = torch.t(prob_positions_logits) - (1 - full_mask_tensor.int().cuda()) * 1e30  # [bsz, seq_len]

            l2r_positions_prob = torch.softmax(l2r_positions_logits, 1)
            _, prelim_l2r_topk_pos = torch.topk(l2r_positions_prob, 2)
            follow_positions_prob = torch.softmax(follow_positions_logits, 1)
            _, prelim_follow_topk_pos = torch.topk(follow_positions_prob, 5)
            ent_positions_prob = torch.softmax(ent_positions_logits, 1)
            _, prelim_ent_topk_pos = torch.topk(ent_positions_prob, 5)
            prob_positions_prob = torch.softmax(prob_positions_logits, 1)
            _, prelim_prob_topk_pos = torch.topk(prob_positions_prob, 5)

            # Create a mask that zero-out all but the topk positions
            prelim_mask_tensor = (torch.ones(batch_size, max_len) * (- 1e30)).cuda()
            prelim_mask_tensor = prelim_mask_tensor.scatter_(1, prelim_l2r_topk_pos, torch.zeros_like(prelim_l2r_topk_pos).float().cuda())
            prelim_mask_tensor = prelim_mask_tensor.scatter_(1, prelim_follow_topk_pos, torch.zeros_like(prelim_follow_topk_pos).float().cuda())
            prelim_mask_tensor = prelim_mask_tensor.scatter_(1, prelim_prob_topk_pos, torch.zeros_like(prelim_prob_topk_pos).float().cuda())
            prelim_mask_tensor = prelim_mask_tensor.scatter_(1, prelim_ent_topk_pos, torch.zeros_like(prelim_ent_topk_pos).float().cuda())

        if mode == 'train':
            order_model.train()

        tensor = order_model('fwd', x=batch.clone(), lengths=lengths, positions=positions, langs=langs, causal=False)
        _, order_logits_pred_full = order_model(
            'predict_wo_targets', tensor=tensor, pred_mask=torch.t(full_mask_tensor).int().byte().cuda()
        )

        order_logits_pred_full = order_logits_pred_full - (1 - full_mask_tensor.int().cuda()) * 1e30
        if restrain_action_space:
            order_logits_pred_full = order_logits_pred_full + prelim_mask_tensor

        if rl_baseline == 'state_value':
            state_value = order_model("predict_baseline_wo_targets", tensor=tensor, src_lens=src_lens)
            all_state_values.append(state_value)

        if decode_order_strategy == 'sample':
            m = Categorical(logits=order_logits_pred_full)
            sampled_order = m.sample()
            all_sampled_order.append(sampled_order.unsqueeze(1))
        elif decode_order_strategy == 'top1':
            _, sampled_order = torch.topk(order_logits_pred_full, 1)
            all_sampled_order.append(sampled_order)
            sampled_order = sampled_order.squeeze(1)
        elif decode_order_strategy == 'predefined':
            sampled_order = predefined_order[:, dec_iter]
            all_sampled_order.append(sampled_order.unsqueeze(1))
        else:
            raise NotImplementedError

        all_order_logits.append(order_logits_pred_full.unsqueeze(1))
        #all_order_loss_masks.append(mask_tensor.unsqueeze(1))

        if dec_iter < dec_len:
            selected_pos.append(sampled_order)
        else:
            sampled_order = selected_pos[dec_iter % dec_len.numpy()]

        # create a prediction mask just for argmax pos
        pred_mask = torch.zeros_like(batch).byte()
        for batch_idx in range(batch_size):
            if gen_type == "src2trg":
                #if src_lens[batch_idx]+ argmax_pos[batch_idx] + 1 < pred_mask.size(0) - 1:
                if sampled_order[batch_idx] < trg_lens[batch_idx] + src_lens[batch_idx] - 1:
                    assert sampled_order[batch_idx] < pred_mask.size(0) - 1
                    pred_mask[sampled_order[batch_idx], batch_idx] = 1
                else:
                    pred_mask[-1, batch_idx] = 1
            elif gen_type == "trg2src":
                if sampled_order[batch_idx] < src_lens[batch_idx] - 1:
                    pred_mask[sampled_order[batch_idx], batch_idx] = 1
                else:
                    pred_mask[src_lens[batch_idx] - 1, batch_idx] = 1
            else:
                sys.exit("something is wrong")

        # re-run the model on the masked batch
        model.eval()
        if params.generation_model_cpu:
            tensor = model('fwd', x=batch.cpu(), lengths=lengths.cpu(), positions=positions.cpu(), langs=langs.cpu(),
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask).cuda()
        else:
            tensor = model('fwd', x=batch, lengths=lengths, positions=positions, langs=langs,
                           causal=False)
            scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask)

        if not separate_order_model:
            scores_pred = scores_pred[0]

        if top_k == 1:
            # select "best" token (argmax) given that best position
            scores_pred = torch.log_softmax(scores_pred, -1)
            score_argmax_tok, argmax_tok = torch.topk(scores_pred, 1, dim=-1)
            argmax_tok = argmax_tok.squeeze(1)
            score_argmax_tok = score_argmax_tok.squeeze(1)
        else:
            indices_to_remove = scores_pred < torch.topk(scores_pred, top_k)[0][..., -1, None]
            scores_pred[indices_to_remove] = -float('Inf')
            w = Categorical(logits=scores_pred)
            argmax_tok = w.sample()
            score_argmax_tok = scores_pred[[torch.arange(batch_size).cuda(), argmax_tok]]

        total_score_argmax_toks += score_argmax_tok # + score_argmax_pos)
        # substitute that token in
        order_pred_loss_mask = torch.zeros(batch_size, 1)

        for batch_idx in range(batch_size):
            if gen_type == "src2trg":
                if pred_mask[-1, batch_idx] != 1 and full_mask_tensor[batch_idx, sampled_order[batch_idx]] == torch.tensor(True):
                    batch[sampled_order[batch_idx], batch_idx] = argmax_tok[batch_idx]
                    order_pred_mask[sampled_order[batch_idx], batch_idx] = 0
                    full_mask_tensor[batch_idx, sampled_order[batch_idx]] = False
                    if torch.sum(full_mask_tensor[batch_idx].int()) == 0:
                        if mode in ['train', 'ppo_sample']:
                            sampled_order[batch_idx] = -100
                    else:
                        order_pred_loss_mask[batch_idx, 0] = 1
                else:
                    sampled_order[batch_idx] = -100
                curr_tokens = batch[src_lens[0]+1:-1]
            elif gen_type == "trg2src":
                if pred_mask[src_lens[batch_idx] - 1, batch_idx] != 1 and full_mask_tensor[batch_idx, sampled_order[batch_idx]] == torch.tensor(True):
                    batch[sampled_order[batch_idx], batch_idx] = argmax_tok[batch_idx]
                    order_pred_mask[sampled_order[batch_idx], batch_idx] = 0
                    full_mask_tensor[batch_idx, sampled_order[batch_idx]] = False
                    if torch.sum(full_mask_tensor[batch_idx].int()) == 0:
                        if mode in ['train', 'ppo_sample']:
                            sampled_order[batch_idx] = -100
                    else:
                        order_pred_loss_mask[batch_idx, 0] = 1
                else:
                    sampled_order[batch_idx] = -100
                curr_tokens = batch[1:src_lens[0]-1]
            else:
                sys.exit("something is wrong")

        if restrain_action_space:
            log_probs_tokens = torch.gather(log_probs_pred, 2, batch.unsqueeze(2))[:, :, 0]

        all_order_loss_masks.append(order_pred_loss_mask)

    total_score_argmax_toks = total_score_argmax_toks / ((iter_mult * dec_len))
    all_sampled_order = torch.cat(all_sampled_order, dim=1)
    all_order_logits = torch.cat(all_order_logits, dim=1)
    all_order_loss_masks = torch.cat(all_order_loss_masks, dim=1)
    all_ob = torch.cat(all_ob, dim=1)
    all_full_mask_tensors = torch.cat(all_full_mask_tensors, dim=1).cuda()
    #exit()
    return batch, all_sampled_order, all_order_logits, all_order_loss_masks, all_state_values, total_score_argmax_toks, \
           all_ob, all_full_mask_tensors


def _evaluate_batch_by_order(model, params, batch, lengths, positions, langs, src_lens, trg_lens, \
                    gen_type, order, iter_mult=1, cuda=True, separate_order_model=True):
    """Run on one example"""
    batch_size = batch.size(1)
    #gen_pos = []
    #n_iter = dec_len * iter_mult

    # log probabilities of previously present tokens at each position
    #log_probs_tokens = torch.zeros(dec_len).cuda()
    # vocab_size = len(dico)
    #mask_tok = dico.word2id[MASK_WORD]

    total_score_argmax_toks = torch.zeros(batch_size)
    if cuda:
        total_score_argmax_toks = total_score_argmax_toks.cuda()
        batch, lengths, positions, langs = to_cuda(batch, lengths, positions, langs)
    #not_chosen_pos = np.arange(dec_len)
    selected_pos = []

    dec_len = trg_lens.max() - 2 if gen_type == 'src2trg' else src_lens.max() - 2
    model.eval()
    # do iter_mult passes over entire sentence
    for dec_iter in range(iter_mult*dec_len):

        argmax_pos = order[:, dec_iter % dec_len.numpy()]

        if dec_iter < dec_len:
            selected_pos.append(argmax_pos)
        else:
            argmax_pos = selected_pos[dec_iter % dec_len.numpy()]

        # create a prediction mask just for argmax pos
        pred_mask = torch.zeros_like(batch).byte()
        for batch_idx in range(batch_size):
            if gen_type == "src2trg":
                if argmax_pos[batch_idx] < trg_lens[batch_idx] - 2:
                    assert src_lens[batch_idx]+ argmax_pos[batch_idx] + 1 < pred_mask.size(0) - 1
                    pred_mask[src_lens[batch_idx]+ argmax_pos[batch_idx] + 1, batch_idx] = 1
                else:
                    pred_mask[-1, batch_idx] = 1
            elif gen_type == "trg2src":
                if argmax_pos[batch_idx] < src_lens[batch_idx] - 2:
                    pred_mask[argmax_pos[batch_idx] + 1,batch_idx] = 1
                else:
                    pred_mask[src_lens[batch_idx] + 1] = 1
            else:
                sys.exit("something is wrong")

        # re-run the model on the masked batch
        tensor = model('fwd', x=batch, lengths=lengths, positions=positions, langs=langs, causal=False)
        scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask)
        if not separate_order_model:
            scores_pred = scores_pred[0]

        # select "best" token (argmax) given that best position
        score_argmax_tok, argmax_tok = torch.topk(scores_pred, 1, dim=-1)
        score_argmax_tok = torch.log_softmax(score_argmax_tok, 1)
        argmax_tok = argmax_tok.squeeze(1)
        score_argmax_tok = score_argmax_tok.squeeze(1)

        total_score_argmax_toks += score_argmax_tok # + score_argmax_pos)
        # substitute that token in
        for batch_idx in range(batch_size):
            if gen_type == "src2trg":
                if pred_mask[-1, batch_idx] != 1:
                    batch[src_lens[batch_idx] + argmax_pos + 1, batch_idx] = argmax_tok[batch_idx]
                curr_tokens = batch[src_lens[0]+1:-1]
            elif gen_type == "trg2src":
                if pred_mask[src_lens[batch_idx] + 1] != 1:
                    batch[argmax_pos + 1, batch_idx] = argmax_tok[batch_idx]
                curr_tokens = batch[1:src_lens[0]-1]
            else:
                sys.exit("something is wrong")

    return batch


def calculate_bleu_scores(batch, src_x, trg_x, src_lens, trg_lens, params, gen_type, scorer, dico, full_x=None):
    bleu_scores_batch = []
    batch_size = batch.size(1)

    for batch_idx in range(batch_size):
        src_len = src_lens[batch_idx].item()
        tgt_len = trg_lens[batch_idx].item()

        if gen_type == "src2trg":
            _generated = batch[src_len:src_len + tgt_len, batch_idx]
            _reference = trg_x[:tgt_len, batch_idx]
        else:
            _generated = batch[:src_len, batch_idx]
            _reference = src_x[:src_len, batch_idx]

        # extra <eos>
        eos_pos = (_generated == params.eos_index).nonzero()
        if eos_pos.shape[0] > 2:
            _generated = _generated[:(eos_pos[1, 0].item() + 1)]
        _generated_text = convert_to_text(
            _generated.unsqueeze(1), \
            torch.Tensor([_generated.shape[0]]).int(), \
            dico, params)

        eos_pos = (_reference == params.eos_index).nonzero()
        # print(src_x[:, batch_idx])
        # print(trg_x[:, batch_idx])
        # print(src_len, tgt_len)
        if eos_pos.shape[0] > 2:
            _reference = _reference[:(eos_pos[1, 0].item() + 1)]
        # print(_reference)
        # print(full_x[:src_len + tgt_len, batch_idx])
        # print(_generated)
        _reference_text = convert_to_text(
            _reference.unsqueeze(1), \
            torch.Tensor([_reference.shape[0]]).int(), \
            dico, params)
        # print(batch[:, batch_idx])
        # print("Ex {0}\nRef: {1}\nHyp: {2}\n".format(batch_idx, _reference_text[0].encode("utf-8"),
        #                                             _generated_text[0].encode("utf-8")))

        #_reference_text = refs[batch_n * batch_size + batch_idx]
        #hypothesis_all_orders.append(_generated_text[0])
        # reference_all_orders.append(_reference_text[0])
        # bleu_score = scorer.score([_reference_text[0]], [_generated_text[0]])

        bleu_score = scorer.score([_reference_text[0]], [_generated_text[0]])
        bleu_scores_batch.append(bleu_score)

    return bleu_scores_batch


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence