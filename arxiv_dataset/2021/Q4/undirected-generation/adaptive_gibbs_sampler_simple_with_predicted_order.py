""" Adaptive Gibbs Sampling where at each iteration t of sampling
    exp (- \alpha \log \frac{H(x_i|x_{\i})}{\log(|V|)} - \beta \log p(x_i^{t-1} | x_{\i}))
    At first iteration only use entropies at timestep 1
    Number of iterations is fixed at depends to target length"""

import os
import sys
import argparse
import subprocess
import pickle as pkl
import json
from tqdm import tqdm

import numpy as np
import torch
import operator

from src.data.loader import load_data
from src.utils import AttrDict, concat_batches, to_cuda, reload_checkpoint, reload_order_checkpoint
from src.utils import create_batch, mask_batch_seq, create_masked_batch, generate_step, restore_segmentation
from src.evaluation.evaluator import convert_to_text, eval_moses_bleu, SingleEvaluator
from src.decoding_utils import _evaluate_batch_while_generate_order, _evaluate_batch_while_generate_order_beam
from src.pybleu import PyBleuScorer
from src.decoding_utils import process_bpe_symbol

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def run(model, order_model, args, params, dico, data, split,
        src_lang, trg_lang,
        gen_type="src2trg", alpha=1., beta=1., gamma=0., uniform=False, iter_mult=1, \
        batch_size=8, gpu_id=0):
    #n_batches = math.ceil(len(srcs) / batch_size)
    if gen_type == "src2trg":
        src_path = params.ref_paths[(trg_lang, src_lang, split)]
        ref_path = params.ref_paths[(src_lang, trg_lang, split)]
    elif gen_type == "trg2src":
        src_path = params.ref_paths[(src_lang, trg_lang, split)]
        ref_path = params.ref_paths[(trg_lang, src_lang, split)]
    srcs = [s.strip() for s in open(src_path, encoding="utf-8").readlines()]
    refs = [s.strip() for s in open(ref_path, encoding="utf-8").readlines()]
    hypothesis, reference, hypothesis_bpe = [], [], []
    hypothesis_selected_pos = []
    scorer = PyBleuScorer(case_insensitive=True, smooth=True)
    hyp_order_file = open('hyp+order.txt', 'w')
    hyp_order_json_list = []
    for batch_n, batch in enumerate(get_iterator(params, data, split, src_lang, trg_lang)):
        (src_x, src_lens), (trg_x, trg_lens) = batch

        batches, batches_src_lens, batches_trg_lens, total_scores = [], [], [], []
        batches_selected_pos = []
        predicted_orders = []

        for i_topk_length in range(params.num_topk_lengths):

            # overwrite source/target lengths according to dataset stats if necessary
            if params.de2en_lengths != None and params.en2de_lengths != None:
                src_lens_item = src_lens[0].item() - 2 # remove BOS, EOS
                trg_lens_item = trg_lens[0].item() - 2 # remove BOS, EOS
                if gen_type == "src2trg":
                    if len(params.de2en_lengths[src_lens_item].keys()) < i_topk_length + 1:
                        break
                    data_trg_lens = sorted(params.de2en_lengths[src_lens_item].items(), key=operator.itemgetter(1))
                    data_trg_lens_item = data_trg_lens[-1-i_topk_length][0] + 2
                    # overwrite trg_lens
                    if args.use_gt_len:
                        trg_lens = torch.ones_like(trg_lens) * trg_lens_item + 2
                    else:
                        trg_lens = torch.ones_like(trg_lens) * data_trg_lens_item

                elif gen_type == "trg2src":
                    if len(params.en2de_lengths[trg_lens_item].keys()) < i_topk_length + 1:
                        break
                    data_src_lens = sorted(params.en2de_lengths[trg_lens_item].items(), key=operator.itemgetter(1))
                    # take i_topk_length most likely length and add BOS, EOS
                    data_src_lens_item = data_src_lens[-1-i_topk_length][0] + 2
                    # overwrite src_lens
                    src_lens = torch.ones_like(src_lens) * data_src_lens_item

            if gen_type == "src2trg":
                sent1_input = src_x
                sent2_input = create_masked_batch(trg_lens, params, dico)
                dec_len = torch.max(trg_lens).item() - 2 # cut BOS, EOS
            elif gen_type == "trg2src":
                sent1_input = create_masked_batch(src_lens, params, dico)
                sent2_input = trg_x
                dec_len = torch.max(src_lens).item() - 2 # cut BOS, EOS

            batch, lengths, positions, langs = concat_batches(sent1_input, src_lens, params.lang2id[src_lang], \
                                                              sent2_input, trg_lens, params.lang2id[trg_lang], \
                                                              params.pad_index, params.eos_index, \
                                                              reset_positions=True,
                                                              assert_eos=True) # not sure about it
            if gpu_id >= 0:
                batch, lengths, positions, langs, src_lens_cuda, trg_lens_cuda = \
                    to_cuda(batch, lengths, positions, langs, src_lens, trg_lens)

            with torch.no_grad():
                # max_src_len = src_lens.max() - 2 if gen_type == "src2trg" else trg_lens.max() - 2
                max_trg_len = trg_lens.max() - 2 if gen_type == "src2trg" else src_lens.max() - 2
                l2r_order = torch.arange(max_trg_len).unsqueeze(0)
                if gen_type == "src2trg":
                    l2r_order = l2r_order.expand(batch.size(1), l2r_order.size(1)) + src_lens.unsqueeze(1) + 1
                else:
                    l2r_order = l2r_order.expand(batch.size(1), l2r_order.size(1)) + 1
                r2l_order = torch.flip(l2r_order, dims=[1])

                if args.order_beam > 1 and trg_lens[0] > 4:
                    batch, predicted_order, _, selected_pos, total_score_argmax_toks = _evaluate_batch_while_generate_order_beam(
                        model, order_model, params, batch, lengths, positions, langs, src_lens, trg_lens, gen_type,
                        separate_order_model=params.separate_order_model, beam_size=args.order_beam,
                        iter_mult=iter_mult, mode='eval', decode_order_strategy='topk',
                    )
                    batch = torch.unbind(batch.clone().unsqueeze(1), dim=2)
                    total_score_argmax_toks = torch.unbind(total_score_argmax_toks.clone().unsqueeze(0), dim=1)
                    batches += batch
                    batches_selected_pos += [selected_pos] * args.order_beam
                    batches_src_lens += [src_lens.clone()] * args.order_beam
                    batches_trg_lens += [trg_lens.clone()] * args.order_beam
                    total_scores += total_score_argmax_toks
                else:
                    batch, predicted_order, _, selected_pos, _, total_score_argmax_toks, _, _ = _evaluate_batch_while_generate_order(
                        model, order_model, params, batch, lengths, positions, langs, src_lens, trg_lens, gen_type,
                        separate_order_model=params.separate_order_model, rl_baseline='',
                        iter_mult=iter_mult, mode='eval',
                        decode_order_strategy='top1',
                        restrain_action_space=False,
                    )
                    predicted_orders.append(predicted_order.cpu().numpy()[0])
                    batches.append(batch.clone())
                    batches_selected_pos.append(selected_pos)
                    batches_src_lens.append(src_lens.clone())
                    batches_trg_lens.append(trg_lens.clone())
                    total_scores.append(total_score_argmax_toks)

        best_score_idx = np.array(total_scores).argmax()
        batch, src_lens, trg_lens = batches[best_score_idx], batches_src_lens[best_score_idx], batches_trg_lens[best_score_idx]
        selected_pos = batches_selected_pos[best_score_idx]
        selected_order = predicted_orders[best_score_idx]
        selected_order = [str(_s) for _s in list(selected_order)]

        if gen_type == "src2trg":
            hypothesis_selected_pos.append([selected_pos, trg_lens.item()-2])
        elif gen_type == "trg2src":
            hypothesis_selected_pos.append([selected_pos, src_lens.item()-2])

        for batch_idx in range(batch_size):
            src_len = src_lens[batch_idx].item()
            tgt_len = trg_lens[batch_idx].item()
            if gen_type == "src2trg":
                generated = batch[src_len:src_len + tgt_len, batch_idx]
            else:
                generated = batch[:src_len, batch_idx]
            # extra <eos>
            eos_pos = (generated == params.eos_index).nonzero()
            if eos_pos.shape[0] > 2:
                generated = generated[:(eos_pos[1,0].item()+1)]

            generated = convert_to_text(generated.unsqueeze(1), \
                                        torch.Tensor([generated.shape[0]]).int(), \
                                        dico, params)
            hypothesis_bpe.extend(generated)
            generated_processed = [process_bpe_symbol(_g, '@@ ') for _g in generated]
            hypothesis.extend(generated_processed)
            reference.append(refs[batch_n * batch_size + batch_idx])

        print("Ex {0}\nRef: {1}\nHyp: {2}\n".format(batch_n, refs[batch_n].encode("utf-8"), hypothesis[-1].encode("utf-8")))
        hyp_order_file.write("Ex {0}\nRef: {1}\nHyp: {2}\n".format(batch_n, refs[batch_n].encode("utf-8"), hypothesis_bpe[-1].encode("utf-8")))
        hyp_order_file.write((',').join(selected_order) + '\n')
        print((',').join(selected_order))
        bleu_score = scorer.score([reference[-1]], [hypothesis[-1]])
        hyp_order_json_list.append({'id': batch_n, 'src': srcs[batch_n], 'ref': refs[batch_n], 'hyp': hypothesis_bpe[-1],
                                    'bleu': bleu_score, 'order': selected_order})

        hyp_order_file.write(str(bleu_score)+'\n\n')
        print(bleu_score)

    with open('hyp+order.json', 'w') as f:
        json.dump(hyp_order_json_list, f, indent=2)

    print(args.order_model_path)

    hyp_name = 'decoding.txt'
    hyp_name_tok = 'decoding.tok.txt'
    hyp_selected_pos_path = os.path.join(params.hyp_path, "selected_pos.pkl")

    hyp_path = os.path.join(params.hyp_path, hyp_name)
    hyp_path_tok = os.path.join(params.hyp_path, hyp_name_tok)

    # export sentences to hypothesis file / restore BPE segmentation
    with open(hyp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypothesis) + '\n')
    with open(hyp_path_tok, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypothesis) + '\n')
    with open(hyp_selected_pos_path, 'wb') as f:
        pkl.dump(hypothesis_selected_pos, f)

    restore_segmentation(hyp_path)

    # evaluate BLEU score
    bleu = eval_moses_bleu(ref_path, hyp_path)
    print("BLEU %s-%s; %s %s : %f" % (src_lang, trg_lang, hyp_path, ref_path, bleu))
    # write BLEU score result to file
    result_path = os.path.join(params.hyp_path, "result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("BLEU %s-%s; %s %s : %f\n" % (src_lang, trg_lang, hyp_path, ref_path, bleu))

    results_save_path = '/'.join(args.order_model_path.split('/')[:-1]) + '/'
    alt_hyp_path = results_save_path + 'hypothesis'
    bleu_file_name = results_save_path + 'bleu_scores'
    if not os.path.exists(alt_hyp_path):
        os.makedirs(alt_hyp_path)
        os.makedirs(bleu_file_name)
    bleu_file_name += '/' + args.order_model_path.split('/')[-1][11:-4] + '_' + gen_type + '_BLEU.txt'
    alt_hyp_path += '/' + args.order_model_path.split('/')[-1][11:-4] + '_' + gen_type + '.txt'
    with open(bleu_file_name, 'w') as f:
        f.write(str(bleu))
    with open(alt_hyp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypothesis) + '\n')
    restore_segmentation(alt_hyp_path)

def load_data_tmp(src_path, trg_path):
    """ Wrapper for loading aligned source and target sentences """

    def _load_file(path):
        with open(path, encoding="utf-8") as data_fh:
            data = data_fh.readlines()
        #data = [sent.replace("</s>", "") for sent in data]
        data = [('</s> %s </s>' % sent.strip()).split() for sent in data]
        return data

    src_sents = _load_file(src_path)
    trg_sents = _load_file(trg_path)

    assert len(src_sents) == len(trg_sents), "Found a different number of source and target sentences!"

    return src_sents, trg_sents


def get_iterator(params, data, data_set, lang1, lang2=None, stream=False):
    """
    Create a new iterator for a dataset.
    """
    assert data_set in ['valid', 'test']
    assert lang1 in params.langs
    assert lang2 is None or lang2 in params.langs
    assert stream is False or lang2 is None

    # hacks to reduce evaluation time when using many languages
    if len(params.langs) > 30:
        eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
        subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
        n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
    elif len(params.langs) > 5:
        subsample = 10 if data_set == 'test' else 5
        n_sentences = 300 if data_set == 'test' else 1500
    else:
        # n_sentences = -1 if data_set == 'valid' else 100
        n_sentences = -1
        subsample = 1

    if lang2 is None:
        if stream:
            iterator = data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
        else:
            iterator = data['mono'][lang1][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences,
            )
    else:
        assert stream is False
        #_lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
        _lang1, _lang2 = (lang1, lang2)
        iterator = data['para'][(_lang1, _lang2)][data_set].get_iterator(
            shuffle=False,
            group_by_size=True, #False to preserve original split order
            n_sentences=n_sentences
        )

    for batch in iterator:
        # yield batch if lang2 is None or lang1 < lang2 else batch[::-1]
        yield batch

def prepare_data(params, data, split, gen_type, alpha, beta, gamma, uniform, iter_mul, use_data_length, num_topk_lengths):
    """ Load data the same as in src/evaluation/evaluator.py """

    def create_reference_files():
        """
        Create reference files for BLEU evaluation.
        """
        params.ref_paths = {}

        for (lang1, lang2), v in data['para'].items():

            #assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in get_iterator(params, data, data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, data['dico'], params))
                    lang2_txt.extend(convert_to_text(sent2, len2, data['dico'], params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    params.hyp_path = os.path.join(params.dump_path, 'gen_beam_simple/hypotheses_{0}_split_{1}_gentype_{2}_alpha{3}_beta{4}_gamma{5}_uniform{6}_beam_4_itermul{7}_usedatalength{8}_numtopklengths{9}'.format('yichenj', split, gen_type, alpha, beta, gamma, uniform, iter_mul, use_data_length, num_topk_lengths))
    subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
    create_reference_files()


def main(arguments):
    """ """
    parser = argparse.ArgumentParser(description='Enumerate over all possible positions to pick the best one')

    parser.add_argument('--model_path', type=str,
                        default='../models/best-valid_de-en_mt_bleu.pth',
                        help='path to pretrained TLM model')
    parser.add_argument('--order_model_path', type=str,
                        required=True,
                        help='path to pretrained TLM model')
    parser.add_argument('--data_path', type=str,
                        default='../data/de-en',
                        help='path to pretrained TLM model')
    parser.add_argument('--src_lang', type=str, default='de', help='source language')
    parser.add_argument('--trg_lang', type=str, default='en', help='target language')
    parser.add_argument('--split', type=str, default='valid', help='use valid/test split of dataset', choices=['valid', 'test'])
    parser.add_argument('--use_data_length', action='store_true', help='use lengths according to dataset statistics')
    parser.add_argument('--num_topk_lengths', type=int, default=1, help='number of topk lengths to use when using dataset statistics')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size to use')
    parser.add_argument('--gen_type', type=str, default="src2trg", \
                        choices=['src2trg', 'trg2src'], \
                        help='generation type to use src2trg (de->en) or trg2src (en->de)')
    parser.add_argument('--print_every', type=int, default=10, help='how often to log progress')
    parser.add_argument('--alpha', type=float, default=1., help='weight to put on entropy')
    parser.add_argument('--beta', type=float, default=1., help='weight to put on log prob')
    parser.add_argument('--gamma', type=float, default=0., help='weight to put on left to right decoding')
    parser.add_argument('--uniform', action='store_true', help='do uniform sampling of positions')
    parser.add_argument('--iter_mult', type=int, default=1, help='iteration multipler (multiply this number by target length)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID, use -1 for CPU')
    parser.add_argument('--out_file', type=str, default="potentials.txt",
                        help='file to write evaluations')
    parser.add_argument('--order_beam', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=1000)
    parser.add_argument('--use_gt_len', type=bool, default=False)
    args = parser.parse_args(arguments)

    if args.uniform:
        args.alpha, args.beta, args.gamma = 0, 0, 0

    # set GPU
    if args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)
    print("Evaluating model at {0}".format(args.model_path))

    # load everything from checkpoint

    params, dico, order_model = reload_order_checkpoint(args.order_model_path)
    if params.separate_order_model:
        _, _, model = reload_checkpoint(args.model_path)
    else:
        model = order_model
        # _, _, model = reload_checkpoint(args.model_path)
        # params.separate_order_model = True

    # put on gpu
    model = model.cuda() if args.gpu_id >= 0 else model
    order_model = order_model.cuda() if args.gpu_id >= 0 else order_model
    # put in eval model
    model = model.eval()
    order_model = order_model.eval( )

    params.data_path = args.data_path

    params.generation_model_cpu = False

    if args.use_data_length:
        params.de2en_lengths = pkl.load(
            open(os.path.join(params.data_path, args.src_lang + '2' + args.trg_lang + '_lengths.pkl'), 'rb'))
        params.en2de_lengths = pkl.load(
            open(os.path.join(params.data_path, args.trg_lang + '2' + args.src_lang + '_lengths.pkl'), 'rb'))
        params.num_topk_lengths = args.num_topk_lengths
    else:
        params.de2en_lengths = None
        params.en2de_lengths = None
        params.num_topk_lengths = 1

    params.dump_path = 'dump'
    lang_pair = args.src_lang + '-' + args.trg_lang
    params.para_dataset = \
        {(args.src_lang, args.trg_lang):
             {'train':
                  (params.data_path + '/train.' + lang_pair + '.' + args.src_lang + '.pth',
                   params.data_path + '/train.' + lang_pair + '.' + args.trg_lang + '.pth'),
              'valid':
                  (params.data_path + '/valid.' + lang_pair + '.' + args.src_lang + '.pth',
                   params.data_path + '/valid.' + lang_pair + '.' + args.trg_lang + '.pth'),
              'test':
                  (params.data_path + '/test.' + lang_pair + '.' + args.src_lang + '.pth',
                   params.data_path + '/test.' + lang_pair + '.' + args.trg_lang + '.pth')
              }
         }

    # load data
    params.eval_only = True
    params.batch_size = args.batch_size
    data = load_data(params, args.max_seq_length)

    prepare_data(params, data, args.split, args.gen_type, args.alpha, args.beta, args.gamma, args.uniform, args.iter_mult, args.use_data_length, args.num_topk_lengths) # creates reference files for BLEU eval

    # evaluate
    run(model, order_model, args, params, dico, data, args.split, \
        args.src_lang, args.trg_lang, args.gen_type, \
        args.alpha, args.beta, args.gamma, args.uniform, args.iter_mult, \
        args.batch_size, args.gpu_id)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
