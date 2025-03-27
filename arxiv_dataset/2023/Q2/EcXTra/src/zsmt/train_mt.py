import datetime
import json
import os
import pickle
import sys
import time
from itertools import chain
from typing import List

import numpy as np
import sacrebleu
import torch.nn as nn
import torch.utils.data as data_utils
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from zsmt import dataset
from zsmt.arg_parser import get_mt_args_parser
from zsmt.loss import SmoothedNLLLoss
from zsmt.seq2seq import Seq2Seq
from zsmt.seq_gen import BeamDecoder, get_outputs_until_eos
from zsmt.textprocessor import TextProcessor
from zsmt.utils import *

class Trainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None,
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5, len_penalty_ratio: float = 0.8,
                 nll_loss: bool = False, rank: int = -1, save_latest: bool = True,
                 is_multi_stream: bool = False, bidi: bool = False, use_rt_bleu: bool = False,
                 foreign_tok_d: dict = {}, src_lines_lc: List[str] = []):
        self.model = model
        # TODO: refactor to define bidirectional for model, not trainer
        self.is_multi_stream = is_multi_stream
        self.bidi = bidi
        self.use_rt_bleu = use_rt_bleu

        self.foreign_tok_d = foreign_tok_d
        self.src_lines_lc = src_lines_lc

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count() if rank < 0 else 1

        self.mask_prob = mask_prob
        if nll_loss:
            self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())
        else:
            self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.rank = rank
        if rank >= 0:
            self.device = torch.device('cuda', rank)
            torch.cuda.set_device(self.device)
            print("The device is", self.device, "with rank", self.rank)

        self.model = self.model.to(self.device)
        self.scaler = GradScaler()

        self.generator = BeamDecoder(self.model, beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                     len_penalty_ratio=len_penalty_ratio)
        if rank >= 0:
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank,
                                                 find_unused_parameters=True)
            self.generator = DistributedDataParallel(self.generator, device_ids=[self.rank], output_device=self.rank,
                                                     find_unused_parameters=True)

        self.reference = None
        self.best_bleu = -1.0
        self.save_latest = save_latest

    def train_epoch(self, step: int, saving_path: str = None, mt_dev_iter: List[data_utils.DataLoader] = None,
                    mt_train_iter: List[data_utils.DataLoader] = None, max_step: int = 300000, accum=1,
                    save_opt: bool = False, eval_steps: int = 5000, eval_every_epoch=True, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(mt_train_iter)
        bleu = 0

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer.zero_grad()
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                try:
                    with autocast():
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_mask = batch["src_pad_mask"].squeeze(0)
                        tgt_inputs = batch["dst_texts"].squeeze(0)
                        tgt_mask = batch["dst_pad_mask"].squeeze(0)

                        if self.is_multi_stream:
                            # Second stream of data in case of multi-stream processing.
                            srct_inputs = batch["srct_texts"].squeeze(0)
                            srct_mask = batch["srct_pad_mask"].squeeze(0)
                        else:
                            srct_inputs = None
                            srct_mask = None
                        if src_inputs.size(0) < self.num_gpu:
                            continue
                        predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs, src_mask=src_mask,
                                                 srct_inputs=srct_inputs, srct_mask=srct_mask,
                                                 tgt_mask=tgt_mask, log_softmax=True)
                        targets = tgt_inputs[:, 1:].contiguous().view(-1)
                        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                        targets = targets[tgt_mask_flat]
                        ntokens = targets.size(0)

                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)
                    if self.rank >= 0: targets = targets.to(self.device)

                    loss = self.criterion(predictions, targets).mean()
                    self.scaler.scale(loss).backward()

                    loss = float(loss.data) * ntokens
                    tokens += ntokens
                    total_tokens += ntokens
                    total_loss += loss
                    cur_loss += loss

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    step += 1
                    if step % accum == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    if step % 50 == 0 and tokens > 0:
                        elapsed = time.time() - start
                        print(self.rank, "->", datetime.datetime.now(),
                              "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (
                                  step, cur_loss / tokens, tokens / elapsed))

                        if mt_dev_iter is not None and step % eval_steps == 0:
                            bleu = self.eval_bleu(mt_dev_iter, saving_path)
                            print("BLEU:", bleu)

                        if self.save_latest and step % 25000 == 0:
                            self.handle_save(model, saving_path + ".latest", save_opt)

                        start, tokens, cur_loss = time.time(), 0, 0

                except RuntimeError as err:
                    print(repr(err))
                    torch.cuda.empty_cache()

            if i == shortest - 1:
                break
            if step >= max_step:
                break

        try:
            if eval_every_epoch:  # self.rank <= 0:
                print("Total loss in this epoch: %f" % (total_loss / total_tokens))
                if self.save_latest:
                    self.handle_save(model, saving_path + ".latest", save_opt)

                if mt_dev_iter is not None:
                    bleu = self.eval_bleu(mt_dev_iter, saving_path)
                    print("BLEU:", bleu)
                elif (mt_dev_iter is None and step == max_step): # save best BLEU at end
                    self.handle_save(model, saving_path, save_opt)

        except RuntimeError as err:
            print(repr(err))

        return step, bleu

    def get_batch_zip(self, mt_train_iter):
        iters = list(chain(*filter(lambda x: x != None, [mt_train_iter])))
        shortest = min(len(l) for l in iters)
        return zip(*iters), shortest

    def handle_save(self, model, saving_path, save_opt=False):
        if self.rank < 0:
            model.cpu().save(saving_path)
            model = model.to(self.device)
        elif self.rank == 0:
            model.save(saving_path)

        if save_opt:
            with open(os.path.join(saving_path, "optim"), "wb") as fp:
                pickle.dump(self.optimizer, fp)

    def do_inference(self, model, dev_data_iter, debug=False):
        outputs_all = []
        src_ids_all = []

        to_en_idxs = []
        to_fo_idxs = []
        curr_num_ex = 0
        to_foreign_ids = [v[1] for k, v in self.foreign_tok_d.items() if k != 'en']

        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    src_pad_idx = batch["src_pad_idx"].squeeze(0)

                    # Second stream of data in case of multi-stream processing.
                    srct_inputs, srct_mask = None, None
                    if self.is_multi_stream:
                        srct_inputs = batch["srct_texts"].squeeze(0)
                        srct_mask = batch["srct_pad_mask"].squeeze(0)

                    if self.bidi:
                        start_toks = src_inputs[:, 1].numpy()
                        to_en_curr = np.where(np.isin(start_toks, to_foreign_ids, invert=True))[0]
                        to_fo_curr = np.where(np.isin(start_toks, to_foreign_ids))[0]
                        to_en_idxs.extend(to_en_curr + curr_num_ex)
                        to_fo_idxs.extend(to_fo_curr + curr_num_ex)
                        curr_num_ex += src_inputs.shape[0]

                    src_ids = get_outputs_until_eos(model.src_eos_id(), src_inputs)
                    src_ids_all += src_ids

                    outputs = self.generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                                             first_tokens=tgt_inputs[:, 0], srct_inputs=srct_inputs,
                                             src_mask=src_mask, srct_mask=srct_mask,
                                             pad_idx=model.text_processor.pad_token_id())

                    if self.num_gpu > 1 and self.rank < 0:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    outputs_all += outputs

            model.train()
        return src_ids_all, outputs_all, to_en_idxs, to_fo_idxs

    def eval_bleu(self, dev_data_iter, saving_path, save_opt: bool = False):
        def tensor_is_zero(x):
            return x.tolist() == [0]

        print(datetime.datetime.now(), 'running inference on validation...')
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()

        src_ids_all, outputs_all, to_en_idxs, to_fo_idxs = self.do_inference(model, dev_data_iter)

        src_text = list(map(lambda src: model.decode_src(src, True), src_ids_all))
        mt_output = list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs_all))

        reference = self.reference
        mt_ref = reference[:len(mt_output)]

        # calculate BLEU scores
        bleu = sacrebleu.corpus_bleu(mt_output, [mt_ref])

        if self.bidi:
            bleu_en, bleu_fo = self.get_bidi_bleu(mt_output, mt_ref, to_fo_idxs, to_en_idxs)
            print(f'bidi BLEU: 2en={bleu_en.score}, 2fo={bleu_fo.score}')
        ###

        # handle round-trip translation BLEU, if specified
        if self.use_rt_bleu:
            print(datetime.datetime.now(), 'running inference on round-trip translation...')
            self.add_foreign_ids(outputs_all, self.src_lines_lc, to_en_idxs)

            rev_is_undef = [x for x in to_en_idxs if self.src_lines_lc[x] == '']
            examples = [(x.tolist() + [model.src_eos_id()], [0]) for x in outputs_all]
            # for those examples where the reverse is undefined, do not translate!
            examples = [x if i not in rev_is_undef else ([0, 2], [0]) for i, x in enumerate(examples)]

            inter_data = dataset.MTDataset(examples=examples,
                                            max_batch_capacity=args.total_capacity, keep_src_pad_idx=True,
                                            max_batch=int(args.batch / (args.beam_width * 2)),
                                            src_pad_idx=model.src_pad_id(),
                                            srct_pad_idx=model.src_pad_id(), # TODO: check this
                                            dst_pad_idx=model.text_processor.pad_token_id())
            dl = data_utils.DataLoader(inter_data, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())
            src_ids_all2, outputs_all2, to_en_idxs2, to_fo_idxs2 = self.do_inference(model, [dl], True)

            src_text2 = list(map(lambda src: model.decode_src(src, True), src_ids_all2))
            # assert src_text2 == mt_output
            mt_output2 = list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs_all2))

            mt_ref2 = [model.decode_src( (src if i not in to_fo_idxs else src[1:]) , True) \
                for i, src in enumerate(src_ids_all[:len(mt_output2)])]

            # as a final step, for those entries where the original src lang was undefined
            # ignore round trip, and use one-pass translation instead
            mt_output_final = [mt_output[i] if tensor_is_zero(x) else mt_output2[i] for i, x in enumerate(src_ids_all2)]
            mt_ref_final = [mt_ref[i] if tensor_is_zero(x) else mt_ref2[i] for i, x in enumerate(src_ids_all2)]

            obleu, obleu_en, obleu_fo = bleu, bleu_en, bleu_fo
            # calculate BLEU scores
            bleu = sacrebleu.corpus_bleu(mt_output_final, [mt_ref_final], lowercase=True, tokenize="intl")
            if self.bidi:
                bleu_en, bleu_fo = self.get_bidi_bleu(mt_output_final, mt_ref_final, to_fo_idxs, to_en_idxs)
                print(f'round-trip bidi BLEU: 2en={bleu_en.score}, 2fo={bleu_fo.score}')

        if self.rank <= 0:
            with open(os.path.join(saving_path, "bleu.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, mt_ref)]))

            if bleu.score > self.best_bleu:
                self.best_bleu = bleu.score
                with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                    writer.write("\n".join(
                        [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                         zip(src_text, mt_output, mt_ref)]))
                print("Saving best BLEU", self.best_bleu)
                self.handle_save(model, saving_path, save_opt)

        return bleu.score

    @staticmethod
    def get_bidi_bleu(mt_output, mt_ref, to_fo_idxs, to_en_idxs):
        fo_out = [x for i, x in enumerate(mt_output) if i in to_fo_idxs]
        fo_ref = [x for i, x in enumerate(mt_ref) if i in to_fo_idxs]
        bleu_fo = 0.0
        if fo_out:
            bleu_fo = sacrebleu.corpus_bleu(fo_out, [fo_ref], lowercase=True, tokenize="intl")

        en_out = [x for i, x in enumerate(mt_output) if i in to_en_idxs]
        en_ref = [x for i, x in enumerate(mt_ref) if i in to_en_idxs]
        bleu_en = 0.0
        if en_out:
            bleu_en = sacrebleu.corpus_bleu(en_out, [en_ref], lowercase=True, tokenize="intl")
        print(len(en_out), len(fo_out), len(mt_output))
        return bleu_en, bleu_fo


    def add_foreign_ids(self, output, src_codes, idxs_to_add):
        # used for round-trip translation. There are two cases:
        # `l1->en`, then we add the foreign tok to `en` to indicate translation to l1
        # `en->l1`, then we do nothing, which indicates translation to en
        for idx in idxs_to_add:
            foreign_tup = self.foreign_tok_d.get(src_codes[idx])
            if not foreign_tup: # in this case, the target language is not in test set -- so skip
                continue
            else:
                start_toks = torch.tensor([output[idx][0], foreign_tup[1]])
                output[idx] = torch.cat((start_toks, output[idx][1:]))


    @staticmethod
    def train(args):
        if args.local_rank <= 0 and not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        # load tokenizers
        srct_text_processor, dst_text_processor = None, None
        if args.tokenizer_path: # load trained tokenizer on srct
            srct_text_processor = TextProcessor(args.tokenizer_path)

        if args.xlmr_dst_tok:
            dst_text_processor = TextProcessor(pretrained_name='xlm-roberta-base')
        elif args.pretrained_decoder and args.tokenizer_path:
            # for when transferring decoder layers, but not decoder embeddings
            dst_text_processor = srct_text_processor
        elif args.pretrained_decoder:
            dst_text_processor = TextProcessor(pretrained_name=args.pretrained_decoder)

        srct_text_processor = srct_text_processor if args.multi_stream else None

        foreign_tok_d = {}
        src_lines_lc = []
        if args.bidi:
            train_path = args.mt_train_path[0]
            parent, name = os.path.split(train_path)
            stem = name.split('.', 1)[0]
            foreign_tok_path = args.foreign_tok_d or os.path.join(parent, f'{stem}.foreign_toks.json')
            with open(foreign_tok_path, 'r') as f:
                foreign_tok_d = json.load(f)

            src_lines_lc_path = os.path.join(parent, f'{stem}.dev_lang_codes.json')
            with open(src_lines_lc_path, 'r') as f:
                src_lines_lc = json.load(f)

        if args.pretrained_decoder is None and args.tokenizer_path is None and not args.xlmr_dst_tok:
            print(f'ERROR: no tokenizers specified')
            sys.exit(-1)
        text_processor = dst_text_processor or srct_text_processor

        if srct_text_processor:
            args.srct_pad_id = srct_text_processor.pad_token_id()
        else:
            args.srct_pad_id = text_processor.pad_token_id()

        # assert text_processor.pad_token_id() == 0
        num_processors = max(torch.cuda.device_count(), 1) if args.local_rank < 0 else 1

        if args.pretrained_path is not None:
            print(f'loading pretrained model from {args.pretrained_path}')
            mt_model = Seq2Seq.load(Seq2Seq, args.pretrained_path, tok_dir=args.tokenizer_path,
                                    pretrained_decoder=args.pretrained_decoder, src_lang_fam=args.src_lang_fam,
                                    load_output=args.load_output, xlm_name=args.xlm_name,
                                    xlmr_dst_tok=args.xlmr_dst_tok,
                                    freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder)
        else:
            mt_model = Seq2Seq(dst_text_processor=text_processor, dec_layer=args.decoder_layer,
                               embed_dim=args.embed_dim, intermediate_dim=args.intermediate_layer_dim,
                               freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder,
                               shallow_encoder=args.shallow_encoder,
                               multi_stream=args.multi_stream, pretrained_decoder=args.pretrained_decoder,
                               srct_text_processor=srct_text_processor, src_lang_fam=args.src_lang_fam,
                               xlm_name=args.xlm_name)
        print(args.local_rank, "Model initialization done!")

        # calc num params
        num_params_train = sum(p.numel() for p in mt_model.parameters() if p.requires_grad)
        print(f'# trainable parameters: {num_params_train}')

        num_params = sum(p.numel() for p in mt_model.parameters())
        print(f'Total # parameters: {num_params}')

        if args.continue_train:
            with open(os.path.join(args.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(mt_model, args.learning_rate, warump_steps=args.warmup)

        trainer = Trainer(model=mt_model, mask_prob=args.mask_prob, optimizer=optimizer, clip=args.clip,
                          beam_width=args.beam_width, max_len_a=args.max_len_a, max_len_b=args.max_len_b,
                          len_penalty_ratio=args.len_penalty_ratio, rank=args.local_rank, save_latest=False,
                          bidi=args.bidi, is_multi_stream=args.multi_stream, use_rt_bleu=args.round_trip_bleu,
                          foreign_tok_d=foreign_tok_d, src_lines_lc=src_lines_lc)

        pin_memory = torch.cuda.is_available()

        mt_train_loader = None
        if args.mt_train_path is not None:
            mt_train_loader = Trainer.get_mt_train_data(mt_model, num_processors, args, pin_memory)

        mt_dev_loader = None
        if args.mt_dev_path is not None:
            mt_dev_loader = Trainer.get_mt_dev_data(mt_model, args, pin_memory, text_processor, trainer)

        step, train_epoch = 0, 1
        best_bleu = 0
        not_improved = 0
        while args.step > 0 and step < args.step:
            print(trainer.rank, "--> train epoch", train_epoch)
            step, bleu = trainer.train_epoch(mt_train_iter=mt_train_loader, max_step=args.step,
                                       mt_dev_iter=mt_dev_loader, saving_path=args.model_path, step=step,
                                       save_opt=args.save_opt, accum=args.accum,
                                       eval_steps=args.eval_steps,
                                       eval_every_epoch=args.eval_every_epoch)
            if bleu > best_bleu:
                not_improved = 0
                best_bleu = bleu
            else:
                not_improved += 1
            # TODO: currently consider early stopping after every epoch,
            # should be changed to be after every eval step
            if args.early_stop and not_improved >= args.early_stop:
                break
            train_epoch += 1

    @staticmethod
    def get_mt_dev_data(mt_model, args, pin_memory, text_processor, trainer):
        mt_dev_loader = []
        dev_paths = args.mt_dev_path.split(",")
        trainer.reference = []
        for dev_path in dev_paths:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                            max_batch_capacity=args.total_capacity, keep_src_pad_idx=True,
                                            max_batch=int(args.batch / (args.beam_width * 2)),
                                            src_pad_idx=mt_model.src_pad_id(),
                                            srct_pad_idx=args.srct_pad_id,
                                            dst_pad_idx=mt_model.text_processor.pad_token_id())
            dl = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
            mt_dev_loader.append(dl)

            print(args.local_rank, "creating reference")

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in dl:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs, remove_first_token=True)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref
        return mt_dev_loader

    @staticmethod
    def get_mt_train_data(mt_model, num_processors, args, pin_memory: bool):
        mt_train_loader = []
        train_paths = args.mt_train_path
        if args.load_separate_train:
            train_paths = [train_paths[args.local_rank]]
            print(f'loading {train_paths[0]} on this device')
        for train_path in train_paths:
            mt_train_data = dataset.MTDataset(batch_pickle_dir=train_path,
                                              max_batch_capacity=int(num_processors * args.total_capacity / 2),
                                              max_batch=int(num_processors * args.batch / 2),
                                              src_pad_idx=mt_model.src_pad_id(),
                                              srct_pad_idx=args.srct_pad_id,
                                              dst_pad_idx=mt_model.text_processor.pad_token_id(),
                                              keep_src_pad_idx=False)
            sampler = None if (args.local_rank < 0 or args.load_separate_train) \
                else DistributedSampler(mt_train_data, rank=args.local_rank)
            mtl = data_utils.DataLoader(mt_train_data, sampler=sampler,
                                        batch_size=1, shuffle=(args.local_rank < 0), pin_memory=pin_memory)
            mt_train_loader.append(mtl)
        return mt_train_loader


if __name__ == "__main__":
    parser = get_mt_args_parser()
    args = parser.parse_args()
    if torch.cuda.is_available() and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    if args.local_rank <= 0:
        print(args)
    init_distributed(args)
    Trainer.train(args=args)
    if args.local_rank >= 0:
        torch.distributed.destroy_process_group()
    print("Finished Training!")
