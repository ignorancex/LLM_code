#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import json
import torch

from transformers import BartTokenizer
from tensorboardX import SummaryWriter
from others.utils import rouge_results_to_str, test_rouge, tile


class Translator(object):

    def __init__(self, args, model, logger=None):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model

        if args.large:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', do_lower_case=True)
        else:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        tensorboard_log_dir = args.model_path
        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")


    def translate(self, data_iter, step):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        example_id_path = self.args.result_path + '.%d.example_id' % step
        self.example_id_file = codecs.open(example_id_path, 'w', 'utf-8')

        with torch.no_grad():
            for batch in data_iter:
                translations = self.translate_batch(batch)
                preds = self.ids_to_toks(translations)
                golds = self.ids_to_toks(batch.tgt)
                srcs = batch.src_str
                example_ids = batch.example_ids
                for i, pred in enumerate(preds):
                    pred_str = pred.strip()
                    gold_str = golds[i].strip()
                    src = '\t'.join(srcs[i])
                    example_id = example_ids[i]
                    # write out
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    self.example_id_file.write(str(example_id) + '\n')
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.example_id_file.flush()
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.example_id_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)


    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def ids_to_toks(self, ids):
        toks = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in ids]
        return toks

    def translate_batch(self, batch, fast=False):
        with torch.no_grad():
            src = batch.src
            mask_src = batch.mask_src
            hypos = self.model.generate(src, 
                        beam_size=self.beam_size,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        early_stopping=True)
        return hypos
