#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
import hydra
from hydra.core.config_store import ConfigStore
import logging
import math
import numpy as np
import os
from omegaconf import OmegaConf
from typing import Optional
import sys

import editdistance
import torch
from torch.utils.data import DataLoader

from hydra.core.hydra_config import HydraConfig

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import FairseqDataclass, FairseqConfig
from omegaconf import open_dict

from espum.data import EvalExtractedDataset
from espum.models.espum import ESPUM_Config, ESPUM

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnsupGenerateConfig(FairseqDataclass):
    fairseq: FairseqConfig = FairseqConfig()
    margin: float = field(
        default=0.0,
        metadata={"help": "margin for boundary classification"}
    )
    espum: ESPUM_Config = ESPUM_Config()
    dict_path: str = field(
        default="",
        metadata={"help": "dictionary used by the ESPUM model"}
    )
    centroid_path: str = field(
        default="",
        metadata={"help": "cluster centroids used by the ESPUM model"}
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={"help": "where to store results"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={"help": "how to post process results"},
    )


def prepare_result_files(cfg: UnsupGenerateConfig):
    def get_res_file(file_prefix, suffix="txt"):
        if cfg.fairseq.dataset.num_shards > 1:
            file_prefix = f"{cfg.fairseq.dataset.shard_id}_{file_prefix}"
        path = os.path.join(
            cfg.results_path,
            "{}{}.{}".format(
                cfg.fairseq.dataset.gen_subset,
                file_prefix,
                suffix,
            ),
        )
        return open(path, "w", buffering=1)

    if not cfg.results_path:
        return None
    
    os.makedirs(cfg.results_path, exist_ok=True)
    return {
        "hypo.words": get_res_file(""),
        "hypo.units": get_res_file("_units"),
        "ref.words": get_res_file("_ref"),
        "ref.units": get_res_file("_ref_units"),
    }


def process_predictions(
    cfg: UnsupGenerateConfig,
    hypos,
    tgt_dict,
    target_tokens,
    res_files,
):
    retval = []
    word_preds = []
    transcriptions = []
    dec_scores = []

    for i, hypo in enumerate(hypos):
        if torch.is_tensor(hypo["tokens"]):
            tokens = hypo["tokens"].int().cpu()
            tokens = tokens[tokens >= tgt_dict.nspecial]
            hyp_pieces = tgt_dict.string(tokens).upper()
        else:
            hyp_pieces = " ".join(hypo["tokens"].upper())

        if "words" in hypo and len(hypo["words"]) > 0:
            hyp_words = " ".join(hypo["words"].upper())
        else:
            hyp_words = post_process(hyp_pieces, cfg.post_process)

        to_write = {}
        if res_files is not None:
            to_write[res_files["hypo.units"]] = hyp_pieces
            to_write[res_files["hypo.words"]] = hyp_words

        tgt_words = ""
        if target_tokens is not None:
            if isinstance(target_tokens, str):
                tgt_pieces = tgt_words = target_tokens
            else:
                tgt_pieces = tgt_dict.string(target_tokens)
                tgt_words = post_process(tgt_pieces, cfg.post_process)

            if res_files is not None:
                to_write[res_files["ref.units"]] = tgt_pieces
                to_write[res_files["ref.words"]] = tgt_words

        if not cfg.fairseq.common_eval.quiet:
            logger.info(f"HYPO {i}:" + hyp_words)
            if tgt_words:
                logger.info("TARGET:" + tgt_words)

            logger.info("___________________")

        hyp_words_arr = hyp_words.split()
        tgt_words_arr = tgt_words.split()

        retval.append(
            (
                editdistance.eval(hyp_words_arr, tgt_words_arr),
                len(hyp_words_arr),
                len(tgt_words_arr),
                hyp_pieces,
                hyp_words,
            )
        )
        word_preds.append(hyp_words_arr)
        transcriptions.append(to_write)
        dec_scores.append(-hypo.get("score", 0))  # negate cuz kaldi returns NLL

    if len(retval) > 1:
        best = None
        for r, t in zip(retval, transcriptions):
            if best is None or r[0] < best[0][0]:
                best = r, t
        for dest, tran in best[1].items():
            print(tran, file=dest)
            dest.flush()
        return best[0]

    assert len(transcriptions) == 1
    for dest, tran in transcriptions[0].items():
        print(tran, file=dest)

    return retval[0]


GenResult = namedtuple(
    "GenResult",
    [
        "count",
        "errs_t",
        "lengths_hyp_unit_t",
        "lengths_hyp_t",
        "lengths_t",
        "num_sentences",
        "num_symbols",
    ],
)


def generate(cfg: UnsupGenerateConfig, test_loader, model, use_cuda):
    num_sentences = 0
    res_files = prepare_result_files(cfg)
    errs_t = 0
    lengths_hyp_t = 0
    lengths_hyp_unit_t = 0
    lengths_t = 0
    count = 0

    targets = None
    tgt_dict = test_loader.dataset.label_dict
    num_symbols = (
        len([s for s in tgt_dict.symbols if not s.startswith("madeup")])
        - tgt_dict.nspecial
    )
    if "<SIL>" in tgt_dict:
        sil_id = tgt_dict.index("<SIL>")
    else:
        sil_id = tgt_dict.index("sil")

    start = 0
    end = len(test_loader)

    for i, sample in enumerate(test_loader):
        if i < start or i >= end:
            continue
        
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        res = model(
            **sample["net_input"],
            dense_x_only=True,
        )
        dense_x = res["logits"]
        padding_mask = res["padding_mask"]
        z = dense_x.argmax(-1)
        z[padding_mask] = tgt_dict.pad()

        to_write = {}
        for i, sample_id in enumerate(sample["id"].tolist()):
            toks = (
                sample["target"][i, :]
                if "target_label" not in sample
                else sample["target_label"][i, :]
            )

            target_tokens = utils.strip_pad(toks, tgt_dict.pad()).int().cpu()

            x = z[i].detach()
            x = x[
                (x >= tgt_dict.nspecial)
                & (x < (num_symbols + tgt_dict.nspecial))
            ]
            if sil_id >= 0:
                x = x[x != sil_id]
            
            # Write output, evaluate results
            hypos = {"tokens": x}
            errs, length_hyp, length, hyp_pieces, hyp_words = process_predictions(
                cfg, [hypos], tgt_dict, target_tokens, res_files,
            )


            errs_t += errs
            lengths_hyp_t += length_hyp
            lengths_hyp_unit_t += (
                len(hyp_pieces) if len(hyp_pieces) > 0 else len(hyp_words)
            )
            lengths_t += length
            count += 1
        
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    return GenResult(
        count,
        errs_t,
        lengths_hyp_unit_t,
        lengths_hyp_t,
        lengths_t,
        num_sentences,
        num_symbols,
    )


def main(cfg: UnsupGenerateConfig, model=None):
    use_cuda = torch.cuda.is_available() and not cfg.fairseq.common.cpu 
    device = 'cuda' if use_cuda else 'cpu'

    testset = EvalExtractedDataset( 
        path=cfg.fairseq.task.data,
        split=cfg.fairseq.dataset.gen_subset,
        dict_path=cfg.dict_path,
        centroid_path=cfg.centroid_path,
        labels=cfg.fairseq.task.labels,
        shuffle=False,
        sort_by_length=False,
    )

    test_loader = DataLoader(
        testset,
        batch_size=1,
        collate_fn=testset.collater,
        shuffle=False,
    )

    cfg.espum.input_dim = testset.n_clus
    model = ESPUM(cfg.espum, testset.label_dict)
    model.load_state_dict(torch.load(cfg.fairseq.common_eval.path)['model'], strict=False)
    model = model.to(device)

    gen_result = generate(cfg, test_loader, model, use_cuda)
    
    wer = None
    if gen_result.lengths_t > 0:
        wer = gen_result.errs_t * 100 / gen_result.lengths_t
        logger.info(f'WER: {wer}')

    res = (
        f"| Generate {cfg.fairseq.dataset.gen_subset}, "
        f"WER: {wer}, length: {gen_result.lengths_hyp_t}"
    )
    logger.info(res)


@hydra.main(
    config_path=os.path.join("./", "config"), config_name="config"
)
def hydra_main(cfg):
    with open_dict(cfg):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        cfg.job_logging_cfg = OmegaConf.to_container(
            HydraConfig.get().job_logging, resolve=True
        )

    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False, enum_to_str=False)
    )
    OmegaConf.set_struct(cfg, True)
    logger.info(cfg)

    utils.import_user_module(cfg.fairseq.common)

    main(cfg)

    
def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=UnsupGenerateConfig)
    hydra_main()


if __name__ == '__main__':
    cli_main()
