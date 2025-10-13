import os
import shutil
import logging
import warnings
import transformers
from pathlib import Path
import datetime

import hydra
from omegaconf import DictConfig

from utils import retrieve_newsemp_file_names, log_info

from paired_texts_modelling import PairedTextModelController
from consistency_modelling import TwoModelsController

logger = logging.getLogger(__name__)

import torch
# MI250X GPU has Tensor core, so recommeded to use high or medium precision
# as opposed to highest precision (default) for faster computation
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="../config", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig):
    transformers.logging.set_verbosity_error()
    logger.info(f"Experiment name: {cfg.expt}")
    logger.info(f"Config: {cfg}")

    parent_log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    expt_name = hydra.core.hydra_config.HydraConfig.get().job.name
    
    # things coming from the config
    seeds = cfg.seeds
    lr = cfg.lr
    train_bsz = cfg.train_bsz
    eval_bsz = cfg.eval_bsz
    delta = cfg.expt.delta
    n_trials = cfg.expt.n_trials
    error_decay_factor = cfg.expt.error_decay_factor
    do_tune = cfg.expt.do_tune
    do_train = cfg.expt.do_train
    overwrite_parent_dir = cfg.expt.overwrite_parent_dir
    approach = cfg.expt.approach
    main_data = cfg.expt.main_data

    is_two_models = cfg.expt.is_two_models

    if approach == "cross-basic":
        plm_names = ["roberta-base"]
    elif approach == "siamese":
        plm_names = ["facebook/bart-base", "facebook/bart-base"]
    elif approach == "bi-prob":
        plm_names = ["roberta-base", "roberta-base"]
    elif approach == "cross-prob":
        if is_two_models:
            plm_names = ["roberta-base", "roberta-base"]
            if plm_names[0] != "roberta-base" or plm_names[1] != "roberta-base":
                warnings.warn("Between-text loss is hardcoded for roberta-base")
        else:
            plm_names = ["roberta-base"]
    else:
        raise ValueError(f"Unknown approach: {approach}")

    newsemp_train_files, newsemp_val_files, newsemp_test_files = retrieve_newsemp_file_names(cfg)
    empstories_train_files = ["data/EmpathicStories/PAIRS (train).csv"]
    empstories_val_files = ["data/EmpathicStories/PAIRS (dev).csv"]
    empstories_test_files = ["data/EmpathicStories/PAIRS (test).csv"]
    if main_data == "newsemp":
        labelled_train_files = newsemp_train_files
        val_files = newsemp_val_files
        test_files = newsemp_test_files
    elif main_data == "empstories":
        labelled_train_files = empstories_train_files
        val_files = empstories_val_files
        test_files = empstories_test_files
    else:
        raise ValueError("main_data should be either newsemp or empstories")
    log_info(logger, f"Train data: {labelled_train_files}\tVal data: {val_files}\tTest data: {test_files}")

    if not do_tune and not do_train and (overwrite_parent_dir is None):
        raise ValueError("Assuming you want to test only, please provide the overwrite_parent_dir")

    if overwrite_parent_dir is not None:
        log_info(logger, f"Using overwrite_logging_dir {overwrite_parent_dir}")
        assert os.path.isdir(overwrite_parent_dir), f"{overwrite_parent_dir} is not a directory \
            Note it must be a **parent** diretory like outputs/yyyy-mm-dd/hh-mm-ss_xx"
        log_info(logger, "If you are resuming training, MAKE SURE you manually DELETE any last directory, for which training was partially completed.")
        current_run_log_dir = parent_log_dir
        parent_log_dir = os.path.normpath(overwrite_parent_dir) # normpath to remove trailing slashes if any
        expt_name = os.path.basename(parent_log_dir)
        expt_name = expt_name[9:] # remove the hh-mm-ss_ prefix

    debug = False
    if "debug" in expt_name.lower():
        debug = True
        log_info(logger, "Debug mode")
        logger.setLevel(logging.DEBUG)
        seeds = seeds[:2] # reduce the number of seeds for debugging
        cfg.max_steps = 50
        cfg.val_check_interval = 5
        n_trials = 2

    common_kwargs = dict(
        labelled_train_files=labelled_train_files,
        val_files=val_files,
        test_files=test_files,
        lr=lr,
        train_bsz=train_bsz,
        eval_bsz=eval_bsz,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        noise_level=cfg.expt.noise_level,
        delta=delta,
        expt_name=expt_name,
        debug=debug,
        do_tune=do_tune,
        do_train=do_train,
        do_test=cfg.expt.do_test,
        error_decay_factor=error_decay_factor,
        lambdas=cfg.expt.lambdas,
        approach=approach,
        main_data=main_data,
        lbl_split=cfg.expt.lbl_split,
        plm_names=plm_names,
        num_passes=cfg.expt.num_passes,
        sanitise_labels=cfg.expt.sanitise_labels,
        add_noise_train=cfg.expt.add_noise_train,
        add_noise_test=cfg.expt.add_noise_test,
        do_augment=cfg.expt.do_augment
    )

    if is_two_models:
        modelling = TwoModelsController(
            is_ucvme=cfg.expt.is_ucvme,
            **common_kwargs
        )
    else:
        if approach != "cross-basic" and do_train:
            assert len(cfg.expt.lambdas) == 3, "Number of lambdas must be 3 for cross-prob" 
        
        modelling = PairedTextModelController(**common_kwargs)

    modelling.tune_train_test(
        n_trials=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )

    # clean-up
    if overwrite_parent_dir is not None:
        time_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # move the log file
        log_file_to_move = list(Path(current_run_log_dir).glob("*.log"))[0]
        new_name = f"new-run_{time_tag}.log"
        shutil.move(log_file_to_move, os.path.join(parent_log_dir, new_name))
        
        # move the .hydra directory
        new_name = f".hydra_{time_tag}"
        shutil.move(os.path.join(current_run_log_dir, ".hydra"), os.path.join(parent_log_dir, new_name))
        
        # delete the log dir
        shutil.rmtree(current_run_log_dir) # because results are saved in earlier log dir
        log_info(logger, f"Deleted {current_run_log_dir}")
    if debug:
        shutil.rmtree(parent_log_dir)
        log_info(logger, f"Deleted {parent_log_dir}")

if __name__ == "__main__":
    main()
