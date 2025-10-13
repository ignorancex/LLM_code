# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os

from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
from tta.tafas import build_adapter
import tta.petsa as petsa
from config import get_norm_module_cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, cfg.TRAIN.CHECKPOINT_DIR.split('./checkpoints/')[-1])
    
    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR)


    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)


    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()
        
    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    if cfg.TTA.ENABLE:


        if 'TAFAS' in cfg.RESULT_DIR:
            print("TAFAS")
            adapter = build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'PETSA' in cfg.RESULT_DIR:
            print("PETSA")
            adapter = petsa.build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
            
            
    if cfg.TEST.ENABLE:
        predictor = Predictor(cfg, model, norm_module=norm_module)
        predictor.predict()


if __name__ == '__main__':
    main()
