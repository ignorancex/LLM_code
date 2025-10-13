# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
import random
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="pkg://scripts/config", config_name="config.yaml", version_base="1.3")
def parse(conf: DictConfig) -> Any:
    # Set environment variable if specified.
    # This is broken in hydra because of issues with variable interpolations.
    # See: https://github.com/facebookresearch/hydra/issues/2800
    if "env" in conf:
        for name, value in conf.env.items():
            os.environ[name] = value

    # Set up the random seeds
    torch.manual_seed(conf.experiment.seed)
    random.seed(conf.experiment.seed)
    np.random.seed(conf.experiment.seed)

    log.info(f"Running experiment {conf.experiment.run_id}")

    run = instantiate(conf.experiment.script, _partial_=True)
    return_values = run(conf)

    return return_values


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    parse()
