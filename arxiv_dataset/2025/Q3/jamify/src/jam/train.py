# Copyright (c) 2025 Project Jamify
#               2025 Declare-Lab
#               2025 AMAAI Lab
#               2025 Renhang Liu (liurenhang0@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from jam.model import CFM, DiT
from jam.trainer import WebDatasetTrainer

from omegaconf import OmegaConf
import sys
import os

os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"

def main():
    cfg_cli = OmegaConf.from_dotlist(sys.argv[1:])
    config_path = cfg_cli.get("config", "configs/default.yaml")
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, cfg_cli)

    model = CFM(
        transformer=DiT(**cfg.model.dit),
        **cfg.model.cfm
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainer = WebDatasetTrainer(
        model=model,
        data_cfg=cfg.data,
        wandb_cfg=cfg.wandb,
        **cfg.training,
        bnb_optimizer=False,
        config=cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()
