#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is copied from
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.4.0/scripts/finetune.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ipex_llm import llm_patch
llm_patch(train=True)
# The following is the original axolotl finetune code (without IPEX-LLM)

"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""
import logging
from pathlib import Path

import fire
import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    do_inference,
    do_merge_lora,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.cli.shard import shard
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train

LOG = logging.getLogger("axolotl.scripts.finetune")


def do_cli(config: Path = Path("examples/"), **kwargs):
    print_axolotl_text_art()
    LOG.warning(
        str(
            PendingDeprecationWarning(
                "scripts/finetune.py will be replaced with calling axolotl.cli.train"
            )
        )
    )
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    if parsed_cli_args.inference:
        do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)
    elif parsed_cli_args.merge_lora:
        do_merge_lora(cfg=parsed_cfg, cli_args=parsed_cli_args)
    elif parsed_cli_args.shard:
        shard(cfg=parsed_cfg, cli_args=parsed_cli_args)
    else:
        dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
        train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
