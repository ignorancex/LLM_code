"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

if "library_root" not in registry.mapping["paths"]:
    registry.register_path("library_root", root_dir)

repo_root = os.path.join(root_dir, "..")
if "repo_root" not in registry.mapping["paths"]:
    registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
if "cache_root" not in registry.mapping["paths"]:
    registry.register_path("cache_root", cache_root)

# 注册其他项时，检查是否已经注册
if "MAX_INT" not in registry.mapping:
    registry.register("MAX_INT", sys.maxsize)

if "SPLIT_NAMES" not in registry.mapping:
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

# registry.register_path("library_root", root_dir)
# repo_root = os.path.join(root_dir, "..")
# registry.register_path("repo_root", repo_root)
# cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
# registry.register_path("cache_root", cache_root)

# registry.register("MAX_INT", sys.maxsize)
# registry.register("SPLIT_NAMES", ["train", "val", "test"])
