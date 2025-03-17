# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import warnings

from typing import Optional, Sequence

import numpy as np

from .expert import get_feature_extractor_parser, UpstreamExpert


def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert audio files to representation files",
        parents=[get_feature_extractor_parser()],
    )
    parser.add_argument(
        "ckpt", type=argparse.FileType("rb"), help="Path to pre-trained model"
    )
    parser.add_argument(
        "out", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )
    options = parser.parse_args(args)

    expert = UpstreamExpert(options.ckpt, options=options)

    out = options.out
    out.write(f"feat_size {expert.encoder.input_size}\n")
    out.write(f"output_size {expert.encoder.output_size}\n")
    out.write(f"downsampling_factor {expert.get_downsample_rates('')}\n")
    out.write(
        f"num_params {sum(np.prod(p.shape) for p in expert.encoder.parameters())}\n"
    )
